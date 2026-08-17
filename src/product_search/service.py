"""Unified, artifact-verified product search service for future transport layers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from pandas import DataFrame

from product_search.config import ProjectSettings, load_settings
from product_search.data.download import sha256_file
from product_search.indexing.dense import EmbeddingProvider, FastEmbedProvider
from product_search.ranking.features import ProductFeatureStore
from product_search.ranking.model import load_relevance_model
from product_search.ranking.reranker import RerankingSearchEngine
from product_search.retrieval.base import SearchEngine, SearchResult
from product_search.retrieval.hybrid import FusionStrategy, HybridSearchEngine
from product_search.retrieval.lexical import LexicalSearchEngine
from product_search.retrieval.semantic import SemanticSearchEngine

SearchMode = Literal["lexical", "semantic", "hybrid", "rerank", "default"]
ResolvedSearchMode = Literal["lexical", "semantic", "hybrid", "rerank"]

MAX_TOP_K = 100
SHORT_DESCRIPTION_LENGTH = 240
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_SELECTION_TO_SERVICE_MODE: dict[str, ResolvedSearchMode] = {
    "lexical": "lexical",
    "semantic": "semantic",
    "hybrid": "hybrid",
    "reranker": "rerank",
}
_SYSTEM_TO_SELECTION_MODE = {
    "lexical": "lexical",
    "semantic": "semantic",
    "hybrid": "hybrid",
    "reranked_hybrid": "reranker",
}


class SearchServiceStartupError(RuntimeError):
    """Raised when required generated artifacts are missing or incompatible."""


class SearchModeUnavailableError(ValueError):
    """Raised when a valid mode was not loaded for this frozen service configuration."""


@dataclass(frozen=True, slots=True)
class ProductDisplayRecord:
    """Catalog fields safe and useful for a search response."""

    product_id: str
    product_name: str
    product_class: str | None
    category_hierarchy: str | None
    short_description: str | None


@dataclass(frozen=True, slots=True)
class SearchExplanation:
    """Deterministic score and text-overlap explanation without generative AI."""

    matched_query_terms_in_title: tuple[str, ...]
    lexical_contribution: float | None
    semantic_contribution: float | None


@dataclass(frozen=True, slots=True)
class ProductSearchResult:
    """One display-ready ranked product returned by the service."""

    product_id: str
    product_name: str
    product_class: str | None
    category_hierarchy: str | None
    short_description: str | None
    rank: int
    final_score: float
    lexical_score: float | None
    semantic_score: float | None
    explanation: SearchExplanation


@dataclass(frozen=True, slots=True)
class SearchResponse:
    """One timed service search and its resolved mode."""

    query: str
    requested_mode: str
    resolved_mode: ResolvedSearchMode
    top_k: int
    latency_ms: float
    results: tuple[ProductSearchResult, ...]


@dataclass(frozen=True, slots=True)
class ServiceBenchmark:
    """Warm-process latency summary for repeated calls through the service boundary."""

    query: str
    requested_mode: str
    resolved_mode: ResolvedSearchMode
    top_k: int
    warmup_count: int
    sample_count: int
    initialization_time_ms: float
    mean_query_latency_ms: float
    median_query_latency_ms: float
    p95_query_latency_ms: float
    result_count: int


class SearchService:
    """Load static artifacts once and expose every supported search mode consistently."""

    def __init__(
        self,
        *,
        products: Mapping[str, ProductDisplayRecord],
        engines: Mapping[ResolvedSearchMode, SearchEngine],
        default_mode: ResolvedSearchMode,
        fusion_strategy: FusionStrategy,
        semantic_weight: float,
        initialization_time_ms: float,
        selection_sha256: str,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if default_mode not in engines:
            raise SearchServiceStartupError(
                f"selected default mode {default_mode!r} was not loaded"
            )
        if not products:
            raise SearchServiceStartupError("the product display catalog is empty")
        self._products = dict(products)
        self._engines = dict(engines)
        self._default_mode = default_mode
        self._fusion_strategy = fusion_strategy
        self._semantic_weight = semantic_weight
        self._clock = clock
        self.initialization_time_ms = initialization_time_ms
        self.selection_sha256 = selection_sha256

    @classmethod
    def load(
        cls,
        settings: ProjectSettings | None = None,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        local_files_only: bool = True,
        clock: Callable[[], float] = time.perf_counter,
    ) -> SearchService:
        """Load and cross-check every static artifact exactly once."""

        started_at = clock()
        resolved_settings = settings or load_settings()
        products_path = resolved_settings.paths.processed_data / "products.parquet"
        lexical_dir = resolved_settings.paths.indexes / "tfidf"
        dense_dir = resolved_settings.paths.embeddings / "dense"
        model_dir = resolved_settings.paths.models / "reranker"
        selection_path = resolved_settings.paths.reports / "final_engine.json"

        _require_generated_file(
            selection_path,
            command=(
                "uv run python -m product_search.evaluation.benchmark_final --local-files-only"
            ),
            artifact_name="final engine selection",
        )
        selection = _read_selection(selection_path)
        default_mode = _selected_mode(selection)
        components = _selection_components(selection)
        reranker_selected = default_mode == "rerank"

        _require_generated_file(
            products_path,
            command="uv run python -m product_search.data.prepare",
            artifact_name="processed product catalog",
        )
        product_hash = sha256_file(products_path)
        _validate_selected_source_hash(selection, "products.parquet", product_hash)
        products_frame, display_records = _load_products(
            products_path, include_product_text=reranker_selected
        )
        ordered_product_ids = tuple(sorted(display_records))

        lexical_metadata_path = lexical_dir / "metadata.json"
        _require_generated_file(
            lexical_metadata_path,
            command="uv run python -m product_search.indexing.build_lexical",
            artifact_name="TF-IDF index",
        )
        _validate_metadata_hash(components, "lexical", lexical_metadata_path)
        try:
            lexical = LexicalSearchEngine.from_index_dir(lexical_dir)
        except Exception as error:
            raise SearchServiceStartupError(
                "unable to load the TF-IDF index; run "
                "`uv run python -m product_search.indexing.build_lexical`: "
                f"{type(error).__name__}: {error}"
            ) from error

        dense_metadata_path = dense_dir / "metadata.json"
        _require_generated_file(
            dense_metadata_path,
            command=("uv run python -m product_search.indexing.build_dense"),
            artifact_name="dense embedding index",
        )
        semantic_component = _component(components, "semantic")
        _validate_metadata_hash(components, "semantic", dense_metadata_path)
        model_name = _required_string(semantic_component, "model_name", owner="semantic")
        expected_dimension = _required_int(
            semantic_component, "embedding_dimension", owner="semantic"
        )
        try:
            provider = embedding_provider or FastEmbedProvider(
                model_name,
                cache_dir=resolved_settings.paths.embeddings / "model_cache",
                local_files_only=local_files_only,
            )
            semantic = SemanticSearchEngine.from_index_dir(
                dense_dir,
                provider=provider,
                expected_dimension=expected_dimension,
            )
        except Exception as error:
            raise SearchServiceStartupError(
                "unable to load the dense index/model; run "
                "`uv run python -m product_search.indexing.build_dense`: "
                f"{type(error).__name__}: {error}"
            ) from error

        _validate_engine_compatibility(
            product_hash=product_hash,
            ordered_product_ids=ordered_product_ids,
            lexical=lexical,
            semantic=semantic,
        )
        hybrid_component = _component(components, "hybrid")
        strategy = _required_string(hybrid_component, "strategy", owner="hybrid")
        if strategy not in {"weighted_normalized", "rrf"}:
            raise SearchServiceStartupError(f"unsupported frozen hybrid strategy: {strategy!r}")
        semantic_weight = _required_float(hybrid_component, "semantic_weight", owner="hybrid")
        candidate_depth = _required_int(hybrid_component, "candidate_depth", owner="hybrid")
        rrf_k = _required_int(hybrid_component, "rrf_k", owner="hybrid")
        hybrid = HybridSearchEngine(
            lexical,
            semantic,
            strategy=cast(FusionStrategy, strategy),
            semantic_weight=semantic_weight,
            candidate_depth=candidate_depth,
            rrf_k=rrf_k,
        )
        engines: dict[ResolvedSearchMode, SearchEngine] = {
            "lexical": lexical,
            "semantic": semantic,
            "hybrid": hybrid,
        }

        if reranker_selected:
            reranker_component = _component(components, "reranked_hybrid")
            if reranker_component.get("eligible") is not True:
                raise SearchServiceStartupError(
                    "final selection requests reranking but validation eligibility is false"
                )
            model_metadata_path = model_dir / "metadata.json"
            _require_generated_file(
                model_metadata_path,
                command=(
                    "uv run python -m product_search.evaluation.benchmark_reranker "
                    "--local-files-only"
                ),
                artifact_name="reranker model",
            )
            expected_model_metadata_hash = _required_string(
                reranker_component,
                "model_metadata_sha256",
                owner="reranked_hybrid",
            )
            if sha256_file(model_metadata_path) != expected_model_metadata_hash:
                raise SearchServiceStartupError(
                    "reranker metadata does not match the immutable final selection"
                )
            product_store = ProductFeatureStore.from_frame(
                products_frame,
                dataset_sha256=product_hash,
            )
            reranker_depth = _required_int(
                reranker_component, "candidate_depth", owner="reranked_hybrid"
            )
            try:
                model = load_relevance_model(
                    model_dir,
                    expected_product_dataset_sha256=product_hash,
                    expected_candidate_depth=reranker_depth,
                )
            except Exception as error:
                raise SearchServiceStartupError(
                    "unable to load the selected reranker; run "
                    "`uv run python -m product_search.evaluation.benchmark_reranker "
                    "--local-files-only`: "
                    f"{type(error).__name__}: {error}"
                ) from error
            engines["rerank"] = RerankingSearchEngine(
                hybrid,
                model,
                product_store,
                candidate_depth=reranker_depth,
            )

        initialization_time_ms = max(0.0, (clock() - started_at) * 1000.0)
        return cls(
            products=display_records,
            engines=engines,
            default_mode=default_mode,
            fusion_strategy=cast(FusionStrategy, strategy),
            semantic_weight=semantic_weight,
            initialization_time_ms=initialization_time_ms,
            selection_sha256=sha256_file(selection_path),
            clock=clock,
        )

    @property
    def default_mode(self) -> ResolvedSearchMode:
        """Return the frozen Stage 8 default mode."""

        return self._default_mode

    @property
    def available_modes(self) -> tuple[str, ...]:
        """Return accepted modes in stable user-facing order."""

        ordered = tuple(
            mode for mode in ("lexical", "semantic", "hybrid", "rerank") if mode in self._engines
        )
        return ("default", *ordered)

    @property
    def product_count(self) -> int:
        return len(self._products)

    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: str = "default",
    ) -> SearchResponse:
        """Search through one loaded engine and enrich results with safe catalog fields."""

        _validate_search_input(query, top_k)
        resolved_mode = self._resolve_mode(mode)
        started_at = self._clock()
        raw_results = self._engines[resolved_mode].search(query, top_k)
        enriched = tuple(
            self._enrich_result(query, result, resolved_mode) for result in raw_results
        )
        latency_ms = max(0.0, (self._clock() - started_at) * 1000.0)
        return SearchResponse(
            query=query,
            requested_mode=mode,
            resolved_mode=resolved_mode,
            top_k=top_k,
            latency_ms=latency_ms,
            results=enriched,
        )

    def benchmark(
        self,
        query: str,
        *,
        top_k: int = 10,
        mode: str = "default",
        runs: int = 10,
    ) -> tuple[ServiceBenchmark, SearchResponse]:
        """Measure repeated warm-process calls without reloading any artifact."""

        if isinstance(runs, bool) or not isinstance(runs, int) or runs <= 0:
            raise ValueError("runs must be a positive integer")
        self.search(query, top_k=top_k, mode=mode)
        responses = tuple(self.search(query, top_k=top_k, mode=mode) for _ in range(runs))
        latencies = np.asarray([response.latency_ms for response in responses], dtype=np.float64)
        final_response = responses[-1]
        return (
            ServiceBenchmark(
                query=query,
                requested_mode=mode,
                resolved_mode=final_response.resolved_mode,
                top_k=top_k,
                warmup_count=1,
                sample_count=runs,
                initialization_time_ms=self.initialization_time_ms,
                mean_query_latency_ms=float(np.mean(latencies)),
                median_query_latency_ms=float(np.median(latencies)),
                p95_query_latency_ms=float(np.percentile(latencies, 95)),
                result_count=len(final_response.results),
            ),
            final_response,
        )

    def _resolve_mode(self, mode: str) -> ResolvedSearchMode:
        if not isinstance(mode, str):
            raise TypeError("mode must be a string")
        if mode == "default":
            return self._default_mode
        if mode not in {"lexical", "semantic", "hybrid", "rerank"}:
            raise ValueError(
                "unsupported search mode; choose default, lexical, semantic, hybrid, or rerank"
            )
        resolved = cast(ResolvedSearchMode, mode)
        if resolved not in self._engines:
            raise SearchModeUnavailableError(
                f"search mode {mode!r} is unavailable; loaded modes: {self.available_modes}"
            )
        return resolved

    def _enrich_result(
        self,
        query: str,
        result: SearchResult,
        mode: ResolvedSearchMode,
    ) -> ProductSearchResult:
        try:
            product = self._products[result.product_id]
        except KeyError as error:
            raise SearchServiceStartupError(
                f"search index returned unknown product ID {result.product_id!r}"
            ) from error
        lexical_score, semantic_score, lexical_contribution, semantic_contribution = (
            _score_breakdown(
                result,
                mode,
                fusion_strategy=self._fusion_strategy,
                semantic_weight=self._semantic_weight,
            )
        )
        return ProductSearchResult(
            product_id=product.product_id,
            product_name=product.product_name,
            product_class=product.product_class,
            category_hierarchy=product.category_hierarchy,
            short_description=product.short_description,
            rank=result.rank,
            final_score=result.score,
            lexical_score=lexical_score,
            semantic_score=semantic_score,
            explanation=SearchExplanation(
                matched_query_terms_in_title=_matched_query_terms(query, product.product_name),
                lexical_contribution=lexical_contribution,
                semantic_contribution=semantic_contribution,
            ),
        )


def _load_products(
    products_path: Path,
    *,
    include_product_text: bool,
) -> tuple[DataFrame, dict[str, ProductDisplayRecord]]:
    columns = [
        "product_id",
        "product_name",
        "product_class",
        "category_hierarchy",
        "product_description",
    ]
    if include_product_text:
        columns.append("product_text")
    try:
        products = pd.read_parquet(products_path, columns=columns)
    except Exception as error:
        raise SearchServiceStartupError(
            "unable to load the processed product catalog; run "
            "`uv run python -m product_search.data.prepare`: "
            f"{type(error).__name__}: {error}"
        ) from error
    if products.empty:
        raise SearchServiceStartupError("processed product catalog is empty")
    if products[["product_id", "product_name"]].isna().any(axis=None):
        raise SearchServiceStartupError("product IDs and names must not be missing")
    product_ids = products["product_id"].astype(str)
    product_names = products["product_name"].astype(str)
    if product_ids.str.strip().eq("").any() or product_ids.duplicated().any():
        raise SearchServiceStartupError("product IDs must be non-blank and unique")
    if product_names.str.strip().eq("").any():
        raise SearchServiceStartupError("product names must be non-blank")
    normalized = products.assign(product_id=product_ids, product_name=product_names)
    records = {
        str(row["product_id"]): ProductDisplayRecord(
            product_id=str(row["product_id"]),
            product_name=str(row["product_name"]),
            product_class=_optional_text(row["product_class"]),
            category_hierarchy=_optional_text(row["category_hierarchy"]),
            short_description=_short_description(row["product_description"]),
        )
        for row in cast(list[dict[str, object]], normalized.to_dict(orient="records"))
    }
    return normalized, records


def _score_breakdown(
    result: SearchResult,
    mode: ResolvedSearchMode,
    *,
    fusion_strategy: FusionStrategy,
    semantic_weight: float,
) -> tuple[float | None, float | None, float | None, float | None]:
    components = result.score_components or {}
    if mode == "lexical":
        score = float(components.get("lexical", result.score))
        return score, None, score, None
    if mode == "semantic":
        score = float(components.get("semantic", result.score))
        return None, score, None, score
    lexical_score = _required_component(components, "lexical_raw")
    semantic_score = _required_component(components, "semantic_raw")
    lexical_normalized = _required_component(components, "lexical_normalized")
    semantic_normalized = _required_component(components, "semantic_normalized")
    if fusion_strategy == "rrf":
        return (
            lexical_score,
            semantic_score,
            0.5 * _required_component(components, "lexical_rrf"),
            0.5 * _required_component(components, "semantic_rrf"),
        )
    return (
        lexical_score,
        semantic_score,
        (1.0 - semantic_weight) * lexical_normalized,
        semantic_weight * semantic_normalized,
    )


def _matched_query_terms(query: str, product_name: str) -> tuple[str, ...]:
    title_tokens = set(_TOKEN_PATTERN.findall(product_name.lower()))
    matched: list[str] = []
    seen: set[str] = set()
    for token in _TOKEN_PATTERN.findall(query.lower()):
        if token in title_tokens and token not in seen:
            matched.append(token)
            seen.add(token)
    return tuple(matched)


def _short_description(value: object) -> str | None:
    normalized = _optional_text(value)
    if normalized is None or len(normalized) <= SHORT_DESCRIPTION_LENGTH:
        return normalized
    return f"{normalized[: SHORT_DESCRIPTION_LENGTH - 3].rstrip()}..."


def _optional_text(value: object) -> str | None:
    if value is None or bool(pd.isna(cast(Any, value))):
        return None
    normalized = " ".join(str(value).split())
    return normalized or None


def _validate_search_input(query: str, top_k: int) -> None:
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if not query.strip():
        raise ValueError("query must not be blank")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be an integer between 1 and {MAX_TOP_K}")


def _read_selection(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as input_file:
            selection = cast(dict[str, Any], json.load(input_file))
    except (OSError, json.JSONDecodeError) as error:
        raise SearchServiceStartupError(
            f"unable to read final engine selection at {path}: {error}"
        ) from error
    if selection.get("schema_version") != 1:
        raise SearchServiceStartupError("unsupported final engine selection schema")
    immutable = selection.get("immutable_configuration")
    if not isinstance(immutable, dict):
        raise SearchServiceStartupError("final engine selection lacks immutable_configuration")
    encoded = json.dumps(immutable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    actual_hash = hashlib.sha256(encoded).hexdigest()
    if selection.get("immutable_configuration_sha256") != actual_hash:
        raise SearchServiceStartupError("final engine immutable configuration hash is invalid")
    return selection


def _selected_mode(selection: Mapping[str, Any]) -> ResolvedSearchMode:
    selected = selection.get("selected_search_mode")
    system = selection.get("system")
    expected = _SYSTEM_TO_SELECTION_MODE.get(str(system))
    if not isinstance(selected, str) or selected not in _SELECTION_TO_SERVICE_MODE:
        raise SearchServiceStartupError(f"unsupported selected search mode: {selected!r}")
    if selected != expected:
        raise SearchServiceStartupError(
            "final engine system and selected_search_mode are inconsistent"
        )
    return _SELECTION_TO_SERVICE_MODE[selected]


def _selection_components(selection: Mapping[str, Any]) -> Mapping[str, Any]:
    immutable = _mapping(selection.get("immutable_configuration"), "immutable_configuration")
    components = _mapping(immutable.get("components"), "immutable_configuration.components")
    missing = {"lexical", "semantic", "hybrid"} - set(components)
    if missing:
        raise SearchServiceStartupError(
            f"final selection is missing required components: {sorted(missing)}"
        )
    return components


def _validate_selected_source_hash(
    selection: Mapping[str, Any], filename: str, actual_hash: str
) -> None:
    immutable = _mapping(selection.get("immutable_configuration"), "immutable_configuration")
    source_hashes = _mapping(immutable.get("source_hashes"), "immutable source_hashes")
    if source_hashes.get(filename) != actual_hash:
        raise SearchServiceStartupError(
            f"{filename} does not match the immutable final engine selection"
        )


def _validate_metadata_hash(
    components: Mapping[str, Any], component_name: str, metadata_path: Path
) -> None:
    component = _component(components, component_name)
    expected = _required_string(component, "metadata_sha256", owner=component_name)
    if sha256_file(metadata_path) != expected:
        raise SearchServiceStartupError(
            f"{component_name} metadata does not match the immutable final selection"
        )


def _validate_engine_compatibility(
    *,
    product_hash: str,
    ordered_product_ids: tuple[str, ...],
    lexical: LexicalSearchEngine,
    semantic: SemanticSearchEngine,
) -> None:
    if lexical.metadata["dataset_sha256"] != product_hash:
        raise SearchServiceStartupError("TF-IDF index was built from a different product dataset")
    if semantic.metadata["dataset_sha256"] != product_hash:
        raise SearchServiceStartupError("dense index was built from a different product dataset")
    if lexical.product_ids != semantic.product_ids:
        raise SearchServiceStartupError("lexical and dense product ID orderings are incompatible")
    if lexical.product_ids != ordered_product_ids:
        raise SearchServiceStartupError("search indexes and product catalog IDs are incompatible")


def _require_generated_file(path: Path, *, command: str, artifact_name: str) -> None:
    if not path.is_file():
        raise SearchServiceStartupError(
            f"missing {artifact_name}: {path}. Build it with `{command}`."
        )


def _component(components: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return _mapping(components.get(name), f"final selection component {name}")


def _mapping(value: object, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SearchServiceStartupError(f"{owner} must be a mapping")
    return cast(Mapping[str, Any], value)


def _required_string(component: Mapping[str, Any], key: str, *, owner: str) -> str:
    value = component.get(key)
    if not isinstance(value, str) or not value:
        raise SearchServiceStartupError(f"{owner}.{key} must be a non-empty string")
    return value


def _required_int(component: Mapping[str, Any], key: str, *, owner: str) -> int:
    value = component.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SearchServiceStartupError(f"{owner}.{key} must be a positive integer")
    return value


def _required_float(component: Mapping[str, Any], key: str, *, owner: str) -> float:
    value = component.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SearchServiceStartupError(f"{owner}.{key} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise SearchServiceStartupError(f"{owner}.{key} must be between 0 and 1")
    return normalized


def _required_component(components: Mapping[str, float], key: str) -> float:
    if key not in components:
        raise SearchServiceStartupError(f"search result lacks required score component {key!r}")
    return float(components[key])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Free-text product query.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--mode",
        default="default",
        choices=("default", "lexical", "semantic", "hybrid", "rerank"),
    )
    parser.add_argument("--benchmark-runs", type=int, default=1)
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the embedding model to be cached locally (default: true).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load the service once, benchmark a query, and print display-safe results."""

    arguments = _build_parser().parse_args(argv)
    service = SearchService.load(local_files_only=arguments.local_files_only)
    benchmark, response = service.benchmark(
        arguments.query,
        top_k=arguments.top_k,
        mode=arguments.mode,
        runs=arguments.benchmark_runs,
    )
    payload = {
        "service": {
            "default_mode": service.default_mode,
            "available_modes": service.available_modes,
            "product_count": service.product_count,
            "selection_sha256": service.selection_sha256,
        },
        "benchmark": asdict(benchmark),
        "results": [asdict(result) for result in response.results],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
