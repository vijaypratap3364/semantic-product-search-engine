"""Tests for the unified, artifact-backed search service."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from pandas import DataFrame

from product_search import service as service_module
from product_search.config import ProjectSettings, load_settings
from product_search.data.download import sha256_file
from product_search.indexing.dense import build_dense_index
from product_search.indexing.tfidf import build_tfidf_index
from product_search.ranking.features import FEATURE_NAMES
from product_search.ranking.model import save_relevance_model, train_relevance_model
from product_search.retrieval.base import SearchResult
from product_search.retrieval.lexical import LexicalSearchEngine
from product_search.retrieval.semantic import SemanticSearchEngine
from product_search.service import (
    ProductDisplayRecord,
    ProductSearchResult,
    SearchExplanation,
    SearchModeUnavailableError,
    SearchResponse,
    SearchService,
    SearchServiceStartupError,
    ServiceBenchmark,
)


class FakeEmbeddingProvider:
    model_name = "BAAI/bge-small-en-v1.5"
    provider_name = "fake"
    provider_version = "1.0"

    def __init__(self, vectors: dict[str, NDArray[np.float32]]) -> None:
        self.vectors = vectors

    def embed_documents(
        self, texts: Sequence[str], *, batch_size: int
    ) -> Iterable[NDArray[np.float32]]:
        return (self.vectors[text] for text in texts)

    def embed_queries(
        self, texts: Sequence[str], *, batch_size: int
    ) -> Iterable[NDArray[np.float32]]:
        return (self.vectors[text] for text in texts)


class StepClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.001
        return self.value


def _vector(index: int) -> NDArray[np.float32]:
    vector = np.zeros(384, dtype=np.float32)
    vector[index] = 1.0
    return vector


def _write_selection(
    settings: ProjectSettings,
    *,
    selected_mode: str,
    include_reranker: bool,
    hybrid_strategy: str = "weighted_normalized",
) -> Path:
    products_path = settings.paths.processed_data / "products.parquet"
    lexical_metadata = settings.paths.indexes / "tfidf" / "metadata.json"
    dense_metadata = settings.paths.embeddings / "dense" / "metadata.json"
    components: dict[str, object] = {
        "lexical": {
            "metadata_sha256": sha256_file(lexical_metadata),
            "dataset_sha256": sha256_file(products_path),
        },
        "semantic": {
            "metadata_sha256": sha256_file(dense_metadata),
            "dataset_sha256": sha256_file(products_path),
            "model_name": "BAAI/bge-small-en-v1.5",
            "embedding_dimension": 384,
        },
        "hybrid": {
            "strategy": hybrid_strategy,
            "semantic_weight": 0.9,
            "candidate_depth": 100,
            "rrf_k": 60,
        },
    }
    if include_reranker:
        model_metadata = settings.paths.models / "reranker" / "metadata.json"
        components["reranked_hybrid"] = {
            "eligible": True,
            "candidate_depth": 100,
            "model_metadata_sha256": sha256_file(model_metadata),
        }
    immutable = {
        "components": components,
        "source_hashes": {"products.parquet": sha256_file(products_path)},
    }
    encoded = json.dumps(immutable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    system = "reranked_hybrid" if selected_mode == "reranker" else selected_mode
    payload = {
        "schema_version": 1,
        "system": system,
        "selected_search_mode": selected_mode,
        "immutable_configuration": immutable,
        "immutable_configuration_sha256": hashlib.sha256(encoded).hexdigest(),
    }
    selection_path = settings.paths.reports / "final_engine.json"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(json.dumps(payload), encoding="utf-8")
    return selection_path


def _build_fixture(
    tmp_path: Path,
    *,
    selected_mode: str = "hybrid",
    hybrid_strategy: str = "weighted_normalized",
) -> tuple[ProjectSettings, FakeEmbeddingProvider, Path]:
    settings = load_settings(project_root=tmp_path)
    products_path = settings.paths.processed_data / "products.parquet"
    products_path.parent.mkdir(parents=True, exist_ok=True)
    product_texts = {
        "p1": "coffee table round wood",
        "p2": "dining table square wood",
        "p3": "coffee desk lamp metal",
    }
    DataFrame(
        {
            "product_id": ["p3", "p1", "p2"],
            "product_name": ["Coffee Desk Lamp", "Round Coffee Table", "Square Dining Table"],
            "product_class": ["Lighting", "Tables", None],
            "category_hierarchy": ["Home/Lighting", "Home/Furniture/Tables", None],
            "product_description": [
                "Metal task lamp",
                "A" * 300,
                None,
            ],
            "product_text": [product_texts["p3"], product_texts["p1"], product_texts["p2"]],
        }
    ).to_parquet(products_path, index=False)
    build_tfidf_index(
        products_path,
        settings.paths.indexes / "tfidf",
        settings=settings.lexical,
    )
    provider = FakeEmbeddingProvider(
        {
            product_texts["p1"]: _vector(0),
            product_texts["p2"]: _vector(1),
            product_texts["p3"]: _vector(2),
            "coffee table": _vector(0),
            "desk lamp": _vector(2),
        }
    )
    build_dense_index(
        products_path,
        settings.paths.embeddings / "dense",
        provider=provider,
        settings=settings.dense,
    )
    if selected_mode == "reranker":
        rows: list[list[float]] = []
        labels: list[int] = []
        for grade in (0, 1, 2):
            for offset in (0.0, 0.05, 0.1, 0.15):
                rows.append([grade + offset] * len(FEATURE_NAMES))
                labels.append(grade)
        model = train_relevance_model(
            np.asarray(rows, dtype=np.float64),
            np.asarray(labels, dtype=np.int64),
            c_value=1.0,
            class_weight="balanced",
            max_iter=500,
            random_seed=42,
            candidate_depth=100,
        )
        save_relevance_model(
            model,
            settings.paths.models / "reranker",
            product_dataset_sha256=sha256_file(products_path),
            source_hashes={},
            train_query_count=3,
            training_row_count=len(rows),
            class_distribution={0: 4, 1: 4, 2: 4},
        )
    selection_path = _write_selection(
        settings,
        selected_mode=selected_mode,
        include_reranker=selected_mode == "reranker",
        hybrid_strategy=hybrid_strategy,
    )
    return settings, provider, selection_path


def test_service_loads_artifacts_once_and_exposes_display_safe_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, provider, _ = _build_fixture(tmp_path)
    counts = {"lexical": 0, "semantic": 0}
    original_lexical_load = LexicalSearchEngine.from_index_dir
    original_semantic_load = SemanticSearchEngine.from_index_dir

    def load_lexical(path: Path) -> LexicalSearchEngine:
        counts["lexical"] += 1
        return original_lexical_load(path)

    def load_semantic(
        path: Path,
        *,
        provider: FakeEmbeddingProvider,
        expected_dimension: int,
    ) -> SemanticSearchEngine:
        counts["semantic"] += 1
        return original_semantic_load(
            path,
            provider=provider,
            expected_dimension=expected_dimension,
        )

    monkeypatch.setattr(
        LexicalSearchEngine,
        "from_index_dir",
        staticmethod(load_lexical),
    )
    monkeypatch.setattr(
        SemanticSearchEngine,
        "from_index_dir",
        staticmethod(load_semantic),
    )
    service = SearchService.load(
        settings,
        embedding_provider=provider,
        clock=StepClock(),
    )

    assert service.default_mode == "hybrid"
    assert service.available_modes == ("default", "lexical", "semantic", "hybrid")
    assert service.product_count == 3
    assert service.initialization_time_ms == pytest.approx(1.0)
    default_response = service.search("coffee table", top_k=2)
    lexical_response = service.search("coffee table", top_k=2, mode="lexical")
    semantic_response = service.search("coffee table", top_k=2, mode="semantic")

    assert counts == {"lexical": 1, "semantic": 1}
    assert default_response.resolved_mode == "hybrid"
    assert default_response.latency_ms == pytest.approx(1.0)
    assert default_response.results[0].product_id == "p1"
    assert default_response.results[0].product_name == "Round Coffee Table"
    assert default_response.results[0].product_class == "Tables"
    assert default_response.results[0].category_hierarchy == "Home/Furniture/Tables"
    assert len(default_response.results[0].short_description or "") == 240
    assert default_response.results[0].lexical_score is not None
    assert default_response.results[0].semantic_score is not None
    assert default_response.results[0].explanation.matched_query_terms_in_title == (
        "coffee",
        "table",
    )
    assert default_response.results[0].explanation.lexical_contribution is not None
    assert default_response.results[0].explanation.semantic_contribution is not None
    assert not hasattr(default_response.results[0], "product_features")
    assert lexical_response.results[0].semantic_score is None
    assert semantic_response.results[0].lexical_score is None

    benchmark, final_response = service.benchmark("desk lamp", mode="semantic", top_k=1, runs=3)
    assert benchmark.sample_count == 3
    assert benchmark.warmup_count == 1
    assert benchmark.median_query_latency_ms == pytest.approx(1.0)
    assert benchmark.initialization_time_ms == pytest.approx(1.0)
    assert final_response.results[0].product_id == "p3"
    assert counts == {"lexical": 1, "semantic": 1}


def test_service_loads_selected_reranker_and_resolves_default(tmp_path: Path) -> None:
    settings, provider, _ = _build_fixture(tmp_path, selected_mode="reranker")

    service = SearchService.load(settings, embedding_provider=provider, clock=StepClock())
    response = service.search("coffee table", top_k=2, mode="default")

    assert service.default_mode == "rerank"
    assert service.available_modes[-1] == "rerank"
    assert response.resolved_mode == "rerank"
    assert len(response.results) == 2
    assert all(result.lexical_score is not None for result in response.results)
    assert all(result.semantic_score is not None for result in response.results)


def test_service_rejects_invalid_inputs_and_unavailable_reranker(tmp_path: Path) -> None:
    settings, provider, _ = _build_fixture(tmp_path)
    service = SearchService.load(settings, embedding_provider=provider)

    with pytest.raises(ValueError, match="must not be blank"):
        service.search(" ")
    with pytest.raises(TypeError, match="query must be a string"):
        service.search(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="between 1 and 100"):
        service.search("coffee", top_k=101)
    with pytest.raises(ValueError, match="between 1 and 100"):
        service.search("coffee", top_k=True)
    with pytest.raises(ValueError, match="unsupported search mode"):
        service.search("coffee", mode="other")
    with pytest.raises(TypeError, match="mode must be a string"):
        service.search("coffee", mode=None)  # type: ignore[arg-type]
    with pytest.raises(SearchModeUnavailableError, match="unavailable"):
        service.search("coffee", mode="rerank")
    with pytest.raises(ValueError, match="positive integer"):
        service.benchmark("coffee", runs=0)

    class MissingComponentsEngine:
        def search(self, query: str, top_k: int) -> list[SearchResult]:
            return [SearchResult(product_id="p1", rank=1, score=0.5, score_components={})]

    malformed_service = SearchService(
        products={"p1": ProductDisplayRecord("p1", "Table", None, None, None)},
        engines={"hybrid": MissingComponentsEngine()},
        default_mode="hybrid",
        fusion_strategy="weighted_normalized",
        semantic_weight=0.9,
        initialization_time_ms=1.0,
        selection_sha256="selection-hash",
    )
    with pytest.raises(SearchServiceStartupError, match="lacks required score component"):
        malformed_service.search("coffee")


def test_service_reports_rrf_rank_contributions(tmp_path: Path) -> None:
    settings, provider, _ = _build_fixture(tmp_path, hybrid_strategy="rrf")
    service = SearchService.load(settings, embedding_provider=provider)

    result = service.search("coffee table", top_k=1, mode="hybrid").results[0]

    assert result.explanation.lexical_contribution is not None
    assert result.explanation.semantic_contribution is not None
    assert result.final_score == pytest.approx(
        result.explanation.lexical_contribution + result.explanation.semantic_contribution
    )


def test_service_missing_artifact_error_names_build_command(tmp_path: Path) -> None:
    settings = load_settings(project_root=tmp_path)

    with pytest.raises(SearchServiceStartupError, match="benchmark_final"):
        SearchService.load(settings, embedding_provider=FakeEmbeddingProvider({}))


def test_service_rejects_tampered_selection_and_metadata(tmp_path: Path) -> None:
    settings, provider, selection_path = _build_fixture(tmp_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["immutable_configuration"]["components"]["hybrid"]["semantic_weight"] = 0.8
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    with pytest.raises(SearchServiceStartupError, match="configuration hash is invalid"):
        SearchService.load(settings, embedding_provider=provider)

    _write_selection(settings, selected_mode="hybrid", include_reranker=False)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["immutable_configuration"]["components"]["lexical"]["metadata_sha256"] = "0" * 64
    immutable = selection["immutable_configuration"]
    encoded = json.dumps(immutable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    selection["immutable_configuration_sha256"] = hashlib.sha256(encoded).hexdigest()
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    with pytest.raises(SearchServiceStartupError, match="lexical metadata does not match"):
        SearchService.load(settings, embedding_provider=provider)


def test_service_cli_reports_initialization_query_latency_and_safe_results(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = ProductSearchResult(
        product_id="p1",
        product_name="Table",
        product_class="Tables",
        category_hierarchy="Home/Tables",
        short_description="Wood table",
        rank=1,
        final_score=1.5,
        lexical_score=0.8,
        semantic_score=0.7,
        explanation=SearchExplanation(("table",), 0.08, 0.63),
    )
    response = SearchResponse("table", "default", "rerank", 1, 2.0, (result,))
    benchmark = ServiceBenchmark("table", "default", "rerank", 1, 1, 2, 50.0, 2.0, 2.0, 2.0, 1)

    class FakeService:
        default_mode = "rerank"
        available_modes = ("default", "lexical", "semantic", "hybrid", "rerank")
        product_count = 3
        selection_sha256 = "selection-hash"

        def benchmark(
            self, *args: object, **kwargs: object
        ) -> tuple[ServiceBenchmark, SearchResponse]:
            return benchmark, response

    monkeypatch.setattr(
        service_module.SearchService,
        "load",
        lambda *args, **kwargs: FakeService(),
    )

    assert (
        service_module.main(
            ["table", "--top-k", "1", "--benchmark-runs", "2", "--local-files-only"]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["benchmark"]["initialization_time_ms"] == 50.0
    assert payload["benchmark"]["median_query_latency_ms"] == 2.0
    assert payload["results"][0]["short_description"] == "Wood table"
    assert "product_features" not in payload["results"][0]
