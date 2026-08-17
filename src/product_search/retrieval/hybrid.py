"""Deterministic fusion of lexical and semantic product rankings."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from product_search.config import load_settings
from product_search.indexing.dense import FastEmbedProvider
from product_search.retrieval.base import JudgedCandidateSearchEngine, SearchResult
from product_search.retrieval.lexical import LexicalSearchEngine
from product_search.retrieval.semantic import SemanticSearchEngine

FusionStrategy = Literal["weighted_normalized", "rrf"]
CONSTANT_SCORE_VALUE = 1.0
MISSING_MODALITY_VALUE = 0.0


class HybridSearchEngine:
    """Fuse a lexical engine and semantic engine without mixing raw score scales."""

    def __init__(
        self,
        lexical_engine: JudgedCandidateSearchEngine,
        semantic_engine: JudgedCandidateSearchEngine,
        *,
        strategy: FusionStrategy = "weighted_normalized",
        semantic_weight: float = 0.5,
        candidate_depth: int = 100,
        rrf_k: int = 60,
    ) -> None:
        _validate_fusion_settings(
            strategy=strategy,
            semantic_weight=semantic_weight,
            candidate_depth=candidate_depth,
            rrf_k=rrf_k,
        )
        self._lexical_engine = lexical_engine
        self._semantic_engine = semantic_engine
        self.strategy = strategy
        self.semantic_weight = semantic_weight
        self.candidate_depth = candidate_depth
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Fuse the union of bounded full-catalog lexical and semantic candidates."""

        _validate_top_k(top_k)
        depth = max(top_k, self.candidate_depth)
        lexical_results = self._lexical_engine.search(query, depth)
        semantic_results = self._semantic_engine.search(query, depth)
        candidate_ids = sorted(
            {result.product_id for result in lexical_results}
            | {result.product_id for result in semantic_results}
        )
        return fuse_rankings(
            lexical_results,
            semantic_results,
            candidate_product_ids=candidate_ids,
            top_k=top_k,
            strategy=self.strategy,
            semantic_weight=self.semantic_weight,
            rrf_k=self.rrf_k,
        )

    def search_candidates(
        self,
        query: str,
        candidate_product_ids: Sequence[str],
        top_k: int,
    ) -> list[SearchResult]:
        """Score every explicit candidate in each modality before deterministic fusion."""

        _validate_top_k(top_k)
        candidate_ids = [str(product_id) for product_id in candidate_product_ids]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate product IDs must be unique")
        if not candidate_ids:
            return []
        lexical_results = self._lexical_engine.search_candidates(
            query, candidate_ids, len(candidate_ids)
        )
        semantic_results = self._semantic_engine.search_candidates(
            query, candidate_ids, len(candidate_ids)
        )
        return fuse_rankings(
            lexical_results,
            semantic_results,
            candidate_product_ids=candidate_ids,
            top_k=top_k,
            strategy=self.strategy,
            semantic_weight=self.semantic_weight,
            rrf_k=self.rrf_k,
        )


def min_max_normalize(scores: Mapping[str, float]) -> dict[str, float]:
    """Normalize one query/modality to [0, 1] with a deterministic constant-set rule.

    When every available score is equal, every available candidate receives 1.0. This preserves
    equal membership evidence without inventing an ordering. Candidates absent from a modality are
    assigned the separate floor 0.0 by ``fuse_rankings``.
    """

    if not scores:
        return {}
    if any(not math.isfinite(score) for score in scores.values()):
        raise ValueError("fusion scores must be finite")
    minimum = min(scores.values())
    maximum = max(scores.values())
    if math.isclose(minimum, maximum, rel_tol=0.0, abs_tol=1e-12):
        return {product_id: CONSTANT_SCORE_VALUE for product_id in scores}
    scale = maximum - minimum
    return {product_id: (score - minimum) / scale for product_id, score in scores.items()}


def fuse_rankings(
    lexical_results: Sequence[SearchResult],
    semantic_results: Sequence[SearchResult],
    *,
    candidate_product_ids: Sequence[str],
    top_k: int,
    strategy: FusionStrategy,
    semantic_weight: float = 0.5,
    rrf_k: int = 60,
) -> list[SearchResult]:
    """Fuse comparable normalized scores or reciprocal-rank contributions."""

    _validate_top_k(top_k)
    _validate_fusion_settings(
        strategy=strategy,
        semantic_weight=semantic_weight,
        candidate_depth=1,
        rrf_k=rrf_k,
    )
    candidate_ids = [str(product_id) for product_id in candidate_product_ids]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate product IDs must be unique")
    if not candidate_ids:
        return []
    allowed_ids = set(candidate_ids)
    lexical_by_id = _validate_and_map_results(lexical_results, allowed_ids, modality="lexical")
    semantic_by_id = _validate_and_map_results(semantic_results, allowed_ids, modality="semantic")
    lexical_raw = {product_id: result.score for product_id, result in lexical_by_id.items()}
    semantic_raw = {product_id: result.score for product_id, result in semantic_by_id.items()}
    lexical_normalized = min_max_normalize(lexical_raw)
    semantic_normalized = min_max_normalize(semantic_raw)
    lexical_rank_scores = {
        product_id: 1.0 / (rrf_k + result.rank) for product_id, result in lexical_by_id.items()
    }
    semantic_rank_scores = {
        product_id: 1.0 / (rrf_k + result.rank) for product_id, result in semantic_by_id.items()
    }

    scored: list[tuple[str, float, dict[str, float]]] = []
    lexical_weight = 1.0 - semantic_weight
    for product_id in candidate_ids:
        lexical_value = lexical_normalized.get(product_id, MISSING_MODALITY_VALUE)
        semantic_value = semantic_normalized.get(product_id, MISSING_MODALITY_VALUE)
        lexical_rrf = lexical_rank_scores.get(product_id, MISSING_MODALITY_VALUE)
        semantic_rrf = semantic_rank_scores.get(product_id, MISSING_MODALITY_VALUE)
        if strategy == "weighted_normalized":
            fused_score = lexical_weight * lexical_value + semantic_weight * semantic_value
        else:
            fused_score = 0.5 * lexical_rrf + 0.5 * semantic_rrf
        components = {
            "lexical_raw": lexical_raw.get(product_id, MISSING_MODALITY_VALUE),
            "semantic_raw": semantic_raw.get(product_id, MISSING_MODALITY_VALUE),
            "lexical_rank": float(
                lexical_by_id[product_id].rank if product_id in lexical_by_id else 0
            ),
            "semantic_rank": float(
                semantic_by_id[product_id].rank if product_id in semantic_by_id else 0
            ),
            "lexical_normalized": lexical_value,
            "semantic_normalized": semantic_value,
            "lexical_rrf": lexical_rrf,
            "semantic_rrf": semantic_rrf,
            "lexical_present": float(product_id in lexical_by_id),
            "semantic_present": float(product_id in semantic_by_id),
            "hybrid": fused_score,
        }
        scored.append((product_id, fused_score, components))

    scored.sort(key=lambda item: (-item[1], item[0]))
    return [
        SearchResult(
            product_id=product_id,
            rank=rank,
            score=score,
            score_components=components,
        )
        for rank, (product_id, score, components) in enumerate(scored[:top_k], start=1)
    ]


def _validate_and_map_results(
    results: Sequence[SearchResult],
    allowed_ids: set[str],
    *,
    modality: str,
) -> dict[str, SearchResult]:
    product_ids = [result.product_id for result in results]
    if len(set(product_ids)) != len(product_ids):
        raise ValueError(f"{modality} results contain duplicate product IDs")
    unexpected = sorted(set(product_ids) - allowed_ids)
    if unexpected:
        raise ValueError(f"{modality} results contain unexpected products: {unexpected[:5]}")
    ordered = sorted(results, key=lambda result: result.rank)
    if [result.rank for result in ordered] != list(range(1, len(ordered) + 1)):
        raise ValueError(f"{modality} result ranks must be contiguous from one")
    return {result.product_id: result for result in ordered}


def _validate_fusion_settings(
    *,
    strategy: str,
    semantic_weight: float,
    candidate_depth: int,
    rrf_k: int,
) -> None:
    if strategy not in {"weighted_normalized", "rrf"}:
        raise ValueError(f"unsupported fusion strategy: {strategy}")
    if not math.isfinite(semantic_weight) or not 0.0 <= semantic_weight <= 1.0:
        raise ValueError("semantic_weight must be between 0.0 and 1.0")
    if (
        isinstance(candidate_depth, bool)
        or not isinstance(candidate_depth, int)
        or candidate_depth <= 0
    ):
        raise ValueError("candidate_depth must be a positive integer")
    if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k <= 0:
        raise ValueError("rrf_k must be a positive integer")


def _validate_top_k(top_k: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Free-text product query.")
    parser.add_argument("--top-k", type=int, help="Maximum result count.")
    parser.add_argument("--strategy", choices=("weighted_normalized", "rrf"))
    parser.add_argument("--semantic-weight", type=float)
    parser.add_argument("--candidate-depth", type=int)
    parser.add_argument("--rrf-k", type=int)
    parser.add_argument("--lexical-index-dir", type=Path)
    parser.add_argument("--dense-index-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load both verified indexes and print fused full-catalog results."""

    arguments = _build_parser().parse_args(argv)
    settings = load_settings()
    provider = FastEmbedProvider(
        settings.dense.model_name,
        cache_dir=settings.paths.embeddings / "model_cache",
        local_files_only=arguments.local_files_only,
    )
    lexical_engine = LexicalSearchEngine.from_index_dir(
        arguments.lexical_index_dir or settings.paths.indexes / "tfidf"
    )
    semantic_engine = SemanticSearchEngine.from_index_dir(
        arguments.dense_index_dir or settings.paths.embeddings / "dense",
        provider=provider,
        expected_dimension=settings.dense.expected_dimension,
    )
    engine = HybridSearchEngine(
        lexical_engine,
        semantic_engine,
        strategy=arguments.strategy or settings.hybrid.strategy,
        semantic_weight=(
            arguments.semantic_weight
            if arguments.semantic_weight is not None
            else settings.hybrid.semantic_weight
        ),
        candidate_depth=arguments.candidate_depth or settings.hybrid.candidate_depth,
        rrf_k=arguments.rrf_k or settings.hybrid.rrf_k,
    )
    results = engine.search(arguments.query, arguments.top_k or settings.default_top_k)
    print(
        json.dumps(
            [
                {
                    "product_id": result.product_id,
                    "rank": result.rank,
                    "score": result.score,
                    "score_components": result.score_components,
                }
                for result in results
            ],
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
