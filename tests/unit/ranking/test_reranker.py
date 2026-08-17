"""Tests for deterministic second-stage reranking."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from pandas import DataFrame

from product_search.ranking.features import FEATURE_NAMES, ProductFeatureStore
from product_search.ranking.reranker import RerankingSearchEngine
from product_search.retrieval.base import SearchResult


class StaticHybridEngine:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.requested_depths: list[int] = []

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        self.requested_depths.append(top_k)
        return self.results[:top_k]

    def search_candidates(
        self,
        query: str,
        candidate_product_ids: Sequence[str],
        top_k: int,
    ) -> list[SearchResult]:
        self.requested_depths.append(top_k)
        allowed = set(candidate_product_ids)
        return [result for result in self.results if result.product_id in allowed][:top_k]


class ReverseLexicalScorer:
    classes = (0, 1, 2)

    def predict_probabilities(self, features: np.ndarray) -> np.ndarray:
        lexical_index = FEATURE_NAMES.index("lexical_similarity")
        lexical = np.clip(features[:, lexical_index], 0.0, 1.0)
        return np.column_stack((lexical, np.zeros_like(lexical), 1.0 - lexical))

    def predict_expected_relevance(self, features: np.ndarray) -> np.ndarray:
        return self.predict_probabilities(features) @ np.asarray(self.classes, dtype=np.float64)


def _candidate(product_id: str, rank: int, lexical: float) -> SearchResult:
    return SearchResult(
        product_id=product_id,
        rank=rank,
        score=lexical,
        score_components={
            "lexical_raw": lexical,
            "semantic_raw": 0.5,
            "lexical_rank": float(rank),
            "semantic_rank": float(rank),
            "lexical_present": 1.0,
            "semantic_present": 1.0,
            "hybrid": lexical,
        },
    )


def _store() -> ProductFeatureStore:
    return ProductFeatureStore.from_frame(
        DataFrame(
            {
                "product_id": ["p1", "p2", "p3"],
                "product_name": ["One", "Two", "Three"],
                "product_description": ["", "", ""],
                "product_text": ["one", "two", "three"],
            }
        )
    )


def test_reranker_orders_by_expected_relevance_and_preserves_components() -> None:
    hybrid = StaticHybridEngine([_candidate("p1", 1, 0.9), _candidate("p2", 2, 0.1)])
    engine = RerankingSearchEngine(hybrid, ReverseLexicalScorer(), _store(), candidate_depth=100)

    results = engine.search("query", 2)

    assert [result.product_id for result in results] == ["p2", "p1"]
    assert [result.rank for result in results] == [1, 2]
    assert results[0].score == pytest.approx(1.8)
    assert results[0].score_components["hybrid"] == 0.1  # type: ignore[index]
    assert results[0].score_components["hybrid_original_rank"] == 2.0  # type: ignore[index]
    assert results[0].score_components["reranker_probability_exact"] == 0.9  # type: ignore[index]
    assert hybrid.requested_depths == [100]


def test_reranker_candidate_mode_is_deterministic_and_bounded() -> None:
    hybrid = StaticHybridEngine(
        [_candidate("p1", 1, 0.9), _candidate("p2", 2, 0.5), _candidate("p3", 3, 0.1)]
    )
    engine = RerankingSearchEngine(hybrid, ReverseLexicalScorer(), _store(), candidate_depth=2)

    first = engine.search_candidates("query", ["p1", "p2", "p3"], 1)
    second = engine.search_candidates("query", ["p1", "p2", "p3"], 1)

    assert first == second
    assert first[0].product_id == "p2"
    assert hybrid.requested_depths == [2, 2]
    assert engine.search_candidates("query", [], 10) == []


def test_reranker_rejects_invalid_candidates_and_top_k() -> None:
    duplicate = [_candidate("p1", 1, 0.9), _candidate("p1", 2, 0.1)]
    engine = RerankingSearchEngine(
        StaticHybridEngine(duplicate), ReverseLexicalScorer(), _store(), candidate_depth=2
    )

    with pytest.raises(ValueError, match="duplicate"):
        engine.search("query", 2)
    with pytest.raises(ValueError, match="unique"):
        engine.search_candidates("query", ["p1", "p1"], 2)
    with pytest.raises(ValueError, match="positive integer"):
        engine.search("query", 0)
