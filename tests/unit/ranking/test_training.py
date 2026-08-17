"""Tests for split-scoped pairwise feature-row construction."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from pandas import DataFrame

from product_search.ranking.features import FEATURE_NAMES, ProductFeatureStore
from product_search.ranking.training import (
    build_pairwise_feature_rows,
    class_distribution,
    model_arrays,
)
from product_search.retrieval.base import SearchResult


class RecordingCandidateEngine:
    def __init__(self) -> None:
        self.seen_queries: list[str] = []

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        return []

    def search_candidates(
        self,
        query: str,
        candidate_product_ids: Sequence[str],
        top_k: int,
    ) -> list[SearchResult]:
        self.seen_queries.append(query)
        return [
            SearchResult(
                product_id=product_id,
                rank=rank,
                score=1.0 / rank,
                score_components={
                    "lexical_raw": 1.0 / rank,
                    "semantic_raw": 1.0 / (rank + 1),
                    "lexical_rank": float(rank),
                    "semantic_rank": float(rank),
                    "lexical_present": 1.0,
                    "semantic_present": 1.0,
                    "hybrid": 1.0 / rank,
                },
            )
            for rank, product_id in enumerate(sorted(candidate_product_ids)[:top_k], start=1)
        ]


def _products() -> ProductFeatureStore:
    return ProductFeatureStore.from_frame(
        DataFrame(
            {
                "product_id": ["p1", "p2", "p3", "p4"],
                "product_name": ["Alpha table", "Beta chair", "Gamma rug", "Delta lamp"],
                "product_description": ["wood", "seat", "floor", "light"],
                "product_text": [
                    "alpha table wood",
                    "beta chair seat",
                    "gamma rug floor",
                    "delta lamp light",
                ],
            }
        )
    )


def test_pairwise_rows_use_only_requested_train_queries_and_allowed_features() -> None:
    queries = DataFrame(
        {
            "query_id": ["train-1", "test-1"],
            "query": ["table", "held out lamp"],
            "query_class": ["Broad", "Exact"],
        }
    )
    judgments = DataFrame(
        {
            "query_id": ["train-1", "train-1", "train-1", "test-1"],
            "product_id": ["p1", "p2", "p3", "p4"],
            "relevance_grade": [2, 1, 0, 2],
        }
    )
    engine = RecordingCandidateEngine()

    rows = build_pairwise_feature_rows(
        engine,
        queries,
        judgments,
        _products(),
        query_ids={"train-1"},
        candidate_depth=2,
    )
    matrix, labels = model_arrays(rows)

    assert engine.seen_queries == ["table"]
    assert set(rows["query_id"]) == {"train-1"}
    assert "query_class" not in rows
    assert len(rows) == 2
    assert matrix.shape == (2, len(FEATURE_NAMES))
    assert labels.tolist() == [2, 1]
    assert class_distribution([0, 0, 1, 2]) == {0: 2, 1: 1, 2: 1}


def test_pairwise_rows_reject_missing_split_queries_and_noncanonical_judgments() -> None:
    queries = DataFrame({"query_id": ["q1"], "query": ["table"]})
    judgments = DataFrame({"query_id": ["q1"], "product_id": ["p1"], "relevance_grade": [2]})

    with pytest.raises(ValueError, match="absent from queries"):
        build_pairwise_feature_rows(
            RecordingCandidateEngine(),
            queries,
            judgments,
            _products(),
            query_ids={"missing"},
            candidate_depth=10,
        )
    duplicate = DataFrame(
        {
            "query_id": ["q1", "q1"],
            "product_id": ["p1", "p1"],
            "relevance_grade": [2, 1],
        }
    )
    with pytest.raises(ValueError, match="one row"):
        build_pairwise_feature_rows(
            RecordingCandidateEngine(),
            queries,
            duplicate,
            _products(),
            query_ids={"q1"},
            candidate_depth=10,
        )
