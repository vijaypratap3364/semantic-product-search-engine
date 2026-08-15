"""Hand-calculated tests for information retrieval metrics."""

from __future__ import annotations

import math

import pytest

from product_search.evaluation.metrics import (
    dcg_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    reciprocal_rank_at_k,
)


def test_perfect_graded_ranking() -> None:
    grades = [2, 1, 0]
    expected_dcg = 3.0 + 1.0 / math.log2(3.0)

    assert dcg_at_k(grades, 3) == pytest.approx(expected_dcg)
    assert ndcg_at_k(grades, 3) == pytest.approx(1.0)
    assert precision_at_k(grades, 3) == pytest.approx(2 / 3)
    assert recall_at_k(grades, grades, 3) == pytest.approx(1.0)
    assert reciprocal_rank(grades) == pytest.approx(1.0)


def test_reversed_ranking_matches_hand_calculation() -> None:
    ranked = [0, 1, 2]
    ideal = [2, 1, 0]
    actual_dcg = 1.0 / math.log2(3.0) + 3.0 / math.log2(4.0)
    ideal_dcg = 3.0 + 1.0 / math.log2(3.0)

    assert dcg_at_k(ranked, 3) == pytest.approx(actual_dcg)
    assert ndcg_at_k(ranked, 3, ideal_relevance=ideal) == pytest.approx(actual_dcg / ideal_dcg)
    assert reciprocal_rank_at_k(ranked, 3) == pytest.approx(0.5)


def test_no_relevant_results_and_no_relevant_judgments_return_zero() -> None:
    assert ndcg_at_k([0, 0], 2) == 0.0
    assert precision_at_k([0, 0], 2) == 0.0
    assert recall_at_k([0, 0], [0, 0], 2) == 0.0
    assert reciprocal_rank([0, 0]) == 0.0
    assert reciprocal_rank_at_k([0, 1], 1) == 0.0
    assert mrr_at_k([], 10) == 0.0


def test_truncated_k_and_missing_result_slots() -> None:
    assert dcg_at_k([2, 1, 0], 1) == 3.0
    assert precision_at_k([1], 3) == pytest.approx(1 / 3)
    assert recall_at_k([1], [1, 1], 1) == pytest.approx(0.5)
    assert reciprocal_rank_at_k([0, 1], 1) == 0.0


def test_binary_relevance_threshold_is_configurable() -> None:
    grades = [1, 2, 0]

    assert precision_at_k(grades, 3, relevant_threshold=1) == pytest.approx(2 / 3)
    assert precision_at_k(grades, 3, relevant_threshold=2) == pytest.approx(1 / 3)
    assert reciprocal_rank(grades, relevant_threshold=2) == pytest.approx(0.5)


def test_mrr_at_k_is_mean_of_per_query_reciprocal_ranks() -> None:
    rankings = [[1, 0], [0, 1], [0, 0]]

    assert mrr_at_k(rankings, 2) == pytest.approx((1.0 + 0.5 + 0.0) / 3)


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_metrics_reject_invalid_k(k: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        dcg_at_k([1], k)


def test_metrics_reject_invalid_relevance_grades() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        dcg_at_k([-1], 1)
    with pytest.raises(ValueError, match="finite and non-negative"):
        dcg_at_k([math.inf], 1)
