"""Dependency-light, hand-verifiable information retrieval metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence

Relevance = int | float


def dcg_at_k(relevance_grades: Sequence[Relevance], k: int) -> float:
    """Return DCG@K using exponential gain and log-base-two discount."""

    _validate_k(k)
    grades = _validated_grades(relevance_grades)
    return float(
        sum(
            (2.0**grade - 1.0) / math.log2(rank + 1.0)
            for rank, grade in enumerate(grades[:k], start=1)
        )
    )


def ndcg_at_k(
    ranked_relevance: Sequence[Relevance],
    k: int,
    *,
    ideal_relevance: Sequence[Relevance] | None = None,
) -> float:
    """Return nDCG@K, or zero when the ideal ranking has no gain."""

    _validate_k(k)
    ranked = _validated_grades(ranked_relevance)
    available = ranked if ideal_relevance is None else _validated_grades(ideal_relevance)
    ideal_dcg = dcg_at_k(sorted(available, reverse=True), k)
    return dcg_at_k(ranked, k) / ideal_dcg if ideal_dcg > 0.0 else 0.0


def precision_at_k(
    ranked_relevance: Sequence[Relevance],
    k: int,
    *,
    relevant_threshold: Relevance = 1,
) -> float:
    """Return relevant results in the first K divided by K.

    Missing result slots count as non-relevant, so an engine returning fewer than K results is
    not rewarded compared with an engine that fills the requested result set.
    """

    _validate_k(k)
    grades = _validated_grades(ranked_relevance)
    relevant_count = sum(grade >= relevant_threshold for grade in grades[:k])
    return relevant_count / k


def recall_at_k(
    ranked_relevance: Sequence[Relevance],
    all_relevance: Sequence[Relevance],
    k: int,
    *,
    relevant_threshold: Relevance = 1,
) -> float:
    """Return relevant products retrieved by K over all known relevant products."""

    _validate_k(k)
    ranked = _validated_grades(ranked_relevance)
    available = _validated_grades(all_relevance)
    relevant_total = sum(grade >= relevant_threshold for grade in available)
    if relevant_total == 0:
        return 0.0
    retrieved_relevant = sum(grade >= relevant_threshold for grade in ranked[:k])
    return retrieved_relevant / relevant_total


def reciprocal_rank(
    ranked_relevance: Sequence[Relevance],
    *,
    relevant_threshold: Relevance = 1,
) -> float:
    """Return the reciprocal rank of the first relevant result, or zero."""

    grades = _validated_grades(ranked_relevance)
    for rank, grade in enumerate(grades, start=1):
        if grade >= relevant_threshold:
            return 1.0 / rank
    return 0.0


def reciprocal_rank_at_k(
    ranked_relevance: Sequence[Relevance],
    k: int,
    *,
    relevant_threshold: Relevance = 1,
) -> float:
    """Return reciprocal rank truncated at K."""

    _validate_k(k)
    return reciprocal_rank(
        _validated_grades(ranked_relevance)[:k],
        relevant_threshold=relevant_threshold,
    )


def mrr_at_k(
    rankings: Sequence[Sequence[Relevance]],
    k: int,
    *,
    relevant_threshold: Relevance = 1,
) -> float:
    """Return mean reciprocal rank at K across query rankings."""

    _validate_k(k)
    if not rankings:
        return 0.0
    return sum(
        reciprocal_rank_at_k(ranking, k, relevant_threshold=relevant_threshold)
        for ranking in rankings
    ) / len(rankings)


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")


def _validated_grades(grades: Sequence[Relevance]) -> list[float]:
    validated = [float(grade) for grade in grades]
    if any(not math.isfinite(grade) or grade < 0.0 for grade in validated):
        raise ValueError("relevance grades must be finite and non-negative")
    return validated
