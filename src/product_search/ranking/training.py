"""Build train/validation query-product feature rows from bounded hybrid candidates."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from product_search.ranking.features import (
    FEATURE_NAMES,
    FloatMatrix,
    ProductFeatureStore,
    extract_query_product_features,
    feature_matrix,
)
from product_search.retrieval.base import JudgedCandidateSearchEngine

PAIR_PROVENANCE_COLUMNS = (
    "query_id",
    "query",
    "product_id",
    "relevance_grade",
    "hybrid_rank",
)


def build_pairwise_feature_rows(
    engine: JudgedCandidateSearchEngine,
    queries: DataFrame,
    canonical_judgments: DataFrame,
    product_store: ProductFeatureStore,
    *,
    query_ids: set[str],
    candidate_depth: int,
) -> DataFrame:
    """Create features only for selected split queries and their top hybrid candidates."""

    if not query_ids:
        raise ValueError("feature query ID set must not be empty")
    if (
        isinstance(candidate_depth, bool)
        or not isinstance(candidate_depth, int)
        or candidate_depth <= 0
    ):
        raise ValueError("candidate_depth must be a positive integer")
    _validate_source_frames(queries, canonical_judgments)
    normalized_queries = queries.loc[:, ["query_id", "query"]].assign(
        query_id=queries["query_id"].astype(str)
    )
    normalized_judgments = canonical_judgments.loc[
        :, ["query_id", "product_id", "relevance_grade"]
    ].assign(
        query_id=canonical_judgments["query_id"].astype(str),
        product_id=canonical_judgments["product_id"].astype(str),
    )
    selected_queries = normalized_queries.loc[
        normalized_queries["query_id"].isin(query_ids)
    ].sort_values("query_id", kind="stable", ignore_index=True)
    actual_ids = set(selected_queries["query_id"])
    if actual_ids != query_ids:
        missing = sorted(query_ids - actual_ids)
        raise ValueError(f"feature query IDs are absent from queries table: {missing[:5]}")
    selected_judgments = normalized_judgments.loc[
        normalized_judgments["query_id"].isin(query_ids)
    ].copy()
    judged_ids = set(selected_judgments["query_id"])
    if judged_ids != query_ids:
        missing = sorted(query_ids - judged_ids)
        raise ValueError(f"feature queries have no canonical judgments: {missing[:5]}")

    judgments_by_query = {
        str(query_id): group
        for query_id, group in selected_judgments.groupby("query_id", sort=True, observed=True)
    }
    rows: list[dict[str, str | int | float]] = []
    for query_record in cast(list[dict[str, object]], selected_queries.to_dict(orient="records")):
        query_id = str(query_record["query_id"])
        query = str(query_record["query"])
        judgments = judgments_by_query[query_id]
        grades = {
            str(product_id): int(grade)
            for product_id, grade in zip(
                judgments["product_id"], judgments["relevance_grade"], strict=True
            )
        }
        candidate_ids = sorted(grades)
        limit = min(candidate_depth, len(candidate_ids))
        results = engine.search_candidates(query, candidate_ids, limit)
        _validate_candidate_results(results, set(candidate_ids), limit)
        for result in results:
            features = extract_query_product_features(
                query,
                product_store.get(result.product_id),
                result,
                candidate_depth=candidate_depth,
            )
            rows.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "product_id": result.product_id,
                    "relevance_grade": grades[result.product_id],
                    "hybrid_rank": result.rank,
                    **features,
                }
            )
    frame = DataFrame(rows, columns=[*PAIR_PROVENANCE_COLUMNS, *FEATURE_NAMES])
    if frame.empty:
        raise ValueError("pairwise feature rows must not be empty")
    if set(frame["query_id"].astype(str)) - query_ids:
        raise RuntimeError("feature rows leaked query IDs outside the requested split")
    return frame


def model_arrays(feature_rows: DataFrame) -> tuple[FloatMatrix, NDArray[np.int64]]:
    """Select only the allowed predictive schema and the separate target vector."""

    required = {*PAIR_PROVENANCE_COLUMNS, *FEATURE_NAMES}
    missing = required - set(feature_rows.columns)
    if missing:
        raise ValueError(f"pairwise feature rows are missing columns: {sorted(missing)}")
    matrix = feature_matrix(
        cast(list[dict[str, Any]], feature_rows.loc[:, FEATURE_NAMES].to_dict(orient="records"))
    )
    grades = feature_rows["relevance_grade"].to_numpy(dtype=np.int64, copy=True)
    return matrix, grades


def class_distribution(relevance_grades: Sequence[int]) -> dict[int, int]:
    """Return deterministic counts for all three expected WANDS grades."""

    labels = np.asarray(relevance_grades, dtype=np.int64)
    return {grade: int(np.sum(labels == grade)) for grade in (0, 1, 2)}


def _validate_source_frames(queries: DataFrame, judgments: DataFrame) -> None:
    missing_queries = {"query_id", "query"} - set(queries.columns)
    if missing_queries:
        raise ValueError(f"queries are missing required columns: {sorted(missing_queries)}")
    missing_judgments = {"query_id", "product_id", "relevance_grade"} - set(judgments.columns)
    if missing_judgments:
        raise ValueError(
            f"canonical judgments are missing required columns: {sorted(missing_judgments)}"
        )
    if queries[["query_id", "query"]].isna().any(axis=None):
        raise ValueError("query IDs and text must not be missing")
    if judgments[["query_id", "product_id", "relevance_grade"]].isna().any(axis=None):
        raise ValueError("canonical judgment keys and grades must not be missing")
    if judgments.duplicated(["query_id", "product_id"]).any():
        raise ValueError("canonical judgments must contain one row per query-product pair")
    grades = judgments["relevance_grade"].astype(float)
    if not np.isfinite(grades).all() or not grades.isin([0.0, 1.0, 2.0]).all():
        raise ValueError("canonical relevance grades must be 0, 1, or 2")


def _validate_candidate_results(
    results: Sequence[object],
    allowed_product_ids: set[str],
    limit: int,
) -> None:
    from product_search.retrieval.base import SearchResult

    if not all(isinstance(result, SearchResult) for result in results):
        raise ValueError("candidate engine returned an invalid result type")
    typed_results = cast(Sequence[SearchResult], results)
    if len(typed_results) > limit:
        raise ValueError("candidate engine returned more results than requested")
    product_ids = [result.product_id for result in typed_results]
    if len(set(product_ids)) != len(product_ids):
        raise ValueError("candidate engine returned duplicate product IDs")
    if set(product_ids) - allowed_product_ids:
        raise ValueError("candidate engine returned products outside canonical judgments")
    if [result.rank for result in typed_results] != list(range(1, len(typed_results) + 1)):
        raise ValueError("candidate engine ranks must be contiguous from one")
