"""Engine-agnostic judged-candidate and known-relevant recovery evaluation."""

from __future__ import annotations

import csv
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict, cast

import numpy as np
from pandas import DataFrame

from product_search.evaluation.metrics import (
    dcg_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from product_search.retrieval.base import (
    JudgedCandidateSearchEngine,
    SearchEngine,
    SearchResult,
)

EvaluationMode = Literal[
    "judged_candidate_evaluation",
    "full_catalog_known_relevant_evaluation",
]
Scalar = str | int | float


class Diagnostic(TypedDict):
    """One machine-readable query warning or failure."""

    query_id: str
    code: str
    message: str


class EvaluationReportPaths(TypedDict):
    """Files written for one evaluation run."""

    per_query_csv: Path
    aggregate_json: Path
    diagnostics_json: Path


@dataclass(frozen=True, slots=True)
class EvaluationRun:
    """Per-query, aggregate, latency, and diagnostic evaluation outputs."""

    mode: EvaluationMode
    top_k: int
    per_query: tuple[dict[str, Scalar], ...]
    aggregate: dict[str, object]
    diagnostics: tuple[Diagnostic, ...]


def judged_candidate_evaluation(
    engine: JudgedCandidateSearchEngine,
    queries: DataFrame,
    canonical_judgments: DataFrame,
    *,
    top_k: int,
    relevant_threshold: int | float = 1,
    clock: Callable[[], float] = time.perf_counter,
) -> EvaluationRun:
    """Rank only the human-judged product candidates for each supplied query."""

    _validate_inputs(queries, canonical_judgments, top_k=top_k)
    judgment_groups = _judgments_by_query(canonical_judgments)
    query_records = cast(
        list[dict[str, object]], queries.loc[:, ["query_id", "query"]].to_dict(orient="records")
    )
    rows: list[dict[str, Scalar]] = []
    diagnostics: list[Diagnostic] = []

    for record in query_records:
        query_id = str(record["query_id"])
        query_text = str(record["query"])
        grades_by_product = judgment_groups.get(query_id, {})
        candidate_ids = sorted(grades_by_product)
        all_grades = list(grades_by_product.values())
        relevant_count = sum(grade >= relevant_threshold for grade in all_grades)

        started_at = clock()
        try:
            results = _validate_results(
                engine.search_candidates(query_text, candidate_ids, top_k),
                top_k=top_k,
                allowed_product_ids=set(candidate_ids),
            )
            status = "ok"
        except Exception as error:
            results = []
            status = "error"
            diagnostics.append(_search_error(query_id, error))
        latency_ms = max(0.0, (clock() - started_at) * 1000.0)
        ranked_grades = [grades_by_product[result.product_id] for result in results]

        rows.append(
            {
                "query_id": query_id,
                "candidate_count": len(candidate_ids),
                "retrieved_count": len(results),
                "relevant_judgment_count": relevant_count,
                "dcg_at_k": dcg_at_k(ranked_grades, top_k),
                "ndcg_at_k": ndcg_at_k(
                    ranked_grades,
                    top_k,
                    ideal_relevance=all_grades,
                ),
                "precision_at_k": precision_at_k(
                    ranked_grades,
                    top_k,
                    relevant_threshold=relevant_threshold,
                ),
                "recall_at_k": recall_at_k(
                    ranked_grades,
                    all_grades,
                    top_k,
                    relevant_threshold=relevant_threshold,
                ),
                "reciprocal_rank_at_k": reciprocal_rank_at_k(
                    ranked_grades,
                    top_k,
                    relevant_threshold=relevant_threshold,
                ),
                "latency_ms": latency_ms,
                "status": status,
            }
        )
        if status == "ok":
            diagnostics.extend(
                _result_diagnostics(
                    query_id,
                    retrieved_count=len(results),
                    requested_count=min(top_k, len(candidate_ids)),
                    relevant_count=relevant_count,
                )
            )

    aggregate = _aggregate(
        rows,
        metric_names={
            "dcg_at_k": "dcg_at_k",
            "ndcg_at_k": "ndcg_at_k",
            "precision_at_k": "precision_at_k",
            "recall_at_k": "recall_at_k",
            "mrr_at_k": "reciprocal_rank_at_k",
        },
    )
    aggregate["binary_relevant_threshold"] = float(relevant_threshold)
    aggregate["unjudged_products_policy"] = "not_applicable_all_candidates_are_judged"
    return EvaluationRun(
        mode="judged_candidate_evaluation",
        top_k=top_k,
        per_query=tuple(rows),
        aggregate=aggregate,
        diagnostics=tuple(diagnostics),
    )


def full_catalog_known_relevant_evaluation(
    engine: SearchEngine,
    queries: DataFrame,
    canonical_judgments: DataFrame,
    *,
    top_k: int,
    relevant_threshold: int | float = 1,
    clock: Callable[[], float] = time.perf_counter,
) -> EvaluationRun:
    """Measure recovery of explicitly relevant judgments from full-catalog search.

    Unjudged retrieved products remain unknown. This mode deliberately does not emit precision,
    DCG, or nDCG because those metrics would require treating unknown products as non-relevant.
    """

    _validate_inputs(queries, canonical_judgments, top_k=top_k)
    judgment_groups = _judgments_by_query(canonical_judgments)
    query_records = cast(
        list[dict[str, object]], queries.loc[:, ["query_id", "query"]].to_dict(orient="records")
    )
    rows: list[dict[str, Scalar]] = []
    diagnostics: list[Diagnostic] = []

    for record in query_records:
        query_id = str(record["query_id"])
        query_text = str(record["query"])
        grades_by_product = judgment_groups.get(query_id, {})
        known_relevant = {
            product_id: grade
            for product_id, grade in grades_by_product.items()
            if grade >= relevant_threshold
        }

        started_at = clock()
        try:
            results = _validate_results(engine.search(query_text, top_k), top_k=top_k)
            status = "ok"
        except Exception as error:
            results = []
            status = "error"
            diagnostics.append(_search_error(query_id, error))
        latency_ms = max(0.0, (clock() - started_at) * 1000.0)
        recovery_grades = [known_relevant.get(result.product_id, 0) for result in results]
        known_relevant_grades = list(known_relevant.values())

        rows.append(
            {
                "query_id": query_id,
                "retrieved_count": len(results),
                "known_relevant_count": len(known_relevant),
                "known_relevant_recovered_at_k": sum(
                    grade >= relevant_threshold for grade in recovery_grades
                ),
                "known_relevant_recall_at_k": recall_at_k(
                    recovery_grades,
                    known_relevant_grades,
                    top_k,
                    relevant_threshold=relevant_threshold,
                ),
                "known_relevant_reciprocal_rank_at_k": reciprocal_rank_at_k(
                    recovery_grades,
                    top_k,
                    relevant_threshold=relevant_threshold,
                ),
                "relevant_judgment_count": len(known_relevant),
                "latency_ms": latency_ms,
                "status": status,
            }
        )
        if status == "ok":
            diagnostics.extend(
                _result_diagnostics(
                    query_id,
                    retrieved_count=len(results),
                    requested_count=top_k,
                    relevant_count=len(known_relevant),
                )
            )

    aggregate = _aggregate(
        rows,
        metric_names={
            "known_relevant_recall_at_k": "known_relevant_recall_at_k",
            "known_relevant_mrr_at_k": "known_relevant_reciprocal_rank_at_k",
        },
    )
    aggregate["binary_relevant_threshold"] = float(relevant_threshold)
    aggregate["unjudged_products_policy"] = "unknown_not_irrelevant"
    return EvaluationRun(
        mode="full_catalog_known_relevant_evaluation",
        top_k=top_k,
        per_query=tuple(rows),
        aggregate=aggregate,
        diagnostics=tuple(diagnostics),
    )


def write_evaluation_reports(
    run: EvaluationRun,
    output_dir: Path,
    *,
    report_name: str | None = None,
) -> EvaluationReportPaths:
    """Write per-query CSV, aggregate JSON, and diagnostics JSON atomically."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    name = report_name or run.mode
    paths = EvaluationReportPaths(
        per_query_csv=output_dir / f"{name}_per_query.csv",
        aggregate_json=output_dir / f"{name}_aggregate.json",
        diagnostics_json=output_dir / f"{name}_diagnostics.json",
    )
    _write_csv_atomic(paths["per_query_csv"], run.per_query)
    _write_json_atomic(
        paths["aggregate_json"],
        {"mode": run.mode, "top_k": run.top_k, **run.aggregate},
    )
    _write_json_atomic(
        paths["diagnostics_json"],
        {"mode": run.mode, "diagnostics": run.diagnostics},
    )
    return paths


def _validate_inputs(queries: DataFrame, judgments: DataFrame, *, top_k: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
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
        raise ValueError("canonical judgments must contain exactly one row per query-product pair")
    grades = judgments["relevance_grade"].astype(float)
    if not np.isfinite(grades).all() or grades.lt(0.0).any():
        raise ValueError("canonical relevance grades must be finite and non-negative")


def _judgments_by_query(judgments: DataFrame) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, float]] = {}
    for query_id, group in judgments.groupby("query_id", sort=True, observed=True):
        grouped[str(query_id)] = {
            str(product_id): float(grade)
            for product_id, grade in zip(group["product_id"], group["relevance_grade"], strict=True)
        }
    return grouped


def _validate_results(
    results: Sequence[SearchResult],
    *,
    top_k: int,
    allowed_product_ids: set[str] | None = None,
) -> list[SearchResult]:
    ranked = sorted(results, key=lambda result: result.rank)
    if len(ranked) > top_k:
        raise ValueError(f"engine returned {len(ranked)} results for top_k={top_k}")
    product_ids = [result.product_id for result in ranked]
    ranks = [result.rank for result in ranked]
    if len(set(product_ids)) != len(product_ids):
        raise ValueError("engine returned duplicate product IDs")
    if ranks != list(range(1, len(ranked) + 1)):
        raise ValueError("engine result ranks must be unique and contiguous from one")
    if allowed_product_ids is not None:
        unexpected = sorted(set(product_ids) - allowed_product_ids)
        if unexpected:
            raise ValueError(
                f"engine returned products outside judged candidates: {unexpected[:5]}"
            )
    return ranked


def _aggregate(
    rows: Sequence[Mapping[str, Scalar]],
    *,
    metric_names: Mapping[str, str],
) -> dict[str, object]:
    all_metrics = {
        aggregate_name: _mean([float(row[row_name]) for row in rows])
        for aggregate_name, row_name in metric_names.items()
    }
    eligible_rows = [row for row in rows if int(row["relevant_judgment_count"]) > 0]
    eligible_metrics = {
        aggregate_name: _mean([float(row[row_name]) for row in eligible_rows])
        for aggregate_name, row_name in metric_names.items()
    }
    latencies = [float(row["latency_ms"]) for row in rows]
    latency_summary = {
        "sample_count": len(latencies),
        "mean": _mean(latencies),
        "median": float(np.median(latencies)) if latencies else 0.0,
        "p95": float(np.percentile(latencies, 95)) if latencies else 0.0,
    }
    return {
        "query_count": len(rows),
        "eligible_query_count": len(eligible_rows),
        "failed_query_count": sum(row["status"] == "error" for row in rows),
        "metrics_all_queries": all_metrics,
        "metrics_relevant_queries_only": eligible_metrics,
        "latency_ms": latency_summary,
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _search_error(query_id: str, error: Exception) -> Diagnostic:
    return Diagnostic(
        query_id=query_id,
        code="search_error",
        message=f"{type(error).__name__}: {error}",
    )


def _result_diagnostics(
    query_id: str,
    *,
    retrieved_count: int,
    requested_count: int,
    relevant_count: int,
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    if relevant_count == 0:
        diagnostics.append(
            Diagnostic(
                query_id=query_id,
                code="no_relevant_judgments",
                message="Query has no judgments at or above the configured relevance threshold.",
            )
        )
    if retrieved_count == 0 and requested_count > 0:
        diagnostics.append(
            Diagnostic(query_id=query_id, code="no_results", message="Engine returned no results.")
        )
    elif retrieved_count < requested_count:
        diagnostics.append(
            Diagnostic(
                query_id=query_id,
                code="fewer_results_than_requested",
                message=(
                    f"Engine returned {retrieved_count} of {requested_count} requested results."
                ),
            )
        )
    return diagnostics


def _write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Scalar]]) -> None:
    temporary_path = path.with_name(f"{path.name}.part")
    fieldnames = list(rows[0]) if rows else []
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as output_file:
            if fieldnames:
                writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        os.replace(temporary_path, path)
    except OSError:
        temporary_path.unlink(missing_ok=True)
        raise


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    temporary_path = path.with_name(f"{path.name}.part")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as output_file:
            json.dump(payload, output_file, indent=2, sort_keys=True)
            output_file.write("\n")
        os.replace(temporary_path, path)
    except OSError:
        temporary_path.unlink(missing_ok=True)
        raise
