"""Tune lexical/semantic fusion on validation queries without touching test queries."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
from pandas import DataFrame

from product_search.config import load_settings
from product_search.data.download import sha256_file
from product_search.evaluation.evaluator import (
    EvaluationReportPaths,
    EvaluationRun,
    full_catalog_known_relevant_evaluation,
    judged_candidate_evaluation,
    write_evaluation_reports,
)
from product_search.retrieval.base import JudgedCandidateSearchEngine, SearchResult
from product_search.retrieval.hybrid import FusionStrategy, HybridSearchEngine


class _MemoizedEngine:
    """Cache component rankings so each tuning query is scored once per modality."""

    def __init__(self, engine: JudgedCandidateSearchEngine) -> None:
        self._engine = engine
        self._catalog: dict[tuple[str, int], tuple[SearchResult, ...]] = {}
        self._candidates: dict[tuple[str, tuple[str, ...], int], tuple[SearchResult, ...]] = {}

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        key = (query, top_k)
        if key not in self._catalog:
            self._catalog[key] = tuple(self._engine.search(query, top_k))
        return list(self._catalog[key])

    def search_candidates(
        self,
        query: str,
        candidate_product_ids: Sequence[str],
        top_k: int,
    ) -> list[SearchResult]:
        key = (query, tuple(candidate_product_ids), top_k)
        if key not in self._candidates:
            self._candidates[key] = tuple(
                self._engine.search_candidates(query, candidate_product_ids, top_k)
            )
        return list(self._candidates[key])


def run_hybrid_validation_benchmark(
    *,
    lexical_engine: JudgedCandidateSearchEngine,
    semantic_engine: JudgedCandidateSearchEngine,
    queries_path: Path,
    judgments_path: Path,
    splits_path: Path,
    weight_search_path: Path,
    report_path: Path,
    semantic_weight_grid: Sequence[float],
    candidate_depth: int,
    rrf_k: int,
    relevant_threshold: int | float = 1,
    timestamp: datetime | None = None,
) -> dict[str, object]:
    """Select fusion on validation nDCG@10 and produce validation-only reports."""

    resolved = [
        queries_path,
        judgments_path,
        splits_path,
        weight_search_path,
        report_path,
    ]
    queries_path, judgments_path, splits_path, weight_search_path, report_path = (
        path.resolve() for path in resolved
    )
    weights = _validate_weights(semantic_weight_grid)
    split_manifest = _read_json(splits_path)
    train_ids, validation_ids, test_ids = _split_ids(split_manifest)
    queries = pd.read_parquet(queries_path)
    judgments = pd.read_parquet(judgments_path)
    validation_queries = _select_validation_queries(queries, validation_ids)
    validation_judgments = judgments.loc[
        judgments["query_id"].astype(str).isin(validation_ids)
    ].copy()
    if validation_judgments.empty:
        raise ValueError("validation split has no canonical judgments")

    cached_lexical = _MemoizedEngine(lexical_engine)
    cached_semantic = _MemoizedEngine(semantic_engine)
    lexical_run = judged_candidate_evaluation(
        cached_lexical,
        validation_queries,
        validation_judgments,
        top_k=10,
        relevant_threshold=relevant_threshold,
    )
    semantic_run = judged_candidate_evaluation(
        cached_semantic,
        validation_queries,
        validation_judgments,
        top_k=10,
        relevant_threshold=relevant_threshold,
    )

    configurations: list[tuple[FusionStrategy, float]] = [
        ("weighted_normalized", weight) for weight in weights
    ]
    configurations.append(("rrf", 0.5))
    search_rows: list[dict[str, str | int | float]] = []
    runs: list[EvaluationRun] = []
    for strategy, semantic_weight in configurations:
        engine = HybridSearchEngine(
            cached_lexical,
            cached_semantic,
            strategy=strategy,
            semantic_weight=semantic_weight,
            candidate_depth=candidate_depth,
            rrf_k=rrf_k,
        )
        run = judged_candidate_evaluation(
            engine,
            validation_queries,
            validation_judgments,
            top_k=10,
            relevant_threshold=relevant_threshold,
        )
        runs.append(run)
        search_rows.append(
            {
                "strategy": strategy,
                "semantic_weight": semantic_weight,
                "lexical_weight": 1.0 - semantic_weight,
                "candidate_depth": candidate_depth,
                "rrf_k": rrf_k,
                "validation_query_count": len(validation_ids),
                "ndcg_at_10": _metric(run, "ndcg_at_k"),
                "precision_at_10": _metric(run, "precision_at_k"),
                "recall_at_10": _metric(run, "recall_at_k"),
                "mrr_at_10": _metric(run, "mrr_at_k"),
                "test_query_count_evaluated": 0,
            }
        )

    best_index = max(
        range(len(runs)),
        key=lambda index: (
            _metric(runs[index], "ndcg_at_k"),
            _metric(runs[index], "recall_at_k"),
            _metric(runs[index], "mrr_at_k"),
            -index,
        ),
    )
    selected_row = search_rows[best_index]
    selected_strategy = cast(FusionStrategy, selected_row["strategy"])
    selected_weight = float(selected_row["semantic_weight"])

    selected_engine = HybridSearchEngine(
        lexical_engine,
        semantic_engine,
        strategy=selected_strategy,
        semantic_weight=selected_weight,
        candidate_depth=candidate_depth,
        rrf_k=rrf_k,
    )
    judged_at_5 = judged_candidate_evaluation(
        selected_engine,
        validation_queries,
        validation_judgments,
        top_k=5,
        relevant_threshold=relevant_threshold,
    )
    judged_at_10 = judged_candidate_evaluation(
        selected_engine,
        validation_queries,
        validation_judgments,
        top_k=10,
        relevant_threshold=relevant_threshold,
    )
    full_catalog_at_10 = full_catalog_known_relevant_evaluation(
        selected_engine,
        validation_queries,
        validation_judgments,
        top_k=10,
        relevant_threshold=relevant_threshold,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    detailed_paths = {
        "judged_at_5": write_evaluation_reports(
            judged_at_5,
            report_path.parent,
            report_name="hybrid_validation_judged_k5",
        ),
        "judged_at_10": write_evaluation_reports(
            judged_at_10,
            report_path.parent,
            report_name="hybrid_validation_judged_k10",
        ),
        "full_catalog_at_10": write_evaluation_reports(
            full_catalog_at_10,
            report_path.parent,
            report_name="hybrid_validation_full_catalog_k10",
        ),
    }
    _write_csv_atomic(weight_search_path, search_rows)

    report: dict[str, object] = {
        "schema_version": 1,
        "created_at": (timestamp or datetime.now(UTC)).astimezone(UTC).isoformat(),
        "split": "validation",
        "selection_metric": "validation_judged_candidate_ndcg_at_10",
        "tie_break_policy": "recall_at_10_then_mrr_at_10_then_grid_order",
        "validation_query_count": len(validation_ids),
        "allowed_train_query_count": len(train_ids),
        "train_query_count_evaluated": 0,
        "held_out_test_query_count": len(test_ids),
        "test_query_count_evaluated": 0,
        "binary_relevant_threshold": float(relevant_threshold),
        "source_hashes": {
            queries_path.name: sha256_file(queries_path),
            judgments_path.name: sha256_file(judgments_path),
            splits_path.name: sha256_file(splits_path),
        },
        "search_space": {
            "weighted_normalized_semantic_weights": list(weights),
            "weighted_normalized_lexical_weight": "1 - semantic_weight",
            "rrf": {"rrf_k": rrf_k, "modality_contributions": "equal"},
            "candidate_depth": candidate_depth,
            "configuration_count": len(search_rows),
        },
        "selected_configuration": dict(selected_row),
        "judged_candidate_evaluation": {
            "query_count": judged_at_10.aggregate["query_count"],
            "ndcg_at_5": _metric(judged_at_5, "ndcg_at_k"),
            "ndcg_at_10": _metric(judged_at_10, "ndcg_at_k"),
            "precision_at_10": _metric(judged_at_10, "precision_at_k"),
            "recall_at_10": _metric(judged_at_10, "recall_at_k"),
            "mrr_at_10": _metric(judged_at_10, "mrr_at_k"),
            "latency_ms_at_10": _latency(judged_at_10),
        },
        "full_catalog_known_relevant_evaluation": {
            "query_count": full_catalog_at_10.aggregate["query_count"],
            "known_relevant_recall_at_10": _metric(
                full_catalog_at_10, "known_relevant_recall_at_k"
            ),
            "known_relevant_mrr_at_10": _metric(full_catalog_at_10, "known_relevant_mrr_at_k"),
            "latency_ms_at_10": _latency(full_catalog_at_10),
            "unjudged_products_policy": "unknown_not_irrelevant",
        },
        "validation_comparison": {
            "lexical": _comparison_metrics(lexical_run),
            "semantic": _comparison_metrics(semantic_run),
            "hybrid": _comparison_metrics(judged_at_10),
        },
        "error_analysis": _error_analysis(
            validation_queries,
            lexical_run,
            semantic_run,
            judged_at_10,
        ),
        "weight_search_csv": weight_search_path.name,
        "detailed_reports": {
            name: _report_path_names(paths) for name, paths in detailed_paths.items()
        },
    }
    _write_json_atomic(report_path, report)
    return report


def _validate_weights(weights: Sequence[float]) -> tuple[float, ...]:
    normalized = tuple(float(weight) for weight in weights)
    if not normalized:
        raise ValueError("semantic weight grid must not be empty")
    if any(weight < 0.0 or weight > 1.0 for weight in normalized):
        raise ValueError("semantic weights must be between 0.0 and 1.0")
    if tuple(sorted(set(normalized))) != normalized:
        raise ValueError("semantic weight grid must be unique and ascending")
    return normalized


def _split_ids(manifest: Mapping[str, Any]) -> tuple[set[str], set[str], set[str]]:
    query_ids = manifest.get("query_ids")
    if not isinstance(query_ids, dict):
        raise ValueError("query split manifest is missing query_ids")
    values = [query_ids.get(name) for name in ("train", "validation", "test")]
    if not all(isinstance(value, list) for value in values):
        raise ValueError("query split manifest must contain train, validation, and test ID lists")
    train, validation, test = (
        {str(item) for item in cast(list[object], value)} for value in values
    )
    if train & validation or train & test or validation & test:
        raise ValueError("train, validation, and test query IDs must be disjoint")
    if not validation:
        raise ValueError("validation query split must not be empty")
    return train, validation, test


def _select_validation_queries(queries: DataFrame, validation_ids: set[str]) -> DataFrame:
    if {"query_id", "query"} - set(queries.columns):
        raise ValueError("queries must contain query_id and query")
    normalized = queries.assign(query_id=queries["query_id"].astype(str))
    selected = normalized.loc[normalized["query_id"].isin(validation_ids)].copy()
    if set(selected["query_id"]) != validation_ids:
        missing = sorted(validation_ids - set(selected["query_id"]))
        raise ValueError(f"validation query IDs are absent from queries table: {missing[:5]}")
    return selected.sort_values("query_id", kind="stable", ignore_index=True)


def _error_analysis(
    queries: DataFrame,
    lexical: EvaluationRun,
    semantic: EvaluationRun,
    hybrid: EvaluationRun,
) -> dict[str, object]:
    query_text = dict(
        zip(
            queries["query_id"].astype(str),
            queries["query"].astype(str),
            strict=True,
        )
    )
    lexical_scores = _per_query_metric(lexical, "ndcg_at_k")
    semantic_scores = _per_query_metric(semantic, "ndcg_at_k")
    hybrid_scores = _per_query_metric(hybrid, "ndcg_at_k")
    categories: dict[str, list[dict[str, object]]] = {
        "lexical_wins": [],
        "semantic_wins": [],
        "hybrid_improves_both": [],
        "fusion_hurts": [],
    }
    for query_id in sorted(query_text):
        lexical_score = lexical_scores[query_id]
        semantic_score = semantic_scores[query_id]
        hybrid_score = hybrid_scores[query_id]
        record = {
            "query_id": query_id,
            "query": query_text[query_id],
            "lexical_ndcg_at_10": lexical_score,
            "semantic_ndcg_at_10": semantic_score,
            "hybrid_ndcg_at_10": hybrid_score,
        }
        if lexical_score > semantic_score and lexical_score > hybrid_score:
            categories["lexical_wins"].append(
                {**record, "margin": lexical_score - max(semantic_score, hybrid_score)}
            )
        if semantic_score > lexical_score and semantic_score > hybrid_score:
            categories["semantic_wins"].append(
                {**record, "margin": semantic_score - max(lexical_score, hybrid_score)}
            )
        if hybrid_score > lexical_score and hybrid_score > semantic_score:
            categories["hybrid_improves_both"].append(
                {**record, "margin": hybrid_score - max(lexical_score, semantic_score)}
            )
        if hybrid_score < lexical_score and hybrid_score < semantic_score:
            categories["fusion_hurts"].append(
                {**record, "margin": min(lexical_score, semantic_score) - hybrid_score}
            )
    analysis: dict[str, object] = {
        "metric": "judged_candidate_ndcg_at_10",
        "strict_comparison_policy": "ties_are_not_assigned_to_a_category",
    }
    analysis.update(
        {
            name: {
                "query_count": len(records),
                "examples": sorted(
                    records,
                    key=lambda record: (
                        -float(cast(Any, record["margin"])),
                        str(record["query_id"]),
                    ),
                )[:5],
            }
            for name, records in categories.items()
        }
    )
    return analysis


def _per_query_metric(run: EvaluationRun, name: str) -> dict[str, float]:
    return {str(row["query_id"]): float(row[name]) for row in run.per_query}


def _comparison_metrics(run: EvaluationRun) -> dict[str, float]:
    return {
        "ndcg_at_10": _metric(run, "ndcg_at_k"),
        "precision_at_10": _metric(run, "precision_at_k"),
        "recall_at_10": _metric(run, "recall_at_k"),
        "mrr_at_10": _metric(run, "mrr_at_k"),
    }


def _metric(run: EvaluationRun, name: str) -> float:
    metrics = cast(Mapping[str, object], run.aggregate["metrics_all_queries"])
    return float(cast(Any, metrics[name]))


def _latency(run: EvaluationRun) -> dict[str, object]:
    return dict(cast(Mapping[str, object], run.aggregate["latency_ms"]))


def _report_path_names(paths: EvaluationReportPaths) -> dict[str, str]:
    return {
        "per_query_csv": paths["per_query_csv"].name,
        "aggregate_json": paths["aggregate_json"].name,
        "diagnostics_json": paths["diagnostics_json"].name,
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        return cast(dict[str, Any], json.load(input_file))


def _write_csv_atomic(
    path: Path,
    rows: Sequence[Mapping[str, str | int | float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.part")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lexical-index-dir", type=Path)
    parser.add_argument("--dense-index-dir", type=Path)
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--judgments", type=Path)
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--weight-search", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load verified indexes and run validation-only fusion selection."""

    from product_search.indexing.dense import FastEmbedProvider
    from product_search.retrieval.lexical import LexicalSearchEngine
    from product_search.retrieval.semantic import SemanticSearchEngine

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
    report = run_hybrid_validation_benchmark(
        lexical_engine=lexical_engine,
        semantic_engine=semantic_engine,
        queries_path=arguments.queries or settings.paths.processed_data / "queries.parquet",
        judgments_path=arguments.judgments
        or settings.paths.processed_data / "evaluation_judgments.parquet",
        splits_path=arguments.splits or settings.paths.processed_data / "query_splits.json",
        weight_search_path=arguments.weight_search
        or settings.paths.reports / "hybrid_weight_search.csv",
        report_path=arguments.report or settings.paths.reports / "hybrid_validation_metrics.json",
        semantic_weight_grid=settings.hybrid.semantic_weight_grid,
        candidate_depth=settings.hybrid.candidate_depth,
        rrf_k=settings.hybrid.rrf_k,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
