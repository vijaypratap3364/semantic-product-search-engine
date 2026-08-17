"""Train on train queries and select a lightweight relevance reranker on validation."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame
from sklearn.metrics import classification_report, confusion_matrix

from product_search.config import load_settings
from product_search.data.download import sha256_file
from product_search.evaluation.evaluator import (
    EvaluationReportPaths,
    EvaluationRun,
    full_catalog_known_relevant_evaluation,
    judged_candidate_evaluation,
    write_evaluation_reports,
)
from product_search.ranking.features import FEATURE_NAMES, ProductFeatureStore
from product_search.ranking.model import (
    ClassWeightName,
    RelevanceModel,
    save_relevance_model,
    train_relevance_model,
)
from product_search.ranking.reranker import RerankingSearchEngine
from product_search.ranking.training import (
    build_pairwise_feature_rows,
    class_distribution,
    model_arrays,
)
from product_search.retrieval.base import JudgedCandidateSearchEngine, SearchResult


class _PrecomputedRankingEngine:
    """Expose validation predictions through the shared judged-candidate evaluator."""

    def __init__(self, rankings: Mapping[str, Sequence[SearchResult]]) -> None:
        self._rankings = {query: tuple(results) for query, results in rankings.items()}

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        return self._select(query, None, top_k)

    def search_candidates(
        self,
        query: str,
        candidate_product_ids: Sequence[str],
        top_k: int,
    ) -> list[SearchResult]:
        return self._select(query, set(candidate_product_ids), top_k)

    def _select(
        self,
        query: str,
        allowed: set[str] | None,
        top_k: int,
    ) -> list[SearchResult]:
        try:
            ranking = self._rankings[query]
        except KeyError as error:
            raise ValueError(f"no precomputed validation ranking for query: {query}") from error
        selected = [
            result for result in ranking if allowed is None or result.product_id in allowed
        ][:top_k]
        return [
            SearchResult(
                product_id=result.product_id,
                rank=rank,
                score=result.score,
                score_components=result.score_components,
            )
            for rank, result in enumerate(selected, start=1)
        ]


def run_reranker_validation_experiment(
    *,
    hybrid_engine: JudgedCandidateSearchEngine,
    product_store: ProductFeatureStore,
    queries_path: Path,
    judgments_path: Path,
    splits_path: Path,
    model_dir: Path,
    model_search_path: Path,
    report_path: Path,
    c_grid: Sequence[float],
    class_weight_options: Sequence[ClassWeightName],
    max_iter: int,
    random_seed: int,
    candidate_depth: int,
    relevant_threshold: int | float = 1,
    force: bool = False,
    timestamp: datetime | None = None,
) -> dict[str, object]:
    """Fit only train rows, select only on validation, and never evaluate test queries."""

    queries_path = queries_path.resolve()
    judgments_path = judgments_path.resolve()
    splits_path = splits_path.resolve()
    model_dir = model_dir.resolve()
    model_search_path = model_search_path.resolve()
    report_path = report_path.resolve()
    split_manifest = _read_json(splits_path)
    train_ids, validation_ids, test_ids = _split_ids(split_manifest)
    queries = pd.read_parquet(queries_path)
    judgments = pd.read_parquet(judgments_path)
    train_queries = _select_queries(queries, train_ids, split_name="train")
    validation_queries = _select_queries(queries, validation_ids, split_name="validation")
    train_judgments = _select_judgments(judgments, train_ids)
    validation_judgments = _select_judgments(judgments, validation_ids)

    train_rows = build_pairwise_feature_rows(
        hybrid_engine,
        train_queries,
        train_judgments,
        product_store,
        query_ids=train_ids,
        candidate_depth=candidate_depth,
    )
    validation_rows = build_pairwise_feature_rows(
        hybrid_engine,
        validation_queries,
        validation_judgments,
        product_store,
        query_ids=validation_ids,
        candidate_depth=candidate_depth,
    )
    train_features, train_labels = model_arrays(train_rows)
    validation_features, validation_labels = model_arrays(validation_rows)

    configurations = _configurations(c_grid, class_weight_options)
    search_rows: list[dict[str, str | int | float]] = []
    models: list[RelevanceModel] = []
    validation_runs: list[EvaluationRun] = []
    classification_summaries: list[dict[str, object]] = []
    for c_value, class_weight in configurations:
        model = train_relevance_model(
            train_features,
            train_labels,
            c_value=c_value,
            class_weight=class_weight,
            max_iter=max_iter,
            random_seed=random_seed,
            candidate_depth=candidate_depth,
        )
        probabilities = model.predict_probabilities(validation_features)
        expected_scores = model.predict_expected_relevance(validation_features)
        predicted_labels = np.asarray(model.classes, dtype=np.int64)[
            np.argmax(probabilities, axis=1)
        ]
        classification = _classification_summary(validation_labels, predicted_labels)
        ranking_engine = _precomputed_engine(validation_rows, expected_scores)
        validation_run = judged_candidate_evaluation(
            ranking_engine,
            validation_queries,
            validation_judgments,
            top_k=10,
            relevant_threshold=relevant_threshold,
        )
        models.append(model)
        validation_runs.append(validation_run)
        classification_summaries.append(classification)
        search_rows.append(
            {
                "C": c_value,
                "class_weight": class_weight,
                "train_query_count": len(train_ids),
                "training_row_count": len(train_rows),
                "validation_query_count": len(validation_ids),
                "validation_row_count": len(validation_rows),
                "validation_ndcg_at_10": _metric(validation_run, "ndcg_at_k"),
                "validation_precision_at_10": _metric(validation_run, "precision_at_k"),
                "validation_recall_at_10": _metric(validation_run, "recall_at_k"),
                "validation_mrr_at_10": _metric(validation_run, "mrr_at_k"),
                "validation_accuracy": float(cast(Any, classification["accuracy"])),
                "validation_macro_f1": float(cast(Any, classification["macro_f1"])),
                "test_query_count_evaluated": 0,
            }
        )

    best_index = max(
        range(len(models)),
        key=lambda index: (
            _metric(validation_runs[index], "ndcg_at_k"),
            _metric(validation_runs[index], "recall_at_k"),
            _metric(validation_runs[index], "mrr_at_k"),
            float(cast(Any, classification_summaries[index]["macro_f1"])),
            -index,
        ),
    )
    selected_model = models[best_index]
    selected_search_row = search_rows[best_index]
    selected_classification = classification_summaries[best_index]

    source_hashes = {
        queries_path.name: sha256_file(queries_path),
        judgments_path.name: sha256_file(judgments_path),
        splits_path.name: sha256_file(splits_path),
    }
    model_metadata = save_relevance_model(
        selected_model,
        model_dir,
        product_dataset_sha256=product_store.dataset_sha256,
        source_hashes=source_hashes,
        train_query_count=len(train_ids),
        training_row_count=len(train_rows),
        class_distribution=class_distribution(train_labels),
        force=force,
        timestamp=timestamp,
    )

    reranker_engine = RerankingSearchEngine(
        hybrid_engine,
        selected_model,
        product_store,
        candidate_depth=candidate_depth,
    )
    hybrid_at_10 = judged_candidate_evaluation(
        hybrid_engine,
        validation_queries,
        validation_judgments,
        top_k=10,
        relevant_threshold=relevant_threshold,
    )
    reranker_at_5 = judged_candidate_evaluation(
        reranker_engine,
        validation_queries,
        validation_judgments,
        top_k=5,
        relevant_threshold=relevant_threshold,
    )
    reranker_at_10 = judged_candidate_evaluation(
        reranker_engine,
        validation_queries,
        validation_judgments,
        top_k=10,
        relevant_threshold=relevant_threshold,
    )
    hybrid_full_at_10 = full_catalog_known_relevant_evaluation(
        hybrid_engine,
        validation_queries,
        validation_judgments,
        top_k=10,
        relevant_threshold=relevant_threshold,
    )
    reranker_full_at_10 = full_catalog_known_relevant_evaluation(
        reranker_engine,
        validation_queries,
        validation_judgments,
        top_k=10,
        relevant_threshold=relevant_threshold,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    detailed_paths = {
        "judged_at_5": write_evaluation_reports(
            reranker_at_5,
            report_path.parent,
            report_name="reranker_validation_judged_k5",
        ),
        "judged_at_10": write_evaluation_reports(
            reranker_at_10,
            report_path.parent,
            report_name="reranker_validation_judged_k10",
        ),
        "full_catalog_at_10": write_evaluation_reports(
            reranker_full_at_10,
            report_path.parent,
            report_name="reranker_validation_full_catalog_k10",
        ),
    }
    _write_csv_atomic(model_search_path, search_rows)

    hybrid_ndcg = _metric(hybrid_at_10, "ndcg_at_k")
    reranker_ndcg = _metric(reranker_at_10, "ndcg_at_k")
    improved = reranker_ndcg > hybrid_ndcg
    report: dict[str, object] = {
        "schema_version": 1,
        "created_at": (timestamp or datetime.now(UTC)).astimezone(UTC).isoformat(),
        "training_split": "train",
        "selection_split": "validation",
        "train_query_count": len(train_ids),
        "validation_query_count": len(validation_ids),
        "held_out_test_query_count": len(test_ids),
        "test_query_count_evaluated": 0,
        "candidate_depth": candidate_depth,
        "binary_relevant_threshold": float(relevant_threshold),
        "source_hashes": source_hashes,
        "feature_schema": {
            "feature_names": list(FEATURE_NAMES),
            "predictive_identifier_features": [],
            "training_row_count": len(train_rows),
            "validation_row_count": len(validation_rows),
        },
        "class_distribution": {
            "train": _named_distribution(class_distribution(train_labels)),
            "validation_candidates": _named_distribution(class_distribution(validation_labels)),
        },
        "model_search": {
            "selection_metric": "validation_judged_candidate_ndcg_at_10",
            "tie_break_policy": "recall_at_10_then_mrr_at_10_then_macro_f1_then_grid_order",
            "configuration_count": len(search_rows),
            "selected_configuration": dict(selected_search_row),
            "csv": model_search_path.name,
        },
        "validation_classification": selected_classification,
        "feature_coefficients": {
            "space": "standardized_features",
            "by_relevance_grade": selected_model.standardized_coefficients(),
        },
        "judged_candidate_evaluation": {
            "reranker_ndcg_at_5": _metric(reranker_at_5, "ndcg_at_k"),
            "hybrid_ndcg_at_10": hybrid_ndcg,
            "reranker_ndcg_at_10": reranker_ndcg,
            "ndcg_at_10_delta": reranker_ndcg - hybrid_ndcg,
            "hybrid_precision_at_10": _metric(hybrid_at_10, "precision_at_k"),
            "reranker_precision_at_10": _metric(reranker_at_10, "precision_at_k"),
            "hybrid_recall_at_10": _metric(hybrid_at_10, "recall_at_k"),
            "reranker_recall_at_10": _metric(reranker_at_10, "recall_at_k"),
            "hybrid_mrr_at_10": _metric(hybrid_at_10, "mrr_at_k"),
            "reranker_mrr_at_10": _metric(reranker_at_10, "mrr_at_k"),
            "reranker_latency_ms_at_10": _latency(reranker_at_10),
        },
        "full_catalog_known_relevant_evaluation": {
            "hybrid_known_relevant_recall_at_10": _metric(
                hybrid_full_at_10, "known_relevant_recall_at_k"
            ),
            "reranker_known_relevant_recall_at_10": _metric(
                reranker_full_at_10, "known_relevant_recall_at_k"
            ),
            "hybrid_known_relevant_mrr_at_10": _metric(
                hybrid_full_at_10, "known_relevant_mrr_at_k"
            ),
            "reranker_known_relevant_mrr_at_10": _metric(
                reranker_full_at_10, "known_relevant_mrr_at_k"
            ),
            "reranker_latency_ms_at_10": _latency(reranker_full_at_10),
            "unjudged_products_policy": "unknown_not_irrelevant",
        },
        "production_decision": {
            "reranker_improves_validation_ndcg_at_10": improved,
            "recommended_default_search_mode": "reranker" if improved else "hybrid",
            "policy": (
                "Use reranker only when validation judged-candidate nDCG@10 strictly exceeds "
                "the hybrid baseline."
            ),
        },
        "model_artifact": {
            "directory": model_dir.name,
            "metadata": model_metadata,
        },
        "detailed_reports": {
            name: _report_path_names(paths) for name, paths in detailed_paths.items()
        },
    }
    _write_json_atomic(report_path, report)
    return report


def _configurations(
    c_grid: Sequence[float],
    class_weights: Sequence[ClassWeightName],
) -> list[tuple[float, ClassWeightName]]:
    c_values = tuple(float(value) for value in c_grid)
    weights = tuple(class_weights)
    if not c_values or any(not np.isfinite(value) or value <= 0 for value in c_values):
        raise ValueError("C grid must contain positive finite values")
    if tuple(sorted(set(c_values))) != c_values:
        raise ValueError("C grid must be unique and ascending")
    if not weights or len(set(weights)) != len(weights):
        raise ValueError("class-weight options must be non-empty and unique")
    if set(weights) - {"none", "balanced"}:
        raise ValueError("class-weight options contain an unsupported value")
    return [(c_value, weight) for c_value in c_values for weight in weights]


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
    if not train or not validation:
        raise ValueError("train and validation query splits must not be empty")
    if train & validation or train & test or validation & test:
        raise ValueError("train, validation, and test query IDs must be disjoint")
    return train, validation, test


def _select_queries(queries: DataFrame, query_ids: set[str], *, split_name: str) -> DataFrame:
    if {"query_id", "query"} - set(queries.columns):
        raise ValueError("queries must contain query_id and query")
    normalized = queries.assign(query_id=queries["query_id"].astype(str))
    selected = normalized.loc[normalized["query_id"].isin(query_ids), ["query_id", "query"]].copy()
    if set(selected["query_id"]) != query_ids:
        missing = sorted(query_ids - set(selected["query_id"]))
        raise ValueError(f"{split_name} query IDs are absent from queries table: {missing[:5]}")
    if selected["query"].astype(str).duplicated().any():
        raise ValueError(f"{split_name} query text must be unique for precomputed evaluation")
    return selected.sort_values("query_id", kind="stable", ignore_index=True)


def _select_judgments(judgments: DataFrame, query_ids: set[str]) -> DataFrame:
    return judgments.loc[judgments["query_id"].astype(str).isin(query_ids)].copy()


def _precomputed_engine(
    rows: DataFrame,
    expected_scores: Sequence[float] | NDArray[np.float64],
) -> _PrecomputedRankingEngine:
    if len(rows) != len(expected_scores):
        raise ValueError("validation scores must align with feature rows")
    scored = rows.loc[:, ["query", "product_id", "hybrid_rank"]].assign(
        expected_relevance=np.asarray(expected_scores, dtype=np.float64)
    )
    rankings: dict[str, list[SearchResult]] = {}
    for query, group in scored.groupby("query", sort=True, observed=True):
        ordered = group.sort_values(
            ["expected_relevance", "hybrid_rank", "product_id"],
            ascending=[False, True, True],
            kind="stable",
        )
        rankings[str(query)] = [
            SearchResult(
                product_id=str(record["product_id"]),
                rank=rank,
                score=float(cast(Any, record["expected_relevance"])),
            )
            for rank, record in enumerate(
                cast(list[dict[str, object]], ordered.to_dict(orient="records")), start=1
            )
        ]
    return _PrecomputedRankingEngine(rankings)


def _classification_summary(
    labels: Sequence[int] | NDArray[np.int64],
    predictions: Sequence[int] | NDArray[np.int64],
) -> dict[str, object]:
    report = cast(
        dict[str, Any],
        classification_report(
            labels,
            predictions,
            labels=[0, 1, 2],
            target_names=["Irrelevant", "Partial", "Exact"],
            output_dict=True,
            zero_division=0,
        ),
    )
    macro = cast(dict[str, Any], report["macro avg"])
    return {
        "row_count": len(labels),
        "accuracy": float(report["accuracy"]),
        "macro_f1": float(macro["f1-score"]),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1, 2]).tolist(),
        "confusion_matrix_label_order": ["Irrelevant", "Partial", "Exact"],
    }


def _named_distribution(distribution: Mapping[int, int]) -> dict[str, int]:
    return {
        "Irrelevant": int(distribution[0]),
        "Partial": int(distribution[1]),
        "Exact": int(distribution[2]),
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
    temporary = path.with_name(f"{path.name}.part")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f"{path.name}.part")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output_file:
            json.dump(payload, output_file, indent=2, sort_keys=True)
            output_file.write("\n")
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products", type=Path)
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--judgments", type=Path)
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--lexical-index-dir", type=Path)
    parser.add_argument("--dense-index-dir", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--model-search", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load local indexes and run the train/validation reranker experiment."""

    from product_search.indexing.dense import FastEmbedProvider
    from product_search.retrieval.hybrid import HybridSearchEngine
    from product_search.retrieval.lexical import LexicalSearchEngine
    from product_search.retrieval.semantic import SemanticSearchEngine

    arguments = _build_parser().parse_args(argv)
    settings = load_settings()
    products_path = arguments.products or settings.paths.processed_data / "products.parquet"
    provider = FastEmbedProvider(
        settings.dense.model_name,
        cache_dir=settings.paths.embeddings / "model_cache",
        local_files_only=arguments.local_files_only,
    )
    lexical = LexicalSearchEngine.from_index_dir(
        arguments.lexical_index_dir or settings.paths.indexes / "tfidf"
    )
    semantic = SemanticSearchEngine.from_index_dir(
        arguments.dense_index_dir or settings.paths.embeddings / "dense",
        provider=provider,
        expected_dimension=settings.dense.expected_dimension,
    )
    hybrid = HybridSearchEngine(
        lexical,
        semantic,
        strategy=settings.hybrid.strategy,
        semantic_weight=settings.hybrid.semantic_weight,
        candidate_depth=settings.hybrid.candidate_depth,
        rrf_k=settings.hybrid.rrf_k,
    )
    report = run_reranker_validation_experiment(
        hybrid_engine=hybrid,
        product_store=ProductFeatureStore.from_parquet(products_path),
        queries_path=arguments.queries or settings.paths.processed_data / "queries.parquet",
        judgments_path=arguments.judgments
        or settings.paths.processed_data / "evaluation_judgments.parquet",
        splits_path=arguments.splits or settings.paths.processed_data / "query_splits.json",
        model_dir=arguments.model_dir or settings.paths.models / "reranker",
        model_search_path=arguments.model_search
        or settings.paths.reports / "reranker_model_search.csv",
        report_path=arguments.report or settings.paths.reports / "reranker_validation_metrics.json",
        c_grid=settings.reranker.c_grid,
        class_weight_options=settings.reranker.class_weight_options,
        max_iter=settings.reranker.max_iter,
        random_seed=settings.random_seed,
        candidate_depth=settings.reranker.candidate_depth,
        force=arguments.force,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
