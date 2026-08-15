"""Benchmark the TF-IDF baseline on validation queries only."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict, cast

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
from product_search.indexing.tfidf import METADATA_FILENAME
from product_search.retrieval.base import SearchResult
from product_search.retrieval.lexical import LexicalSearchEngine


class RelevanceInfo(TypedDict):
    """Canonical label information used by error-analysis examples."""

    label: str
    grade: int


def run_lexical_validation_benchmark(
    *,
    index_dir: Path,
    products_path: Path,
    queries_path: Path,
    judgments_path: Path,
    splits_path: Path,
    report_path: Path,
    relevant_threshold: int | float = 1,
    timestamp: datetime | None = None,
) -> dict[str, object]:
    """Run judged and full-catalog validation evaluation without touching test queries."""

    index_dir = index_dir.resolve()
    products_path = products_path.resolve()
    queries_path = queries_path.resolve()
    judgments_path = judgments_path.resolve()
    splits_path = splits_path.resolve()
    report_path = report_path.resolve()

    split_manifest = _read_json(splits_path)
    validation_ids, test_ids = _split_ids(split_manifest)
    queries = pd.read_parquet(queries_path)
    judgments = pd.read_parquet(judgments_path)
    products = pd.read_parquet(
        products_path,
        columns=["product_id", "product_name", "product_text"],
    )
    validation_queries = _select_validation_queries(queries, validation_ids)
    validation_judgments = judgments.loc[
        judgments["query_id"].astype(str).isin(validation_ids)
    ].copy()

    engine = LexicalSearchEngine.from_index_dir(index_dir)
    _warm_engine(engine, validation_queries)
    judged_at_5 = judged_candidate_evaluation(
        engine,
        validation_queries,
        validation_judgments,
        top_k=5,
        relevant_threshold=relevant_threshold,
    )
    judged_at_10 = judged_candidate_evaluation(
        engine,
        validation_queries,
        validation_judgments,
        top_k=10,
        relevant_threshold=relevant_threshold,
    )
    full_catalog_at_10 = full_catalog_known_relevant_evaluation(
        engine,
        validation_queries,
        validation_judgments,
        top_k=10,
        relevant_threshold=relevant_threshold,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    detailed_paths: dict[str, EvaluationReportPaths] = {
        "judged_at_5": write_evaluation_reports(
            judged_at_5,
            report_path.parent,
            report_name="lexical_validation_judged_k5",
        ),
        "judged_at_10": write_evaluation_reports(
            judged_at_10,
            report_path.parent,
            report_name="lexical_validation_judged_k10",
        ),
        "full_catalog_at_10": write_evaluation_reports(
            full_catalog_at_10,
            report_path.parent,
            report_name="lexical_validation_full_catalog_k10",
        ),
    }
    error_analysis = _build_error_analysis(
        engine=engine,
        validation_queries=validation_queries,
        validation_judgments=validation_judgments,
        products=products,
        judged_run=judged_at_10,
        relevant_threshold=relevant_threshold,
    )

    report: dict[str, object] = {
        "schema_version": 1,
        "created_at": (timestamp or datetime.now(UTC)).astimezone(UTC).isoformat(),
        "split": "validation",
        "validation_query_count": len(validation_ids),
        "test_query_count_evaluated": 0,
        "held_out_test_query_count": len(test_ids),
        "binary_relevant_threshold": float(relevant_threshold),
        "source_hashes": {
            "index_metadata.json": sha256_file(index_dir / METADATA_FILENAME),
            queries_path.name: sha256_file(queries_path),
            judgments_path.name: sha256_file(judgments_path),
            splits_path.name: sha256_file(splits_path),
        },
        "index": {
            "product_count": engine.metadata["product_count"],
            "vocabulary_size": engine.metadata["vocabulary_size"],
            "matrix_shape": engine.metadata["matrix_shape"],
            "matrix_dtype": engine.metadata["matrix_dtype"],
            "dataset_sha256": engine.metadata["dataset_sha256"],
            "vectorizer_parameters": engine.metadata["vectorizer_parameters"],
            "artifact_hashes": {
                filename: artifact["sha256"]
                for filename, artifact in engine.metadata["artifacts"].items()
            },
        },
        "judged_candidate_evaluation": {
            "query_count": judged_at_10.aggregate["query_count"],
            "eligible_query_count": judged_at_10.aggregate["eligible_query_count"],
            "ndcg_at_5": _metric(judged_at_5, "ndcg_at_k"),
            "ndcg_at_10": _metric(judged_at_10, "ndcg_at_k"),
            "precision_at_10": _metric(judged_at_10, "precision_at_k"),
            "recall_at_10": _metric(judged_at_10, "recall_at_k"),
            "mrr_at_10": _metric(judged_at_10, "mrr_at_k"),
            "latency_ms_at_10": _latency(judged_at_10),
        },
        "full_catalog_known_relevant_evaluation": {
            "query_count": full_catalog_at_10.aggregate["query_count"],
            "eligible_query_count": full_catalog_at_10.aggregate["eligible_query_count"],
            "known_relevant_recall_at_10": _metric(
                full_catalog_at_10, "known_relevant_recall_at_k"
            ),
            "known_relevant_mrr_at_10": _metric(full_catalog_at_10, "known_relevant_mrr_at_k"),
            "latency_ms_at_10": _latency(full_catalog_at_10),
            "unjudged_products_policy": "unknown_not_irrelevant",
        },
        "median_query_latency_ms": _latency(full_catalog_at_10)["median"],
        "p95_query_latency_ms": _latency(full_catalog_at_10)["p95"],
        "detailed_reports": {
            run_name: _report_path_names(paths) for run_name, paths in detailed_paths.items()
        },
        "error_analysis": error_analysis,
    }
    _write_json_atomic(report_path, report)
    return report


def _build_error_analysis(
    *,
    engine: LexicalSearchEngine,
    validation_queries: DataFrame,
    validation_judgments: DataFrame,
    products: DataFrame,
    judged_run: EvaluationRun,
    relevant_threshold: int | float,
) -> dict[str, object]:
    query_text = dict(
        zip(
            validation_queries["query_id"].astype(str),
            validation_queries["query"].astype(str),
            strict=True,
        )
    )
    products_by_id = products.assign(product_id=products["product_id"].astype(str)).set_index(
        "product_id"
    )
    judgments_by_query = {
        str(query_id): group.assign(product_id=group["product_id"].astype(str))
        for query_id, group in validation_judgments.groupby("query_id", sort=True, observed=True)
    }
    records: list[dict[str, object]] = []
    for metric_row in judged_run.per_query:
        query_id = str(metric_row["query_id"])
        query = query_text[query_id]
        query_judgments = judgments_by_query.get(query_id, DataFrame())
        candidate_ids = (
            query_judgments["product_id"].astype(str).tolist() if not query_judgments.empty else []
        )
        results = engine.search_candidates(query, candidate_ids, 10)
        vocabulary = engine.vocabulary_analysis(query)
        unigrams = tuple(token for token in vocabulary["tokens"] if " " not in token)
        out_of_vocabulary = tuple(
            token for token in vocabulary["out_of_vocabulary_tokens"] if " " not in token
        )
        relevance_by_product: dict[str, RelevanceInfo] = {
            str(row["product_id"]): {
                "label": str(row["label"]),
                "grade": _as_int(row["relevance_grade"]),
            }
            for row in cast(list[dict[str, object]], query_judgments.to_dict(orient="records"))
        }
        relevant_products = [
            product_id
            for product_id, relevance in relevance_by_product.items()
            if relevance["grade"] >= relevant_threshold
        ]
        result_details = _result_details(results[:3], relevance_by_product, products_by_id)
        top_grade = int(relevance_by_product[results[0].product_id]["grade"]) if results else -1
        top_text = (
            str(products_by_id.loc[results[0].product_id, "product_text"]).lower()
            if results
            else ""
        )
        relevant_title_overlap = _maximum_relevant_title_overlap(
            unigrams, relevant_products, products_by_id
        )
        records.append(
            {
                "query_id": query_id,
                "query": query,
                "ndcg_at_10": float(metric_row["ndcg_at_k"]),
                "precision_at_10": float(metric_row["precision_at_k"]),
                "recall_at_10": float(metric_row["recall_at_k"]),
                "mrr_at_10": float(metric_row["reciprocal_rank_at_k"]),
                "oov_unigrams": out_of_vocabulary,
                "oov_unigram_fraction": (
                    len(out_of_vocabulary) / len(unigrams) if unigrams else 0.0
                ),
                "top_result_relevance_grade": top_grade,
                "exact_query_phrase_in_top_product_text": query.lower() in top_text,
                "maximum_relevant_title_token_overlap": relevant_title_overlap,
                "top_judged_results": result_details,
                "known_relevant_examples": _known_relevant_details(
                    relevant_products, relevance_by_product, products_by_id
                ),
            }
        )

    eligible = [
        record
        for record, metric_row in zip(records, judged_run.per_query, strict=True)
        if int(metric_row["relevant_judgment_count"]) > 0
    ]
    high = sorted(
        eligible,
        key=lambda record: (-_as_float(record["ndcg_at_10"]), str(record["query_id"])),
    )[:3]
    poor = sorted(
        eligible,
        key=lambda record: (_as_float(record["ndcg_at_10"]), str(record["query_id"])),
    )[:3]
    vocabulary_mismatch = sorted(
        (record for record in eligible if _as_float(record["oov_unigram_fraction"]) > 0.0),
        key=lambda record: (
            -_as_float(record["oov_unigram_fraction"]),
            _as_float(record["ndcg_at_10"]),
            str(record["query_id"]),
        ),
    )[:3]
    exact_keyword_successes = sorted(
        (
            record
            for record in eligible
            if _as_int(record["top_result_relevance_grade"]) >= relevant_threshold
            and bool(record["exact_query_phrase_in_top_product_text"])
        ),
        key=lambda record: (-_as_float(record["ndcg_at_10"]), str(record["query_id"])),
    )[:3]
    synonym_failures = sorted(
        (
            record
            for record in eligible
            if _as_float(record["ndcg_at_10"]) < 0.5
            and _as_int(record["top_result_relevance_grade"]) < relevant_threshold
            and _as_float(record["maximum_relevant_title_token_overlap"]) == 0.0
        ),
        key=lambda record: (_as_float(record["ndcg_at_10"]), str(record["query_id"])),
    )[:3]
    return {
        "selection_policy": {
            "high_performing": "Highest judged-candidate nDCG@10 among relevant queries.",
            "poorly_performing": "Lowest judged-candidate nDCG@10 among relevant queries.",
            "vocabulary_mismatch": (
                "At least one query unigram is absent from the fitted vocabulary."
            ),
            "exact_keyword_success": (
                "Top judged result is relevant and contains the exact normalized query phrase."
            ),
            "synonym_failure": (
                "nDCG@10 below 0.5, non-relevant top result, and no query-token overlap with "
                "known-relevant product titles; examples require human interpretation."
            ),
        },
        "high_performing_queries": high,
        "poorly_performing_queries": poor,
        "vocabulary_mismatch_queries": vocabulary_mismatch,
        "exact_keyword_successes": exact_keyword_successes,
        "synonym_failure_candidates": synonym_failures,
    }


def _result_details(
    results: Sequence[SearchResult],
    relevance_by_product: Mapping[str, RelevanceInfo],
    products_by_id: DataFrame,
) -> list[dict[str, object]]:
    return [
        {
            "rank": result.rank,
            "product_id": result.product_id,
            "product_name": str(products_by_id.loc[result.product_id, "product_name"]),
            "score": result.score,
            "label": relevance_by_product[result.product_id]["label"],
            "relevance_grade": relevance_by_product[result.product_id]["grade"],
        }
        for result in results
    ]


def _known_relevant_details(
    product_ids: Sequence[str],
    relevance_by_product: Mapping[str, RelevanceInfo],
    products_by_id: DataFrame,
) -> list[dict[str, object]]:
    ordered = sorted(
        product_ids,
        key=lambda product_id: (
            -relevance_by_product[product_id]["grade"],
            product_id,
        ),
    )[:3]
    return [
        {
            "product_id": product_id,
            "product_name": str(products_by_id.loc[product_id, "product_name"]),
            "label": relevance_by_product[product_id]["label"],
            "relevance_grade": relevance_by_product[product_id]["grade"],
        }
        for product_id in ordered
    ]


def _maximum_relevant_title_overlap(
    query_unigrams: Sequence[str],
    relevant_product_ids: Sequence[str],
    products_by_id: DataFrame,
) -> float:
    query_tokens = set(query_unigrams)
    if not query_tokens or not relevant_product_ids:
        return 0.0
    overlaps = []
    for product_id in relevant_product_ids:
        title_tokens = set(str(products_by_id.loc[product_id, "product_name"]).lower().split())
        overlaps.append(len(query_tokens & title_tokens) / len(query_tokens))
    return max(overlaps, default=0.0)


def _metric(run: EvaluationRun, name: str) -> float:
    metrics = cast(Mapping[str, object], run.aggregate["metrics_all_queries"])
    return _as_float(metrics[name])


def _latency(run: EvaluationRun) -> dict[str, object]:
    return dict(cast(Mapping[str, object], run.aggregate["latency_ms"]))


def _report_path_names(paths: EvaluationReportPaths) -> dict[str, str]:
    return {
        "per_query_csv": paths["per_query_csv"].name,
        "aggregate_json": paths["aggregate_json"].name,
        "diagnostics_json": paths["diagnostics_json"].name,
    }


def _as_float(value: object) -> float:
    return float(cast(Any, value))


def _as_int(value: object) -> int:
    return int(cast(Any, value))


def _split_ids(manifest: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    query_ids = manifest.get("query_ids")
    if not isinstance(query_ids, dict):
        raise ValueError("query split manifest is missing query_ids")
    validation = query_ids.get("validation")
    test = query_ids.get("test")
    if not isinstance(validation, list) or not isinstance(test, list):
        raise ValueError("query split manifest must contain validation and test ID lists")
    validation_ids = {str(query_id) for query_id in validation}
    test_ids = {str(query_id) for query_id in test}
    if validation_ids & test_ids:
        raise ValueError("validation and test query IDs overlap")
    return validation_ids, test_ids


def _select_validation_queries(queries: DataFrame, validation_ids: set[str]) -> DataFrame:
    if {"query_id", "query"} - set(queries.columns):
        raise ValueError("queries must contain query_id and query")
    normalized = queries.assign(query_id=queries["query_id"].astype(str))
    selected = normalized.loc[normalized["query_id"].isin(validation_ids)].copy()
    if set(selected["query_id"]) != validation_ids:
        missing = sorted(validation_ids - set(selected["query_id"]))
        raise ValueError(f"validation query IDs are absent from queries table: {missing[:5]}")
    return selected.sort_values("query_id", kind="stable", ignore_index=True)


def _warm_engine(engine: LexicalSearchEngine, validation_queries: DataFrame) -> None:
    for query in validation_queries["query"].astype(str).head(3):
        engine.search(query, 10)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        return cast(dict[str, Any], json.load(input_file))


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
    parser.add_argument("--index-dir", type=Path)
    parser.add_argument("--products", type=Path)
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--judgments", type=Path)
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the lexical validation benchmark from configured local artifacts."""

    arguments = _build_parser().parse_args(argv)
    settings = load_settings()
    report = run_lexical_validation_benchmark(
        index_dir=arguments.index_dir or settings.paths.indexes / "tfidf",
        products_path=arguments.products or settings.paths.processed_data / "products.parquet",
        queries_path=arguments.queries or settings.paths.processed_data / "queries.parquet",
        judgments_path=arguments.judgments
        or settings.paths.processed_data / "evaluation_judgments.parquet",
        splits_path=arguments.splits or settings.paths.processed_data / "query_splits.json",
        report_path=arguments.report or settings.paths.reports / "lexical_validation_metrics.json",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
