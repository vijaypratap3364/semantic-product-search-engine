"""Benchmark dense semantic retrieval on validation queries and compare TF-IDF."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict, cast

import pandas as pd
from pandas import DataFrame
from scipy import sparse
from scipy.sparse import csr_matrix

from product_search.config import load_settings
from product_search.data.download import sha256_file
from product_search.evaluation.evaluator import (
    EvaluationReportPaths,
    EvaluationRun,
    full_catalog_known_relevant_evaluation,
    judged_candidate_evaluation,
    write_evaluation_reports,
)
from product_search.indexing.dense import (
    METADATA_FILENAME as DENSE_METADATA_FILENAME,
)
from product_search.indexing.dense import EmbeddingProvider
from product_search.indexing.tfidf import (
    MATRIX_FILENAME as LEXICAL_MATRIX_FILENAME,
)
from product_search.indexing.tfidf import (
    METADATA_FILENAME as LEXICAL_METADATA_FILENAME,
)
from product_search.retrieval.lexical import LexicalSearchEngine
from product_search.retrieval.semantic import SemanticSearchEngine


class RelevanceInfo(TypedDict):
    """Canonical label information used by qualitative examples."""

    label: str
    grade: int


def run_semantic_validation_benchmark(
    *,
    provider: EmbeddingProvider,
    index_dir: Path,
    lexical_index_dir: Path,
    products_path: Path,
    queries_path: Path,
    judgments_path: Path,
    splits_path: Path,
    lexical_report_path: Path,
    lexical_per_query_path: Path,
    report_path: Path,
    expected_dimension: int,
    relevant_threshold: int | float = 1,
    timestamp: datetime | None = None,
) -> dict[str, object]:
    """Evaluate validation only, preserving test isolation and the Stage 3 methodology."""

    paths = [
        index_dir,
        lexical_index_dir,
        products_path,
        queries_path,
        judgments_path,
        splits_path,
        lexical_report_path,
        lexical_per_query_path,
        report_path,
    ]
    (
        index_dir,
        lexical_index_dir,
        products_path,
        queries_path,
        judgments_path,
        splits_path,
        lexical_report_path,
        lexical_per_query_path,
        report_path,
    ) = (path.resolve() for path in paths)

    split_manifest = _read_json(splits_path)
    validation_ids, test_ids = _split_ids(split_manifest)
    queries = pd.read_parquet(queries_path)
    judgments = pd.read_parquet(judgments_path)
    products = pd.read_parquet(products_path, columns=["product_id", "product_name"])
    validation_queries = _select_validation_queries(queries, validation_ids)
    validation_judgments = judgments.loc[
        judgments["query_id"].astype(str).isin(validation_ids)
    ].copy()

    engine = SemanticSearchEngine.from_index_dir(
        index_dir,
        provider=provider,
        expected_dimension=expected_dimension,
    )
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
            report_name="semantic_validation_judged_k5",
        ),
        "judged_at_10": write_evaluation_reports(
            judged_at_10,
            report_path.parent,
            report_name="semantic_validation_judged_k10",
        ),
        "full_catalog_at_10": write_evaluation_reports(
            full_catalog_at_10,
            report_path.parent,
            report_name="semantic_validation_full_catalog_k10",
        ),
    }

    lexical_report = _read_json(lexical_report_path)
    lexical_per_query = pd.read_csv(lexical_per_query_path, dtype={"query_id": str})
    lexical_engine = LexicalSearchEngine.from_index_dir(lexical_index_dir)
    qualitative_examples = _semantic_low_overlap_successes(
        semantic_engine=engine,
        lexical_engine=lexical_engine,
        validation_queries=validation_queries,
        validation_judgments=validation_judgments,
        products=products,
        semantic_run=judged_at_10,
        lexical_per_query=lexical_per_query,
        relevant_threshold=relevant_threshold,
    )
    disk_and_memory = _resource_comparison(
        semantic_engine=engine,
        semantic_index_dir=index_dir,
        lexical_index_dir=lexical_index_dir,
    )

    semantic_ndcg_10 = _metric(judged_at_10, "ndcg_at_k")
    semantic_recall_10 = _metric(judged_at_10, "recall_at_k")
    semantic_latency = _latency(full_catalog_at_10)
    lexical_judged = _mapping(lexical_report["judged_candidate_evaluation"])
    lexical_ndcg_10 = _as_float(lexical_judged["ndcg_at_10"])
    lexical_recall_10 = _as_float(lexical_judged["recall_at_10"])
    lexical_latency = _mapping(
        _mapping(lexical_report["full_catalog_known_relevant_evaluation"])["latency_ms_at_10"]
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
            "dense_metadata.json": sha256_file(index_dir / DENSE_METADATA_FILENAME),
            queries_path.name: sha256_file(queries_path),
            judgments_path.name: sha256_file(judgments_path),
            splits_path.name: sha256_file(splits_path),
            lexical_report_path.name: sha256_file(lexical_report_path),
        },
        "index": {
            "product_count": engine.metadata["product_count"],
            "model_name": engine.metadata["model_name"],
            "embedding_dimension": engine.metadata["embedding_dimension"],
            "embedding_dtype": engine.metadata["embedding_dtype"],
            "embedding_normalization": engine.metadata["embedding_normalization"],
            "matrix_shape": engine.metadata["matrix_shape"],
            "dataset_sha256": engine.metadata["dataset_sha256"],
            "artifact_hashes": {
                filename: artifact["sha256"]
                for filename, artifact in engine.metadata["artifacts"].items()
            },
        },
        "judged_candidate_evaluation": {
            "query_count": judged_at_10.aggregate["query_count"],
            "eligible_query_count": judged_at_10.aggregate["eligible_query_count"],
            "ndcg_at_5": _metric(judged_at_5, "ndcg_at_k"),
            "ndcg_at_10": semantic_ndcg_10,
            "precision_at_10": _metric(judged_at_10, "precision_at_k"),
            "recall_at_10": semantic_recall_10,
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
            "latency_ms_at_10": semantic_latency,
            "unjudged_products_policy": "unknown_not_irrelevant",
        },
        "median_query_latency_ms": semantic_latency["median"],
        "p95_query_latency_ms": semantic_latency["p95"],
        "comparison": {
            "judged_candidate_ndcg_at_10": _comparison(lexical_ndcg_10, semantic_ndcg_10),
            "judged_candidate_recall_at_10": _comparison(lexical_recall_10, semantic_recall_10),
            "full_catalog_latency_ms": {
                "lexical_median": _as_float(lexical_latency["median"]),
                "semantic_median": _as_float(semantic_latency["median"]),
                "lexical_p95": _as_float(lexical_latency["p95"]),
                "semantic_p95": _as_float(semantic_latency["p95"]),
            },
            **disk_and_memory,
        },
        "qualitative_examples": {
            "selection_policy": (
                "Validation queries where semantic judged-candidate nDCG@10 exceeds lexical, "
                "a semantic top-10 result is relevant and outranks its lexical position, and "
                "query/title unigram overlap is at most 0.25."
            ),
            "semantic_low_lexical_overlap_successes": qualitative_examples,
        },
        "detailed_reports": {
            run_name: _report_path_names(paths) for run_name, paths in detailed_paths.items()
        },
    }
    _write_json_atomic(report_path, report)
    return report


def _semantic_low_overlap_successes(
    *,
    semantic_engine: SemanticSearchEngine,
    lexical_engine: LexicalSearchEngine,
    validation_queries: DataFrame,
    validation_judgments: DataFrame,
    products: DataFrame,
    semantic_run: EvaluationRun,
    lexical_per_query: DataFrame,
    relevant_threshold: int | float,
) -> list[dict[str, object]]:
    products_by_id = products.assign(product_id=products["product_id"].astype(str)).set_index(
        "product_id"
    )
    semantic_metrics = {
        str(row["query_id"]): _as_float(row["ndcg_at_k"]) for row in semantic_run.per_query
    }
    lexical_metrics = {
        str(row["query_id"]): _as_float(row["ndcg_at_k"])
        for row in cast(list[dict[str, object]], lexical_per_query.to_dict(orient="records"))
    }
    judgments_by_query = {
        str(query_id): group.assign(product_id=group["product_id"].astype(str))
        for query_id, group in validation_judgments.groupby("query_id", sort=True, observed=True)
    }
    records: list[dict[str, object]] = []
    for row in cast(list[dict[str, object]], validation_queries.to_dict(orient="records")):
        query_id = str(row["query_id"])
        query = str(row["query"])
        semantic_ndcg = semantic_metrics[query_id]
        lexical_ndcg = lexical_metrics[query_id]
        if semantic_ndcg <= lexical_ndcg:
            continue
        query_judgments = judgments_by_query[query_id]
        candidates = query_judgments["product_id"].astype(str).tolist()
        relevance: dict[str, RelevanceInfo] = {
            str(record["product_id"]): {
                "label": str(record["label"]),
                "grade": _as_int(record["relevance_grade"]),
            }
            for record in cast(list[dict[str, object]], query_judgments.to_dict(orient="records"))
        }
        semantic_results = semantic_engine.search_candidates(query, candidates, top_k=10)
        lexical_results = lexical_engine.search_candidates(query, candidates, top_k=len(candidates))
        lexical_by_product = {result.product_id: result for result in lexical_results}
        for semantic_result in semantic_results:
            product_relevance = relevance[semantic_result.product_id]
            if product_relevance["grade"] < relevant_threshold:
                continue
            product_name = str(products_by_id.loc[semantic_result.product_id, "product_name"])
            overlap = _unigram_overlap(query, product_name)
            lexical_result = lexical_by_product.get(semantic_result.product_id)
            lexical_rank = (
                lexical_result.rank if lexical_result is not None else len(candidates) + 1
            )
            lexical_score = lexical_result.score if lexical_result is not None else 0.0
            if overlap <= 0.25 and semantic_result.rank < lexical_rank:
                records.append(
                    {
                        "query_id": query_id,
                        "query": query,
                        "product_id": semantic_result.product_id,
                        "product_name": product_name,
                        "label": product_relevance["label"],
                        "relevance_grade": product_relevance["grade"],
                        "query_title_unigram_overlap": overlap,
                        "semantic_rank": semantic_result.rank,
                        "semantic_score": semantic_result.score,
                        "lexical_rank": lexical_rank,
                        "lexical_score": lexical_score,
                        "semantic_ndcg_at_10": semantic_ndcg,
                        "lexical_ndcg_at_10": lexical_ndcg,
                        "ndcg_at_10_delta": semantic_ndcg - lexical_ndcg,
                    }
                )
                break
    return sorted(
        records,
        key=lambda record: (-_as_float(record["ndcg_at_10_delta"]), str(record["query_id"])),
    )[:5]


def _resource_comparison(
    *,
    semantic_engine: SemanticSearchEngine,
    semantic_index_dir: Path,
    lexical_index_dir: Path,
) -> dict[str, object]:
    lexical_metadata = _read_json(lexical_index_dir / LEXICAL_METADATA_FILENAME)
    semantic_artifact_bytes = (
        sum(
            int(artifact["byte_size"])
            for artifact in semantic_engine.metadata["artifacts"].values()
        )
        + (semantic_index_dir / DENSE_METADATA_FILENAME).stat().st_size
    )
    lexical_artifacts = _mapping(lexical_metadata["artifacts"])
    lexical_artifact_bytes = (
        sum(_as_int(_mapping(artifact)["byte_size"]) for artifact in lexical_artifacts.values())
        + (lexical_index_dir / LEXICAL_METADATA_FILENAME).stat().st_size
    )
    lexical_matrix = cast(
        csr_matrix,
        sparse.load_npz(lexical_index_dir / LEXICAL_MATRIX_FILENAME),
    )
    lexical_matrix_memory = (
        lexical_matrix.data.nbytes + lexical_matrix.indices.nbytes + lexical_matrix.indptr.nbytes
    )
    model_cache = semantic_index_dir.parent / "model_cache"
    model_cache_bytes = sum(
        path.stat().st_size for path in model_cache.rglob("*") if path.is_file()
    )
    return {
        "index_artifact_disk_bytes": {
            "lexical": lexical_artifact_bytes,
            "semantic": semantic_artifact_bytes,
            "semantic_model_cache": model_cache_bytes,
        },
        "matrix_memory_bytes": {
            "lexical_csr_arrays": lexical_matrix_memory,
            "semantic_float32_matrix": semantic_engine.embedding_matrix_nbytes,
            "measurement_policy": "array payload only; excludes Python/model runtime overhead",
        },
    }


def _unigram_overlap(query: str, product_name: str) -> float:
    query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    title_tokens = set(re.findall(r"[a-z0-9]+", product_name.lower()))
    return len(query_tokens & title_tokens) / len(query_tokens) if query_tokens else 0.0


def _metric(run: EvaluationRun, name: str) -> float:
    metrics = cast(Mapping[str, object], run.aggregate["metrics_all_queries"])
    return _as_float(metrics[name])


def _latency(run: EvaluationRun) -> dict[str, object]:
    return dict(cast(Mapping[str, object], run.aggregate["latency_ms"]))


def _comparison(lexical: float, semantic: float) -> dict[str, float]:
    return {"lexical": lexical, "semantic": semantic, "delta": semantic - lexical}


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


def _mapping(value: object) -> Mapping[str, object]:
    return cast(Mapping[str, object], value)


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


def _warm_engine(engine: SemanticSearchEngine, validation_queries: DataFrame) -> None:
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
    parser.add_argument("--lexical-index-dir", type=Path)
    parser.add_argument("--products", type=Path)
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--judgments", type=Path)
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--lexical-report", type=Path)
    parser.add_argument("--lexical-per-query", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the semantic validation benchmark from configured local artifacts."""

    from product_search.indexing.dense import FastEmbedProvider

    arguments = _build_parser().parse_args(argv)
    settings = load_settings()
    provider = FastEmbedProvider(
        settings.dense.model_name,
        cache_dir=settings.paths.embeddings / "model_cache",
        local_files_only=arguments.local_files_only,
    )
    report = run_semantic_validation_benchmark(
        provider=provider,
        index_dir=arguments.index_dir or settings.paths.embeddings / "dense",
        lexical_index_dir=arguments.lexical_index_dir or settings.paths.indexes / "tfidf",
        products_path=arguments.products or settings.paths.processed_data / "products.parquet",
        queries_path=arguments.queries or settings.paths.processed_data / "queries.parquet",
        judgments_path=arguments.judgments
        or settings.paths.processed_data / "evaluation_judgments.parquet",
        splits_path=arguments.splits or settings.paths.processed_data / "query_splits.json",
        lexical_report_path=arguments.lexical_report
        or settings.paths.reports / "lexical_validation_metrics.json",
        lexical_per_query_path=arguments.lexical_per_query
        or settings.paths.reports / "lexical_validation_judged_k10_per_query.csv",
        report_path=arguments.report or settings.paths.reports / "semantic_validation_metrics.json",
        expected_dimension=settings.dense.expected_dimension,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
