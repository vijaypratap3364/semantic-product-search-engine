"""One-time held-out test benchmark for validation-frozen search systems."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import sklearn
from pandas import DataFrame

from product_search.config import ProjectSettings, load_settings
from product_search.data.download import sha256_file
from product_search.evaluation.evaluator import (
    EvaluationRun,
    full_catalog_known_relevant_evaluation,
    judged_candidate_evaluation,
)
from product_search.evaluation.final_reporting import (
    FINAL_OUTPUT_FILENAMES,
    SYSTEM_LABELS,
    write_final_report_files,
)
from product_search.retrieval.base import JudgedCandidateSearchEngine

SYSTEM_ORDER = ("lexical", "semantic", "hybrid", "reranked_hybrid")
SEARCH_MODES = {
    "lexical": "lexical",
    "semantic": "semantic",
    "hybrid": "hybrid",
    "reranked_hybrid": "reranker",
}


def freeze_final_configurations(
    settings: ProjectSettings,
    *,
    products_path: Path,
    queries_path: Path,
    judgments_path: Path,
    splits_path: Path,
    lexical_metadata_path: Path,
    dense_metadata_path: Path,
    reranker_metadata_path: Path,
    hybrid_validation_report_path: Path,
    reranker_validation_report_path: Path,
) -> dict[str, object]:
    """Validate and serialize the train/validation-selected configuration before test use."""

    paths = {
        "products.parquet": products_path.resolve(),
        "queries.parquet": queries_path.resolve(),
        "evaluation_judgments.parquet": judgments_path.resolve(),
        "query_splits.json": splits_path.resolve(),
        "lexical_metadata.json": lexical_metadata_path.resolve(),
        "dense_metadata.json": dense_metadata_path.resolve(),
        "reranker_metadata.json": reranker_metadata_path.resolve(),
        "hybrid_validation_metrics.json": hybrid_validation_report_path.resolve(),
        "reranker_validation_metrics.json": reranker_validation_report_path.resolve(),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"required frozen input is missing ({label}): {path}")

    lexical = _read_json(paths["lexical_metadata.json"])
    dense = _read_json(paths["dense_metadata.json"])
    reranker = _read_json(paths["reranker_metadata.json"])
    hybrid_validation = _read_json(paths["hybrid_validation_metrics.json"])
    reranker_validation = _read_json(paths["reranker_validation_metrics.json"])
    source_hashes = {name: sha256_file(path) for name, path in paths.items()}

    product_hash = source_hashes["products.parquet"]
    _require_equal("lexical product dataset hash", lexical.get("dataset_sha256"), product_hash)
    _require_equal("dense product dataset hash", dense.get("dataset_sha256"), product_hash)
    _require_equal(
        "reranker product dataset hash", reranker.get("product_dataset_sha256"), product_hash
    )

    expected_lexical = {
        "analyzer": settings.lexical.analyzer,
        "dtype": "float32",
        "lowercase": settings.lexical.lowercase,
        "max_features": settings.lexical.max_features,
        "min_df": settings.lexical.min_df,
        "ngram_range": [settings.lexical.ngram_min, settings.lexical.ngram_max],
        "norm": settings.lexical.norm,
        "sublinear_tf": settings.lexical.sublinear_tf,
    }
    _require_equal(
        "lexical vectorizer parameters", lexical.get("vectorizer_parameters"), expected_lexical
    )
    _require_equal("dense model", dense.get("model_name"), settings.dense.model_name)
    _require_equal(
        "dense embedding dimension",
        dense.get("embedding_dimension"),
        settings.dense.expected_dimension,
    )
    _require_equal(
        "dense normalization",
        dense.get("embedding_normalization"),
        "project_applied_l2_unit_normalization",
    )

    _validate_validation_provenance(
        hybrid_validation,
        report_name="hybrid validation report",
        source_hashes=source_hashes,
    )
    hybrid_selected = _mapping(hybrid_validation.get("selected_configuration"))
    _require_equal("hybrid strategy", hybrid_selected.get("strategy"), settings.hybrid.strategy)
    _require_close(
        "hybrid semantic weight",
        hybrid_selected.get("semantic_weight"),
        settings.hybrid.semantic_weight,
    )
    _require_equal(
        "hybrid candidate depth",
        hybrid_selected.get("candidate_depth"),
        settings.hybrid.candidate_depth,
    )
    _require_equal("hybrid RRF constant", hybrid_selected.get("rrf_k"), settings.hybrid.rrf_k)

    _validate_validation_provenance(
        reranker_validation,
        report_name="reranker validation report",
        source_hashes=source_hashes,
    )
    decision = _mapping(reranker_validation.get("production_decision"))
    reranker_eligible = (
        decision.get("reranker_improves_validation_ndcg_at_10") is True
        and decision.get("recommended_default_search_mode") == "reranker"
    )
    if settings.reranker.default_search_mode == "reranker" and not reranker_eligible:
        raise ValueError(
            "configuration enables the reranker although the validation decision did not"
        )
    _require_equal(
        "reranker candidate depth",
        reranker.get("candidate_depth"),
        settings.reranker.candidate_depth,
    )
    _require_equal("reranker class order", reranker.get("classes"), [0, 1, 2])
    _require_equal(
        "reranker expected relevance formula",
        reranker.get("expected_relevance_formula"),
        "P(Partial) * 1 + P(Exact) * 2",
    )
    selected_model = _mapping(
        _mapping(reranker_validation.get("model_search")).get("selected_configuration")
    )
    hyperparameters = _mapping(reranker.get("hyperparameters"))
    _require_close("reranker C", hyperparameters.get("C"), selected_model.get("C"))
    _require_equal(
        "reranker class weight",
        hyperparameters.get("class_weight"),
        selected_model.get("class_weight"),
    )
    _require_equal("reranker solver", hyperparameters.get("solver"), "lbfgs")
    _validate_source_hash_mapping(
        _mapping(reranker.get("source_hashes")),
        source_hashes,
        owner="reranker metadata",
    )

    lexical_artifacts = _artifact_hashes(lexical)
    dense_artifacts = _artifact_hashes(dense)
    reranker_artifacts = _artifact_hashes(reranker)
    return {
        "schema_version": 1,
        "frozen_before_test_evaluation": True,
        "source_hashes": source_hashes,
        "lexical": {
            "vectorizer_parameters": expected_lexical,
            "product_count": lexical.get("product_count"),
            "vocabulary_size": lexical.get("vocabulary_size"),
            "dataset_sha256": product_hash,
            "metadata_sha256": source_hashes["lexical_metadata.json"],
            "artifact_sha256": lexical_artifacts,
        },
        "semantic": {
            "model_name": settings.dense.model_name,
            "embedding_dimension": settings.dense.expected_dimension,
            "embedding_normalization": dense.get("embedding_normalization"),
            "embedding_dtype": dense.get("embedding_dtype"),
            "product_count": dense.get("product_count"),
            "dataset_sha256": product_hash,
            "metadata_sha256": source_hashes["dense_metadata.json"],
            "artifact_sha256": dense_artifacts,
        },
        "hybrid": {
            "strategy": settings.hybrid.strategy,
            "semantic_weight": settings.hybrid.semantic_weight,
            "lexical_weight": 1.0 - settings.hybrid.semantic_weight,
            "candidate_depth": settings.hybrid.candidate_depth,
            "rrf_k": settings.hybrid.rrf_k,
            "selection_split": "validation",
            "selection_metric": hybrid_validation.get("selection_metric"),
            "validation_report_sha256": source_hashes["hybrid_validation_metrics.json"],
        },
        "reranked_hybrid": {
            "eligible": reranker_eligible,
            "eligibility_policy": decision.get("policy"),
            "candidate_depth": settings.reranker.candidate_depth,
            "model_type": reranker.get("model_type"),
            "classes": reranker.get("classes"),
            "expected_relevance_formula": reranker.get("expected_relevance_formula"),
            "feature_schema_sha256": reranker.get("feature_schema_sha256"),
            "hyperparameters": dict(hyperparameters),
            "model_metadata_sha256": source_hashes["reranker_metadata.json"],
            "model_artifact_sha256": reranker_artifacts,
            "validation_report_sha256": source_hashes["reranker_validation_metrics.json"],
        },
    }


def run_final_test_benchmark(
    *,
    engines: Mapping[str, JudgedCandidateSearchEngine],
    frozen_configurations: Mapping[str, object],
    queries_path: Path,
    judgments_path: Path,
    splits_path: Path,
    output_dir: Path,
    relevant_threshold: int | float = 1,
    timestamp: datetime | None = None,
    clock: Callable[[], float] = time.perf_counter,
    warmup_query_count: int = 3,
    hardware: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Evaluate only held-out test queries once and write the complete final report family."""

    if frozen_configurations.get("frozen_before_test_evaluation") is not True:
        raise ValueError("final configurations must be frozen before test evaluation")
    reranker_config = _mapping(frozen_configurations.get("reranked_hybrid"))
    reranker_eligible = reranker_config.get("eligible") is True
    expected_systems = set(SYSTEM_ORDER[: 4 if reranker_eligible else 3])
    if set(engines) != expected_systems:
        raise ValueError(
            "final engine set does not match validation eligibility; "
            f"expected {sorted(expected_systems)}"
        )
    _assert_outputs_absent(output_dir)
    if warmup_query_count < 0:
        raise ValueError("warmup_query_count must not be negative")

    split_manifest = _read_json(splits_path.resolve())
    train_ids, validation_ids, test_ids = _split_ids(split_manifest)
    queries = pd.read_parquet(queries_path.resolve())
    judgments = pd.read_parquet(judgments_path.resolve())
    test_queries = _select_queries(queries, test_ids)
    test_judgments = _select_judgments(judgments, test_ids)

    warmup_queries = test_queries["query"].astype(str).head(warmup_query_count).tolist()
    for system in SYSTEM_ORDER:
        if system not in engines:
            continue
        for query in warmup_queries:
            engines[system].search(query, 10)

    system_reports: dict[str, object] = {}
    aggregate_rows: list[dict[str, object]] = []
    per_query_rows: list[dict[str, object]] = []
    for system in SYSTEM_ORDER:
        if system not in engines:
            continue
        engine = engines[system]
        judged_at_5 = judged_candidate_evaluation(
            engine,
            test_queries,
            test_judgments,
            top_k=5,
            relevant_threshold=relevant_threshold,
            clock=clock,
        )
        judged_at_10 = judged_candidate_evaluation(
            engine,
            test_queries,
            test_judgments,
            top_k=10,
            relevant_threshold=relevant_threshold,
            clock=clock,
        )
        full_at_10 = full_catalog_known_relevant_evaluation(
            engine,
            test_queries,
            test_judgments,
            top_k=10,
            relevant_threshold=relevant_threshold,
            clock=clock,
        )
        _require_successful_run(system, judged_at_5)
        _require_successful_run(system, judged_at_10)
        _require_successful_run(system, full_at_10)
        row = _aggregate_row(system, judged_at_5, judged_at_10, full_at_10)
        aggregate_rows.append(row)
        per_query_rows.extend(_per_query_rows(system, judged_at_5, judged_at_10, full_at_10))
        system_reports[system] = _system_report(judged_at_5, judged_at_10, full_at_10)

    selected_row = select_default_engine(aggregate_rows)
    effective_timestamp = timestamp or datetime.now(UTC)
    final_engine = _final_engine_payload(
        selected_row, frozen_configurations, timestamp=effective_timestamp
    )
    created_at = effective_timestamp.astimezone(UTC).isoformat()
    report: dict[str, object] = {
        "schema_version": 1,
        "created_at": created_at,
        "split": "test",
        "test_query_count": len(test_ids),
        "train_query_count_evaluated": 0,
        "validation_query_count_evaluated": 0,
        "binary_relevant_threshold": float(relevant_threshold),
        "binary_relevance_policy": "Exact_and_Partial_relevant_Irrelevant_not_relevant",
        "judged_candidate_policy": (
            "primary controlled benchmark; every ranked candidate has a WANDS judgment"
        ),
        "full_catalog_policy": (
            "known-relevant recovery only; unjudged retrieved products are unknown, not irrelevant"
        ),
        "latency_measurement": {
            "boundary": "end_to_end_engine.search(query, top_k=10)",
            "context": "full_catalog_known_relevant_evaluation",
            "warm_process": True,
            "warmup_query_count_per_system": len(warmup_queries),
            "model_and_index_initialization_excluded": True,
            "query_preprocessing_encoding_scoring_top_k_and_ranking_included": True,
        },
        "selection_policy": {
            "primary": "highest_test_judged_candidate_ndcg_at_10",
            "tie_break": "lower_full_catalog_median_latency_then_frozen_system_order",
            "tuning_after_test": False,
        },
        "split_counts": {
            "train": len(train_ids),
            "validation": len(validation_ids),
            "test": len(test_ids),
        },
        "source_hashes": dict(_mapping(frozen_configurations.get("source_hashes"))),
        "frozen_configurations": dict(frozen_configurations),
        "reranker_included": reranker_eligible,
        "systems": system_reports,
        "final_engine": final_engine,
        "hardware": dict(hardware or collect_hardware_metadata()),
        "generated_files": list(FINAL_OUTPUT_FILENAMES),
    }
    write_final_report_files(report, aggregate_rows, per_query_rows, output_dir)
    return report


def select_default_engine(rows: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    """Select on ranking quality first, then steady-state latency, without parameter tuning."""

    if not rows:
        raise ValueError("at least one final system metric row is required")
    names = [str(row["system"]) for row in rows]
    if len(set(names)) != len(names) or any(name not in SYSTEM_ORDER for name in names):
        raise ValueError("final metric rows must contain unique supported systems")
    order = {name: index for index, name in enumerate(SYSTEM_ORDER)}
    return max(
        rows,
        key=lambda row: (
            _as_float(row["ndcg_at_10"]),
            -_as_float(row["median_latency_ms"]),
            -order[str(row["system"])],
        ),
    )


def collect_hardware_metadata() -> dict[str, object]:
    """Capture the local warm-process benchmark environment without external tooling."""

    processor = platform.processor().strip() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown")
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": processor,
        "logical_cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
    }


def _aggregate_row(
    system: str,
    judged_at_5: EvaluationRun,
    judged_at_10: EvaluationRun,
    full_at_10: EvaluationRun,
) -> dict[str, object]:
    latency = _mapping(full_at_10.aggregate["latency_ms"])
    return {
        "system": system,
        "test_query_count": _as_int(judged_at_10.aggregate["query_count"]),
        "ndcg_at_5": _metric(judged_at_5, "ndcg_at_k"),
        "ndcg_at_10": _metric(judged_at_10, "ndcg_at_k"),
        "precision_at_5": _metric(judged_at_5, "precision_at_k"),
        "precision_at_10": _metric(judged_at_10, "precision_at_k"),
        "recall_at_5": _metric(judged_at_5, "recall_at_k"),
        "recall_at_10": _metric(judged_at_10, "recall_at_k"),
        "mrr_at_10": _metric(judged_at_10, "mrr_at_k"),
        "known_relevant_recall_at_10": _metric(full_at_10, "known_relevant_recall_at_k"),
        "known_relevant_mrr_at_10": _metric(full_at_10, "known_relevant_mrr_at_k"),
        "median_latency_ms": float(latency["median"]),
        "p95_latency_ms": float(latency["p95"]),
        "latency_sample_count": int(latency["sample_count"]),
    }


def _system_report(
    judged_at_5: EvaluationRun,
    judged_at_10: EvaluationRun,
    full_at_10: EvaluationRun,
) -> dict[str, object]:
    return {
        "judged_candidate_evaluation": {
            "query_count": judged_at_10.aggregate["query_count"],
            "eligible_query_count": judged_at_10.aggregate["eligible_query_count"],
            "ndcg_at_5": _metric(judged_at_5, "ndcg_at_k"),
            "ndcg_at_10": _metric(judged_at_10, "ndcg_at_k"),
            "precision_at_5": _metric(judged_at_5, "precision_at_k"),
            "precision_at_10": _metric(judged_at_10, "precision_at_k"),
            "recall_at_5": _metric(judged_at_5, "recall_at_k"),
            "recall_at_10": _metric(judged_at_10, "recall_at_k"),
            "mrr_at_10": _metric(judged_at_10, "mrr_at_k"),
        },
        "full_catalog_known_relevant_evaluation": {
            "query_count": full_at_10.aggregate["query_count"],
            "eligible_query_count": full_at_10.aggregate["eligible_query_count"],
            "known_relevant_recall_at_10": _metric(full_at_10, "known_relevant_recall_at_k"),
            "known_relevant_mrr_at_10": _metric(full_at_10, "known_relevant_mrr_at_k"),
            "latency_ms_at_10": dict(_mapping(full_at_10.aggregate["latency_ms"])),
            "unjudged_products_policy": "unknown_not_irrelevant",
        },
        "diagnostics": {
            "judged_at_5": list(judged_at_5.diagnostics),
            "judged_at_10": list(judged_at_10.diagnostics),
            "full_catalog_at_10": list(full_at_10.diagnostics),
        },
    }


def _per_query_rows(
    system: str,
    judged_at_5: EvaluationRun,
    judged_at_10: EvaluationRun,
    full_at_10: EvaluationRun,
) -> list[dict[str, object]]:
    at_5 = {str(row["query_id"]): row for row in judged_at_5.per_query}
    at_10 = {str(row["query_id"]): row for row in judged_at_10.per_query}
    full = {str(row["query_id"]): row for row in full_at_10.per_query}
    if set(at_5) != set(at_10) or set(at_5) != set(full):
        raise RuntimeError("per-query final evaluation rows are not aligned")
    return [
        {
            "system": system,
            "query_id": query_id,
            "ndcg_at_5": float(at_5[query_id]["ndcg_at_k"]),
            "ndcg_at_10": float(at_10[query_id]["ndcg_at_k"]),
            "precision_at_5": float(at_5[query_id]["precision_at_k"]),
            "precision_at_10": float(at_10[query_id]["precision_at_k"]),
            "recall_at_5": float(at_5[query_id]["recall_at_k"]),
            "recall_at_10": float(at_10[query_id]["recall_at_k"]),
            "mrr_at_10": float(at_10[query_id]["reciprocal_rank_at_k"]),
            "known_relevant_recall_at_10": float(full[query_id]["known_relevant_recall_at_k"]),
            "known_relevant_mrr_at_10": float(
                full[query_id]["known_relevant_reciprocal_rank_at_k"]
            ),
            "full_catalog_latency_ms": float(full[query_id]["latency_ms"]),
        }
        for query_id in sorted(at_5)
    ]


def _final_engine_payload(
    selected: Mapping[str, object],
    frozen: Mapping[str, object],
    *,
    timestamp: datetime | None,
) -> dict[str, object]:
    system = str(selected["system"])
    dependencies = {
        "lexical": ("lexical",),
        "semantic": ("semantic",),
        "hybrid": ("lexical", "semantic", "hybrid"),
        "reranked_hybrid": ("lexical", "semantic", "hybrid", "reranked_hybrid"),
    }[system]
    immutable_configuration = {
        "components": {name: frozen[name] for name in dependencies},
        "source_hashes": frozen["source_hashes"],
    }
    encoded = json.dumps(immutable_configuration, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return {
        "schema_version": 1,
        "created_at": (timestamp or datetime.now(UTC)).astimezone(UTC).isoformat(),
        "system": system,
        "selected_search_mode": SEARCH_MODES[system],
        "selection_policy": (
            "highest held-out judged-candidate nDCG@10; lower full-catalog median latency "
            "breaks an exact quality tie; frozen system order is the final deterministic tie-break"
        ),
        "selection_rationale": (
            f"{SYSTEM_LABELS[system]} had the highest held-out judged-candidate nDCG@10 "
            f"({_as_float(selected['ndcg_at_10']):.6f}). Its measured full-catalog median latency "
            f"was {_as_float(selected['median_latency_ms']):.3f} ms; ranking quality remained the "
            "primary criterion."
        ),
        "selected_metrics": dict(selected),
        "immutable_configuration": immutable_configuration,
        "immutable_configuration_sha256": hashlib.sha256(encoded).hexdigest(),
        "latency_considered": True,
    }


def _validate_validation_provenance(
    report: Mapping[str, Any],
    *,
    report_name: str,
    source_hashes: Mapping[str, str],
) -> None:
    split = report.get("split", report.get("selection_split"))
    _require_equal(f"{report_name} split", split, "validation")
    _require_equal(f"{report_name} test query count", report.get("test_query_count_evaluated"), 0)
    _validate_source_hash_mapping(
        _mapping(report.get("source_hashes")), source_hashes, owner=report_name
    )


def _validate_source_hash_mapping(
    recorded: Mapping[str, Any], actual: Mapping[str, str], *, owner: str
) -> None:
    for filename in ("queries.parquet", "evaluation_judgments.parquet", "query_splits.json"):
        _require_equal(
            f"{owner} source hash for {filename}", recorded.get(filename), actual[filename]
        )


def _artifact_hashes(metadata: Mapping[str, Any]) -> dict[str, str]:
    artifacts = _mapping(metadata.get("artifacts"))
    return {
        str(filename): str(_mapping(details).get("sha256"))
        for filename, details in artifacts.items()
    }


def _split_ids(manifest: Mapping[str, Any]) -> tuple[set[str], set[str], set[str]]:
    query_ids = manifest.get("query_ids")
    if not isinstance(query_ids, dict):
        raise ValueError("query split manifest is missing query_ids")
    values = [query_ids.get(name) for name in ("train", "validation", "test")]
    if any(not isinstance(value, list) for value in values):
        raise ValueError("query split manifest must contain train, validation, and test lists")
    train, validation, test = (set(map(str, cast(list[object], value))) for value in values)
    if not train or not validation or not test:
        raise ValueError("final evaluation requires non-empty train, validation, and test splits")
    if train & validation or train & test or validation & test:
        raise ValueError("train, validation, and test query IDs must be disjoint")
    return train, validation, test


def _select_queries(queries: DataFrame, test_ids: set[str]) -> DataFrame:
    missing = {"query_id", "query"} - set(queries.columns)
    if missing:
        raise ValueError(f"queries are missing required columns: {sorted(missing)}")
    normalized = queries.loc[:, ["query_id", "query"]].assign(
        query_id=queries["query_id"].astype(str)
    )
    selected = normalized.loc[normalized["query_id"].isin(test_ids)].copy()
    if set(selected["query_id"]) != test_ids or len(selected) != len(test_ids):
        raise ValueError("test query IDs must appear exactly once in the queries table")
    return selected.sort_values("query_id", kind="stable", ignore_index=True)


def _select_judgments(judgments: DataFrame, test_ids: set[str]) -> DataFrame:
    missing = {"query_id", "product_id", "relevance_grade"} - set(judgments.columns)
    if missing:
        raise ValueError(f"judgments are missing required columns: {sorted(missing)}")
    normalized = judgments.assign(query_id=judgments["query_id"].astype(str))
    selected = normalized.loc[normalized["query_id"].isin(test_ids)].copy()
    if set(selected["query_id"]) != test_ids:
        raise ValueError("every held-out test query must have canonical judgments")
    return selected.sort_values(["query_id", "product_id"], kind="stable", ignore_index=True)


def _require_successful_run(system: str, run: EvaluationRun) -> None:
    failed = _as_int(run.aggregate["failed_query_count"])
    if failed:
        raise RuntimeError(f"{system} produced {failed} failed queries during final evaluation")


def _metric(run: EvaluationRun, name: str) -> float:
    return float(_mapping(run.aggregate["metrics_all_queries"])[name])


def _as_float(value: object) -> float:
    return float(cast(Any, value))


def _as_int(value: object) -> int:
    return int(cast(Any, value))


def _assert_outputs_absent(output_dir: Path) -> None:
    existing = sorted(
        name for name in FINAL_OUTPUT_FILENAMES if (output_dir.resolve() / name).exists()
    )
    if existing:
        raise FileExistsError(
            f"final evaluation outputs already exist and are immutable: {existing}"
        )


def _require_equal(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise ValueError(f"frozen {label} mismatch: expected {expected!r}, got {actual!r}")


def _require_close(label: str, actual: object, expected: object) -> None:
    try:
        matches = np.isclose(
            float(cast(Any, actual)), float(cast(Any, expected)), rtol=0, atol=1e-12
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"frozen {label} is not numeric: {actual!r}") from error
    if not bool(matches):
        raise ValueError(f"frozen {label} mismatch: expected {expected!r}, got {actual!r}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as input_file:
            return cast(dict[str, Any], json.load(input_file))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read final evaluation input {path}: {error}") from error


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("expected a mapping in final evaluation metadata")
    return cast(Mapping[str, Any], value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products", type=Path)
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--judgments", type=Path)
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--lexical-index-dir", type=Path)
    parser.add_argument("--dense-index-dir", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--hybrid-validation-report", type=Path)
    parser.add_argument("--reranker-validation-report", type=Path)
    parser.add_argument("--reports-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Verify frozen artifacts, then optionally run the one-time held-out benchmark."""

    arguments = _build_parser().parse_args(argv)
    settings = load_settings()
    products_path = arguments.products or settings.paths.processed_data / "products.parquet"
    queries_path = arguments.queries or settings.paths.processed_data / "queries.parquet"
    judgments_path = (
        arguments.judgments or settings.paths.processed_data / "evaluation_judgments.parquet"
    )
    splits_path = arguments.splits or settings.paths.processed_data / "query_splits.json"
    lexical_dir = arguments.lexical_index_dir or settings.paths.indexes / "tfidf"
    dense_dir = arguments.dense_index_dir or settings.paths.embeddings / "dense"
    model_dir = arguments.model_dir or settings.paths.models / "reranker"
    hybrid_report = (
        arguments.hybrid_validation_report
        or settings.paths.reports / "hybrid_validation_metrics.json"
    )
    reranker_report = (
        arguments.reranker_validation_report
        or settings.paths.reports / "reranker_validation_metrics.json"
    )
    reports_dir = arguments.reports_dir or settings.paths.reports
    frozen = freeze_final_configurations(
        settings,
        products_path=products_path,
        queries_path=queries_path,
        judgments_path=judgments_path,
        splits_path=splits_path,
        lexical_metadata_path=lexical_dir / "metadata.json",
        dense_metadata_path=dense_dir / "metadata.json",
        reranker_metadata_path=model_dir / "metadata.json",
        hybrid_validation_report_path=hybrid_report,
        reranker_validation_report_path=reranker_report,
    )
    if arguments.verify_only:
        print(json.dumps(frozen, indent=2, sort_keys=True))
        return 0
    _assert_outputs_absent(reports_dir)

    from product_search.indexing.dense import FastEmbedProvider
    from product_search.ranking.features import ProductFeatureStore
    from product_search.ranking.model import load_relevance_model
    from product_search.ranking.reranker import RerankingSearchEngine
    from product_search.retrieval.hybrid import HybridSearchEngine
    from product_search.retrieval.lexical import LexicalSearchEngine
    from product_search.retrieval.semantic import SemanticSearchEngine

    provider = FastEmbedProvider(
        settings.dense.model_name,
        cache_dir=settings.paths.embeddings / "model_cache",
        local_files_only=arguments.local_files_only,
    )
    lexical = LexicalSearchEngine.from_index_dir(lexical_dir)
    semantic = SemanticSearchEngine.from_index_dir(
        dense_dir,
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
    engines: dict[str, JudgedCandidateSearchEngine] = {
        "lexical": lexical,
        "semantic": semantic,
        "hybrid": hybrid,
    }
    if _mapping(frozen["reranked_hybrid"])["eligible"] is True:
        product_store = ProductFeatureStore.from_parquet(products_path)
        model = load_relevance_model(
            model_dir,
            expected_product_dataset_sha256=product_store.dataset_sha256,
            expected_candidate_depth=settings.reranker.candidate_depth,
        )
        engines["reranked_hybrid"] = RerankingSearchEngine(
            hybrid,
            model,
            product_store,
            candidate_depth=settings.reranker.candidate_depth,
        )
    report = run_final_test_benchmark(
        engines=engines,
        frozen_configurations=frozen,
        queries_path=queries_path,
        judgments_path=judgments_path,
        splits_path=splits_path,
        output_dir=reports_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
