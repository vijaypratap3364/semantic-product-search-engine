"""Validated view model for the generated held-out benchmark report."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


class BenchmarkReportError(RuntimeError):
    """The verified final benchmark report is missing or incompatible."""


@dataclass(frozen=True, slots=True)
class BenchmarkRow:
    """Dashboard fields sourced from one evaluated search system."""

    system: str
    display_name: str
    ndcg_at_10: float
    recall_at_10: float
    mrr_at_10: float
    median_latency_ms: float


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Small display-safe projection of final_test_metrics.json."""

    created_at: str
    test_query_count: int
    rows: tuple[BenchmarkRow, ...]


SYSTEM_DISPLAY_NAMES = {
    "lexical": "Lexical",
    "semantic": "Semantic",
    "hybrid": "Hybrid",
    "reranked_hybrid": "Reranked hybrid",
}


def load_benchmark_report(path: Path) -> BenchmarkReport:
    """Load real Stage 8 metrics, rejecting missing or malformed reports."""

    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise BenchmarkReportError(
            "Final benchmark metrics are unavailable. Run "
            "`uv run python -m product_search.evaluation.benchmark_final --local-files-only`."
        ) from error
    except json.JSONDecodeError as error:
        raise BenchmarkReportError("Final benchmark metrics are not valid JSON.") from error

    root = _object(decoded, "report")
    if root.get("split") != "test":
        raise BenchmarkReportError("Final benchmark report must use the held-out test split.")
    systems = _object(root.get("systems"), "systems")
    rows = tuple(
        _parse_system(system_name, _object(systems.get(system_name), system_name))
        for system_name in SYSTEM_DISPLAY_NAMES
        if system_name in systems
    )
    if not rows:
        raise BenchmarkReportError("Final benchmark report contains no recognized search systems.")
    return BenchmarkReport(
        created_at=_string(root, "created_at"),
        test_query_count=_integer(root, "test_query_count"),
        rows=rows,
    )


def _parse_system(system_name: str, payload: dict[str, Any]) -> BenchmarkRow:
    quality = _object(payload.get("judged_candidate_evaluation"), "judged evaluation")
    catalog = _object(
        payload.get("full_catalog_known_relevant_evaluation"),
        "full-catalog evaluation",
    )
    latency = _object(catalog.get("latency_ms_at_10"), "latency")
    return BenchmarkRow(
        system=system_name,
        display_name=SYSTEM_DISPLAY_NAMES[system_name],
        ndcg_at_10=_metric(quality, "ndcg_at_10"),
        recall_at_10=_metric(quality, "recall_at_10"),
        mrr_at_10=_metric(quality, "mrr_at_10"),
        median_latency_ms=_metric(latency, "median"),
    )


def _object(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkReportError(f"Final benchmark {context} must be an object.")
    return cast(dict[str, Any], value)


def _string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise BenchmarkReportError(f"Final benchmark {key} must be a string.")
    return value


def _integer(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BenchmarkReportError(f"Final benchmark {key} must be a non-negative integer.")
    return value


def _metric(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise BenchmarkReportError(f"Final benchmark metric {key} must be numeric.")
    metric = float(value)
    if not math.isfinite(metric) or metric < 0.0:
        raise BenchmarkReportError(f"Final benchmark metric {key} must be non-negative and finite.")
    return metric
