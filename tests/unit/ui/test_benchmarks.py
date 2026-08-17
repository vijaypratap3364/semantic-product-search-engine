"""Tests for verified benchmark report transformation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from product_search.ui.benchmarks import BenchmarkReportError, load_benchmark_report


def _report() -> dict[str, object]:
    return {
        "split": "test",
        "created_at": "2026-08-17T02:48:14+00:00",
        "test_query_count": 72,
        "systems": {
            "lexical": {
                "judged_candidate_evaluation": {
                    "ndcg_at_10": 0.74,
                    "recall_at_10": 0.10,
                    "mrr_at_10": 0.93,
                },
                "full_catalog_known_relevant_evaluation": {"latency_ms_at_10": {"median": 196.3}},
            },
            "semantic": {
                "judged_candidate_evaluation": {
                    "ndcg_at_10": 0.81,
                    "recall_at_10": 0.11,
                    "mrr_at_10": 0.96,
                },
                "full_catalog_known_relevant_evaluation": {"latency_ms_at_10": {"median": 7.85}},
            },
        },
    }


def _write_report(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_benchmark_report_projects_verified_metrics(tmp_path: Path) -> None:
    path = tmp_path / "final_test_metrics.json"
    _write_report(path, _report())

    report = load_benchmark_report(path)

    assert report.created_at == "2026-08-17T02:48:14+00:00"
    assert report.test_query_count == 72
    assert [row.system for row in report.rows] == ["lexical", "semantic"]
    assert report.rows[0].display_name == "Lexical"
    assert report.rows[0].ndcg_at_10 == 0.74
    assert report.rows[0].recall_at_10 == 0.10
    assert report.rows[0].mrr_at_10 == 0.93
    assert report.rows[0].median_latency_ms == 196.3


def test_missing_report_has_actionable_build_command(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkReportError, match="benchmark_final --local-files-only"):
        load_benchmark_report(tmp_path / "missing.json")


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("not json", "not valid JSON"),
        ({"split": "validation", "systems": {}}, "held-out test split"),
        (
            {
                "split": "test",
                "created_at": "now",
                "test_query_count": 1,
                "systems": {"unknown": {}},
            },
            "no recognized search systems",
        ),
    ],
)
def test_invalid_report_envelopes_are_rejected(
    tmp_path: Path,
    payload: object,
    message: str,
) -> None:
    path = tmp_path / "report.json"
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        _write_report(path, payload)

    with pytest.raises(BenchmarkReportError, match=message):
        load_benchmark_report(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ndcg_at_10", "high", "must be numeric"),
        ("recall_at_10", -0.1, "non-negative and finite"),
        ("mrr_at_10", float("inf"), "non-negative and finite"),
    ],
)
def test_invalid_metric_values_are_rejected(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    payload = _report()
    systems = payload["systems"]
    assert isinstance(systems, dict)
    lexical = systems["lexical"]
    assert isinstance(lexical, dict)
    quality = lexical["judged_candidate_evaluation"]
    assert isinstance(quality, dict)
    quality[field] = value
    path = tmp_path / "report.json"
    _write_report(path, payload)

    with pytest.raises(BenchmarkReportError, match=message):
        load_benchmark_report(path)
