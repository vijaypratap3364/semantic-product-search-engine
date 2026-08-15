"""Tests for deterministic WANDS product document construction and outputs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from product_search.data.prepare import (
    build_product_text,
    normalize_whitespace,
    prepare_wands_data,
)
from product_search.data.prepare import main as prepare_main

RELEVANCE_MAPPING = {"Exact": 2, "Partial": 1, "Irrelevant": 0}


def test_product_text_normalizes_whitespace_and_missing_fields() -> None:
    product = {
        "product_name": "  Modern   Lamp ",
        "product_class": "Lighting",
        "category_hierarchy": None,
        "product_description": float("nan"),
        "product_features": "Color: Black\nStyle: Modern",
    }

    assert build_product_text(product) == "Modern Lamp Lighting Color: Black Style: Modern"
    assert normalize_whitespace(None) == ""
    assert normalize_whitespace(" multiple\tspaces ") == "multiple spaces"


def test_prepare_writes_parquet_tables_and_actual_summary(
    tmp_path: Path,
    raw_wands_dir: Path,
) -> None:
    processed_dir = tmp_path / "processed"
    report_path = tmp_path / "reports" / "data_summary.json"
    timestamp = datetime(2026, 2, 3, 4, 5, tzinfo=UTC)

    summary = prepare_wands_data(
        raw_dir=raw_wands_dir,
        processed_dir=processed_dir,
        report_path=report_path,
        relevance_mapping=RELEVANCE_MAPPING,
        timestamp=timestamp,
    )

    products = pd.read_parquet(processed_dir / "products.parquet")
    queries = pd.read_parquet(processed_dir / "queries.parquet")
    labels = pd.read_parquet(processed_dir / "labels.parquet")
    saved_summary = json.loads(report_path.read_text(encoding="utf-8"))

    assert list(products["product_id"]) == ["1001", "1002", "1003"]
    assert "nan" not in " ".join(products["product_text"]).lower()
    assert products.loc[1, "product_text"].startswith("Round Coffee Table")
    assert len(queries) == 2
    assert list(labels["relevance_grade"]) == [2, 0, 1, 0]
    assert summary["product_count"] == 3
    assert summary["query_count"] == 2
    assert summary["label_count"] == 4
    assert summary["relevance_distribution"] == {
        "Exact": 1,
        "Partial": 1,
        "Irrelevant": 2,
    }
    assert summary["missing_value_counts"]["products"]["product_description"] == 1
    assert summary["duplicate_counts"]["label_query_product_pairs"] == 0
    assert summary["created_at"] == timestamp.isoformat()
    assert summary["source_manifest_sha256"] is None
    assert saved_summary == summary


def test_prepare_cli_uses_explicit_paths(
    tmp_path: Path,
    raw_wands_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    processed_dir = tmp_path / "cli-processed"
    report_path = tmp_path / "cli-reports" / "summary.json"

    exit_code = prepare_main(
        [
            "--raw-dir",
            str(raw_wands_dir),
            "--processed-dir",
            str(processed_dir),
            "--report",
            str(report_path),
        ]
    )

    assert exit_code == 0
    assert '"product_count": 3' in capsys.readouterr().out
    assert (processed_dir / "products.parquet").is_file()
    assert report_path.is_file()
