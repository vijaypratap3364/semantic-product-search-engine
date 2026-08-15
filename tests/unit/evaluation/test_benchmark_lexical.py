"""Tests for validation-only lexical benchmark orchestration."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pandas import DataFrame

from product_search.config import LexicalSettings
from product_search.evaluation.benchmark_lexical import main as benchmark_main
from product_search.evaluation.benchmark_lexical import run_lexical_validation_benchmark
from product_search.indexing.tfidf import build_tfidf_index


def test_benchmark_uses_validation_queries_and_writes_actual_reports(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    products_path = tmp_path / "products.parquet"
    queries_path = tmp_path / "queries.parquet"
    judgments_path = tmp_path / "evaluation_judgments.parquet"
    splits_path = tmp_path / "query_splits.json"
    index_dir = tmp_path / "tfidf"
    report_path = tmp_path / "reports" / "lexical_validation_metrics.json"

    DataFrame(
        {
            "product_id": ["p1", "p2", "p3"],
            "product_name": ["Round table", "Blue rug", "Desk lamp"],
            "product_text": ["round coffee table", "blue outdoor rug", "black desk lamp"],
        }
    ).to_parquet(products_path, index=False)
    DataFrame(
        {
            "query_id": ["v1", "v2", "t1"],
            "query": ["round table", "blue rug", "desk lamp"],
        }
    ).to_parquet(queries_path, index=False)
    DataFrame(
        {
            "query_id": ["v1", "v1", "v2", "v2", "t1"],
            "product_id": ["p1", "p2", "p2", "p3", "p3"],
            "label": ["Exact", "Irrelevant", "Exact", "Irrelevant", "Exact"],
            "relevance_grade": [2, 0, 2, 0, 2],
        }
    ).to_parquet(judgments_path, index=False)
    splits_path.write_text(
        json.dumps(
            {
                "query_ids": {
                    "train": [],
                    "validation": ["v1", "v2"],
                    "test": ["t1"],
                }
            }
        ),
        encoding="utf-8",
    )
    build_tfidf_index(
        products_path,
        index_dir,
        settings=LexicalSettings(min_df=1, max_features=None),
    )

    report = run_lexical_validation_benchmark(
        index_dir=index_dir,
        products_path=products_path,
        queries_path=queries_path,
        judgments_path=judgments_path,
        splits_path=splits_path,
        report_path=report_path,
        timestamp=datetime(2026, 8, 14, 12, 0, tzinfo=UTC),
    )

    judged = report["judged_candidate_evaluation"]
    full_catalog = report["full_catalog_known_relevant_evaluation"]
    assert report["split"] == "validation"
    assert report["validation_query_count"] == 2
    assert report["test_query_count_evaluated"] == 0
    assert report["held_out_test_query_count"] == 1
    assert judged["ndcg_at_5"] == 1.0  # type: ignore[index]
    assert judged["ndcg_at_10"] == 1.0  # type: ignore[index]
    assert full_catalog["known_relevant_recall_at_10"] == 1.0  # type: ignore[index]
    assert report["error_analysis"]  # generated from actual fixture results
    assert json.loads(report_path.read_text(encoding="utf-8"))["split"] == "validation"

    per_query_path = report_path.parent / "lexical_validation_judged_k10_per_query.csv"
    with per_query_path.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    assert {row["query_id"] for row in rows} == {"v1", "v2"}
    assert "t1" not in {row["query_id"] for row in rows}

    cli_report = report_path.parent / "cli_metrics.json"
    exit_code = benchmark_main(
        [
            "--index-dir",
            str(index_dir),
            "--products",
            str(products_path),
            "--queries",
            str(queries_path),
            "--judgments",
            str(judgments_path),
            "--splits",
            str(splits_path),
            "--report",
            str(cli_report),
        ]
    )
    assert exit_code == 0
    assert cli_report.is_file()
    assert '"split": "validation"' in capsys.readouterr().out
