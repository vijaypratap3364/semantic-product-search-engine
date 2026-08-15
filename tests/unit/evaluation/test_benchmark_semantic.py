"""Tests for validation-only semantic benchmark orchestration and comparison."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from pandas import DataFrame

from product_search.config import DenseSettings, LexicalSettings
from product_search.evaluation import benchmark_semantic as benchmark_semantic_module
from product_search.evaluation.benchmark_lexical import run_lexical_validation_benchmark
from product_search.evaluation.benchmark_semantic import run_semantic_validation_benchmark
from product_search.indexing import dense as dense_module
from product_search.indexing.dense import build_dense_index
from product_search.indexing.tfidf import build_tfidf_index


class RecordingFakeEmbeddingProvider:
    model_name = "fake/benchmark"
    provider_name = "fake"
    provider_version = "1.0"

    def __init__(self) -> None:
        self.seen_queries: list[str] = []
        self.vectors = {
            "round coffee table": [1.0, 0.0, 0.0],
            "blue outdoor rug": [0.0, 1.0, 0.0],
            "black desk lamp": [0.0, 0.0, 1.0],
            "seating": [1.0, 0.0, 0.0],
            "round table": [1.0, 0.0, 0.0],
            "blue rug": [0.0, 1.0, 0.0],
            "desk lamp": [0.0, 0.0, 1.0],
        }

    def embed_documents(
        self, texts: Sequence[str], *, batch_size: int
    ) -> Iterable[NDArray[np.float32]]:
        return (np.asarray(self.vectors[text], dtype=np.float32) for text in texts)

    def embed_queries(
        self, texts: Sequence[str], *, batch_size: int
    ) -> Iterable[NDArray[np.float32]]:
        self.seen_queries.extend(texts)
        return (np.asarray(self.vectors[text], dtype=np.float32) for text in texts)


def test_semantic_benchmark_uses_validation_only_and_compares_lexical(tmp_path: Path) -> None:
    products_path = tmp_path / "products.parquet"
    queries_path = tmp_path / "queries.parquet"
    judgments_path = tmp_path / "evaluation_judgments.parquet"
    splits_path = tmp_path / "query_splits.json"
    lexical_index_dir = tmp_path / "indexes" / "tfidf"
    semantic_index_dir = tmp_path / "embeddings" / "dense"
    reports_dir = tmp_path / "reports"
    lexical_report_path = reports_dir / "lexical_validation_metrics.json"
    semantic_report_path = reports_dir / "semantic_validation_metrics.json"

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
            "query": ["seating", "blue rug", "desk lamp"],
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
        json.dumps({"query_ids": {"train": [], "validation": ["v1", "v2"], "test": ["t1"]}}),
        encoding="utf-8",
    )
    build_tfidf_index(
        products_path,
        lexical_index_dir,
        settings=LexicalSettings(min_df=1, max_features=None),
    )
    run_lexical_validation_benchmark(
        index_dir=lexical_index_dir,
        products_path=products_path,
        queries_path=queries_path,
        judgments_path=judgments_path,
        splits_path=splits_path,
        report_path=lexical_report_path,
    )
    provider = RecordingFakeEmbeddingProvider()
    build_dense_index(
        products_path,
        semantic_index_dir,
        provider=provider,
        settings=DenseSettings(model_name="fake/benchmark", expected_dimension=3, batch_size=2),
    )

    report = run_semantic_validation_benchmark(
        provider=provider,
        index_dir=semantic_index_dir,
        lexical_index_dir=lexical_index_dir,
        products_path=products_path,
        queries_path=queries_path,
        judgments_path=judgments_path,
        splits_path=splits_path,
        lexical_report_path=lexical_report_path,
        lexical_per_query_path=reports_dir / "lexical_validation_judged_k10_per_query.csv",
        report_path=semantic_report_path,
        expected_dimension=3,
        timestamp=datetime(2026, 8, 14, 12, 0, tzinfo=UTC),
    )

    assert report["split"] == "validation"
    assert report["validation_query_count"] == 2
    assert report["test_query_count_evaluated"] == 0
    assert report["held_out_test_query_count"] == 1
    assert "desk lamp" not in provider.seen_queries
    assert report["comparison"]
    examples = report["qualitative_examples"][  # type: ignore[index]
        "semantic_low_lexical_overlap_successes"
    ]
    assert examples
    assert examples[0]["query"] == "seating"
    assert examples[0]["query_title_unigram_overlap"] == 0.0
    assert json.loads(semantic_report_path.read_text(encoding="utf-8"))["split"] == "validation"
    with (reports_dir / "semantic_validation_judged_k10_per_query.csv").open(
        encoding="utf-8", newline=""
    ) as input_file:
        rows = list(csv.DictReader(input_file))
    assert {row["query_id"] for row in rows} == {"v1", "v2"}


def test_semantic_benchmark_cli_wires_configured_paths(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    provider = RecordingFakeEmbeddingProvider()
    provider.model_name = "BAAI/bge-small-en-v1.5"
    captured: dict[str, object] = {}

    def fake_run(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"split": "validation", "test_query_count_evaluated": 0}

    monkeypatch.setattr(dense_module, "FastEmbedProvider", lambda *args, **kwargs: provider)
    monkeypatch.setattr(
        benchmark_semantic_module,
        "run_semantic_validation_benchmark",
        fake_run,
    )

    exit_code = benchmark_semantic_module.main([])

    assert exit_code == 0
    assert captured["provider"] is provider
    assert captured["expected_dimension"] == 384
    assert '"split": "validation"' in capsys.readouterr().out
