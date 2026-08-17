"""Tests for validation-only hybrid fusion selection and reporting."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pandas import DataFrame

from product_search.evaluation import benchmark_hybrid as benchmark_hybrid_module
from product_search.evaluation.benchmark_hybrid import (
    _MemoizedEngine,
    run_hybrid_validation_benchmark,
)
from product_search.indexing import dense as dense_module
from product_search.retrieval import lexical as lexical_module
from product_search.retrieval import semantic as semantic_module
from product_search.retrieval.base import SearchResult


class RecordingScoreEngine:
    def __init__(self, scores: dict[str, dict[str, float]]) -> None:
        self.scores = scores
        self.seen_queries: list[str] = []
        self.catalog_calls = 0
        self.candidate_calls = 0

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        self.seen_queries.append(query)
        self.catalog_calls += 1
        return self._rank(query, list(self.scores[query]), top_k)

    def search_candidates(
        self,
        query: str,
        candidate_product_ids: list[str],
        top_k: int,
    ) -> list[SearchResult]:
        self.seen_queries.append(query)
        self.candidate_calls += 1
        return self._rank(query, candidate_product_ids, top_k)

    def _rank(self, query: str, product_ids: list[str], top_k: int) -> list[SearchResult]:
        ordered = sorted(
            product_ids, key=lambda product_id: (-self.scores[query][product_id], product_id)
        )
        return [
            SearchResult(product_id=product_id, rank=rank, score=self.scores[query][product_id])
            for rank, product_id in enumerate(ordered[:top_k], start=1)
        ]


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    queries_path = tmp_path / "queries.parquet"
    judgments_path = tmp_path / "evaluation_judgments.parquet"
    splits_path = tmp_path / "query_splits.json"
    DataFrame(
        {
            "query_id": ["train-1", "valid-1", "valid-2", "test-1"],
            "query": ["training query", "round table", "blue rug", "held out query"],
        }
    ).to_parquet(queries_path, index=False)
    DataFrame(
        {
            "query_id": ["valid-1", "valid-1", "valid-2", "valid-2", "test-1"],
            "product_id": ["p1", "p2", "p2", "p3", "p3"],
            "relevance_grade": [2, 0, 2, 0, 2],
            "label": ["Exact", "Irrelevant", "Exact", "Irrelevant", "Exact"],
        }
    ).to_parquet(judgments_path, index=False)
    splits_path.write_text(
        json.dumps(
            {
                "query_ids": {
                    "train": ["train-1"],
                    "validation": ["valid-1", "valid-2"],
                    "test": ["test-1"],
                }
            }
        ),
        encoding="utf-8",
    )
    return queries_path, judgments_path, splits_path


def test_hybrid_benchmark_searches_validation_only_and_records_every_configuration(
    tmp_path: Path,
) -> None:
    queries_path, judgments_path, splits_path = _write_fixture(tmp_path)
    reports_dir = tmp_path / "reports"
    search_path = reports_dir / "hybrid_weight_search.csv"
    report_path = reports_dir / "hybrid_validation_metrics.json"
    lexical = RecordingScoreEngine(
        {
            "round table": {"p1": 10.0, "p2": 0.0, "p3": 0.0},
            "blue rug": {"p1": 0.0, "p2": 0.0, "p3": 10.0},
        }
    )
    semantic = RecordingScoreEngine(
        {
            "round table": {"p1": 0.0, "p2": 10.0, "p3": 0.0},
            "blue rug": {"p1": 0.0, "p2": 10.0, "p3": 0.0},
        }
    )

    report = run_hybrid_validation_benchmark(
        lexical_engine=lexical,
        semantic_engine=semantic,
        queries_path=queries_path,
        judgments_path=judgments_path,
        splits_path=splits_path,
        weight_search_path=search_path,
        report_path=report_path,
        semantic_weight_grid=[index / 10 for index in range(11)],
        candidate_depth=3,
        rrf_k=60,
        timestamp=datetime(2026, 8, 16, 12, 0, tzinfo=UTC),
    )

    assert report["split"] == "validation"
    assert report["validation_query_count"] == 2
    assert report["allowed_train_query_count"] == 1
    assert report["train_query_count_evaluated"] == 0
    assert report["held_out_test_query_count"] == 1
    assert report["test_query_count_evaluated"] == 0
    assert "held out query" not in lexical.seen_queries
    assert "held out query" not in semantic.seen_queries
    assert "training query" not in lexical.seen_queries
    assert "training query" not in semantic.seen_queries
    assert report["selected_configuration"]["strategy"] == "weighted_normalized"  # type: ignore[index]
    assert report["selected_configuration"]["semantic_weight"] == 0.5  # type: ignore[index]
    assert report["judged_candidate_evaluation"]["ndcg_at_10"] == 1.0  # type: ignore[index]
    assert set(report["error_analysis"]) == {  # type: ignore[arg-type]
        "metric",
        "strict_comparison_policy",
        "lexical_wins",
        "semantic_wins",
        "hybrid_improves_both",
        "fusion_hurts",
    }
    with search_path.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    assert len(rows) == 12
    assert {row["strategy"] for row in rows} == {"weighted_normalized", "rrf"}
    assert {row["test_query_count_evaluated"] for row in rows} == {"0"}
    assert json.loads(report_path.read_text(encoding="utf-8"))["split"] == "validation"
    assert (reports_dir / "hybrid_validation_judged_k10_per_query.csv").is_file()
    assert (reports_dir / "hybrid_validation_full_catalog_k10_aggregate.json").is_file()


def test_memoized_engine_reuses_catalog_and_candidate_rankings() -> None:
    underlying = RecordingScoreEngine({"query": {"p1": 1.0, "p2": 0.0}})
    cached = _MemoizedEngine(underlying)

    assert cached.search("query", 2) == cached.search("query", 2)
    assert cached.search_candidates("query", ["p1", "p2"], 2) == cached.search_candidates(
        "query", ["p1", "p2"], 2
    )
    assert underlying.catalog_calls == 1
    assert underlying.candidate_calls == 1


@pytest.mark.parametrize(
    "weights, message",
    [
        ([], "must not be empty"),
        ([-0.1, 0.5], "between 0.0 and 1.0"),
        ([0.5, 0.5], "unique and ascending"),
        ([0.7, 0.2], "unique and ascending"),
    ],
)
def test_hybrid_benchmark_rejects_invalid_weight_grids(
    tmp_path: Path,
    weights: list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        run_hybrid_validation_benchmark(
            lexical_engine=RecordingScoreEngine({}),
            semantic_engine=RecordingScoreEngine({}),
            queries_path=tmp_path / "missing-queries.parquet",
            judgments_path=tmp_path / "missing-judgments.parquet",
            splits_path=tmp_path / "missing-splits.json",
            weight_search_path=tmp_path / "weights.csv",
            report_path=tmp_path / "report.json",
            semantic_weight_grid=weights,
            candidate_depth=10,
            rrf_k=60,
        )


def test_hybrid_benchmark_rejects_split_leakage_and_missing_validation_queries(
    tmp_path: Path,
) -> None:
    queries_path, judgments_path, splits_path = _write_fixture(tmp_path)
    splits_path.write_text(
        json.dumps(
            {
                "query_ids": {
                    "train": ["train-1"],
                    "validation": ["valid-1"],
                    "test": ["valid-1"],
                }
            }
        ),
        encoding="utf-8",
    )
    common = {
        "lexical_engine": RecordingScoreEngine({}),
        "semantic_engine": RecordingScoreEngine({}),
        "queries_path": queries_path,
        "judgments_path": judgments_path,
        "splits_path": splits_path,
        "weight_search_path": tmp_path / "weights.csv",
        "report_path": tmp_path / "report.json",
        "semantic_weight_grid": [0.5],
        "candidate_depth": 10,
        "rrf_k": 60,
    }
    with pytest.raises(ValueError, match="must be disjoint"):
        run_hybrid_validation_benchmark(**common)

    splits_path.write_text(
        json.dumps(
            {
                "query_ids": {
                    "train": ["train-1"],
                    "validation": ["absent"],
                    "test": ["test-1"],
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="absent from queries table"):
        run_hybrid_validation_benchmark(**common)


def test_hybrid_benchmark_cli_wires_configured_engines(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    lexical_engine = object()
    semantic_engine = object()
    provider = object()
    captured: dict[str, object] = {}

    monkeypatch.setattr(dense_module, "FastEmbedProvider", lambda *args, **kwargs: provider)
    monkeypatch.setattr(
        lexical_module.LexicalSearchEngine,
        "from_index_dir",
        lambda *args, **kwargs: lexical_engine,
    )
    monkeypatch.setattr(
        semantic_module.SemanticSearchEngine,
        "from_index_dir",
        lambda *args, **kwargs: semantic_engine,
    )

    def fake_run(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"split": "validation", "test_query_count_evaluated": 0}

    monkeypatch.setattr(benchmark_hybrid_module, "run_hybrid_validation_benchmark", fake_run)

    assert benchmark_hybrid_module.main(["--local-files-only"]) == 0
    assert captured["lexical_engine"] is lexical_engine
    assert captured["semantic_engine"] is semantic_engine
    assert '"test_query_count_evaluated": 0' in capsys.readouterr().out
