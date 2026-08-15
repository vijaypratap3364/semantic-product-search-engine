"""Tests for engine-independent evaluation modes and report generation."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pytest
from pandas import DataFrame

from product_search.evaluation.evaluator import (
    full_catalog_known_relevant_evaluation,
    judged_candidate_evaluation,
    write_evaluation_reports,
)
from product_search.retrieval.base import SearchResult


class FakeEngine:
    """Deterministic engine that records the controlled candidate sets it receives."""

    def __init__(self) -> None:
        self.candidate_calls: list[tuple[str, tuple[str, ...], int]] = []

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        rankings = {
            "lamp": ["unjudged", "p1"],
            "table": ["p3", "unjudged"],
        }
        return [
            SearchResult(product_id=product_id, rank=rank, score=1.0 / rank)
            for rank, product_id in enumerate(rankings[query][:top_k], start=1)
        ]

    def search_candidates(
        self,
        query: str,
        candidate_product_ids: Sequence[str],
        top_k: int,
    ) -> list[SearchResult]:
        candidates = tuple(candidate_product_ids)
        self.candidate_calls.append((query, candidates, top_k))
        preferred = {"lamp": ["p1", "p2"], "table": ["p3"]}[query]
        ranked = [product_id for product_id in preferred if product_id in candidates][:top_k]
        return [
            SearchResult(product_id=product_id, rank=rank, score=1.0 / rank)
            for rank, product_id in enumerate(ranked, start=1)
        ]


def _queries() -> DataFrame:
    return DataFrame({"query_id": ["q1", "q2"], "query": ["lamp", "table"]})


def _judgments() -> DataFrame:
    return DataFrame(
        {
            "query_id": ["q1", "q1", "q2"],
            "product_id": ["p1", "p2", "p3"],
            "relevance_grade": [2, 0, 0],
        }
    )


def test_judged_candidate_evaluation_ranks_only_labeled_products() -> None:
    engine = FakeEngine()

    run = judged_candidate_evaluation(engine, _queries(), _judgments(), top_k=2)

    first = run.per_query[0]
    assert run.mode == "judged_candidate_evaluation"
    assert engine.candidate_calls == [
        ("lamp", ("p1", "p2"), 2),
        ("table", ("p3",), 2),
    ]
    assert first["ndcg_at_k"] == pytest.approx(1.0)
    assert first["precision_at_k"] == pytest.approx(0.5)
    assert first["recall_at_k"] == pytest.approx(1.0)
    assert run.aggregate["eligible_query_count"] == 1
    assert run.aggregate["unjudged_products_policy"] == "not_applicable_all_candidates_are_judged"
    assert any(diagnostic["code"] == "no_relevant_judgments" for diagnostic in run.diagnostics)


def test_full_catalog_mode_only_reports_known_relevant_recovery() -> None:
    run = full_catalog_known_relevant_evaluation(FakeEngine(), _queries(), _judgments(), top_k=2)

    first = run.per_query[0]
    metrics = run.aggregate["metrics_all_queries"]
    assert run.mode == "full_catalog_known_relevant_evaluation"
    assert first["known_relevant_recovered_at_k"] == 1
    assert first["known_relevant_recall_at_k"] == pytest.approx(1.0)
    assert first["known_relevant_reciprocal_rank_at_k"] == pytest.approx(0.5)
    assert "precision_at_k" not in first
    assert metrics == {
        "known_relevant_recall_at_k": pytest.approx(0.5),
        "known_relevant_mrr_at_k": pytest.approx(0.25),
    }
    assert run.aggregate["unjudged_products_policy"] == "unknown_not_irrelevant"


def test_relevance_threshold_is_configurable() -> None:
    judgments = _judgments().copy()
    judgments.loc[judgments["product_id"].eq("p1"), "relevance_grade"] = 1

    run = judged_candidate_evaluation(
        FakeEngine(), _queries(), judgments, top_k=2, relevant_threshold=2
    )

    assert run.per_query[0]["relevant_judgment_count"] == 0
    assert run.aggregate["binary_relevant_threshold"] == 2.0


class InvalidEngine(FakeEngine):
    def search_candidates(
        self,
        query: str,
        candidate_product_ids: Sequence[str],
        top_k: int,
    ) -> list[SearchResult]:
        return [SearchResult(product_id="outside", rank=2, score=1.0)]


def test_engine_contract_failure_is_recorded_without_aborting_other_queries() -> None:
    run = judged_candidate_evaluation(InvalidEngine(), _queries(), _judgments(), top_k=2)

    assert all(row["status"] == "error" for row in run.per_query)
    assert run.aggregate["failed_query_count"] == 2
    assert [diagnostic["code"] for diagnostic in run.diagnostics] == [
        "search_error",
        "search_error",
    ]
    assert "contiguous" in run.diagnostics[0]["message"]


def test_report_writer_creates_csv_aggregate_latency_and_diagnostics(tmp_path: Path) -> None:
    run = judged_candidate_evaluation(FakeEngine(), _queries(), _judgments(), top_k=2)

    paths = write_evaluation_reports(run, tmp_path, report_name="fixture")

    with paths["per_query_csv"].open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    aggregate = json.loads(paths["aggregate_json"].read_text(encoding="utf-8"))
    diagnostics = json.loads(paths["diagnostics_json"].read_text(encoding="utf-8"))
    assert len(rows) == 2
    assert aggregate["mode"] == "judged_candidate_evaluation"
    assert aggregate["latency_ms"]["sample_count"] == 2
    assert diagnostics["diagnostics"]


@pytest.mark.parametrize(
    ("queries", "judgments", "message"),
    [
        (DataFrame({"query_id": ["q1"]}), _judgments(), "queries are missing"),
        (_queries(), DataFrame({"query_id": ["q1"]}), "judgments are missing"),
        (
            _queries(),
            pd.concat([_judgments(), _judgments().iloc[[0]]], ignore_index=True),
            "exactly one row",
        ),
        (
            _queries(),
            DataFrame({"query_id": ["q1"], "product_id": ["p1"], "relevance_grade": [-1]}),
            "finite and non-negative",
        ),
    ],
)
def test_invalid_evaluation_inputs_fail_clearly(
    queries: DataFrame, judgments: DataFrame, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        judged_candidate_evaluation(FakeEngine(), queries, judgments, top_k=2)


def test_invalid_top_k_fails_clearly() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        judged_candidate_evaluation(FakeEngine(), _queries(), _judgments(), top_k=0)
