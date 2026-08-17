"""Tests for local search and feedback persistence."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from product_search.analytics.database import SQLiteAnalyticsDatabase
from product_search.analytics.repository import (
    AnalyticsRepository,
    ProductNotInSearchError,
    QueryLoggingDisabledError,
    SearchEventNotFoundError,
)

TIMESTAMP = datetime(2026, 8, 16, 12, 0, tzinfo=UTC)


def _repository(
    tmp_path: Path,
    *,
    ids: list[str] | None = None,
    query_logging_enabled: bool = True,
) -> tuple[AnalyticsRepository, Path]:
    database_path = tmp_path / "search_analytics.sqlite"
    generated_ids = iter(ids or ["search-1", "feedback-1", "event-3", "event-4"])
    repository = AnalyticsRepository(
        SQLiteAnalyticsDatabase(database_path),
        query_logging_enabled=query_logging_enabled,
        clock=lambda: TIMESTAMP,
        id_factory=lambda: next(generated_ids),
    )
    repository.initialize()
    return repository, database_path


def test_search_logging_round_trip(tmp_path: Path) -> None:
    repository, _ = _repository(tmp_path)

    event = repository.log_search(
        query="modern black desk lamp",
        mode="hybrid",
        top_k=2,
        latency_ms=4.25,
        returned_product_ids=("p1", "p2"),
        session_id="demo-session",
    )

    assert event is not None
    assert event.search_id == "search-1"
    assert event.timestamp == TIMESTAMP.isoformat()
    assert event.returned_product_ids == ("p1", "p2")
    assert repository.get_search("search-1") == event


def test_feedback_logging_requires_known_search_and_returned_product(tmp_path: Path) -> None:
    repository, _ = _repository(tmp_path)
    search = repository.log_search(
        query="lamp",
        mode="semantic",
        top_k=1,
        latency_ms=3.0,
        returned_product_ids=("p1",),
    )
    assert search is not None

    feedback = repository.log_feedback(
        search_id=search.search_id,
        product_id="p1",
        feedback_type="relevant",
    )

    assert feedback.feedback_id == "feedback-1"
    assert feedback.search_id == search.search_id
    assert feedback.product_id == "p1"
    assert feedback.feedback_type == "relevant"
    with pytest.raises(SearchEventNotFoundError, match="unknown search ID"):
        repository.log_feedback(
            search_id="missing-search",
            product_id="p1",
            feedback_type="clicked",
        )
    with pytest.raises(ProductNotInSearchError, match="was not returned"):
        repository.log_feedback(
            search_id=search.search_id,
            product_id="p2",
            feedback_type="not_relevant",
        )


def test_query_logging_disabled_does_not_persist_searches(tmp_path: Path) -> None:
    repository, database_path = _repository(tmp_path, query_logging_enabled=False)

    event = repository.log_search(
        query="private query",
        mode="lexical",
        top_k=10,
        latency_ms=1.0,
        returned_product_ids=("p1",),
    )

    assert database_path.is_file()
    assert event is None
    assert repository.get_search("search-1") is None
    assert repository.summary().search_count == 0
    with pytest.raises(QueryLoggingDisabledError, match="disabled"):
        repository.log_feedback(
            search_id="search-1",
            product_id="p1",
            feedback_type="clicked",
        )


def test_analytics_summary_contains_only_aggregates(tmp_path: Path) -> None:
    repository, _ = _repository(
        tmp_path,
        ids=["search-1", "search-2", "feedback-1", "feedback-2"],
    )
    first = repository.log_search(
        query="private first query",
        mode="hybrid",
        top_k=2,
        latency_ms=10.0,
        returned_product_ids=("p1", "p2"),
    )
    second = repository.log_search(
        query="private second query",
        mode="semantic",
        top_k=1,
        latency_ms=20.0,
        returned_product_ids=("p3",),
    )
    assert first is not None and second is not None
    repository.log_feedback(
        search_id=first.search_id,
        product_id="p1",
        feedback_type="clicked",
    )
    repository.log_feedback(
        search_id=second.search_id,
        product_id="p3",
        feedback_type="relevant",
    )

    summary = repository.summary()

    assert summary.query_logging_enabled is True
    assert summary.search_count == 2
    assert summary.feedback_count == 2
    assert summary.average_latency_ms == pytest.approx(15.0)
    assert summary.searches_by_mode == {"hybrid": 1, "semantic": 1}
    assert summary.feedback_by_type == {"clicked": 1, "relevant": 1}
    assert "private" not in repr(summary)
