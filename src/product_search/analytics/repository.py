"""Typed persistence operations for local search and feedback analytics."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, cast
from uuid import uuid4

from product_search.analytics.database import SQLiteAnalyticsDatabase

FeedbackType = Literal["relevant", "not_relevant", "clicked"]
VALID_FEEDBACK_TYPES = frozenset({"relevant", "not_relevant", "clicked"})


class AnalyticsRepositoryError(RuntimeError):
    """Base class for local analytics persistence errors."""


class SearchEventNotFoundError(AnalyticsRepositoryError):
    """Raised when feedback references an unknown search event."""


class ProductNotInSearchError(AnalyticsRepositoryError):
    """Raised when feedback references a product absent from the search results."""


class QueryLoggingDisabledError(AnalyticsRepositoryError):
    """Raised when feedback is requested while query logging is disabled."""


@dataclass(frozen=True, slots=True)
class SearchEvent:
    """One locally persisted successful search."""

    search_id: str
    timestamp: str
    query: str
    mode: str
    top_k: int
    latency_ms: float
    returned_product_ids: tuple[str, ...]
    session_id: str | None


@dataclass(frozen=True, slots=True)
class FeedbackEvent:
    """One explicit product-level feedback event."""

    feedback_id: str
    search_id: str
    timestamp: str
    product_id: str
    feedback_type: FeedbackType


@dataclass(frozen=True, slots=True)
class AnalyticsSummary:
    """Aggregate-only local demo statistics without queries or product IDs."""

    query_logging_enabled: bool
    search_count: int
    feedback_count: int
    average_latency_ms: float
    searches_by_mode: dict[str, int]
    feedback_by_type: dict[str, int]


class AnalyticsRepository:
    """Persist local demo analytics with no long-lived shared connection."""

    def __init__(
        self,
        database: SQLiteAnalyticsDatabase,
        *,
        query_logging_enabled: bool = True,
        clock: Callable[[], datetime] | None = None,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._database = database
        self.query_logging_enabled = query_logging_enabled
        self._clock = clock or (lambda: datetime.now(UTC))
        self._id_factory = id_factory or (lambda: str(uuid4()))

    def initialize(self) -> None:
        """Create the database schema even when query persistence is disabled."""

        self._database.initialize()

    def log_search(
        self,
        *,
        query: str,
        mode: str,
        top_k: int,
        latency_ms: float,
        returned_product_ids: Sequence[str],
        session_id: str | None = None,
    ) -> SearchEvent | None:
        """Persist a successful search, or return ``None`` when logging is disabled."""

        if not self.query_logging_enabled:
            return None
        normalized_query = _required_text(query, "query")
        normalized_mode = _required_text(mode, "mode")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if not math.isfinite(latency_ms) or latency_ms < 0:
            raise ValueError("latency_ms must be a non-negative finite number")
        product_ids = tuple(
            _required_text(product_id, "product_id") for product_id in returned_product_ids
        )
        if len(product_ids) != len(set(product_ids)):
            raise ValueError("returned product IDs must be unique")
        normalized_session = _optional_text(session_id)
        event = SearchEvent(
            search_id=_required_text(self._id_factory(), "search_id"),
            timestamp=_utc_timestamp(self._clock()),
            query=normalized_query,
            mode=normalized_mode,
            top_k=top_k,
            latency_ms=float(latency_ms),
            returned_product_ids=product_ids,
            session_id=normalized_session,
        )
        with self._database.connection() as connection:
            connection.execute(
                """
                INSERT INTO search_events (
                    search_id, timestamp, query, mode, top_k, latency_ms,
                    returned_product_ids, session_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.search_id,
                    event.timestamp,
                    event.query,
                    event.mode,
                    event.top_k,
                    event.latency_ms,
                    json.dumps(event.returned_product_ids, separators=(",", ":")),
                    event.session_id,
                ),
            )
        return event

    def get_search(self, search_id: str) -> SearchEvent | None:
        """Return one event for feedback validation and deterministic tests."""

        normalized_id = _required_text(search_id, "search_id")
        with self._database.connection() as connection:
            row = connection.execute(
                """
                SELECT search_id, timestamp, query, mode, top_k, latency_ms,
                       returned_product_ids, session_id
                FROM search_events
                WHERE search_id = ?
                """,
                (normalized_id,),
            ).fetchone()
        return None if row is None else _search_event_from_row(row)

    def log_feedback(
        self,
        *,
        search_id: str,
        product_id: str,
        feedback_type: FeedbackType,
    ) -> FeedbackEvent:
        """Persist validated feedback tied to a known returned product."""

        if not self.query_logging_enabled:
            raise QueryLoggingDisabledError("query logging is disabled")
        normalized_search_id = _required_text(search_id, "search_id")
        normalized_product_id = _required_text(product_id, "product_id")
        if feedback_type not in VALID_FEEDBACK_TYPES:
            raise ValueError(f"unsupported feedback type: {feedback_type!r}")
        search = self.get_search(normalized_search_id)
        if search is None:
            raise SearchEventNotFoundError(f"unknown search ID: {normalized_search_id}")
        if normalized_product_id not in search.returned_product_ids:
            raise ProductNotInSearchError(
                f"product {normalized_product_id!r} was not returned by search "
                f"{normalized_search_id!r}"
            )
        event = FeedbackEvent(
            feedback_id=_required_text(self._id_factory(), "feedback_id"),
            search_id=normalized_search_id,
            timestamp=_utc_timestamp(self._clock()),
            product_id=normalized_product_id,
            feedback_type=feedback_type,
        )
        with self._database.connection() as connection:
            connection.execute(
                """
                INSERT INTO feedback_events (
                    feedback_id, search_id, timestamp, product_id, feedback_type
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event.feedback_id,
                    event.search_id,
                    event.timestamp,
                    event.product_id,
                    event.feedback_type,
                ),
            )
        return event

    def summary(self) -> AnalyticsSummary:
        """Return aggregate counts only; never return query or product-level data."""

        with self._database.connection() as connection:
            search_row = connection.execute(
                """
                SELECT COUNT(*) AS search_count,
                       COALESCE(AVG(latency_ms), 0.0) AS average_latency_ms
                FROM search_events
                """
            ).fetchone()
            feedback_row = connection.execute(
                "SELECT COUNT(*) AS feedback_count FROM feedback_events"
            ).fetchone()
            mode_rows = connection.execute(
                "SELECT mode, COUNT(*) AS count FROM search_events GROUP BY mode ORDER BY mode"
            ).fetchall()
            feedback_rows = connection.execute(
                """
                SELECT feedback_type, COUNT(*) AS count
                FROM feedback_events
                GROUP BY feedback_type
                ORDER BY feedback_type
                """
            ).fetchall()
        if search_row is None or feedback_row is None:
            raise AnalyticsRepositoryError("analytics aggregate query returned no row")
        return AnalyticsSummary(
            query_logging_enabled=self.query_logging_enabled,
            search_count=int(search_row["search_count"]),
            feedback_count=int(feedback_row["feedback_count"]),
            average_latency_ms=float(search_row["average_latency_ms"]),
            searches_by_mode={str(row["mode"]): int(row["count"]) for row in mode_rows},
            feedback_by_type={
                str(row["feedback_type"]): int(row["count"]) for row in feedback_rows
            },
        )


def _search_event_from_row(row: Any) -> SearchEvent:
    raw_ids = json.loads(str(row["returned_product_ids"]))
    if not isinstance(raw_ids, list) or not all(isinstance(value, str) for value in raw_ids):
        raise AnalyticsRepositoryError("stored returned_product_ids is not a string list")
    return SearchEvent(
        search_id=str(row["search_id"]),
        timestamp=str(row["timestamp"]),
        query=str(row["query"]),
        mode=str(row["mode"]),
        top_k=int(row["top_k"]),
        latency_ms=float(row["latency_ms"]),
        returned_product_ids=tuple(cast(list[str], raw_ids)),
        session_id=None if row["session_id"] is None else str(row["session_id"]),
    )


def _utc_timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("analytics timestamps must be timezone-aware")
    return value.astimezone(UTC).isoformat()


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")
    return value.strip()


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    return _required_text(value, "session_id")
