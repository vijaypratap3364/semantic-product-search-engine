"""Tests for the local SQLite analytics schema."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from product_search.analytics.database import SCHEMA_VERSION, SQLiteAnalyticsDatabase


def test_database_initialization_creates_expected_schema(tmp_path: Path) -> None:
    database_path = tmp_path / "nested" / "search_analytics.sqlite"
    database = SQLiteAnalyticsDatabase(database_path)

    database.initialize()
    database.initialize()

    assert database_path.is_file()
    with sqlite3.connect(database_path) as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            )
        }
        search_columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(search_events)")
        }
        feedback_columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(feedback_events)")
        }
        schema_version = int(connection.execute("PRAGMA user_version").fetchone()[0])

    assert tables == {"feedback_events", "search_events"}
    assert search_columns == {
        "search_id",
        "timestamp",
        "query",
        "mode",
        "top_k",
        "latency_ms",
        "returned_product_ids",
        "session_id",
    }
    assert feedback_columns == {
        "feedback_id",
        "search_id",
        "timestamp",
        "product_id",
        "feedback_type",
    }
    assert schema_version == SCHEMA_VERSION
