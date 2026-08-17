"""SQLite schema creation and short-lived connection management."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = 1

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS search_events (
    search_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    query TEXT NOT NULL,
    mode TEXT NOT NULL,
    top_k INTEGER NOT NULL CHECK (top_k > 0),
    latency_ms REAL NOT NULL CHECK (latency_ms >= 0),
    returned_product_ids TEXT NOT NULL,
    session_id TEXT
);

CREATE TABLE IF NOT EXISTS feedback_events (
    feedback_id TEXT PRIMARY KEY,
    search_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    product_id TEXT NOT NULL,
    feedback_type TEXT NOT NULL CHECK (
        feedback_type IN ('relevant', 'not_relevant', 'clicked')
    ),
    FOREIGN KEY (search_id) REFERENCES search_events(search_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_feedback_events_search_id
ON feedback_events(search_id);
"""


@dataclass(frozen=True, slots=True)
class SQLiteAnalyticsDatabase:
    """A local SQLite file opened only for the duration of each operation."""

    path: Path
    timeout_seconds: float = 5.0

    def initialize(self) -> None:
        """Create the parent directory and idempotent schema."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            connection.executescript(SCHEMA_SQL)
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Yield one foreign-key-enabled connection and close it deterministically."""

        connection = sqlite3.connect(self.path, timeout=self.timeout_seconds)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
