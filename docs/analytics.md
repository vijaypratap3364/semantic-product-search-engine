# Local search analytics and feedback

This is a local portfolio demo, not a production analytics service. It uses Python's standard
`sqlite3` module and does not require PostgreSQL, a hosted database, credentials, or network
access. The default database is `data/local/search_analytics.sqlite`; SQLite files and sidecar
files are ignored by Git.

## Privacy and configuration

Query logging is enabled by default because feedback must reference a persisted search. Raw query
text therefore remains on the local machine in the SQLite file. Do not enter sensitive or personal
information into the demo.

Disable query persistence in `configs/base.toml`:

```toml
[analytics]
query_logging_enabled = false
database_path = "data/local/search_analytics.sqlite"
```

When disabled, searches still succeed but return `search_id: null`; no search row is written and
new feedback cannot be associated with that search.

## Stored data

`search_events` contains:

- search ID and UTC timestamp;
- raw query, resolved search mode, requested `top_k`, and measured service latency;
- returned product IDs serialized as a JSON array; and
- optional caller-provided session ID.

`feedback_events` contains:

- feedback ID, referenced search ID, and UTC timestamp;
- product ID; and
- feedback type: `relevant`, `not_relevant`, or `clicked`.

SQLite foreign keys reject feedback for an unknown search. The repository also rejects feedback
for products that were not returned by that search.

## API behavior

A successful logged `/search` response includes its `search_id`. Submit feedback with:

```json
{
  "search_id": "6e91b735-2fb3-4957-8931-13dc71bd6ceb",
  "product_id": "12345",
  "feedback_type": "relevant"
}
```

`GET /analytics/summary` returns only aggregate search count, feedback count, average latency,
counts by mode, and counts by feedback type. It never returns query text, session IDs, or product
IDs.

SQLite initialization or write failures are logged inside the local process. They do not turn a
successful product search into an API failure; the response instead contains `search_id: null`.
