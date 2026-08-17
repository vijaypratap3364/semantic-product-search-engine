# Product search API

The FastAPI application in `product_search.api.main` is a thin transport over `SearchService`.
FastAPI lifespan loads and validates the static search service once per process; endpoint handlers
reuse it and never rebuild or reload indexes.

## Local development

Prepare the selected artifacts described in [the service documentation](service.md), synchronize
the environment, and run:

```powershell
uv sync
uv run uvicorn product_search.api.main:app --reload
```

The local OpenAPI documentation is available at `http://127.0.0.1:8000/docs` while the server is
running. Model loading is local-only at API startup.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Process liveness and process-local request counters |
| `GET` | `/ready` | Whether the verified search artifacts loaded successfully |
| `GET` | `/model` | Safe immutable selected-model and artifact metadata |
| `GET` | `/modes` | Search modes loaded by the service |
| `POST` | `/search` | Execute one product search |

`/model` returns the resolved default mode, embedding model, product count, immutable configuration
hash as the artifact version, and final-selection build timestamp. It does not expose artifact
paths.

## Search contract

Example request:

```json
{
  "query": "modern black desk lamp",
  "top_k": 10,
  "mode": "hybrid"
}
```

Validation rules are:

- `query` is trimmed, non-empty, and at most 500 characters;
- `top_k` is a strict integer from 1 through 100; and
- `mode` is one of `default`, `lexical`, `semantic`, `hybrid`, or `rerank`.

The response contains the normalized query, resolved mode, service latency, result count, and
ranked display-safe results. Result objects contain the Stage 9 service fields and deterministic
score explanations. They do not expose `product_text`, product feature strings, local paths, or
other internal artifact data.

## Readiness and errors

Process liveness and search readiness are separate. If required artifacts are missing or fail
compatibility validation, application startup completes so `/health` remains available, while
`/ready` returns `ready: false`. Search, model, and modes requests then return HTTP 503.

All errors use a stable envelope:

```json
{
  "error": {
    "code": "request_validation_error",
    "message": "Request validation failed.",
    "details": [
      {
        "location": ["body", "query"],
        "message": "String should have at least 1 character",
        "error_type": "string_too_short"
      }
    ]
  }
}
```

Validation details contain only location, message, and error type. Startup failures and unexpected
errors return sanitized messages without filesystem paths, exception strings, or stack traces.

The process keeps request count, error count, and average end-to-end HTTP latency in memory. These
counters are informational, reset on process restart, and require no Prometheus service.
