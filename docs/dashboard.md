# Streamlit search dashboard

The Streamlit dashboard is a presentation client of the local FastAPI application. It imports no
retrieval, indexing, ranking, service, or analytics implementation. Every search, metadata lookup,
and feedback action goes through the HTTP API in `product_search.ui.api_client`.

## Run locally

Prepare the generated search artifacts first. Then start FastAPI from the repository root in one
PowerShell terminal:

```powershell
uv run uvicorn product_search.api.main:app --reload
```

After `GET http://127.0.0.1:8000/ready` reports ready, start Streamlit in a second terminal:

```powershell
uv run streamlit run src/product_search/ui/app.py
```

The default API origin and request timeout are configured in `configs/base.toml` under `[ui]`.
FastAPI and Streamlit remain separate processes; Streamlit never loads the TF-IDF matrix, dense
embeddings, FastEmbed model, or reranker.

## Pages

- **Search** provides a query box, retrieval mode, bounded top-K control, ranked product cards,
  component scores, and deterministic “Why this result?” evidence. Relevant and Not relevant
  feedback buttons call `POST /feedback` and are disabled when the API did not return a
  `search_id`.
- **Comparison mode** sends the same query and top-K to lexical, semantic, and hybrid API modes,
  then displays the independent responses in tabs. It does not combine or recompute scores in the
  browser.
- **Benchmarks** reads `artifacts/reports/final_test_metrics.json` at runtime. nDCG@10, Recall@10,
  and MRR@10 are the held-out judged-candidate metrics. Median latency is the held-out
  full-catalog end-to-end latency at K=10. Missing or malformed reports produce an actionable
  message; no fallback values are embedded in the UI.
- **System** reads `/health`, `/ready`, and `/model` to show API health, selected default engine,
  embedding model, indexed product count, build timestamp, and immutable artifact version.

## Failure and privacy behavior

API connection failures and structured API errors are shown without exposing response internals.
Searches still follow the local-demo analytics policy documented in [analytics.md](analytics.md):
query logging can be disabled, and feedback is available only for a logged search event. The UI
does not access SQLite directly.
