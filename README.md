# Semantic Product Search Engine

[![CI](https://github.com/vijaypratap3364/semantic-product-search-engine/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/vijaypratap3364/semantic-product-search-engine/actions/workflows/ci.yml)

A portfolio-quality, CPU-friendly product search system built on the official Wayfair WANDS
catalog. The project will compare TF-IDF lexical retrieval, dense semantic retrieval, hybrid
retrieval, and a lightweight supervised relevance reranker under a shared offline evaluation
framework.

Implementation is in progress. The repository foundation, reproducible WANDS ingestion, canonical
evaluation judgments, deterministic query splits, offline evaluation framework, TF-IDF lexical
baseline, dense semantic retrieval, validation-tuned hybrid retrieval, and a train-only lightweight
relevance reranker are complete. The frozen one-time held-out test evaluation is also complete; the
artifact-verified Python search service, FastAPI search API, local SQLite analytics, and Streamlit
search dashboard are complete.

## Prepare WANDS data

Download only the three required official files, then validate and prepare local Parquet tables:

```powershell
uv run python -m product_search.data.download
uv run python -m product_search.data.prepare
uv run python -m product_search.evaluation.judgments
uv run python -m product_search.evaluation.splits
uv run python -m product_search.indexing.build_lexical
uv run python -m product_search.indexing.build_dense
uv run python -m product_search.retrieval.lexical "round coffee table"
uv run python -m product_search.retrieval.semantic "round coffee table"
uv run python -m product_search.retrieval.hybrid "round coffee table" --local-files-only
uv run python -m product_search.ranking.reranker "round coffee table" --local-files-only
uv run python -m product_search.evaluation.benchmark_lexical
uv run python -m product_search.evaluation.benchmark_semantic --local-files-only
uv run python -m product_search.evaluation.benchmark_hybrid --local-files-only
uv run python -m product_search.evaluation.benchmark_reranker --local-files-only --force
uv run python -m product_search.evaluation.benchmark_final --verify-only --local-files-only
uv run python -m product_search.evaluation.benchmark_final --local-files-only
uv run python -m product_search.evaluation.benchmark_performance --local-files-only
uv run python -m product_search.service "round coffee table" --mode default --top-k 10
uv run uvicorn product_search.api.main:app --reload
```

With FastAPI running, start the dashboard from a second PowerShell terminal:

```powershell
uv run streamlit run src/product_search/ui/app.py
```

Raw data, processed tables, manifests, and generated reports remain local and are excluded from
Git. See [the data-source documentation](docs/data-source.md) for provenance, license, schemas,
verified counts, and limitations. See [the evaluation documentation](docs/evaluation.md) for the
canonical judgment policy, query partitions, metric definitions, evaluation modes, and reporting.
See [the lexical baseline report](docs/lexical-baseline.md) for index configuration, measured
validation metrics, latency, and error analysis. The [dense semantic report](docs/semantic-search.md)
and [hybrid retrieval report](docs/hybrid-search.md) document their measured validation results and
selection policies. See [the lightweight reranker report](docs/reranker.md) for its leakage controls,
model-selection grid, classification diagnostics, ranking comparison, and production decision.
The [search model card](docs/search-model-card.md) records the frozen held-out comparison, selected
default, latency boundary, hardware, limitations, failure modes, biases, and unjudged-product
caveat. See [the service-layer documentation](docs/service.md) for the Python interface, artifact
validation, response contract, deterministic explanations, and measured startup/query latency.
See [the API documentation](docs/api.md) for endpoints, validation limits, safe error responses,
and the local development command. See [the analytics documentation](docs/analytics.md) for the
local-only privacy boundary, database schema, feedback contract, and query-logging toggle.
See [the dashboard guide](docs/dashboard.md) for the two-process local demo, comparison mode,
feedback controls, benchmark provenance, and UI failure behavior.
See [the performance analysis](docs/performance.md) for the bounded benchmark protocol, measured
component/API latency, memory and disk usage, profile-guided optimization, factual error categories,
and exact-search scaling boundary.
See [the continuous-integration guide](docs/continuous-integration.md) for the fixture-only GitHub
Actions contract, offline model/data safeguards, coverage threshold, and matching local commands.

## Development

The project targets Python 3.12 and uses [`uv`](https://docs.astral.sh/uv/) for dependency and
environment management.

```powershell
uv sync
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
```

See [the implementation plan](docs/implementation-plan.md) for the approved architecture and
staged delivery approach.
