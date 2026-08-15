# Semantic Product Search Engine

A portfolio-quality, CPU-friendly product search system built on the official Wayfair WANDS
catalog. The project will compare TF-IDF lexical retrieval, dense semantic retrieval, hybrid
retrieval, and a lightweight supervised relevance reranker under a shared offline evaluation
framework.

Implementation is in progress. The repository foundation and reproducible WANDS ingestion pipeline
are complete. Search indexes, ranking models, the API, and the user interface have not been built
yet.

## Prepare WANDS data

Download only the three required official files, then validate and prepare local Parquet tables:

```powershell
uv run python -m product_search.data.download
uv run python -m product_search.data.prepare
```

Raw data, processed tables, manifests, and generated reports remain local and are excluded from
Git. See [the data-source documentation](docs/data-source.md) for provenance, license, schemas,
verified counts, and limitations.

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
