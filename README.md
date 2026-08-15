# Semantic Product Search Engine

A portfolio-quality, CPU-friendly product search system built on the official Wayfair WANDS
catalog. The project will compare TF-IDF lexical retrieval, dense semantic retrieval, hybrid
retrieval, and a lightweight supervised relevance reranker under a shared offline evaluation
framework.

Implementation is in progress. The current foundation establishes the Python package,
configuration contract, development tooling, and test harness. It does not yet download WANDS,
build search indexes, expose an API, or provide a user interface.

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
