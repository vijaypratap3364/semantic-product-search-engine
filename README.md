# Semantic Product Search Engine

A portfolio-quality, CPU-friendly product search system built on the official Wayfair WANDS
catalog. The project will compare TF-IDF lexical retrieval, dense semantic retrieval, hybrid
retrieval, and a lightweight supervised relevance reranker under a shared offline evaluation
framework.

Implementation is in progress. The repository foundation, reproducible WANDS ingestion, canonical
evaluation judgments, deterministic query splits, offline evaluation framework, TF-IDF lexical
baseline, dense semantic retrieval, and validation-tuned hybrid retrieval are complete. Ranking
models, the API, and the user interface have not been built yet.

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
uv run python -m product_search.evaluation.benchmark_lexical
uv run python -m product_search.evaluation.benchmark_semantic --local-files-only
uv run python -m product_search.evaluation.benchmark_hybrid --local-files-only
```

Raw data, processed tables, manifests, and generated reports remain local and are excluded from
Git. See [the data-source documentation](docs/data-source.md) for provenance, license, schemas,
verified counts, and limitations. See [the evaluation documentation](docs/evaluation.md) for the
canonical judgment policy, query partitions, metric definitions, evaluation modes, and reporting.
See [the lexical baseline report](docs/lexical-baseline.md) for index configuration, measured
validation metrics, latency, and error analysis. The [dense semantic report](docs/semantic-search.md)
and [hybrid retrieval report](docs/hybrid-search.md) document their measured validation results and
selection policies.

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
