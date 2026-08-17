# Search service layer

`product_search.service.SearchService` is the single Python boundary between the generated search
artifacts and future transports such as FastAPI. It loads and verifies static artifacts once in
`SearchService.load()`; repeated calls to `search()` reuse the same lexical matrix, dense embedding
matrix, embedding provider, hybrid engine, product display catalog, and selected reranker.

## Interface and modes

```python
service = SearchService.load()
response = service.search(query="round coffee table", top_k=10, mode="default")
```

Supported modes are `lexical`, `semantic`, `hybrid`, `default`, and `rerank` when the frozen final
selection includes an eligible reranker. `default` resolves to the immutable Stage 8 selection; it
is not a second configurable search strategy. The current frozen selection resolves it to
`rerank`. A request cannot change fusion weights, candidate depth, model parameters, or another
frozen retrieval setting.

Each result contains only display-oriented catalog fields and ranking information:

- product ID, name, class, category hierarchy, and a whitespace-normalized description truncated
  to 240 characters;
- rank and final score;
- raw lexical and semantic scores when those modalities apply; and
- matched query terms in the title plus lexical and semantic fusion contributions.

The explanation is deterministic. It uses token overlap and existing score components, not an LLM.
For a reranked result, `final_score` is the expected relevance predicted by the three-class model;
the lexical and semantic contributions describe the hybrid candidate score feeding that model.
`product_text`, raw product features, and full descriptions are not exposed.

Queries must be non-blank and `top_k` must be an integer from 1 through 100. The service reports its
resolved mode and end-to-end service latency in the response.

## Artifact compatibility and startup failures

Startup validates the immutable final-selection hash, product data hash, selected component
metadata hashes, dense model name and dimension, artifact hashes, product counts, and identical
lexical/dense/catalog product ordering. When reranking is selected, it additionally validates
eligibility, model metadata, feature schema, product data hash, and candidate depth. Corrupt,
incompatible, or partially rebuilt artifacts fail startup rather than mixing versions.

The required local artifacts are produced by these commands:

```powershell
uv run python -m product_search.data.prepare
uv run python -m product_search.indexing.build_lexical
uv run python -m product_search.indexing.build_dense
uv run python -m product_search.evaluation.benchmark_reranker --local-files-only
uv run python -m product_search.evaluation.benchmark_final --local-files-only
```

Missing-artifact errors name the relevant command. Model loading is local-only by default, so a
service process does not silently download model files at startup.

## Measured service benchmark

The Stage 9 measurement used the real 42,994-product frozen artifacts and selected reranker on an
AMD Ryzen 5 8640HS CPU (6 cores, 12 logical processors), Windows 11 10.0.22631, and Python 3.12.13.
The command was:

```powershell
uv run python -m product_search.service "round coffee table" --mode default --top-k 3 --benchmark-runs 20 --local-files-only
```

Actual results were:

- initialization: 3,016.228 ms;
- measured calls: 20 after one excluded warm-up;
- mean query latency: 236.576 ms;
- median query latency: 233.376 ms; and
- p95 query latency: 265.246 ms.

Initialization measures settings resolution, artifact hashing and validation, product loading,
lexical/dense index loading, embedding-provider setup, and selected reranker loading. Per-query
latency starts before the selected engine call and ends after result enrichment, so it includes
query embedding, lexical and dense scoring across their configured candidates, fusion, reranking,
display-field lookup, score explanation construction, and response assembly. It excludes service
and artifact initialization. These are single-query measurements on one local machine, not a
throughput or concurrency benchmark.
