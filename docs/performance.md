# Performance and search-quality analysis

Stage 13 measures warm-process search performance on the actual development computer and produces
two ignored, reproducible reports:

- `artifacts/reports/benchmark.json` contains timing boundaries, sample provenance, hardware,
  memory, artifact sizes, build timing, hashes, and category counts.
- `artifacts/reports/error_analysis.md` contains catalog- and judgment-backed examples. It is
  regenerated from processed WANDS data and does not contain invented failure descriptions.

Run the bounded benchmark with locally cached model files:

```powershell
uv run python -m product_search.evaluation.benchmark_performance --local-files-only
```

Existing outputs are protected. Pass `--force` to replace them. The default workload is capped at
20 deterministic queries times five repeats, or 100 timed observations per component. The command
rejects more than 200 timed query repetitions. It warms two queries, uses `top_k=10`, and selects
the sample with seed 42 using proportional query-length stratification. The exact query IDs and a
SHA-256 of their ordered list are recorded in `benchmark.json`.

## Measured results

The final run used Windows 11, Python 3.12.13, NumPy 2.5.2, scikit-learn 1.9.0, and an AMD64 CPU
with 12 logical processors. Initialization is excluded from every per-query distribution.

| Timed boundary | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---:|---:|---:|
| Lexical query | 21.604 | 26.775 | 28.019 |
| Semantic query embedding | 5.085 | 6.023 | 6.605 |
| Semantic full-catalog retrieval | 2.758 | 3.121 | 3.361 |
| Hybrid search | 30.965 | 35.022 | 39.085 |
| Reranking stage | 16.243 | 22.562 | 25.302 |
| Loopback API, default rerank mode | 53.183 | 61.851 | 64.794 |

Each row has 100 observations, so p99 is reported. Semantic embedding includes FastEmbed encoding
and project L2 normalization. Semantic retrieval starts with a precomputed query embedding and
includes exact dot products over all 42,994 products and top-K selection. Hybrid includes both base
retrievers and fusion. Reranking starts with an already generated hybrid candidate pool and includes
feature extraction, logistic probabilities, expected-relevance scoring, and reordering. API latency
uses a real loopback HTTP `POST /search` through Uvicorn and FastAPI, response serialization and
decoding, and the default reranked search; analytics logging is disabled for this boundary.

SearchService initialized in 3,388.051 ms. Windows working-set measurements were 212,410,368 bytes
before service loading, 734,228,480 bytes after loading, and 747,761,664 bytes after the timed run;
the maximum sampled during timed queries was 747,884,544 bytes.

| Artifact family | Bytes | MiB |
|---|---:|---:|
| TF-IDF index | 68,829,120 | 65.64 |
| Dense index | 66,587,631 | 63.50 |
| Reranker model and metadata | 4,844 | <0.01 |

A temporary full-catalog TF-IDF build took 40.254 seconds for 42,994 products and a 100,000-term
vocabulary. That boundary includes Parquet loading, fitting and transformation, serialization,
hashing, and atomic publication. The existing dense artifact predates build-duration instrumentation.
Re-embedding the full catalog merely to reconstruct that missing elapsed time was deliberately
skipped to keep this benchmark bounded; the report marks dense build duration unavailable instead
of estimating it.

## Profile-guided optimization

The pre-change spot profile measured roughly 192–199 ms per lexical query. Across five queries,
0.919 of 0.991 seconds accumulated while SciPy converted the full sparse product-matrix transpose
from CSC to CSR for repeated sparse multiplication. This was the measured bottleneck.

Lexical scoring now converts only the single transformed query row to a dense vector and multiplies
the existing CSR product matrix by that vector. The 42,994-product TF-IDF matrix remains sparse and
is never converted wholesale to a dense matrix. The final 100-observation lexical p50 is 21.604 ms;
hybrid and API timing benefit from the same change. No vector database or cache was added.

## Search-quality findings

The error analysis evaluates all 72 held-out test queries with canonical judged-candidate nDCG@10.
Categories use strict comparisons, are evaluated independently, and can overlap.

| Category | Qualifying queries | Example query IDs |
|---|---:|---|
| Lexical better than semantic | 10 | 130, 403, 374 |
| Semantic better than lexical | 32 | 29, 246, 129 |
| Hybrid better than both | 9 | 60, 389, 319 |
| Lowest reranked tail | 5 | 20, 277, 142, 224, 130 |
| Partial ranked first while an Exact exists | 10 | 130, 135, 15 |
| Reranking helps hybrid | 18 | 97, 29, 135 |
| Reranking hurts hybrid | 15 | 129, 277, 71 |

The generated Markdown report records the actual query text, four nDCG values, top judgment grades,
the reranked top product, and the first Exact rank for each selected example. The tail category is
defined as the five lowest reranked-hybrid nDCG@10 queries; it is descriptive, not an assertion that
every result for those queries is irrelevant.

## Scaling boundary

Exact dense scoring remains appropriate for this 42,994-product catalog: its measured p50 is 2.758
ms and the dense artifact is 63.50 MiB. Its work and storage grow linearly with catalog size. At the
current 384 float32 dimensions, embeddings alone require about 1.43 GiB per million products and
14.31 GiB per ten million products, before IDs and process overhead.

Brute-force search stops being appropriate when the embedding matrix no longer fits comfortably in
RAM or measured p95 retrieval latency/throughput misses the service target. Catalog size alone is not
used as an arbitrary cutoff. At materially larger scale, rerun this benchmark against the expected
load and introduce an approximate-nearest-neighbor index only if those measurements justify its
recall, complexity, and infrastructure tradeoffs.
