# Dense semantic search

## Model and runtime

The Stage 5 engine uses **`BAAI/bge-small-en-v1.5`** through FastEmbed 0.8.0 and
ONNX Runtime on CPU. FastEmbed's installed supported-model registry describes this exact model as
English, 384-dimensional, limited to 512 input tokens, approximately 0.067 GB, and MIT licensed.
The model is listed in the
[FastEmbed supported models](https://qdrant.github.io/fastembed/examples/Supported_Models/).

The project deliberately does not install PyTorch, sentence-transformers, FAISS, or a vector
database. At this catalog size, exact NumPy dot-product search is both simpler and fast enough.

## Index construction

The builder reads only `product_id` and `product_text` after inspecting the Parquet schema. Products
are sorted deterministically by string product ID. FastEmbed's passage embedding method generates
one vector per `product_text`; the project then explicitly applies L2 normalization and writes each
float32 vector directly into a preallocated NumPy `.npy` memory map.

The configured build batch is `1`. This apparently conservative value was selected from a local
fixed 256-product CPU benchmark after model warm-up. The products were stably scheduled by
character length for all grouped cases to reduce padding bias:

| Batch size | Documents/second |
| ---: | ---: |
| 1 | 10.738 |
| 2 | 10.219 |
| 4 | 9.415 |
| 8 | 8.858 |
| 16 | 8.706 |
| 32 | 8.352 |
| 64 | 7.650 |

Mixed product lengths made larger transformer batches slower on this CPU because they required more
padding. Batch 1 also had the best observed memory profile: roughly 184–244 MB after warm-up during
the full build, compared with about 1.01 GB initially at batch 16 and 2.04 GB at batch 64. Batch size
remains configurable in `configs/base.toml` for other hardware.

Build the ignored local index with:

```text
uv run python -m product_search.indexing.build_dense
```

Use `--local-files-only` to refuse model downloads or `--force` to replace an existing index.

## Persisted artifacts

`artifacts/embeddings/dense/` contains:

- `embeddings.npy`: `(42,994, 384)` float32, L2-normalized matrix;
- `product_ids.json`: deterministic row-to-product mapping;
- `metadata.json`: model, dimension, normalization, dataset hash, creation timestamp, package
  versions, batch size, artifact byte sizes, and SHA-256 hashes.

The completed matrix is 66,038,912 bytes including its NPY header. Its SHA-256 is
`303cc59a0cfdbc3d32d08e19bae6ac2c34c6d4fa7951a741f5d74fe75fc68502`. The loader checks every
artifact hash, byte size, shape, dtype, model name, dimension, product ordering, finite values, and
unit norms before exposing a memory-mapped index.

## Retrieval

Queries use FastEmbed's query embedding method and the same explicit L2 normalization. Full-catalog
scores are exact cosine-equivalent dot products against the memory-mapped matrix. NumPy
`argpartition` limits selection work, followed by deterministic sorting by descending score and
product ID for ties. Judged-candidate search uses the same vector and scores only the supplied WANDS
products.

Search locally with:

```text
uv run python -m product_search.retrieval.semantic "round coffee table"
```

## Validation results

The benchmark used the same Stage 3 evaluator and the same 72 validation queries as the TF-IDF
baseline. It evaluated **zero** of the 72 held-out test queries. Exact and Partial labels count as
binary relevant; unjudged full-catalog products remain unknown rather than being treated as
irrelevant.

| Validation measure | Lexical | Semantic | Semantic delta |
| --- | ---: | ---: | ---: |
| Judged-candidate nDCG@10 | 0.710720 | 0.790632 | +0.079911 |
| Judged-candidate Recall@10 | 0.075085 | 0.078734 | +0.003649 |
| Full-catalog known-relevant Recall@10 | 0.048850 | 0.061829 | +0.012979 |
| Median full-catalog latency | 168.544 ms | 7.829 ms | -160.715 ms |
| p95 full-catalog latency | 173.801 ms | 9.445 ms | -164.355 ms |

Additional semantic results were nDCG@5 `0.782899`, Precision@10 `0.965278`, MRR@10 `0.986111`,
and full-catalog known-relevant MRR@10 `0.915278`.

Resource payload comparison:

| Resource | Lexical | Semantic |
| --- | ---: | ---: |
| Index artifact disk bytes | 68,829,120 | 66,587,631 |
| Matrix payload memory bytes | 109,675,124 | 66,038,784 |
| Additional local model cache | n/a | 67,181,330 bytes |

Matrix memory excludes Python and model runtime overhead. Disk figures include index metadata; the
semantic model cache is shown separately.

The complete machine-readable report is generated locally at
`artifacts/reports/semantic_validation_metrics.json`. Re-run it with:

```text
uv run python -m product_search.evaluation.benchmark_semantic --local-files-only
```

## Actual low-lexical-overlap successes

Examples were selected only when semantic judged-candidate nDCG@10 exceeded lexical, a relevant
semantic top-10 product outranked its lexical position, and query/title unigram overlap was at most
0.25. They were derived from the validation outputs rather than invented:

- `full metal bed rose gold` retrieved Partial product `lyster platform bed` at semantic rank 1;
  lexical rank was 86 and query/title overlap was 0.20.
- `pennfield playhouse` retrieved Partial product `all around playtime patio` at semantic rank 6;
  lexical rank was 18 and query/title overlap was 0.00.
- `turquoise pillows` retrieved Partial product `tennessee throw pillow` at semantic rank 8;
  lexical rank was 438 and query/title overlap was 0.00.

These are examples of improved ranking under human judgments, not proof that every semantic result
is correct. WANDS judgments are incomplete, especially in full-catalog mode.

## Testing policy

Most dense tests use deterministic fake providers and never download a model. The optional test
marked `embedding` constructs the configured FastEmbed provider with `local_files_only=True`; it
skips when the model is absent and runs only from an existing local cache.
