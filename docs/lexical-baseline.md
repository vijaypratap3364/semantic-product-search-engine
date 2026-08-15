# TF-IDF Lexical Baseline

## Configuration and rationale

The first retrieval engine is a deliberately straightforward TF-IDF baseline fitted only on the
prepared `product_text` field. Its reviewed defaults are:

| Parameter | Value | Rationale |
| --- | --- | --- |
| Analyzer | Word | Direct, explainable lexical matching |
| Lowercase | `true` | Case-insensitive catalog/query matching |
| N-grams | `(1, 2)` | Preserve individual terms and common product phrases |
| Sublinear TF | `true` | Reduce domination by repeated catalog/marketing terms |
| `min_df` | `2` | Remove one-document terms while retaining uncommon catalog language |
| `max_features` | `100,000` | Bound memory and sparse matrix size on a modest CPU |
| Normalization | L2 | Makes sparse dot product cosine similarity |
| Matrix dtype | `float32` | Halves value storage compared with float64 |

These values are configurable in `configs/base.toml`; they were not chosen using test queries. No
test query was searched or scored during this stage.

## Reproducible commands

Build the ignored local index:

```powershell
uv run python -m product_search.indexing.build_lexical
```

Search it:

```powershell
uv run python -m product_search.retrieval.lexical "round coffee table"
```

Run the validation-only benchmark:

```powershell
uv run python -m product_search.evaluation.benchmark_lexical
```

Existing artifacts are protected unless the index builder receives `--force`. Loading verifies
the recorded size and SHA-256 of the fitted vectorizer, CSR matrix, and product ordering before
deserialization. It also rejects incompatible matrix dimensions, dtypes, product IDs, vocabulary,
and metadata schema.

Full-catalog search multiplies a sparse query row by the sparse product matrix and partially
selects top nonzero scores. It never converts the complete product matrix or catalog score vector
to dense form. Judged-candidate evaluation may materialize only the small score vector for that
query's explicitly judged candidate set.

## Actual index snapshot

The WANDS index built on August 14, 2026 contained:

| Measure | Actual value |
| --- | ---: |
| Products | 42,994 |
| Vocabulary features | 100,000 |
| Sparse matrix shape | 42,994 × 100,000 |
| Sparse matrix dtype | float32 |
| Matrix artifact size | 66,992,014 bytes |
| Vectorizer artifact size | 1,288,110 bytes |
| Product ordering size | 547,838 bytes |

The source `products.parquet` SHA-256 was
`d103410ebeb387a647a7515e8cdf0ea8a0bd1049fc37fcc12a374af249fb5233`. Artifact SHA-256 values
were:

- product matrix: `7b044792302ada5cd0cca82e19070596956768e8411af6c9327bce088e68039e`
- vectorizer: `7a2f6819f10a987efe5dba99cf823b3328f84de81894fb884a1124a005649c86`
- product ordering: `4d3b23e8c36b16c1876087ce74898c2a40ea343dd146478f55283ee167e988d1`

Generated index files remain under `artifacts/indexes/tfidf/` and are ignored by Git.

## Actual validation results

The benchmark evaluated all 72 validation queries and zero test queries. The binary threshold was
grade `>= 1`, so Exact and Partial were relevant. Macro metrics across all validation queries were:

| Judged-candidate metric | Actual value |
| --- | ---: |
| nDCG@5 | 0.690261 |
| nDCG@10 | 0.710720 |
| Precision@10 | 0.904167 |
| Recall@10 | 0.075085 |
| MRR@10 | 0.903241 |

All 72 validation queries contained at least one relevant canonical judgment. High Precision and
MRR show that the lexical baseline commonly places a relevant item near the top. Recall@10 is much
lower because many queries have substantially more than ten relevant judgments, while only ten
results can be recovered.

Full-catalog known-relevant recovery, where unjudged products remain unknown, measured:

| Carefully scoped full-catalog measure | Actual value |
| --- | ---: |
| Known-relevant Recall@10 | 0.048850 |
| Known-relevant MRR@10 | 0.737765 |
| Median query latency | 168.544 ms |
| p95 query latency | 173.801 ms |

Latency is warm-process engine search time over 72 validation queries and excludes index loading.
It includes query vectorization, sparse full-catalog scoring, and deterministic top-K selection.
The generated source-of-truth report is
`artifacts/reports/lexical_validation_metrics.json`; it remains ignored by Git.

## Evidence-based error analysis

The examples below come from actual validation judgments and rankings. They are not invented demo
queries.

### High-performing and exact-keyword cases

- Query 102, `flamingo`, achieved nDCG@10 = 1.0. The top result was the Exact-labeled `lopp iron
  flamingo garden art`.
- Query 11, `ombre rug`, achieved nDCG@10 = 1.0. The top result was the Exact-labeled `traci ombre
  braided cotton aqua area rug`.
- Query 176, `mexican art`, achieved nDCG@10 = 1.0. Its top result contained the exact query phrase
  and was labeled Exact.
- Query 255, `desk and chair set`, also achieved nDCG@10 = 1.0 with an Exact top result containing
  the complete phrase.

### Poorly performing cases

- Query 84, `full metal bed rose gold`, had nDCG@10 = 0.0. The top result was an Irrelevant rose
  gold table lamp; known relevant results included full and metal beds.
- Query 409, `teal chair`, had nDCG@10 = 0.146973. An Irrelevant teal nightstand ranked first and
  an Irrelevant teal rug ranked second; the first relevant chair appeared at rank five.
- Query 304, `merlyn 6`, had nDCG@10 = 0.161043. Products matching the name `merlyn` were judged
  Irrelevant, while relevant products largely matched only the number `6`. This also exposes noisy
  or ambiguous query intent rather than a simple model error.

### Synonym and terminology failures

- Query 123, `entrance table`, had nDCG@10 = 0.336696. Irrelevant products with the literal word
  `entrance` ranked first and second, while the Exact-labeled `gaiana 12.6 '' console table` ranked
  third. The lexical model cannot inherently equate entrance table with console table.
- Query 35, `enclosed shoe rack`, had nDCG@10 = 0.333333. Open products named `shoe rack` occupied
  the first five positions with Partial labels, while Exact judgments included products described
  as shoe storage cabinets or benches. The desired enclosed/storage concept is expressed using
  different catalog terminology.
- Query 409, `teal chair`, favored products containing the exact color token. Several known-relevant
  titles use the compound `armchair`, which does not contribute the separate unigram `chair` under
  the current word analyzer.

### Vocabulary mismatch

- Query 19, `gurney slade 56`, had `gurney` and `slade` outside the capped fitted vocabulary. It
  still achieved nDCG@10 = 0.721519 by matching `56`.
- Query 353, `pennfield playhouse`, had `pennfield` out of vocabulary and nDCG@10 = 0.694356.
- Query 77, `sancroft armchair`, had `sancroft` out of vocabulary and nDCG@10 = 0.694356.

These observations motivate the later dense and hybrid stages; they do not change or tune this
baseline after seeing validation results.
