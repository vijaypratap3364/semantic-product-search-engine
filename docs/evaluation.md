# Offline Search Evaluation

## Purpose and order of operations

The evaluation framework is implemented before any retrieval model so later lexical, dense,
hybrid, and reranked systems use the same judgments, query partitions, metrics, and reporting
rules. The reproducible order is:

```powershell
uv run python -m product_search.evaluation.judgments
uv run python -m product_search.evaluation.splits
```

These commands require the processed Stage 2 Parquet tables. They create ignored local artifacts;
they never modify `data/processed/labels.parquet`.

## Repeated-judgment inspection and canonicalization

The original processed label table is the source-faithful representation and remains unchanged.
Evaluation uses a separate `data/processed/evaluation_judgments.parquet` table with exactly one row
per `(query_id, product_id)`. Its audit metadata is written to
`artifacts/reports/judgment_canonicalization.json`.

The pinned WANDS snapshot contains 233,448 judgment rows and 231,873 distinct query-product pairs.
The 1,575 repeated rows beyond the first consist of:

- 1,561 duplicate rows whose query ID, product ID, and label are identical to another row; and
- 14 genuinely conflicting query-product pairs.

Conflict combinations are:

| Observed labels | Conflicting pairs |
| --- | ---: |
| `Exact` and `Partial` | 9 |
| `Partial` and `Irrelevant` | 5 |

Canonicalization uses an explicit two-step policy:

1. Select the most frequent label for the query-product pair.
2. If multiple labels tie for the highest frequency, select the tied label with the highest
   configured relevance grade.

Majority vote uses repeated evidence when it exists. The tie rule is intentionally inclusive: it
preserves an assessor's positive relevance evidence rather than turning disagreement into a false
negative. This makes known-relevant recovery more demanding, while the retained audit columns make
the choice inspectable. A future policy comparison may be performed on validation queries, but the
test set must not be used to select a policy.

Each canonical row records the selected label and grade, selected-label count, total judgment
count, distinct-label count, all observed labels, and one of `single`,
`identical_duplicate_collapse`, `majority_vote`, or `tie_highest_relevance`. In this snapshot the
231,873 canonical rows comprise 230,406 single judgments, 1,453 identical-duplicate collapses, two
majority-vote conflicts, and 12 tied conflicts resolved to the highest relevance grade.

The source processed-label SHA-256 recorded by the run was
`c90e184d4522279556344b2fbb68dfc7918738d5d360aaa3538d3fb1b5e95185`.

## Query partitions

Splits operate on unique query IDs, never on individual judgment rows. IDs are sorted first so
input row order cannot affect membership, then permuted with NumPy's seeded generator using seed
42. Integer counts are assigned with the largest-remainder method and stable
train/validation/test tie-breaking.

The generated `data/processed/query_splits.json` records requested and actual proportions, counts,
seed, IDs, and SHA-256 hashes of both source tables. The actual 480-query split is:

| Split | Queries | Proportion | Allowed use |
| --- | ---: | ---: | --- |
| Train | 336 | 70% | Model fitting |
| Validation | 72 | 15% | Configuration and hyperparameter selection |
| Test | 72 | 15% | One-time final evaluation after configuration is frozen |

The recorded source hashes are:

- `queries.parquet`:
  `386685f032f1417a618add78f0335b036f1cc8f07d678897d808e69e0e2d07d1`
- `evaluation_judgments.parquet`:
  `e3ede6c3b4989e4e5c5d6e64babb1a37df91d29f4133674d23bd958e52777286`

The split writer rejects canonical judgments referencing unknown queries. Tests also assert pairwise
split disjointness and complete query coverage. Test queries must never be used for tuning,
threshold selection, conflict-policy selection, or feature design.

## Relevance definitions and metrics

Graded relevance is fixed by project configuration:

- `Exact = 2`
- `Partial = 1`
- `Irrelevant = 0`

DCG@K uses exponential gain and logarithmic discount:

```text
DCG@K = sum((2^grade_i - 1) / log2(i + 1)) for ranks i = 1..K
```

nDCG@K divides DCG by the DCG of the ideal grade ordering over the controlled candidate set. It is
zero when the ideal DCG is zero.

Binary metrics use a configurable numeric relevance threshold. The default is grade `>= 1`, so
`Exact` and `Partial` are relevant and `Irrelevant` is not. Precision@K divides relevant retrieved
results by K; if an engine returns fewer than K results, missing slots count as non-relevant.
Recall@K divides relevant retrieved results by all explicitly relevant judgments. Reciprocal Rank
is the inverse rank of the first relevant result, and MRR@K is its macro mean after truncating each
ranking at K.

For a query with no relevant judgments at the configured threshold, nDCG, Recall, and Reciprocal
Rank are defined as zero and the query is flagged in diagnostics. Aggregate JSON reports both a
macro mean across all queries and a separate relevant-queries-only mean, plus both query counts.
This prevents eligibility policy from being hidden.

Metric tests use hand-calculated perfect, reversed, empty-relevance, truncated-K, and graded
examples rather than trusting an external ranking-metric implementation.

## Engine contracts

Every future engine implements:

```python
search(query: str, top_k: int) -> list[SearchResult]
```

`SearchResult` contains `product_id`, one-based `rank`, finite `score`, and optional finite score
components. The evaluator rejects duplicate product IDs, non-contiguous ranks, and excess results.

The controlled benchmark additionally requires `search_candidates(query, candidate_product_ids,
top_k)`. This extension makes candidate restriction explicit and prevents the evaluator from
pretending a full-catalog top-K contains a ranking of every judged candidate.

## Evaluation modes

### `judged_candidate_evaluation`

This is the primary benchmark. For each query, the engine scores and ranks only products with a
canonical human judgment for that query. Because every candidate is judged, the framework reports
DCG@K, nDCG@K, Precision@K, Recall@K, and Reciprocal Rank@K per query, with MRR@K in the aggregate.

### `full_catalog_known_relevant_evaluation`

This diagnostic searches the entire product catalog and measures recovery of products explicitly
judged at or above the configured relevance threshold. It reports
`known_relevant_recall_at_k` and `known_relevant_mrr_at_k`.

Unjudged results are unknown, not irrelevant. Therefore this mode deliberately does not report
precision, DCG, or nDCG. Its carefully scoped names must not be presented as complete-catalog
relevance metrics.

## Reports, latency, and failures

`write_evaluation_reports` produces three files for either mode:

- `<name>_per_query.csv` with counts, metrics, latency, and status;
- `<name>_aggregate.json` with all-query and eligible-query macro metrics, mean/median/p95 search
  latency, relevance threshold, and failure count; and
- `<name>_diagnostics.json` with query-level no-relevance, no-result, truncated-result, and search
  failure diagnostics.

Latency uses a monotonic high-resolution process clock around the engine search call and result
contract validation. No search-quality or latency numbers are documented yet because no retrieval
model has been implemented or benchmarked.
