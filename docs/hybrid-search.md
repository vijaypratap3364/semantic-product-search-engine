# Hybrid retrieval

## Fusion design

The Stage 6 engine forms the union of the top 100 lexical and top 100 semantic full-catalog
candidates. Candidate depth is configurable. For judged-candidate evaluation, both component
engines score every explicitly judged product, so the comparison remains the controlled WANDS
benchmark defined in Stage 3.

Raw TF-IDF cosine scores and dense cosine scores are not added directly. Weighted fusion first
applies min-max normalization independently to each modality for each query:

```text
normalized_score = (score - query_min) / (query_max - query_min)
hybrid_score = (1 - semantic_weight) * lexical_normalized
             + semantic_weight * semantic_normalized
```

If all available scores from one modality are constant, all of those available candidates receive
`1.0`; this preserves equal membership evidence without inventing an order. A product absent from
one side of the bounded full-catalog union receives that modality's documented floor, `0.0`.
Results are sorted by descending fused score and then product ID, making ties deterministic.

The comparison strategy is equal-contribution reciprocal-rank fusion (RRF):

```text
rrf_score = 0.5 / (60 + lexical_rank) + 0.5 / (60 + semantic_rank)
```

An absent modality rank contributes zero. The offset `60` and candidate depth `100` remain typed,
configurable settings.

## Validation-only selection

The search evaluated semantic weights `0.0, 0.1, ..., 1.0`, with lexical weight constrained to
`1 - semantic_weight`, plus equal-contribution RRF. The primary criterion was macro
judged-candidate validation nDCG@10. Ties would be resolved by Recall@10, then MRR@10, then the
declared grid order.

All 12 configurations used the same 72 validation queries and canonical judgments. No model is
fitted during fusion, so the 336 training queries were not required for parameter estimation. Zero
training queries and **zero of the 72 held-out test queries** were evaluated during selection. The
test split remains untouched for a later final evaluation.

The selected configuration is:

| Setting | Selected value |
| --- | ---: |
| Strategy | weighted per-query min-max fusion |
| Semantic weight | 0.9 |
| Lexical weight | 0.1 |
| Full-catalog candidate depth per modality | 100 |
| RRF offset used in comparison | 60 |

The best result only narrowly exceeds semantic retrieval alone. That is still the selected setting
because validation nDCG@10 was declared as the primary criterion before the search; no secondary
metric or qualitative example was used to override it.

## Actual validation results

Exact and Partial labels count as binary relevant. Metrics below are macro averages over all 72
validation queries.

| Judged-candidate metric | Lexical | Semantic | Selected hybrid |
| --- | ---: | ---: | ---: |
| nDCG@10 | 0.710720 | 0.790632 | **0.791515** |
| Precision@10 | 0.904167 | **0.965278** | 0.962500 |
| Recall@10 | 0.075085 | **0.078734** | 0.078675 |
| MRR@10 | 0.903241 | **0.986111** | 0.976852 |

Selected hybrid nDCG@5 was `0.785145`. In full-catalog known-relevant evaluation, where unjudged
products remain unknown rather than being treated as irrelevant, hybrid known-relevant Recall@10
was `0.062249` and MRR@10 was `0.915278`. Semantic known-relevant Recall@10 was `0.061829` in the
Stage 5 run.

RRF achieved nDCG@10 `0.776789`, Precision@10 `0.951389`, Recall@10 `0.079699`, and MRR@10
`0.968750`. It had the highest Recall@10 in the fusion search, but it did not win the predeclared
nDCG@10 criterion.

## Latency boundary

The selected engine's uncached full-catalog search had median latency `213.850 ms` and p95 latency
`270.005 ms` across 72 queries. Each timed call includes query handling in both engines, TF-IDF
query transformation and sparse scoring, FastEmbed query generation and L2 normalization, exact
dense similarity against all 42,994 products, each modality's top-100 selection, candidate union,
score normalization, fusion, and final top-10 ranking. Model and index construction/loading are
outside the per-query timer. The process was already warm from tuning before selected-configuration
measurement.

Judged-candidate median latency was `15.860 ms` and p95 was `52.991 ms`; this mode scores only the
human-judged candidates for each query and is therefore not comparable to full-catalog latency.

## Actual error analysis

Categories use strict per-query judged-candidate nDCG@10 comparisons; ties are not assigned:

- Lexical beat both semantic and hybrid for 15 queries. The largest case was `almost heaven sauna`:
  lexical `0.614042`, semantic `0.375747`, hybrid `0.377503`.
- Semantic beat both lexical and hybrid for 8 queries. The largest case was
  `full metal bed rose gold`: lexical `0.000000`, semantic `0.574103`, hybrid `0.345963`.
- Hybrid beat both component engines for 11 queries. The largest case was
  `ceramic tile sea shell`: lexical `0.803094`, semantic `0.727383`, hybrid `0.974301`.
- Fusion hurt relative to both component engines for 3 queries. The largest case was
  `parsons chairs`: lexical `0.943238`, semantic `0.907425`, hybrid `0.853272`.

These examples are generated from the validation rankings and canonical human judgments; they are
not invented demonstrations.

## Reproduction and artifacts

Run the selected engine:

```text
uv run python -m product_search.retrieval.hybrid "round coffee table" --local-files-only
```

Re-run selection and validation reporting from local verified indexes:

```text
uv run python -m product_search.evaluation.benchmark_hybrid --local-files-only
```

The complete 12-row search is written to `artifacts/reports/hybrid_weight_search.csv`. Selected
metrics, source hashes, split counts, latency, detailed per-query reports, and error analysis are in
`artifacts/reports/hybrid_validation_metrics.json`. Generated reports remain excluded from Git.
