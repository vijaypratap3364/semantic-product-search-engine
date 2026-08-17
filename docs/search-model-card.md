# Search Model Card

## Model summary

This project ranks products from the official Wayfair ANnotation Dataset (WANDS) with four frozen
CPU search systems: TF-IDF lexical retrieval, dense semantic retrieval, score-level hybrid
retrieval, and hybrid retrieval followed by a small supervised reranker. The selected default is
**reranked hybrid** because it achieved the highest primary held-out judged-candidate nDCG@10.

The final evaluation ran once after all retrieval and model settings had been selected on train and
validation queries. No retrieval parameter, fusion weight, feature, or model hyperparameter was
changed after test metrics were observed. The selected immutable-configuration SHA-256 is
`8fe9ef94f00a1ae00dd11ac7e6694a1b004c061f840d6714f4d3183871f1b487`.

## Dataset and intended use

WANDS is an English e-commerce product-search relevance dataset published by Wayfair for offline
research and benchmarking. The pinned snapshot contains 42,994 products, 480 queries, and 233,448
source annotation rows. Evaluation uses a separate canonical table of 231,873 query-product pairs
so every pair has exactly one relevance grade while the source-faithful labels remain unchanged.

Query IDs are split deterministically with seed 42:

| Split | Queries | Use |
| --- | ---: | --- |
| Train | 336 | Reranker fitting only |
| Validation | 72 | Fusion and reranker selection only |
| Test | 72 | One-time final reporting only |

The system is intended as a reproducible portfolio and research baseline for product-search
ranking. It is not a live Wayfair service and should not be used for inventory, pricing,
availability, safety-critical decisions, or individualized recommendations.

## Frozen retrieval methods

### Lexical

TF-IDF uses product text only, lowercasing, word unigrams and bigrams, sublinear term frequency,
L2 normalization, `min_df=2`, and at most 100,000 features. Query scoring uses sparse similarity
without densifying the product matrix.

### Semantic

Dense retrieval uses FastEmbed `0.8.0` with `BAAI/bge-small-en-v1.5`, 384-dimensional `float32`
embeddings, project-applied L2 normalization, and exact NumPy similarity against all 42,994
products. No ANN service or vector database is used.

### Hybrid

The hybrid takes the union of the top 100 lexical and semantic candidates, applies per-query
min-max normalization to each modality, and combines scores with frozen weights:

```text
hybrid = 0.1 * normalized_lexical + 0.9 * normalized_semantic
```

This configuration was selected by validation judged-candidate nDCG@10.

### Reranked hybrid

The second stage reorders the hybrid top 100 with a standardized three-class multinomial logistic
regression (`C=1.0`, balanced class weights, `lbfgs`). It scores a pair as:

```text
P(Partial) * 1 + P(Exact) * 2
```

Its 13 predictive inputs are retrieval scores/ranks and query/catalog-text overlap or length
features available for an arbitrary free-form query. Query IDs, product IDs, `query_class`, and
label-derived information are not predictive inputs. The reranker was admitted to final evaluation
only because it strictly improved validation judged-candidate nDCG@10 over hybrid.

## Evaluation methodology

The primary controlled benchmark ranks only products with a canonical WANDS judgment for each
test query. nDCG uses graded relevance (`Exact=2`, `Partial=1`, `Irrelevant=0`). Precision, Recall,
and MRR treat Exact and Partial as relevant. Metrics are macro-averaged across all 72 held-out test
queries; all 72 had at least one relevant judgment.

The full-catalog diagnostic searches all products and measures recovery of explicitly judged
relevant products. It does not report full-catalog precision or nDCG because an unjudged retrieved
product is unknown, not irrelevant.

## Held-out test results

Primary judged-candidate results:

| System | nDCG@5 | nDCG@10 | Precision@5 | Precision@10 | Recall@5 | Recall@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Lexical | 0.725720 | 0.743282 | 0.886111 | 0.866667 | 0.066113 | 0.102704 | 0.936921 |
| Semantic | 0.810471 | 0.815459 | 0.952778 | 0.941667 | 0.069452 | 0.110821 | 0.969907 |
| Hybrid | 0.814875 | 0.817523 | 0.955556 | 0.944444 | 0.069422 | 0.110977 | 0.976852 |
| **Reranked hybrid** | **0.824809** | **0.827633** | 0.952778 | 0.941667 | 0.069155 | 0.110600 | 0.972222 |

Full-catalog known-relevant recovery and warm-process latency:

| System | Known-relevant Recall@10 | Known-relevant MRR@10 | Median latency | p95 latency |
| --- | ---: | ---: | ---: | ---: |
| Lexical | 0.060728 | 0.774151 | 196.306 ms | 203.228 ms |
| Semantic | 0.071097 | 0.882077 | 7.857 ms | 8.895 ms |
| Hybrid | **0.071833** | 0.881173 | 206.788 ms | 214.336 ms |
| **Reranked hybrid** | 0.071272 | 0.869907 | 231.112 ms | 265.917 ms |

Reranking improved the primary nDCG@10 by `0.010109` over hybrid, but hybrid retained slightly
higher Precision@10, Recall@10, MRR@10, and known-relevant Recall@10. Reranking also added about
`24.324 ms` to hybrid median latency. The default therefore represents an explicit ranking-quality
choice, not a claim that the reranker dominates every metric. Semantic retrieval remains the best
latency-sensitive option on the measured machine.

## Latency environment and boundary

Measurements were made on Windows 11 (`Windows-11-10.0.22631-SP0`) with an `AMD64 Family 25 Model
117 Stepping 2, AuthenticAMD` processor and 12 logical CPUs, using Python `3.12.13`, NumPy `2.5.2`,
and scikit-learn `1.9.0`.

Each latency sample surrounds warm-process, end-to-end `engine.search(query, top_k=10)` over the
full catalog. It includes query preprocessing/encoding, scoring, candidate selection, fusion where
applicable, feature generation and reranking where applicable, and final top-K ordering. Index,
embedding-model, and reranker-model initialization are excluded. Each system used three warm-up
test queries before 72 timed samples.

## Limitations and failure modes

- WANDS is a static catalog and query snapshot. It does not model current inventory, price,
  popularity, seasonality, personalization, or changing customer intent.
- Only explicitly judged products can establish known relevance. Full-catalog recovery values are
  lower bounds against incomplete judgments, not complete relevance assessments.
- Judged-candidate evaluation isolates ranking quality but is easier than discovering relevant
  products from the entire catalog.
- Recall values use all explicitly relevant judgments as the denominator, which can be much larger
  than K; low Recall@K is therefore expected and should be interpreted with candidate counts.
- Lexical retrieval can fail on synonyms, paraphrases, spelling variation, and vocabulary gaps.
- Dense retrieval can overgeneralize semantically and miss exact model numbers, brands, or rare
  catalog tokens.
- Min-max hybrid normalization is query-local and can amplify small score differences.
- The reranker only reorders the hybrid candidate pool. A relevant product absent from that pool
  cannot be recovered by the second stage.
- The reranker improved nDCG but slightly regressed several binary/recovery metrics and increased
  latency. Distribution shift may reverse its validation and test gains.
- Artifact loading intentionally fails on missing, corrupt, hash-mismatched, dimensionally
  incompatible, or feature-schema-incompatible files. Empty or unsupported queries may return no
  lexical evidence even when semantic retrieval still produces candidates.

## Known biases

- Product coverage and labels reflect Wayfair's catalog taxonomy, assortment, content quality, and
  annotation process rather than all retailers or all customer populations.
- English text and the English embedding model disadvantage other languages and code-switched
  queries.
- Catalog descriptions and features may carry merchandising language, omissions, and societal or
  supplier biases into retrieval scores.
- Majority/tie resolution for the 14 conflicting query-product judgments preserves positive
  evidence in ties; another documented policy could produce different evaluation values.
- Macro averages weight each query equally, regardless of frequency or commercial importance.

## Reproducibility and artifacts

Run the metadata-only freeze audit before the final benchmark:

```powershell
uv run python -m product_search.evaluation.benchmark_final --verify-only --local-files-only
```

The actual one-time benchmark command was:

```powershell
uv run python -m product_search.evaluation.benchmark_final --local-files-only
```

Generated outputs are local and ignored by Git:

- `artifacts/reports/final_test_metrics.json`
- `artifacts/reports/final_test_metrics.csv`
- `artifacts/reports/final_test_per_query_metrics.csv`
- `artifacts/reports/final_comparison.md`
- `artifacts/reports/final_engine.json`
- `artifacts/reports/final_system_comparison.svg`
- `artifacts/reports/final_ndcg_distribution.svg`
- `artifacts/reports/final_latency_comparison.svg`

See [the data documentation](data-source.md) for WANDS attribution and its MIT license, and
[the evaluation documentation](evaluation.md) for canonical judgment and metric definitions.
