# Lightweight supervised relevance reranker

## Architecture and leakage controls

Stage 7 adds no neural model. The first stage uses the validation-selected hybrid engine to produce
at most 100 candidates. A scikit-learn pipeline then standardizes fixed numeric features and applies
multinomial `LogisticRegression`. Products are reranked by expected relevance:

```text
expected_relevance = P(Partial) * 1 + P(Exact) * 2
```

Training used only canonical judgments belonging to the 336 train query IDs. Model selection and
all metrics in this document used the 72 validation queries. **Zero of the 72 test queries were
searched, featurized, fitted, or evaluated.** The test partition remains untouched.

The bounded hybrid pools produced 32,021 train rows and 6,809 validation rows; some queries have
fewer than 100 judged candidates. IDs and `query_class` remain row provenance only and never enter
the model matrix.

## Inference-available feature schema

Training and online inference share the same ordered 13-feature extractor:

- lexical similarity, semantic similarity, and hybrid score;
- lexical and semantic ranks;
- query/title and query/description token overlap;
- exact normalized query phrase in the title;
- query-token coverage in the title and complete `product_text`;
- query, title, and product-text token lengths.

The schema explicitly rejects `query_id`, `product_id`, `query_class`, labels, judgment counts,
resolution fields, and any label-derived value. A missing modality uses rank `candidate_depth + 1`.
Feature names, definitions, tokenization, ordering, schema version, and schema SHA-256 are persisted
with the model.

## Training data

The actual top-100 candidate class distributions were:

| Split | Irrelevant | Partial | Exact | Rows |
| --- | ---: | ---: | ---: | ---: |
| Train | 3,130 | 20,111 | 8,780 | 32,021 |
| Validation | 531 | 4,729 | 1,549 | 6,809 |

## Model selection

The search compared six configurations: `C` values `0.1`, `1.0`, and `10.0`, each with no class
weighting and balanced class weighting. Validation judged-candidate nDCG@10 was the primary metric;
ties used Recall@10, MRR@10, macro F1, then declared grid order.

| C | Class weight | nDCG@10 | Recall@10 | MRR@10 | Accuracy | Macro F1 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0.1 | none | 0.802630 | 0.077177 | 0.976852 | 0.773388 | 0.623526 |
| 0.1 | balanced | 0.804909 | 0.077498 | 0.975694 | 0.592157 | 0.531223 |
| **1.0** | **balanced** | **0.805658** | **0.077498** | **0.976852** | 0.592011 | 0.531261 |
| 1.0 | none | 0.802512 | 0.077177 | 0.976852 | 0.773094 | 0.623122 |
| 10.0 | none | 0.802512 | 0.077177 | 0.976852 | 0.773094 | 0.623122 |
| 10.0 | balanced | 0.805658 | 0.077498 | 0.976852 | 0.592011 | 0.531261 |

Balanced `C=1.0` and `C=10.0` tied on all recorded selection metrics, so the declared grid order
selected the simpler `C=1.0` configuration.

## Validation classification diagnostics

The selected model's candidate-level accuracy was `0.592011` and macro F1 was `0.531261`. This
classification view is diagnostic; the production decision uses query-level ranking nDCG@10.

Confusion matrix rows are actual and columns predicted, ordered Irrelevant, Partial, Exact:

| Actual \ Predicted | Irrelevant | Partial | Exact |
| --- | ---: | ---: | ---: |
| Irrelevant | 374 | 134 | 23 |
| Partial | 1,253 | 2,642 | 834 |
| Exact | 47 | 487 | 1,015 |

The largest positive standardized coefficients for the Exact class were product-text query-token
coverage (`0.723986`), exact phrase in title (`0.598086`), semantic similarity (`0.581865`), and
hybrid score (`0.353126`). Coefficients are associations in standardized feature space, not causal
effects.

## Actual ranking comparison and decision

| Validation measure | Hybrid | Reranker | Delta |
| --- | ---: | ---: | ---: |
| Judged-candidate nDCG@10 | 0.791515 | **0.805658** | **+0.014143** |
| Judged-candidate Precision@10 | **0.962500** | 0.961111 | -0.001389 |
| Judged-candidate Recall@10 | **0.078675** | 0.077498 | -0.001177 |
| Judged-candidate MRR@10 | 0.976852 | 0.976852 | 0.000000 |
| Full-catalog known-relevant Recall@10 | **0.062249** | 0.058789 | -0.003460 |
| Full-catalog known-relevant MRR@10 | **0.915278** | 0.913426 | -0.001852 |

Reranker nDCG@5 was `0.806445`. Its full-catalog median latency was `249.474 ms` and p95 was
`304.660 ms`, compared with Stage 6 hybrid values of `213.850 ms` and `270.005 ms`. Timed reranker
search includes hybrid candidate generation, feature extraction, logistic probabilities, expected
relevance conversion, and final sorting; model and index loading are excluded.

The reranker is recommended as the default search mode because it strictly improved the declared
primary validation metric. This decision does not hide its modest regressions in Precision@10,
Recall@10, full-catalog known-relevant recovery, or latency. Hybrid search remains available as the
simpler and faster mode. No test-set result has influenced this decision.

## Persistence and reproduction

The ignored `artifacts/models/reranker/` directory contains `model.joblib` and `metadata.json`. The
model artifact is 2,001 bytes with SHA-256
`63575a0b6f342de0293208ed40917b75342a7ebc2ebea234e39c776630c44b65`. Loading verifies its byte
size and hash, feature schema, three-class order, product dataset hash, and candidate depth.

Recreate the model and validation report from existing local indexes:

```text
uv run python -m product_search.evaluation.benchmark_reranker --local-files-only --force
```

Search with the persisted reranker:

```text
uv run python -m product_search.ranking.reranker "round coffee table" --local-files-only
```

The six-row search is written to `artifacts/reports/reranker_model_search.csv`; the complete report
is `artifacts/reports/reranker_validation_metrics.json`. Generated models and reports stay out of
Git.
