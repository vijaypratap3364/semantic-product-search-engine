# WANDS Data Source and Preprocessing

## Official source

This project uses the [Wayfair ANnotation Dataset (WANDS)](https://github.com/wayfair/WANDS),
published by Wayfair as a companion to the ECIR 2022 paper *WANDS: Dataset for Product Search
Relevance Assessment*. It is intended for objective benchmarking and evaluation of e-commerce
product search systems.

Downloads are pinned to official repository commit
`3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5`. The downloader retrieves only these files from the
repository's `dataset/` directory:

- `product.csv`
- `query.csv`
- `label.csv`

The files use tab delimiters despite their `.csv` extension. The repository is not cloned, and no
models or unrelated repository files are downloaded.

## License and attribution

The official WANDS repository distributes the dataset under the
[MIT License](https://github.com/wayfair/WANDS/blob/3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5/LICENSE),
with copyright `2021 ecir2022`.

The source repository requests citation of:

> Yan Chen, Shujian Liu, Zheng Liu, Weiyi Sun, Linas Baltrunas, and Benjamin Schroeder. “WANDS:
> Dataset for Product Search Relevance Assessment.” Proceedings of the 44th European Conference
> on Information Retrieval, 2022.

The upstream license applies to WANDS. This project's own license does not replace or modify the
upstream attribution and license terms.

## Source fields

### Products

`product.csv` contains:

- `product_id`: product identifier;
- `product_name`: display name;
- `product_class`: product category/class;
- `category hierarchy`: slash-delimited parent categories in the raw file;
- `product_description`: catalog description;
- `product_features`: pipe-delimited attribute/value features;
- `rating_count`: number of ratings;
- `average_rating`: average rating; and
- `review_count`: number of reviews.

The raw `category hierarchy` header is normalized to `category_hierarchy` in the canonical product
table.

### Queries

`query.csv` contains `query_id`, `query`, and `query_class`. `query_class` is preserved for dataset
inspection but will not be used as an online ranking feature because it is not naturally available
for arbitrary user queries.

### Judgments

`label.csv` contains annotation `id`, `query_id`, `product_id`, and `label`. Valid labels are
`Exact`, `Partial`, and `Irrelevant`. Preparation also creates `relevance_grade` using the committed
mapping `Exact = 2`, `Partial = 1`, and `Irrelevant = 0`.

## Reproducible acquisition

Run:

```powershell
uv run python -m product_search.data.download
```

The downloader:

1. requests only the three allow-listed files at the pinned revision;
2. reuses non-empty cached files without network requests;
3. replaces existing files only when `--force` is passed;
4. writes through temporary files before atomically replacing destinations;
5. computes SHA-256 hashes and byte sizes; and
6. records source URLs, timestamp, row counts, and columns in `data/raw/manifest.json`.

Raw CSVs and their generated manifest remain local and are ignored by Git.

## Validation and preprocessing

Run:

```powershell
uv run python -m product_search.data.prepare
```

Validation rejects missing required columns or identifiers, duplicate product or query IDs,
unknown label foreign keys, and labels outside the three official relevance values. It reports
duplicate annotation IDs and repeated or conflicting query-product judgments without silently
discarding them.

Preparation preserves source columns and creates `product_text` deterministically from:

1. `product_name`;
2. `product_class`;
3. `category_hierarchy`;
4. `product_description`; and
5. `product_features`.

Missing values contribute an empty string, not the literal text `nan`. Whitespace is collapsed,
but stemming and aggressive linguistic normalization are intentionally deferred. The resulting
tables are written to `data/processed/products.parquet`, `queries.parquet`, and `labels.parquet`.
Actual counts and quality observations are written to `artifacts/reports/data_summary.json`. These
generated artifacts remain ignored by Git.

## Verified snapshot

The pinned source was downloaded and prepared successfully on August 14, 2026. Observed values
were:

| Measure | Actual value |
| --- | ---: |
| Products | 42,994 |
| Queries | 480 |
| Annotation rows | 233,448 |
| Exact labels | 25,614 |
| Partial labels | 146,633 |
| Irrelevant labels | 61,201 |
| Duplicate product IDs | 0 |
| Duplicate query IDs | 0 |
| Duplicate annotation IDs | 0 |
| Repeated query-product rows beyond the first | 1,575 |
| Query-product pairs with conflicting labels | 14 |

Pinned file integrity values were:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `product.csv` | 90,621,131 | `d993926254572e6eba96c8fd87cc549a17fb91ad3748308036eee4cf92b10ac6` |
| `query.csv` | 19,942 | `63b61660560fecc33ec490804c7e2b81402ee3e7c31a9cbb5e03736639f68e95` |
| `label.csv` | 5,736,234 | `c11fe81ad62f17f56f316b0ec9630ebe8fbe1393578cb0ca4f05c17253a180ef` |

Observed product missing-value counts were 9,452 each for `rating_count`, `average_rating`, and
`review_count`; 6,008 for `product_description`; 2,852 for `product_class`; and 1,556 for
`category_hierarchy`. Six queries had missing `query_class`. Required identifiers, query text,
product names, product features, and label values had no missing values.

## Limitations

- WANDS judgments cover selected query-product pairs, not every product for every query. An
  unjudged product is unknown rather than necessarily irrelevant.
- The 1,575 repeated query-product rows include 14 pairs with conflicting labels. A later
  evaluation stage must define and test an aggregation policy without using test results for
  tuning.
- Catalog fields contain natural missingness and may contain noisy formatting or marketing text.
- `query_class` is dataset metadata, not a feature available for a new free-form search query.
- The dataset is a static research snapshot and does not represent live inventory, price,
  availability, or changing user behavior.
