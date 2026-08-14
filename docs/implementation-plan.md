# Semantic Product Search Engine: Implementation Plan

## 1. Purpose and constraints

This project will build a reproducible, portfolio-quality product search system on the official Wayfair WANDS dataset. It will compare four ranking approaches under one evaluation contract:

1. TF-IDF lexical retrieval.
2. FastEmbed dense semantic retrieval.
3. A weighted hybrid of lexical and dense scores.
4. Hybrid retrieval followed by a small supervised relevance reranker.

The implementation will run natively on Windows with Python 3.12 and `uv`. It will remain CPU-friendly and zero-cost: no hosted APIs, large language models, GPU-only dependencies, vector database, container platform, or external database are required. The only planned model download is a small FastEmbed-compatible English embedding model, initially `BAAI/bge-small-en-v1.5`. The implementation stage must inspect the installed FastEmbed version and its supported API before model behavior is encoded.

Stage 0 defines the design only. It does not download WANDS, build indexes, train a model, or create fabricated metrics.

## 2. Overall architecture

The system will separate offline preparation from online serving.

```text
Official WANDS CSV files
        |
        v
Schema validation and normalization
        |
        +------------------------+
        |                        |
        v                        v
Canonical product table     Query-level split manifest
        |                        |
        +----------+-------------+
                   |
          Offline index/training jobs
          |        |             |
          v        v             v
       TF-IDF   Dense matrix   Reranker model
          \        /             /
           \      /             /
            Online search service
                    |
             FastAPI application
               /           \
              v             v
       Streamlit UI     SQLite event log
```

Offline commands will create versioned, validated artifacts from committed scripts and configuration. The online service will load those artifacts once at startup, verify their metadata and dimensional consistency, and fail with a useful error when required artifacts are absent or corrupt. Retrieval and ranking will live in library modules so evaluation, the API, and tests exercise the same business logic.

## 3. Dataset flow

The source will be the official `wayfair/WANDS` repository and specifically its `product.csv`, `query.csv`, and `label.csv` files. A documented acquisition script will fetch or accept a local checkout of those files and record source provenance, but raw data will not be committed.

The preparation pipeline will:

1. Validate that all required files and columns are present before transforming data.
2. Normalize identifiers to one stable type and check key uniqueness where expected.
3. Join relevance labels to valid query and product identifiers while reporting orphaned records.
4. Preserve the original user-facing product text while creating normalized search text separately.
5. Convert WANDS labels to explicit grades: `Exact = 2`, `Partial = 1`, and `Irrelevant = 0`.
6. Split unique query IDs with seed 42 into 70% train, 15% validation, and 15% test partitions. Rows inherit the partition of their query ID; individual query-product judgments are never split independently.
7. Save canonical processed tables and a split manifest with source hashes, row counts, schema version, creation time, configuration, and code version.

The split manifest will be created once by a deterministic script and reused by training and evaluation. Training queries supply reranker examples. Validation queries are used to select hybrid weights, candidate depths, thresholds, and reranker settings. Test queries remain untouched until the final selected configuration is evaluated. Assertions will detect query leakage across all partitions.

## 4. Product representation

One canonical product record will contain the product ID and display fields needed by the API, plus searchable text composed from the product name, category hierarchy, description, and product features. Field boundaries will be retained so the reranker can compute title-, description-, and feature-specific overlap without reverse-engineering the combined text.

Text normalization will be deterministic and shared where appropriate: Unicode and whitespace cleanup, null handling, and conservative case normalization. The project will avoid aggressive stemming or transformations that make product names unreadable. Lexical and dense encoders may apply their own model-specific tokenization, but both will consume the same semantically complete source fields.

The combined representation will identify fields explicitly, for example `name: ... category: ... description: ... features: ...`, so dense embeddings retain context. Lexical field contribution will initially be represented through documented field repetition or separate sparse matrices with configured weights; the validation set will decide any weighting. Raw and normalized values will remain distinguishable.

## 5. Lexical indexing and retrieval

The lexical baseline will use scikit-learn's `TfidfVectorizer` and cosine similarity. The fitted vectorizer and a CSR product matrix will be persisted with a stable product-row mapping. Initial tokenization settings will be simple, explainable, and configuration-driven; candidate choices such as unigrams versus limited bigrams and minimum document frequency will be selected only on validation queries.

At query time, the service will transform the query with the fitted vectorizer and score it against the sparse product matrix. Efficient sparse operations and partial top-k selection will avoid sorting the entire catalog when possible. An all-zero query vector will return a clear, deterministic result rather than silently inventing scores.

Index metadata will include vectorizer parameters, vocabulary size, matrix shape and dtype, product mapping hash, source data hash, and package versions.

## 6. Dense semantic indexing and retrieval

The dense baseline will use FastEmbed with a small ONNX-backed English model, initially `BAAI/bge-small-en-v1.5`. Before implementation, the installed FastEmbed version, constructor options, query/document embedding methods, batching behavior, and normalization behavior will be inspected. No unsupported API behavior will be assumed.

Product embeddings will be generated offline in CPU-conscious batches and stored as a contiguous NumPy array, preferably `float32`. Rows will share the same product mapping used by lexical retrieval. If the model does not return normalized vectors, the indexing job will L2-normalize them once. Query vectors will be normalized at request time, making matrix-vector dot product equivalent to cosine similarity.

For roughly 43,000 products, exact NumPy matrix-vector scoring is simple, reproducible, and sufficiently lightweight; a vector database is unnecessary. The benchmark will record embedding time separately from similarity-search time. Dense artifact metadata will include model name and revision when available, FastEmbed version, dimensions, normalization policy, dtype, batch size, source hash, and mapping hash.

## 7. Hybrid retrieval

Hybrid retrieval will form the union of the top lexical and dense candidates, then combine comparable per-query scores:

```text
hybrid_score = alpha * normalized_lexical_score
             + (1 - alpha) * normalized_dense_score
```

Score normalization will be deterministic and robust to constant-score candidate sets. The exact method, candidate depth, and `alpha` will be configuration values chosen using validation nDCG@10, never the test set. Missing modality scores within the candidate union will receive that modality's documented floor. Stable tie-breaking by product ID will make rankings reproducible.

The implementation will also retain each component score in result objects. This supports evaluation, debugging, API transparency, and reranker feature construction without computing the same retrieval work twice.

## 8. Supervised relevance reranking

The reranker will be a small pointwise scikit-learn relevance model trained only from judged WANDS rows whose query IDs are in the training partition. It will rerank a bounded hybrid candidate set rather than score the full catalog. A suitable initial design is regularized multinomial logistic regression, with an expected relevance score calculated from class probabilities; a small tree-based alternative may be compared if it remains explainable and materially improves validation performance.

Features must be computable for an arbitrary user query and catalog product at request time:

- lexical similarity;
- dense similarity;
- hybrid score;
- query-title token overlap and coverage;
- query-description token overlap and coverage;
- query-feature token overlap;
- exact query phrase match indicators;
- product name token length;
- query token length; and
- simple length or coverage interactions established before test evaluation.

The reranker will not use `query_class`, query IDs, label-derived aggregates, test statistics, or any field unavailable for a new query. Feature code will be shared by offline training and online inference. A fitted feature schema, preprocessing pipeline, model, class mapping, training split hash, and package versions will be saved together. Loading will reject missing features, mismatched schema versions, or incompatible product/index mappings.

Class imbalance handling and model hyperparameters will be selected using validation queries. Reported comparison tables will include the hybrid baseline alongside the reranked results so any gain or regression is visible rather than assumed.

## 9. Evaluation methodology

### Metrics and relevance definitions

nDCG@10 will be the primary metric and will use graded relevance values 2, 1, and 0. The project will also report nDCG@5, Precision@5, Precision@10, Recall@5, Recall@10, and MRR@10.

For binary metrics, `Exact` and `Partial` count as relevant and `Irrelevant` does not. MRR@10 uses the rank of the first `Exact` or `Partial` result, or zero if none occurs in the top 10. Metrics will be computed per query and macro-averaged so queries with many judgments do not dominate. Reports will include the number of eligible queries and explain how queries with no relevant judgments are treated.

### Context A: judged-candidate ranking

This is the primary controlled evaluation. For each query, every product with a WANDS judgment for that query forms its candidate set. Each method scores and ranks exactly those candidates, and all candidates have known grades. This isolates ranking quality from incomplete catalog judgments and enables fair metric comparisons.

### Context B: full-catalog retrieval

Each method will search the complete product catalog. Metrics will measure whether known judged-relevant products are recovered in the top results. Unjudged products will be treated as unknown, not asserted to be irrelevant. Consequently, these results measure recovery against incomplete judgments and will be labeled separately from the controlled judged-candidate results.

### Tuning, reporting, and latency

All architecture choices and hyperparameters will be fixed using the validation partition. The test partition will be evaluated once after the configuration is frozen. Reports will present lexical, dense, hybrid, and reranked systems under identical splits and candidate contexts, with configuration and artifact identifiers attached.

Latency measurement will use warm-process repeated queries on the target CPU. It will record query encoding, lexical scoring, dense scoring, fusion, reranking, and end-to-end latency. Reports will include sample count and median and p95 latency, and will distinguish cold startup/index loading from steady-state requests. No benchmark number will be documented until it has actually been measured.

## 10. Search API

FastAPI will expose thin transport endpoints over an injected search service. Planned endpoints are:

- `GET /health` for process and artifact readiness;
- `GET /metadata` for safe index/model version information;
- `POST /search` for query, retrieval mode, result count, and ranked results; and
- `POST /feedback` for optional user feedback tied to a search event.

Pydantic models will validate non-empty queries, supported modes, and bounded `top_k`. Search responses will include product identifiers and display fields, final score, rank, latency, and component scores when the selected mode provides them. Internal file paths and sensitive environment details will not be exposed. Startup will load and validate artifacts once; request handlers will not rebuild indexes.

## 11. Streamlit dashboard

Streamlit will be a presentation client of the FastAPI service, not a second search implementation. The dashboard will allow a user to enter a natural-language product query, choose lexical, dense, hybrid, or reranked search, and inspect ranked product cards with response latency and concise score information. It may provide side-by-side comparisons for demonstration, but each result will still come through the service contract.

The UI will handle API unavailability, validation errors, and empty results visibly. Feedback controls will call the API and will not write directly to SQLite. Configuration will provide the local API base URL without committing secrets.

## 12. SQLite query and feedback logging

The API analytics layer will use Python's standard `sqlite3` module unless later requirements justify more abstraction. Migrations or explicit schema-version initialization will create tables for search events, returned result impressions, and optional feedback.

Search logs will record a generated event ID, UTC timestamp, query text or a documented privacy-conscious alternative, retrieval mode, requested result count, latency, artifact version, and status. Result impressions will record product ID and rank. Feedback will reference the search event and product where applicable. Writes will use parameterized SQL, short transactions, and a concurrency-safe connection strategy. Analytics failures will be observable but will not corrupt search artifacts. Local database files, journals, and backups will stay out of Git.

## 13. Testing strategy

Tests will be layered and deterministic:

- Unit tests will cover text normalization, label mapping, query-level splits, overlap features, score normalization, top-k behavior, metric formulas, stable tie-breaking, configuration validation, and artifact metadata checks.
- Retrieval tests will use tiny synthetic matrices and committed WANDS-shaped fixtures so expected rankings are exact and fast.
- Reranker tests will verify feature parity between training and inference, prohibit unavailable features, and check model-schema validation.
- Integration tests will build tiny temporary artifacts, start the search service in-process, exercise FastAPI with `httpx`, and verify search and feedback persistence.
- Contract tests will confirm the Streamlit-facing API schema without requiring the UI or downloaded model in CI.
- Failure tests will cover missing, corrupted, incompatible, or dimensionally inconsistent artifacts and confirm that errors are explicit.

Random behavior will use seed 42. Tests will not download WANDS or embedding models. Dense interfaces will be injectable so deterministic fake embeddings can validate orchestration; separate opt-in local smoke tests may exercise the real FastEmbed model after it has been downloaded intentionally.

## 14. GitHub Actions strategy

GitHub Actions will run on supported Python 3.12 runners and use `uv` with the committed lockfile. Pull requests and pushes to `main` will run:

1. dependency synchronization from the lockfile;
2. Ruff formatting and lint checks;
3. mypy on the application package;
4. pytest unit and integration suites with coverage reporting if configured; and
5. a build or package import check.

CI will use only small committed fixtures and synthetic embeddings. It will not fetch the WANDS dataset or a model, generate full indexes, start external services, or depend on credentials. Dependency and cache keys will include the lockfile. Workflow permissions will be minimal.

## 15. Generated artifact and reproducibility strategy

Committed code and configuration must be sufficient to recreate every generated file. Offline scripts will have explicit inputs and outputs and will support a documented order such as acquire, prepare, split, build lexical index, build dense index, train reranker, and evaluate.

Each artifact family will include a small machine-readable manifest containing schema version, creation time, command/configuration, seed, source file hashes, product-row mapping hash, relevant package/model versions, shapes and dtypes, and code commit when available. Artifacts loaded together must have compatible dataset and mapping identifiers. Writes should use temporary files followed by an atomic rename so interrupted jobs do not appear complete.

The following will remain local and be excluded from Git:

- raw and processed WANDS data;
- TF-IDF matrices and serialized vectorizers;
- dense product embedding arrays;
- trained reranker binaries;
- generated evaluation reports, plots, and benchmark outputs unless a deliberately reviewed small report is later chosen for documentation;
- SQLite databases and journal files;
- downloaded model caches;
- logs, caches, coverage files, and temporary files; and
- local environment and secret files.

Small schemas, configuration files, fixture data, artifact manifest examples, and scripts will be committed. Checksums and metadata can be published without publishing the large payloads.

## 16. Hardware-conscious design

The catalog size permits exact CPU retrieval. Sparse TF-IDF storage avoids densifying lexical vectors. Dense embeddings use `float32`, batch generation, memory mapping when useful, and one NumPy matrix-vector operation per query. Product text is prepared once rather than rebuilt per request. The API loads shared read-only artifacts once and returns a bounded result set. Reranking is limited to the hybrid candidate pool.

Configuration will expose batch size and candidate depth so a modest Windows laptop can trade build time against memory. Scripts will report estimated or observed shapes and memory usage before large work. No multiprocessing design will assume Unix `fork`; any parallelism must work with Windows process semantics. Full rebuilds remain explicit offline actions, never side effects of API or UI startup.

## 17. Planned repository structure

```text
semantic-product-search-engine/
|-- src/
|   `-- product_search/
|       |-- __init__.py
|       |-- config.py
|       |-- data/
|       |-- indexing/
|       |-- retrieval/
|       |-- ranking/
|       |-- evaluation/
|       |-- api/
|       |-- analytics/
|       `-- ui/
|-- tests/
|   |-- unit/
|   |-- integration/
|   `-- fixtures/
|-- configs/
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- fixtures/
|-- artifacts/
|   |-- indexes/
|   |-- embeddings/
|   |-- models/
|   `-- reports/
|-- docs/
|-- scripts/
|-- .github/
|   `-- workflows/
|-- pyproject.toml
|-- uv.lock
|-- .gitignore
|-- README.md
|-- AGENTS.md
`-- LICENSE
```

Directory responsibilities are:

- `src/product_search/`: installable application package with no notebook-only business logic.
- `config.py`: typed paths, defaults, environment overrides, and cross-field validation.
- `data/`: dataset schema, acquisition, preparation, text normalization, and split logic.
- `indexing/`: reproducible lexical and dense artifact builders and manifest validation.
- `retrieval/`: lexical, dense, and hybrid search implementations behind shared result interfaces.
- `ranking/`: online-safe feature extraction, reranker training, persistence, and inference.
- `evaluation/`: metric implementations, evaluation contexts, latency measurement, and report generation.
- `api/`: FastAPI construction, request/response models, dependencies, and route handlers.
- `analytics/`: SQLite schema management and query, impression, and feedback repositories.
- `ui/`: Streamlit presentation code and API client only.
- `tests/unit/`: fast isolated behavior tests.
- `tests/integration/`: multi-module, API, artifact-loading, and SQLite tests.
- `tests/fixtures/`: small committed inputs and expected outputs designed for tests.
- `configs/`: reviewed, versioned settings for preparation, indexing, training, evaluation, and serving.
- `data/raw/`: ignored official source CSV files.
- `data/processed/`: ignored canonical tables and split manifests produced locally.
- `data/fixtures/`: optional small documented samples for demonstrations distinct from test internals.
- `artifacts/indexes/`: ignored sparse lexical matrices, vectorizers, and row mappings.
- `artifacts/embeddings/`: ignored dense arrays and embedding manifests.
- `artifacts/models/`: ignored trained reranker payloads and model manifests.
- `artifacts/reports/`: ignored generated metric tables, charts, and latency results by default.
- `docs/`: architecture, data, evaluation, operations, and decision documentation.
- `scripts/`: thin, reproducible command entry points that call package modules.
- `.github/workflows/`: CI definitions.
- `pyproject.toml` and `uv.lock`: project metadata, tools, and reproducible dependency resolution.
- `.gitignore`: exclusions for generated artifacts, local data, databases, environments, and caches.
- `README.md`: setup, architecture overview, commands, measured results, and demo guidance.
- `AGENTS.md`: repository-specific contributor and automation instructions.
- `LICENSE`: public repository license.

Directories will be added only when their stage begins; empty placeholder trees are unnecessary.

## 18. Final demo workflow

The final documented local demo will be reproducible in this order:

1. Install Python 3.12 and `uv`, then synchronize locked dependencies.
2. Acquire the official WANDS files into the ignored raw-data directory and verify their hashes/schema.
3. Run preparation to create the canonical catalog, graded judgments, and query split manifest.
4. Build lexical and dense indexes with recorded metadata.
5. Train and validate the relevance reranker, freeze configuration, and run the one-time test evaluation.
6. Review generated metric and latency reports; copy only real, clearly sourced results into public documentation.
7. Start FastAPI locally and verify artifact readiness through the health endpoint.
8. Start Streamlit configured to call the API.
9. Search examples such as `modern black desk lamp`, `small round coffee table`, `white storage cabinet`, and `blue outdoor rug`; compare retrieval modes and optionally submit feedback.
10. Run Ruff, mypy, pytest, and the package/build check before publishing changes.

The README will distinguish quick fixture-based verification from the full data/model workflow and will state expected download, indexing time, memory, and measured search latency only after those values have been observed.

## 19. Incremental delivery principles

Future stages will use one feature branch at a time, review existing work before modification, and produce small conventional commits that pair behavior with tests. Each stage will report files changed, commands and real results, commits, and unresolved issues. Generated data and indexes will never be committed merely to make a demo appear complete, and no retrieval result, metric, screenshot, or benchmark will be fabricated.
