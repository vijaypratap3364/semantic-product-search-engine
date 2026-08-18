# Final audit and release readiness

Audit date: 2026-08-17 (America/Chicago)

Branch: `main`

Candidate release: `v1.0.0`

## Passed

### Required quality gates

| Command | Actual result |
|---|---|
| `uv sync` | Resolved and checked 96 packages successfully from the clean pre-audit state. |
| `uv sync --locked --all-groups` | Passed for the final candidate and installed the local package as `1.0.0`. |
| `uv run ruff format --check .` | Passed; 103 files already formatted. |
| `uv run ruff check .` | Passed with no lint findings. |
| `uv run mypy src` | Passed; no issues in 46 source files. |
| `uv run pytest` | Passed; 282 tests, 38 dependency deprecation warnings, 90.13% coverage. |
| `uv run --locked python -c "import product_search; print(product_search.__version__)"` | Passed and printed `1.0.0`. |
| `uv build` | Built the `1.0.0` source distribution and wheel successfully. |

The latest remote GitHub Actions run before this audit also passed on Linux for commit `8301c7c`: [CI run 32094216900](https://github.com/vijaypratap3364/semantic-product-search-engine/actions/runs/32094216900).

### Documented command verification

| Area | Verification and result |
|---|---|
| Data download | `uv run python -m product_search.data.download` reused all three cached, non-empty WANDS files without `--force`. It reported 42,994 products, 480 queries, and 233,448 label rows with the pinned upstream revision and SHA-256 hashes. |
| Data preparation | The documented module was run against `tests/fixtures/wands` with temporary output arguments. It produced 3 products, 2 queries, and 4 validated labels without touching the full processed cache. |
| Lexical index | The documented builder created a temporary 3-product sparse index, persisted all expected artifacts, and reported a `(3, 2)` float32 matrix. The production metadata and hashes were separately verified. |
| Dense index | The documented builder created a temporary `(3, 384)` float32 index using `BAAI/bge-small-en-v1.5`, batch size 2, and `--local-files-only`. It used the existing model cache; no model was downloaded or deleted. |
| Evaluation | `uv run python -m product_search.evaluation.benchmark_final --verify-only --local-files-only` validated the full frozen data, split, lexical, dense, reranker, and validation-report hashes. It confirmed zero test-query access during selection. |
| FastAPI | Uvicorn started on isolated port 8765. `/health` returned `ok`, `/ready` returned ready, and `/model` reported mode `rerank`, model `BAAI/bge-small-en-v1.5`, and 42,994 products. The audit process was stopped and the port had no remaining listener. |
| Streamlit | Streamlit started headlessly on isolated port 8766 and `/_stcore/health` returned HTTP 200 with `ok`. The audit process was stopped and the port had no remaining listener. |

Temporary fixture outputs and server logs were written outside the repository. The cached full dataset, embedding model, full indexes, models, and generated reports were not deleted.

### Ignore and artifact review

`git check-ignore -v` confirmed that representative files in every required class are ignored:

- raw WANDS CSVs and manifests under `data/raw/`
- processed Parquet tables and split manifests under `data/processed/`
- dense embeddings and model cache under `artifacts/embeddings/`
- TF-IDF matrices and vectorizer artifacts under `artifacts/indexes/`
- reranker models and generated reports under `artifacts/models/` and `artifacts/reports/`
- SQLite files, `.venv`, pytest/mypy/Ruff caches, `.env` files, and local caches

Only `.gitkeep` placeholders are tracked under generated data and artifact directories. No tracked file exceeds 1 MiB.

### Security and quality review

- **Secrets:** high-signal credential and private-key scans found zero matches in the current tree and across all 67 commits. No `.env`, private-key, or certificate files are tracked.
- **Developer paths:** the tracked tree contains no absolute Windows user-profile, macOS user, or Linux home-directory paths.
- **Hidden hosted APIs:** search and serving run entirely against local static artifacts. Network access is limited to the explicit offline WANDS downloader and initial FastEmbed model acquisition; API startup uses `local_files_only=True`.
- **Bounded input:** both the public API schema and `SearchService` enforce `1 <= top_k <= 100`; queries are stripped and limited to 500 characters.
- **SQL safety:** analytics uses fixed schema/query text and parameterized values. The only formatted SQL is an internal integer `PRAGMA user_version` built from the module constant, not user input.
- **Error safety:** request validation and catch-all handlers return stable structured messages. Tests verify that local paths, raw exceptions, and tracebacks are not returned.
- **Evaluation leakage:** hybrid selection is validation-only, reranker training uses train queries and validation-only selection, and the final benchmark validates frozen provenance before reading held-out test queries.
- **Search ownership:** retrieval lives in the retrieval/ranking modules, composition lives in `SearchService`, FastAPI delegates to the service, and Streamlit communicates through the reusable API client. No UI retrieval implementation was found.
- **Metrics provenance:** the runtime dashboard reads `artifacts/reports/final_test_metrics.json`; it contains no fallback metric constants. README and portfolio figures were cross-checked against generated reports.
- **README commands:** dependency, data, index, evaluation, FastAPI, Streamlit, and test commands resolve to existing modules and were exercised directly or through their safe fixture/verification modes.

### Git review

The pre-audit repository contained 67 non-merge commits with meaningful conventional stages: foundation, data, evaluation, lexical retrieval, semantic retrieval, hybrid fusion, reranking, held-out evaluation, service/API, analytics, UI, performance, CI, and portfolio documentation. The two Stage 16 commits bring the release candidate to 69 commits. History has not been rewritten for release presentation. No tag exists yet.

### Final project summary

| Item | Verified value | Source/boundary |
|---|---:|---|
| Products | 42,994 | `data_summary.json` |
| Queries | 480 | `data_summary.json` |
| Relevance judgments | 233,448 source rows; 231,873 canonical pairs | data and canonicalization reports |
| Selected engine | Reranked hybrid (`rerank`) | `final_engine.json` |
| Held-out test queries | 72 | `final_test_metrics.json` |
| Test nDCG@10 | 0.827633 | judged-candidate evaluation |
| Test Recall@10 | 0.110600 | Exact and Partial treated as relevant |
| Test MRR@10 | 0.972222 | judged-candidate evaluation |
| Median search latency | 53.183 ms | Stage 13 loopback FastAPI default search, 100 observations |
| p95 search latency | 61.851 ms | same boundary; initialization excluded and analytics disabled |
| Tests | 282 passed | Stage 16 local run |
| Coverage | 90.13% | branch-aware pytest coverage |
| Lexical index | 68,829,120 bytes (65.641 MiB) | `benchmark.json` |
| Dense index | 66,587,631 bytes (63.503 MiB) | `benchmark.json` |
| Reranker model | 4,844 bytes (4.730 KiB) | `benchmark.json` |
| Combined search artifacts | 135,421,595 bytes (129.148 MiB) | sum of the three measured artifact families |

## Failed

None. No essential release check failed.

## Warning

- The full 42,994-product dense index was not rebuilt during this audit. Re-embedding was unnecessary because the real dense artifact, product ordering, model metadata, dataset hash, and artifact hashes all passed frozen-provenance verification; the actual builder was exercised with the cached model on a tiny fixture.
- Pytest emitted 38 upstream deprecation warnings: one Starlette `TestClient`/httpx warning and NumPy-shape warnings from joblib. They do not fail the suite but should be revisited during dependency upgrades.
- The frozen Stage 8 selection report records an older engine-only median/p95 boundary of 231.112/265.917 ms over 72 test queries. The release summary above uses the later Stage 13 reproducible loopback API benchmark (53.183/61.851 ms over 100 observations). The boundaries are documented separately and must not be compared as identical measurements.
- Query logging is enabled by default for the local portfolio demo. It can be disabled in configuration and should be reviewed before any non-local deployment.
- Real dashboard screenshots remain intentionally absent until the user captures the running application according to `docs/screenshots/README.md`.

## Not verified

- A from-scratch full-catalog rebuild on a second clean machine was not performed; doing so would redownload data/model assets and repeat 42,994 embeddings without adding evidence beyond the verified hashes and fixture build paths.
- External dependency-vulnerability databases, penetration testing, load testing under concurrent users, and browser-by-browser visual QA were outside this local release audit.
- GitHub Actions cannot verify the Stage 16 commits until the user pushes them. The latest pre-audit `main` run is green.

## Future improvements

- Resolve upstream deprecation warnings during a controlled dependency update.
- Capture and review the three real dashboard screenshots, then add them without editing result values.
- Run a fresh-machine reproducibility drill before production use, retaining hashes and elapsed build times.
- Add dependency vulnerability scanning and a modest concurrent API load test if the project moves beyond a local portfolio demo.
- Reassess approximate nearest-neighbor retrieval only when measured latency or memory fails a defined target at larger catalog scale.

## Release decision

Essential checks pass. `v1.0.0` is an appropriate suggested tag after the Stage 16 commits are pushed and their GitHub Actions run succeeds. This audit does not create, push, or tag a release.

### Direct-main workflow

Use this path when continuing the repository's established main-only workflow:

```powershell
git status --short --branch
git push origin main
$headSha = git rev-parse HEAD
$runId = gh run list --commit $headSha --workflow CI --limit 1 --json databaseId --jq '.[0].databaseId'
gh run watch $runId --exit-status
git tag -a v1.0.0 -m "Semantic Product Search Engine v1.0.0"
git push origin v1.0.0
```

Create and push the tag only after the pushed commit's CI run reports success.

### Pull-request workflow

Use this instead of pushing `main` directly if a final review is required. It publishes the current local `main` commit to a remote release branch without switching the local working branch:

```powershell
git push origin HEAD:refs/heads/release/v1.0.0
$prUrl = gh pr create --base main --head release/v1.0.0 --title "chore: release semantic search v1.0.0" --body-file docs/final-audit.md
gh pr checks $prUrl --watch
gh pr merge $prUrl --merge --delete-branch
git pull --ff-only origin main
$headSha = git rev-parse HEAD
$runId = gh run list --commit $headSha --workflow CI --limit 1 --json databaseId --jq '.[0].databaseId'
gh run watch $runId --exit-status
git tag -a v1.0.0 -m "Semantic Product Search Engine v1.0.0"
git push origin v1.0.0
```

Do not run both push workflows. Do not tag until the merge commit is on local `main` and CI is green.
