# Repository Instructions

## Project constraints

- Keep the project zero-cost and suitable for a student portfolio on a modest Windows CPU.
- Use native Python 3.12 and `uv`.
- Do not use Docker, WSL, Kubernetes, cloud infrastructure, hosted APIs, external databases,
  vector database servers, or GPU-only dependencies.
- Do not add heavyweight infrastructure or libraries when NumPy, SciPy, scikit-learn, SQLite,
  or the Python standard library can meet the requirement.
- Do not install FastEmbed, FastAPI, or Streamlit before the stage that introduces each tool.
- Never fabricate metrics, benchmarks, screenshots, relevance labels, test results, or generated
  data.

## Required commands

Run these checks before completing a code stage:

```powershell
uv sync
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
```

Run targeted checks while iterating, but finish with the full command set. Report only commands
that actually ran and their real results.

## Architecture boundaries

- Keep data preparation, indexing, retrieval, ranking, evaluation, API, analytics, and UI logic
  in their corresponding `src/product_search/` modules.
- Keep business logic out of Streamlit pages and transport handlers.
- Make Streamlit consume the search service rather than reimplement retrieval.
- Share feature generation between reranker training and inference.
- Keep test-query data isolated from tuning and split by query ID, never by judgment row.
- Treat missing or incompatible generated artifacts as explicit errors; do not silently rebuild
  or recover from corruption.
- Generated data, indexes, embeddings, trained models, reports, model caches, and SQLite files stay
  out of Git. Commit the scripts and metadata contracts needed to recreate them.

## Tests and quality

- Add or update tests with every meaningful behavior change.
- Use type hints and keep `mypy src` clean.
- Keep Ruff formatting and lint checks clean.
- Use seed 42 wherever randomness exists.
- Use small committed fixtures and deterministic fake embeddings in automated tests; do not make
  tests download WANDS or pretrained models.
- Include failure-path tests for validation and artifact loading behavior.

## Git workflow

- Start each stage by checking status and the current branch; pull `main` only when a configured
  remote is available and authentication already works.
- Work on the requested feature branch and do not merge it automatically.
- Preserve unrelated user changes and review the diff before each commit.
- Use small coherent commits with one of: `feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`,
  `ci:`, or `perf:`.
- Never commit generated embeddings, large indexes, local databases, caches, `.env` files, or
  secrets.
- Never backdate, fabricate, force-push, or rewrite published history.
- End each stage with final checks, status, created commits, changed files, and unresolved issues,
  then stop.
