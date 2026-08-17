# Continuous integration

The GitHub Actions workflow is named **CI** and runs for every pull request and every push to
`main`. It uses an Ubuntu runner with Python 3.12, the committed `uv.lock`, read-only repository
permissions, a 20-minute timeout, and concurrency cancellation for superseded runs.

The workflow performs these checks as separate visible steps:

1. install Python 3.12 and synchronize all locked dependency groups;
2. check Ruff formatting;
3. run Ruff linting;
4. run strict mypy over `src`;
5. run pytest with branch coverage;
6. import the installed `product_search` package; and
7. build both source and wheel distributions.

Pytest configuration enforces the repository's existing 90% aggregate branch-coverage threshold.
CI explicitly excludes the optional test marked `embedding`. Normal dense tests inject deterministic
fake embedding providers and build only tiny temporary arrays. Data tests use tiny committed fixture
CSVs or in-memory fakes.

The required workflow does not invoke the WANDS downloader, dense-index builder, evaluation scripts,
or performance benchmarks. `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` provide an additional
network guard for model tooling. Dependency installation still downloads the packages pinned by the
lockfile on a fresh runner; it does not download WANDS data or embedding-model weights.

Run the CI commands locally from the repository root:

```powershell
uv python install 3.12
uv sync --locked --all-groups
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked mypy src
uv run --locked pytest -m "not embedding"
uv run --locked python -c "import product_search; print(product_search.__version__)"
uv build
```

The local-only FastEmbed integration test can still be run intentionally when the model is already
cached:

```powershell
uv run pytest -m embedding
```

It uses `local_files_only=True` and skips when the configured model is absent.
