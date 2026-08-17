"""Guard the required lightweight GitHub Actions contract."""

from pathlib import Path

WORKFLOW_PATH = Path(__file__).parents[2] / ".github" / "workflows" / "ci.yml"


def test_ci_workflow_runs_required_checks_without_heavy_offline_jobs() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "pull_request:" in workflow
    assert "push:" in workflow
    assert "      - main" in workflow
    assert "  contents: read" in workflow

    required_commands = (
        "uv sync --locked --all-groups",
        "uv run --locked ruff format --check .",
        "uv run --locked ruff check .",
        "uv run --locked mypy src",
        'uv run --locked pytest -m "not embedding"',
        'uv run --locked python -c "import product_search;',
        "        run: uv build\n",
    )
    for command in required_commands:
        assert command in workflow

    forbidden_jobs = (
        "product_search.data.download",
        "product_search.indexing.build_dense",
        "product_search.evaluation.benchmark_",
        "uv build --locked",
    )
    for command in forbidden_jobs:
        assert command not in workflow

    assert 'HF_HUB_OFFLINE: "1"' in workflow
    assert 'TRANSFORMERS_OFFLINE: "1"' in workflow
