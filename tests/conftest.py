"""Shared deterministic fixtures for the test suite."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

WANDS_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "wands"


@pytest.fixture
def raw_wands_dir(tmp_path: Path) -> Path:
    """Copy the tiny WANDS-shaped CSV fixture into an isolated raw directory."""

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for source in WANDS_FIXTURE_DIR.glob("*.csv"):
        shutil.copy2(source, raw_dir / source.name)
    return raw_dir
