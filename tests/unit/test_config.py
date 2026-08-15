"""Tests for the typed project configuration."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from product_search.config import ProjectSettings, load_settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = PROJECT_ROOT / "configs" / "base.toml"


def test_load_base_configuration() -> None:
    settings = load_settings(BASE_CONFIG)

    assert settings.random_seed == 42
    assert settings.wands_repository == "wayfair/WANDS"
    assert settings.default_top_k == 10
    assert settings.relevance_mapping.exact == 2
    assert settings.relevance_mapping.partial == 1
    assert settings.relevance_mapping.irrelevant == 0
    assert settings.splits.train == pytest.approx(0.70)
    assert settings.splits.validation == pytest.approx(0.15)
    assert settings.splits.test == pytest.approx(0.15)
    assert settings.lexical.ngram_min == 1
    assert settings.lexical.ngram_max == 2
    assert settings.lexical.min_df == 2
    assert settings.lexical.max_features == 100_000
    assert settings.lexical.norm == "l2"
    assert settings.dense.model_name == "BAAI/bge-small-en-v1.5"
    assert settings.dense.expected_dimension == 384
    assert settings.dense.batch_size == 64
    assert settings.dense.normalization == "l2"
    assert settings.paths.raw_data == PROJECT_ROOT / "data" / "raw"
    assert settings.paths.indexes == PROJECT_ROOT / "artifacts" / "indexes"
    assert all(
        path.is_absolute()
        for path in (
            settings.paths.raw_data,
            settings.paths.processed_data,
            settings.paths.indexes,
            settings.paths.embeddings,
            settings.paths.models,
            settings.paths.reports,
        )
    )


def test_project_root_can_be_overridden(tmp_path: Path) -> None:
    settings = load_settings(BASE_CONFIG, project_root=tmp_path)

    assert settings.paths.raw_data == tmp_path / "data" / "raw"
    assert settings.paths.reports == tmp_path / "artifacts" / "reports"


def test_invalid_split_total_is_rejected() -> None:
    with BASE_CONFIG.open("rb") as config_file:
        config = tomllib.load(config_file)
    config["splits"]["test"] = 0.20

    with pytest.raises(ValidationError, match=r"split proportions must sum to 1\.0"):
        ProjectSettings.model_validate(config)


def test_invalid_relevance_order_is_rejected() -> None:
    with BASE_CONFIG.open("rb") as config_file:
        config = tomllib.load(config_file)
    config["relevance_mapping"]["Exact"] = 0

    with pytest.raises(ValidationError, match="Exact > Partial > Irrelevant"):
        ProjectSettings.model_validate(config)


def test_invalid_lexical_ngram_range_is_rejected() -> None:
    with BASE_CONFIG.open("rb") as config_file:
        config = tomllib.load(config_file)
    config["lexical"]["ngram_min"] = 3
    config["lexical"]["ngram_max"] = 1

    with pytest.raises(ValidationError, match="ngram_min must not exceed ngram_max"):
        ProjectSettings.model_validate(config)


def test_missing_configuration_has_context(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.toml"

    with pytest.raises(FileNotFoundError, match=re.escape(str(missing_path))):
        load_settings(missing_path)


def test_invalid_toml_has_context(tmp_path: Path) -> None:
    invalid_path = tmp_path / "invalid.toml"
    invalid_path.write_text("[invalid", encoding="utf-8")

    with pytest.raises(ValueError, match="configuration file is not valid TOML"):
        load_settings(invalid_path)
