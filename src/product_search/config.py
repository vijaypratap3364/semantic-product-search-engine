"""Typed project configuration loaded from a committed TOML file."""

from __future__ import annotations

import math
import tomllib
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "base.toml"


class ImmutableModel(BaseModel):
    """Base model for validated configuration sections."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectPaths(ImmutableModel):
    """Filesystem locations for local data and generated artifacts."""

    raw_data: Path
    processed_data: Path
    indexes: Path
    embeddings: Path
    models: Path
    reports: Path

    def resolve_against(self, project_root: Path) -> Self:
        """Return paths made absolute relative to ``project_root``."""

        def resolve(path: Path) -> Path:
            return path if path.is_absolute() else (project_root / path).resolve()

        return self.model_copy(
            update={
                "raw_data": resolve(self.raw_data),
                "processed_data": resolve(self.processed_data),
                "indexes": resolve(self.indexes),
                "embeddings": resolve(self.embeddings),
                "models": resolve(self.models),
                "reports": resolve(self.reports),
            }
        )


class RelevanceMapping(ImmutableModel):
    """Numeric grades for the official WANDS relevance labels."""

    exact: int = Field(alias="Exact")
    partial: int = Field(alias="Partial")
    irrelevant: int = Field(alias="Irrelevant")

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        """Require strictly descending grades from exact to irrelevant."""

        if not self.exact > self.partial > self.irrelevant:
            message = "relevance grades must satisfy Exact > Partial > Irrelevant"
            raise ValueError(message)
        return self


class SplitProportions(ImmutableModel):
    """Query-level train, validation, and test proportions."""

    train: float = Field(gt=0.0, lt=1.0)
    validation: float = Field(gt=0.0, lt=1.0)
    test: float = Field(gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_total(self) -> Self:
        """Require partitions to cover the complete query set."""

        total = self.train + self.validation + self.test
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            message = f"split proportions must sum to 1.0; got {total}"
            raise ValueError(message)
        return self


class LexicalSettings(ImmutableModel):
    """Configurable, CPU-conscious TF-IDF baseline parameters."""

    lowercase: bool = True
    analyzer: Literal["word"] = "word"
    ngram_min: int = Field(default=1, ge=1, le=3)
    ngram_max: int = Field(default=2, ge=1, le=3)
    sublinear_tf: bool = True
    min_df: int = Field(default=2, ge=1)
    max_features: int | None = Field(default=100_000, ge=1)
    norm: Literal["l2"] = "l2"

    @model_validator(mode="after")
    def validate_ngram_range(self) -> Self:
        """Require an ordered n-gram range."""

        if self.ngram_min > self.ngram_max:
            raise ValueError("lexical ngram_min must not exceed ngram_max")
        return self


class DenseSettings(ImmutableModel):
    """Lightweight FastEmbed index and model settings."""

    model_name: str = Field(default="BAAI/bge-small-en-v1.5", min_length=1)
    expected_dimension: int = Field(default=384, ge=1)
    batch_size: int = Field(default=1, ge=1, le=1024)
    normalization: Literal["l2"] = "l2"


class HybridSettings(ImmutableModel):
    """Interpretable lexical/semantic fusion settings."""

    strategy: Literal["weighted_normalized", "rrf"] = "weighted_normalized"
    semantic_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    candidate_depth: int = Field(default=100, ge=1, le=10_000)
    rrf_k: int = Field(default=60, ge=1)
    semantic_weight_grid: tuple[float, ...] = tuple(index / 10 for index in range(11))

    @model_validator(mode="after")
    def validate_weight_grid(self) -> Self:
        """Require a unique, ascending, bounded, interpretable search grid."""

        if not self.semantic_weight_grid:
            raise ValueError("hybrid semantic_weight_grid must not be empty")
        if any(weight < 0.0 or weight > 1.0 for weight in self.semantic_weight_grid):
            raise ValueError("hybrid semantic weights must be between 0.0 and 1.0")
        if tuple(sorted(set(self.semantic_weight_grid))) != self.semantic_weight_grid:
            raise ValueError("hybrid semantic_weight_grid must be unique and ascending")
        return self


class RerankerSettings(ImmutableModel):
    """Small validation-selected logistic relevance model settings."""

    candidate_depth: int = Field(default=100, ge=10, le=1_000)
    c_grid: tuple[float, ...] = (0.1, 1.0, 10.0)
    class_weight_options: tuple[Literal["none", "balanced"], ...] = (
        "none",
        "balanced",
    )
    max_iter: int = Field(default=500, ge=100, le=10_000)
    default_search_mode: Literal["hybrid", "reranker"] = "hybrid"

    @model_validator(mode="after")
    def validate_search_grid(self) -> Self:
        """Require small, deterministic, positive model-selection grids."""

        if not self.c_grid or any(not math.isfinite(value) or value <= 0 for value in self.c_grid):
            raise ValueError("reranker c_grid must contain positive finite values")
        if tuple(sorted(set(self.c_grid))) != self.c_grid:
            raise ValueError("reranker c_grid must be unique and ascending")
        if not self.class_weight_options:
            raise ValueError("reranker class_weight_options must not be empty")
        if len(set(self.class_weight_options)) != len(self.class_weight_options):
            raise ValueError("reranker class_weight_options must be unique")
        return self


class AnalyticsSettings(ImmutableModel):
    """Privacy-conscious local SQLite analytics settings."""

    query_logging_enabled: bool = True
    database_path: Path = Path("data/local/search_analytics.sqlite")

    def resolve_against(self, project_root: Path) -> Self:
        """Return the database path made absolute relative to ``project_root``."""

        database_path = (
            self.database_path
            if self.database_path.is_absolute()
            else (project_root / self.database_path).resolve()
        )
        return self.model_copy(update={"database_path": database_path})


class UiSettings(ImmutableModel):
    """Local Streamlit-to-FastAPI connection settings."""

    api_base_url: str = "http://127.0.0.1:8000"
    request_timeout_seconds: float = Field(default=30.0, gt=0.0, le=120.0)

    @field_validator("api_base_url")
    @classmethod
    def validate_api_base_url(cls, value: str) -> str:
        """Require an explicit HTTP(S) origin without a trailing slash."""

        normalized = value.strip().rstrip("/")
        if not normalized.startswith(("http://", "https://")):
            raise ValueError("ui api_base_url must use http:// or https://")
        if normalized in {"http:", "https:"}:
            raise ValueError("ui api_base_url must include a host")
        return normalized


class ProjectSettings(BaseSettings):
    """Validated settings shared by offline and online project components."""

    model_config = SettingsConfigDict(
        env_prefix="PRODUCT_SEARCH_",
        env_nested_delimiter="__",
        extra="forbid",
        frozen=True,
    )

    random_seed: int = Field(ge=0)
    wands_repository: str = Field(min_length=1)
    analytics: AnalyticsSettings
    ui: UiSettings
    paths: ProjectPaths
    default_top_k: int = Field(gt=0, le=100)
    relevance_mapping: RelevanceMapping
    splits: SplitProportions
    lexical: LexicalSettings
    dense: DenseSettings
    hybrid: HybridSettings
    reranker: RerankerSettings

    def resolve_paths(self, project_root: Path) -> Self:
        """Return settings with all configured paths made absolute."""

        root = project_root.resolve()
        return self.model_copy(
            update={
                "analytics": self.analytics.resolve_against(root),
                "paths": self.paths.resolve_against(root),
            }
        )


def _read_toml(config_path: Path) -> dict[str, Any]:
    try:
        with config_path.open("rb") as config_file:
            return tomllib.load(config_file)
    except FileNotFoundError as error:
        message = f"configuration file does not exist: {config_path}"
        raise FileNotFoundError(message) from error
    except tomllib.TOMLDecodeError as error:
        message = f"configuration file is not valid TOML: {config_path}"
        raise ValueError(message) from error


def load_settings(
    config_path: Path | None = None,
    *,
    project_root: Path | None = None,
) -> ProjectSettings:
    """Load and validate settings, resolving relative paths from the project root."""

    selected_path = (config_path or DEFAULT_CONFIG_PATH).resolve()
    root = project_root.resolve() if project_root is not None else selected_path.parent.parent
    settings = ProjectSettings.model_validate(_read_toml(selected_path))
    return settings.resolve_paths(root)
