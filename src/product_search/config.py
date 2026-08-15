"""Typed project configuration loaded from a committed TOML file."""

from __future__ import annotations

import math
import tomllib
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
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
    paths: ProjectPaths
    default_top_k: int = Field(gt=0, le=100)
    relevance_mapping: RelevanceMapping
    splits: SplitProportions

    def resolve_paths(self, project_root: Path) -> Self:
        """Return settings with all configured paths made absolute."""

        return self.model_copy(update={"paths": self.paths.resolve_against(project_root.resolve())})


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
