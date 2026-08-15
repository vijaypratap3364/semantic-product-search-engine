"""Schema and integrity validation for official WANDS CSV tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame, Series

PRODUCT_COLUMNS = (
    "product_id",
    "product_name",
    "product_class",
    "category_hierarchy",
    "product_description",
    "product_features",
    "rating_count",
    "average_rating",
    "review_count",
)
QUERY_COLUMNS = ("query_id", "query", "query_class")
LABEL_COLUMNS = ("id", "query_id", "product_id", "label")
VALID_RELEVANCE_LABELS = frozenset({"Exact", "Partial", "Irrelevant"})
RAW_PRODUCT_COLUMN_RENAMES = {"category hierarchy": "category_hierarchy"}


class DataValidationError(ValueError):
    """Raised when WANDS data violates its expected schema or integrity rules."""


@dataclass(frozen=True)
class WandsTables:
    """Raw WANDS tables loaded with stable identifier dtypes."""

    products: DataFrame
    queries: DataFrame
    labels: DataFrame


@dataclass(frozen=True)
class ValidationReport:
    """Observed data-quality counts recorded after successful validation."""

    missing_value_counts: dict[str, dict[str, int]]
    duplicate_counts: dict[str, int]


def load_wands_tables(raw_dir: Path) -> WandsTables:
    """Load the three required CSVs while preserving identifiers as strings."""

    raw_dir = raw_dir.resolve()
    try:
        products = pd.read_csv(
            raw_dir / "product.csv",
            sep="\t",
            dtype={"product_id": "string"},
        ).rename(columns=RAW_PRODUCT_COLUMN_RENAMES)
        queries = pd.read_csv(raw_dir / "query.csv", sep="\t", dtype={"query_id": "string"})
        labels = pd.read_csv(
            raw_dir / "label.csv",
            sep="\t",
            dtype={"id": "string", "query_id": "string", "product_id": "string"},
        )
    except (OSError, pd.errors.ParserError) as error:
        message = f"unable to read required WANDS CSV files from {raw_dir}: {error}"
        raise DataValidationError(message) from error
    return WandsTables(products=products, queries=queries, labels=labels)


def validate_wands_tables(tables: WandsTables) -> ValidationReport:
    """Validate schemas, keys, relevance values, and foreign-key integrity."""

    _require_columns(tables.products, PRODUCT_COLUMNS, table_name="products")
    _require_columns(tables.queries, QUERY_COLUMNS, table_name="queries")
    _require_columns(tables.labels, LABEL_COLUMNS, table_name="labels")

    _require_values(tables.products["product_id"], "products.product_id")
    _require_values(tables.queries["query_id"], "queries.query_id")
    _require_values(tables.queries["query"], "queries.query")
    for column in LABEL_COLUMNS:
        _require_values(tables.labels[column], f"labels.{column}")

    duplicate_counts = _duplicate_counts(tables)
    if duplicate_counts["product_ids"]:
        message = f"products contains {duplicate_counts['product_ids']} duplicate product IDs"
        raise DataValidationError(message)
    if duplicate_counts["query_ids"]:
        message = f"queries contains {duplicate_counts['query_ids']} duplicate query IDs"
        raise DataValidationError(message)

    observed_labels = set(tables.labels["label"].astype(str))
    invalid_labels = observed_labels - VALID_RELEVANCE_LABELS
    if invalid_labels:
        message = f"labels contains invalid relevance values: {sorted(invalid_labels)}"
        raise DataValidationError(message)

    known_query_ids = set(tables.queries["query_id"])
    labelled_query_ids = set(tables.labels["query_id"])
    unknown_query_ids = labelled_query_ids - known_query_ids
    if unknown_query_ids:
        message = _foreign_key_message("query IDs", unknown_query_ids)
        raise DataValidationError(message)

    known_product_ids = set(tables.products["product_id"])
    labelled_product_ids = set(tables.labels["product_id"])
    unknown_product_ids = labelled_product_ids - known_product_ids
    if unknown_product_ids:
        message = _foreign_key_message("product IDs", unknown_product_ids)
        raise DataValidationError(message)

    return ValidationReport(
        missing_value_counts={
            "products": _missing_counts(tables.products),
            "queries": _missing_counts(tables.queries),
            "labels": _missing_counts(tables.labels),
        },
        duplicate_counts=duplicate_counts,
    )


def load_and_validate_wands(raw_dir: Path) -> tuple[WandsTables, ValidationReport]:
    """Load WANDS data from disk and validate it in one operation."""

    tables = load_wands_tables(raw_dir)
    return tables, validate_wands_tables(tables)


def _require_columns(frame: DataFrame, expected: tuple[str, ...], *, table_name: str) -> None:
    missing = sorted(set(expected) - set(frame.columns))
    if missing:
        message = f"{table_name} is missing required columns: {missing}"
        raise DataValidationError(message)


def _require_values(series: Series[Any], field_name: str) -> None:
    missing = series.isna() | series.astype("string").str.strip().eq("")
    count = int(missing.sum())
    if count:
        message = f"{field_name} contains {count} missing or blank values"
        raise DataValidationError(message)


def _missing_counts(frame: DataFrame) -> dict[str, int]:
    return {str(column): int(count) for column, count in frame.isna().sum().items()}


def _duplicate_counts(tables: WandsTables) -> dict[str, int]:
    label_counts_by_pair = tables.labels.groupby(["query_id", "product_id"])["label"].nunique()
    return {
        "product_ids": int(tables.products["product_id"].duplicated().sum()),
        "query_ids": int(tables.queries["query_id"].duplicated().sum()),
        "label_ids": int(tables.labels["id"].duplicated().sum()),
        "label_query_product_pairs": int(
            tables.labels.duplicated(subset=["query_id", "product_id"]).sum()
        ),
        "conflicting_label_query_product_pairs": int((label_counts_by_pair > 1).sum()),
    }


def _foreign_key_message(key_name: str, unknown_values: set[object]) -> str:
    examples = sorted(str(value) for value in unknown_values)[:5]
    return f"labels references {len(unknown_values)} unknown {key_name}; examples: {examples}"
