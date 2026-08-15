"""Tests for WANDS schema and integrity validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from product_search.data.validate import (
    DataValidationError,
    WandsTables,
    load_and_validate_wands,
    validate_wands_tables,
)


def test_fixture_schema_and_integrity_are_valid(raw_wands_dir: Path) -> None:
    tables, report = load_and_validate_wands(raw_wands_dir)

    assert len(tables.products) == 3
    assert len(tables.queries) == 2
    assert len(tables.labels) == 4
    assert report.duplicate_counts["product_ids"] == 0
    assert report.missing_value_counts["products"]["product_description"] == 1


def test_missing_raw_files_have_clear_context(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="unable to read required WANDS CSV files"):
        load_and_validate_wands(tmp_path / "missing")


def test_missing_required_column_is_rejected(raw_wands_dir: Path) -> None:
    tables, _ = load_and_validate_wands(raw_wands_dir)
    invalid = WandsTables(
        products=tables.products.drop(columns="product_name"),
        queries=tables.queries,
        labels=tables.labels,
    )

    with pytest.raises(DataValidationError, match="product_name"):
        validate_wands_tables(invalid)


def test_duplicate_product_ids_are_rejected(raw_wands_dir: Path) -> None:
    tables, _ = load_and_validate_wands(raw_wands_dir)
    duplicated_products = tables.products.copy()
    duplicated_products.loc[1, "product_id"] = duplicated_products.loc[0, "product_id"]

    with pytest.raises(DataValidationError, match="duplicate product IDs"):
        validate_wands_tables(WandsTables(duplicated_products, tables.queries, tables.labels))


def test_duplicate_query_ids_are_rejected(raw_wands_dir: Path) -> None:
    tables, _ = load_and_validate_wands(raw_wands_dir)
    duplicated_queries = tables.queries.copy()
    duplicated_queries.loc[1, "query_id"] = duplicated_queries.loc[0, "query_id"]

    with pytest.raises(DataValidationError, match="duplicate query IDs"):
        validate_wands_tables(WandsTables(tables.products, duplicated_queries, tables.labels))


def test_invalid_relevance_label_is_rejected(raw_wands_dir: Path) -> None:
    tables, _ = load_and_validate_wands(raw_wands_dir)
    labels = tables.labels.copy()
    labels.loc[0, "label"] = "Unknown"

    with pytest.raises(DataValidationError, match="invalid relevance values"):
        validate_wands_tables(WandsTables(tables.products, tables.queries, labels))


def test_conflicting_duplicate_judgments_are_reported(raw_wands_dir: Path) -> None:
    tables, _ = load_and_validate_wands(raw_wands_dir)
    conflicting = tables.labels.iloc[[0]].copy()
    conflicting.loc[:, "id"] = "additional-annotation"
    conflicting.loc[:, "label"] = "Partial"
    labels = pd.concat([tables.labels, conflicting], ignore_index=True)

    report = validate_wands_tables(WandsTables(tables.products, tables.queries, labels))

    assert report.duplicate_counts["label_query_product_pairs"] == 1
    assert report.duplicate_counts["conflicting_label_query_product_pairs"] == 1


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("query_id", "missing-query", "unknown query IDs"),
        ("product_id", "missing-product", "unknown product IDs"),
    ],
)
def test_label_foreign_keys_are_enforced(
    raw_wands_dir: Path,
    column: str,
    value: str,
    message: str,
) -> None:
    tables, _ = load_and_validate_wands(raw_wands_dir)
    labels = tables.labels.copy()
    labels.loc[0, column] = value

    with pytest.raises(DataValidationError, match=message):
        validate_wands_tables(WandsTables(tables.products, tables.queries, labels))


def test_missing_query_text_is_rejected(raw_wands_dir: Path) -> None:
    tables, _ = load_and_validate_wands(raw_wands_dir)
    queries = tables.queries.copy()
    queries.loc[0, "query"] = "  "

    with pytest.raises(DataValidationError, match=r"queries\.query"):
        validate_wands_tables(WandsTables(tables.products, queries, tables.labels))
