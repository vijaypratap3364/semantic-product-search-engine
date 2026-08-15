"""Prepare validated WANDS tables for later search indexing and evaluation."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from numbers import Real
from pathlib import Path
from typing import TypedDict, cast

import pandas as pd
from pandas import DataFrame

from product_search.config import load_settings
from product_search.data.download import sha256_file
from product_search.data.validate import ValidationReport, load_and_validate_wands

PRODUCT_TEXT_FIELDS = (
    "product_name",
    "product_class",
    "category_hierarchy",
    "product_description",
    "product_features",
)


class DataSummary(TypedDict):
    """Actual counts and quality observations from one preparation run."""

    created_at: str
    source_manifest_sha256: str | None
    product_count: int
    query_count: int
    label_count: int
    relevance_distribution: dict[str, int]
    missing_value_counts: dict[str, dict[str, int]]
    duplicate_counts: dict[str, int]


def normalize_whitespace(value: object) -> str:
    """Convert one scalar to clean text without emitting missing-value sentinels."""

    if value is None or value is pd.NA or value is pd.NaT:
        return ""
    if isinstance(value, Real) and math.isnan(float(value)):
        return ""
    return " ".join(str(value).split())


def build_product_text(product: Mapping[str, object]) -> str:
    """Build deterministic searchable text from legitimate catalog fields."""

    values = (normalize_whitespace(product.get(field)) for field in PRODUCT_TEXT_FIELDS)
    return " ".join(value for value in values if value)


def construct_product_documents(products: DataFrame) -> DataFrame:
    """Return products with a deterministic ``product_text`` column."""

    prepared = products.copy()
    records = cast(
        list[dict[str, object]],
        prepared.loc[:, list(PRODUCT_TEXT_FIELDS)].to_dict(orient="records"),
    )
    prepared["product_text"] = [build_product_text(record) for record in records]
    return prepared


def prepare_wands_data(
    raw_dir: Path,
    processed_dir: Path,
    report_path: Path,
    *,
    relevance_mapping: Mapping[str, int],
    timestamp: datetime | None = None,
) -> DataSummary:
    """Validate raw WANDS data, write Parquet tables, and record an actual summary."""

    raw_dir = raw_dir.resolve()
    processed_dir = processed_dir.resolve()
    report_path = report_path.resolve()
    tables, validation_report = load_and_validate_wands(raw_dir)

    products = construct_product_documents(tables.products)
    queries = tables.queries.copy()
    labels = tables.labels.copy()
    labels["relevance_grade"] = labels["label"].map(relevance_mapping).astype("int8")

    processed_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet_atomic(products, processed_dir / "products.parquet")
    _write_parquet_atomic(queries, processed_dir / "queries.parquet")
    _write_parquet_atomic(labels, processed_dir / "labels.parquet")

    summary = _build_summary(
        products=products,
        queries=queries,
        labels=labels,
        validation_report=validation_report,
        raw_manifest=raw_dir / "manifest.json",
        timestamp=timestamp,
        relevance_labels=tuple(relevance_mapping),
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(report_path, summary)
    return summary


def _build_summary(
    *,
    products: DataFrame,
    queries: DataFrame,
    labels: DataFrame,
    validation_report: ValidationReport,
    raw_manifest: Path,
    timestamp: datetime | None,
    relevance_labels: tuple[str, ...],
) -> DataSummary:
    counts = labels["label"].value_counts()
    relevance_distribution = {label: int(counts.get(label, 0)) for label in relevance_labels}
    created_at = (timestamp or datetime.now(UTC)).astimezone(UTC).isoformat()
    manifest_hash = sha256_file(raw_manifest) if raw_manifest.is_file() else None
    return DataSummary(
        created_at=created_at,
        source_manifest_sha256=manifest_hash,
        product_count=len(products),
        query_count=len(queries),
        label_count=len(labels),
        relevance_distribution=relevance_distribution,
        missing_value_counts=validation_report.missing_value_counts,
        duplicate_counts=validation_report.duplicate_counts,
    )


def _write_parquet_atomic(frame: DataFrame, path: Path) -> None:
    temporary_path = path.with_name(f"{path.name}.part")
    try:
        frame.to_parquet(temporary_path, engine="pyarrow", index=False)
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _write_json_atomic(path: Path, payload: DataSummary) -> None:
    temporary_path = path.with_name(f"{path.name}.part")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as output_file:
            json.dump(payload, output_file, indent=2, sort_keys=True)
            output_file.write("\n")
        os.replace(temporary_path, path)
    except OSError:
        temporary_path.unlink(missing_ok=True)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, help="Directory containing the raw WANDS CSVs.")
    parser.add_argument("--processed-dir", type=Path, help="Output directory for Parquet tables.")
    parser.add_argument("--report", type=Path, help="Output path for the data summary JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the WANDS preparation command-line interface."""

    arguments = _build_parser().parse_args(argv)
    settings = load_settings()
    summary = prepare_wands_data(
        raw_dir=arguments.raw_dir or settings.paths.raw_data,
        processed_dir=arguments.processed_dir or settings.paths.processed_data,
        report_path=arguments.report or settings.paths.reports / "data_summary.json",
        relevance_mapping=settings.relevance_mapping.model_dump(by_alias=True),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
