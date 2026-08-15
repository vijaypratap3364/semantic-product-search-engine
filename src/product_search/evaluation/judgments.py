"""Build an auditable one-row-per-pair evaluation judgment table."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from pandas import DataFrame

from product_search.config import load_settings
from product_search.data.download import sha256_file

JUDGMENT_KEYS = ("query_id", "product_id")
CONFLICT_RESOLUTION_POLICY = (
    "Choose the most frequent label for each query-product pair; when labels tie, choose "
    "the label with the highest configured relevance grade."
)


class JudgmentCanonicalizationReport(TypedDict):
    """Counts and provenance for one canonicalization run."""

    created_at: str
    source_labels_sha256: str
    original_judgment_count: int
    canonical_judgment_count: int
    repeated_rows_beyond_first: int
    identical_duplicate_rows: int
    conflicting_pair_count: int
    conflict_label_combinations: dict[str, int]
    resolution_counts: dict[str, int]
    conflict_resolution_policy: str


class JudgmentQuality(TypedDict):
    """In-memory quality counts independent of file provenance."""

    original_judgment_count: int
    canonical_judgment_count: int
    repeated_rows_beyond_first: int
    identical_duplicate_rows: int
    conflicting_pair_count: int
    conflict_label_combinations: dict[str, int]
    resolution_counts: dict[str, int]
    conflict_resolution_policy: str


def canonicalize_judgments(
    labels: DataFrame,
    *,
    relevance_mapping: Mapping[str, int],
) -> tuple[DataFrame, JudgmentQuality]:
    """Return deterministic canonical judgments and their data-quality counts.

    Majority vote uses every source row. A tie is resolved toward the highest observed
    relevance grade so assessor evidence of relevance is not converted to a false negative.
    Provenance columns make every collapse and conflict resolution auditable.
    """

    _require_columns(labels, {*JUDGMENT_KEYS, "label"})
    working = labels.loc[:, [*JUDGMENT_KEYS, "label"]].copy()
    working["label"] = working["label"].astype("string")

    observed_labels = set(working["label"].dropna().astype(str))
    unknown_labels = observed_labels - set(relevance_mapping)
    if unknown_labels:
        message = (
            f"judgments contain labels absent from relevance mapping: {sorted(unknown_labels)}"
        )
        raise ValueError(message)
    if working[[*JUDGMENT_KEYS, "label"]].isna().any(axis=None):
        raise ValueError("judgment keys and labels must not be missing")

    label_counts = (
        working.groupby([*JUDGMENT_KEYS, "label"], sort=True, observed=True)
        .size()
        .rename("label_count")
        .reset_index()
    )
    label_counts["relevance_grade"] = label_counts["label"].map(relevance_mapping).astype("int64")
    ranked_labels = label_counts.sort_values(
        [*JUDGMENT_KEYS, "label_count", "relevance_grade", "label"],
        ascending=[True, True, False, False, True],
        kind="stable",
    )
    winners = ranked_labels.drop_duplicates(list(JUDGMENT_KEYS), keep="first")

    pair_summary = (
        working.groupby(list(JUDGMENT_KEYS), sort=True, observed=True)
        .agg(judgment_count=("label", "size"), distinct_label_count=("label", "nunique"))
        .reset_index()
    )
    observed = (
        label_counts.sort_values(
            [*JUDGMENT_KEYS, "relevance_grade", "label"],
            ascending=[True, True, False, True],
            kind="stable",
        )
        .groupby(list(JUDGMENT_KEYS), sort=True, observed=True)["label"]
        .agg("|".join)
        .rename("observed_labels")
        .reset_index()
    )
    maximum_counts = (
        label_counts.groupby(list(JUDGMENT_KEYS), sort=True, observed=True)["label_count"]
        .max()
        .rename("maximum_label_count")
        .reset_index()
    )
    tied_winner_counts = (
        label_counts.merge(maximum_counts, on=list(JUDGMENT_KEYS), how="left")
        .loc[lambda frame: frame["label_count"] == frame["maximum_label_count"]]
        .groupby(list(JUDGMENT_KEYS), sort=True, observed=True)
        .size()
        .rename("tied_winner_count")
        .reset_index()
    )

    canonical = (
        winners.loc[:, [*JUDGMENT_KEYS, "label", "relevance_grade", "label_count"]]
        .rename(columns={"label_count": "selected_label_count"})
        .merge(pair_summary, on=list(JUDGMENT_KEYS), validate="one_to_one")
        .merge(observed, on=list(JUDGMENT_KEYS), validate="one_to_one")
        .merge(tied_winner_counts, on=list(JUDGMENT_KEYS), validate="one_to_one")
    )
    canonical["resolution"] = canonical.apply(_resolution_name, axis="columns")
    canonical = canonical.drop(columns="tied_winner_count").sort_values(
        list(JUDGMENT_KEYS), kind="stable", ignore_index=True
    )
    canonical["relevance_grade"] = canonical["relevance_grade"].astype("int8")
    canonical["selected_label_count"] = canonical["selected_label_count"].astype("int32")
    canonical["judgment_count"] = canonical["judgment_count"].astype("int32")
    canonical["distinct_label_count"] = canonical["distinct_label_count"].astype("int8")

    conflict_combinations = (
        canonical.loc[canonical["distinct_label_count"].gt(1), "observed_labels"]
        .value_counts()
        .sort_index()
    )
    resolution_counts = canonical["resolution"].value_counts().sort_index()
    quality = JudgmentQuality(
        original_judgment_count=len(working),
        canonical_judgment_count=len(canonical),
        repeated_rows_beyond_first=len(working) - len(canonical),
        identical_duplicate_rows=len(working) - len(label_counts),
        conflicting_pair_count=int(canonical["distinct_label_count"].gt(1).sum()),
        conflict_label_combinations={
            str(combination): int(count) for combination, count in conflict_combinations.items()
        },
        resolution_counts={
            str(resolution): int(count) for resolution, count in resolution_counts.items()
        },
        conflict_resolution_policy=CONFLICT_RESOLUTION_POLICY,
    )
    return canonical, quality


def write_canonical_judgments(
    labels_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    relevance_mapping: Mapping[str, int],
    timestamp: datetime | None = None,
) -> JudgmentCanonicalizationReport:
    """Canonicalize a processed label table and atomically write data and report files."""

    labels_path = labels_path.resolve()
    output_path = output_path.resolve()
    report_path = report_path.resolve()
    labels = pd.read_parquet(labels_path)
    canonical, quality = canonicalize_judgments(labels, relevance_mapping=relevance_mapping)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_parquet_atomic(canonical, output_path)
    report = JudgmentCanonicalizationReport(
        created_at=(timestamp or datetime.now(UTC)).astimezone(UTC).isoformat(),
        source_labels_sha256=sha256_file(labels_path),
        original_judgment_count=int(quality["original_judgment_count"]),
        canonical_judgment_count=int(quality["canonical_judgment_count"]),
        repeated_rows_beyond_first=int(quality["repeated_rows_beyond_first"]),
        identical_duplicate_rows=int(quality["identical_duplicate_rows"]),
        conflicting_pair_count=int(quality["conflicting_pair_count"]),
        conflict_label_combinations=quality["conflict_label_combinations"],
        resolution_counts=quality["resolution_counts"],
        conflict_resolution_policy=str(quality["conflict_resolution_policy"]),
    )
    _write_json_atomic(report_path, report)
    return report


def _resolution_name(row: pd.Series[Any]) -> str:
    distinct_count = int(row["distinct_label_count"])
    judgment_count = int(row["judgment_count"])
    tied_winner_count = int(row["tied_winner_count"])
    if distinct_count == 1:
        return "single" if judgment_count == 1 else "identical_duplicate_collapse"
    return "tie_highest_relevance" if tied_winner_count > 1 else "majority_vote"


def _require_columns(frame: DataFrame, required: set[str]) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"judgments are missing required columns: {missing}")


def _write_parquet_atomic(frame: DataFrame, path: Path) -> None:
    temporary_path = path.with_name(f"{path.name}.part")
    try:
        frame.to_parquet(temporary_path, engine="pyarrow", index=False)
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _write_json_atomic(path: Path, payload: JudgmentCanonicalizationReport) -> None:
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
    parser.add_argument("--labels", type=Path, help="Processed labels Parquet input.")
    parser.add_argument("--output", type=Path, help="Canonical judgments Parquet output.")
    parser.add_argument("--report", type=Path, help="Canonicalization report JSON output.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run canonical judgment generation from the command line."""

    arguments = _build_parser().parse_args(argv)
    settings = load_settings()
    report = write_canonical_judgments(
        labels_path=arguments.labels or settings.paths.processed_data / "labels.parquet",
        output_path=arguments.output
        or settings.paths.processed_data / "evaluation_judgments.parquet",
        report_path=arguments.report or settings.paths.reports / "judgment_canonicalization.json",
        relevance_mapping=settings.relevance_mapping.model_dump(by_alias=True),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
