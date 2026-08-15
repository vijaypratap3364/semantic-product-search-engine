"""Tests for auditable evaluation-judgment canonicalization."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from pandas import DataFrame

from product_search.evaluation.judgments import (
    CONFLICT_RESOLUTION_POLICY,
    canonicalize_judgments,
    write_canonical_judgments,
)

RELEVANCE_MAPPING = {"Exact": 2, "Partial": 1, "Irrelevant": 0}


def _labels() -> DataFrame:
    return DataFrame(
        [
            ("q1", "p1", "Exact"),
            ("q1", "p1", "Exact"),
            ("q1", "p2", "Partial"),
            ("q2", "p3", "Partial"),
            ("q2", "p3", "Irrelevant"),
            ("q3", "p4", "Exact"),
            ("q3", "p4", "Partial"),
            ("q3", "p4", "Partial"),
        ],
        columns=["query_id", "product_id", "label"],
    )


def test_identical_duplicates_are_collapsed_and_counted() -> None:
    canonical, quality = canonicalize_judgments(_labels(), relevance_mapping=RELEVANCE_MAPPING)

    row = canonical.set_index(["query_id", "product_id"]).loc[("q1", "p1")]
    assert row["label"] == "Exact"
    assert row["judgment_count"] == 2
    assert row["distinct_label_count"] == 1
    assert row["resolution"] == "identical_duplicate_collapse"
    # This includes the repeated Partial vote inside q3's conflicting group.
    assert quality["identical_duplicate_rows"] == 2


def test_conflicts_use_majority_then_highest_relevance_for_ties() -> None:
    canonical, quality = canonicalize_judgments(_labels(), relevance_mapping=RELEVANCE_MAPPING)
    indexed = canonical.set_index(["query_id", "product_id"])

    tied = indexed.loc[("q2", "p3")]
    assert tied["label"] == "Partial"
    assert tied["observed_labels"] == "Partial|Irrelevant"
    assert tied["resolution"] == "tie_highest_relevance"

    majority = indexed.loc[("q3", "p4")]
    assert majority["label"] == "Partial"
    assert majority["selected_label_count"] == 2
    assert majority["resolution"] == "majority_vote"
    assert quality["conflicting_pair_count"] == 2
    assert quality["conflict_label_combinations"] == {
        "Exact|Partial": 1,
        "Partial|Irrelevant": 1,
    }
    assert quality["conflict_resolution_policy"] == CONFLICT_RESOLUTION_POLICY


def test_canonicalization_is_deterministic_and_has_one_row_per_pair() -> None:
    labels = _labels()
    first, _ = canonicalize_judgments(labels, relevance_mapping=RELEVANCE_MAPPING)
    shuffled, _ = canonicalize_judgments(
        labels.sample(frac=1.0, random_state=97),
        relevance_mapping=RELEVANCE_MAPPING,
    )

    pd.testing.assert_frame_equal(first, shuffled)
    assert not first.duplicated(["query_id", "product_id"]).any()
    assert len(first) == labels[["query_id", "product_id"]].drop_duplicates().shape[0]


def test_canonical_output_is_separate_and_records_counts(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.parquet"
    output_path = tmp_path / "evaluation_judgments.parquet"
    report_path = tmp_path / "report.json"
    timestamp = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)
    source = _labels()
    source.to_parquet(labels_path, index=False)

    report = write_canonical_judgments(
        labels_path,
        output_path,
        report_path,
        relevance_mapping=RELEVANCE_MAPPING,
        timestamp=timestamp,
    )

    assert pd.read_parquet(labels_path).equals(source)
    assert len(pd.read_parquet(output_path)) == 4
    assert report["original_judgment_count"] == 8
    assert report["canonical_judgment_count"] == 4
    assert report["repeated_rows_beyond_first"] == 4
    assert report["created_at"] == timestamp.isoformat()
    assert len(report["source_labels_sha256"]) == 64
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        (DataFrame({"query_id": ["q1"]}), "missing required columns"),
        (
            DataFrame({"query_id": ["q1"], "product_id": ["p1"], "label": ["Unknown"]}),
            "absent from relevance mapping",
        ),
        (
            DataFrame({"query_id": ["q1"], "product_id": [None], "label": ["Exact"]}),
            "must not be missing",
        ),
    ],
)
def test_invalid_judgments_fail_clearly(labels: DataFrame, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        canonicalize_judgments(labels, relevance_mapping=RELEVANCE_MAPPING)
