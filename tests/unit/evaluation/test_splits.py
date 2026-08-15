"""Tests for deterministic, query-disjoint dataset splits."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pandas import DataFrame

from product_search.evaluation.splits import (
    build_query_split_manifest,
    split_query_ids,
    write_query_splits,
)

PROPORTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}


def test_query_split_is_deterministic_and_input_order_independent() -> None:
    query_ids = [f"q{index:03d}" for index in range(20)]

    first = split_query_ids(query_ids, proportions=PROPORTIONS, seed=42)
    second = split_query_ids(list(reversed(query_ids)), proportions=PROPORTIONS, seed=42)

    assert first == second
    assert {name: len(values) for name, values in first.items()} == {
        "train": 14,
        "validation": 3,
        "test": 3,
    }


def test_query_split_has_complete_coverage_without_leakage() -> None:
    query_ids = [f"q{index}" for index in range(7)]
    memberships = split_query_ids(query_ids, proportions=PROPORTIONS, seed=42)
    split_sets = [set(memberships[name]) for name in ("train", "validation", "test")]

    assert split_sets[0].isdisjoint(split_sets[1])
    assert split_sets[0].isdisjoint(split_sets[2])
    assert split_sets[1].isdisjoint(split_sets[2])
    assert set().union(*split_sets) == set(query_ids)
    assert sum(map(len, split_sets)) == len(query_ids)


def test_manifest_records_counts_proportions_seed_and_hashes() -> None:
    queries = DataFrame({"query_id": [f"q{index}" for index in range(10)]})

    manifest = build_query_split_manifest(
        queries,
        proportions=PROPORTIONS,
        seed=42,
        source_hashes={"queries.parquet": "abc", "judgments.parquet": "def"},
    )

    assert manifest["seed"] == 42
    assert manifest["counts"] == {"train": 7, "validation": 2, "test": 1}
    assert manifest["actual_proportions"] == {"train": 0.7, "validation": 0.2, "test": 0.1}
    assert manifest["source_hashes"] == {
        "judgments.parquet": "def",
        "queries.parquet": "abc",
    }


def test_write_query_splits_hashes_inputs_and_checks_foreign_keys(tmp_path: Path) -> None:
    queries_path = tmp_path / "queries.parquet"
    judgments_path = tmp_path / "evaluation_judgments.parquet"
    output_path = tmp_path / "query_splits.json"
    DataFrame({"query_id": [f"q{index}" for index in range(10)]}).to_parquet(
        queries_path, index=False
    )
    DataFrame({"query_id": ["q1", "q2"], "product_id": ["p1", "p2"]}).to_parquet(
        judgments_path, index=False
    )

    manifest = write_query_splits(
        queries_path,
        judgments_path,
        output_path,
        proportions=PROPORTIONS,
        seed=42,
    )

    assert all(len(digest) == 64 for digest in manifest["source_hashes"].values())
    assert json.loads(output_path.read_text(encoding="utf-8")) == manifest

    DataFrame({"query_id": ["unknown"]}).to_parquet(judgments_path, index=False)
    with pytest.raises(ValueError, match="unknown query IDs"):
        write_query_splits(
            queries_path,
            judgments_path,
            output_path,
            proportions=PROPORTIONS,
            seed=42,
        )


@pytest.mark.parametrize(
    ("query_ids", "proportions", "message"),
    [
        (["q1", "q1"], PROPORTIONS, "must be unique"),
        (["q1", " "], PROPORTIONS, "must not be blank"),
        (["q1"], {"train": 0.8, "test": 0.2}, "exactly these keys"),
        (["q1"], {"train": 1.0, "validation": 0.0, "test": 0.0}, "between zero"),
        (["q1"], {"train": 0.5, "validation": 0.2, "test": 0.2}, "sum to 1.0"),
    ],
)
def test_invalid_split_inputs_fail_clearly(
    query_ids: list[str], proportions: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        split_query_ids(query_ids, proportions=proportions, seed=42)


def test_manifest_rejects_missing_query_column_and_values() -> None:
    with pytest.raises(ValueError, match="missing required column"):
        build_query_split_manifest(
            DataFrame({"query": ["lamp"]}),
            proportions=PROPORTIONS,
            seed=42,
            source_hashes={},
        )
    with pytest.raises(ValueError, match="must not be missing"):
        build_query_split_manifest(
            DataFrame({"query_id": [None]}),
            proportions=PROPORTIONS,
            seed=42,
            source_hashes={},
        )
