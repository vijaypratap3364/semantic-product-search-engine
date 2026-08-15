"""Create deterministic, query-disjoint train, validation, and test splits."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
from pandas import DataFrame

from product_search.config import load_settings
from product_search.data.download import sha256_file

SPLIT_NAMES = ("train", "validation", "test")


class QuerySplitManifest(TypedDict):
    """Serializable metadata and memberships for query-level partitions."""

    schema_version: int
    seed: int
    requested_proportions: dict[str, float]
    actual_proportions: dict[str, float]
    counts: dict[str, int]
    source_hashes: dict[str, str]
    query_ids: dict[str, list[str]]


def split_query_ids(
    query_ids: Sequence[str],
    *,
    proportions: Mapping[str, float],
    seed: int,
) -> dict[str, list[str]]:
    """Assign sorted unique query IDs using a seeded permutation and largest remainders."""

    _validate_proportions(proportions)
    normalized_ids = [str(query_id).strip() for query_id in query_ids]
    if any(not query_id for query_id in normalized_ids):
        raise ValueError("query IDs must not be blank")
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("query IDs must be unique before splitting")

    ordered_ids = np.asarray(sorted(normalized_ids), dtype=object)
    shuffled = np.random.default_rng(seed).permutation(ordered_ids).tolist()
    counts = _allocate_counts(len(shuffled), proportions)

    memberships: dict[str, list[str]] = {}
    start = 0
    for split_name in SPLIT_NAMES:
        stop = start + counts[split_name]
        memberships[split_name] = sorted(str(query_id) for query_id in shuffled[start:stop])
        start = stop
    return memberships


def build_query_split_manifest(
    queries: DataFrame,
    *,
    proportions: Mapping[str, float],
    seed: int,
    source_hashes: Mapping[str, str],
) -> QuerySplitManifest:
    """Build a complete split manifest from a query table and source hashes."""

    if "query_id" not in queries.columns:
        raise ValueError("queries are missing required column: query_id")
    if queries["query_id"].isna().any():
        raise ValueError("queries.query_id must not be missing")

    query_ids = queries["query_id"].astype(str).tolist()
    memberships = split_query_ids(query_ids, proportions=proportions, seed=seed)
    total = len(query_ids)
    counts = {name: len(memberships[name]) for name in SPLIT_NAMES}
    actual = {name: counts[name] / total if total else 0.0 for name in SPLIT_NAMES}
    return QuerySplitManifest(
        schema_version=1,
        seed=seed,
        requested_proportions={name: float(proportions[name]) for name in SPLIT_NAMES},
        actual_proportions=actual,
        counts=counts,
        source_hashes=dict(sorted(source_hashes.items())),
        query_ids=memberships,
    )


def write_query_splits(
    queries_path: Path,
    judgments_path: Path,
    output_path: Path,
    *,
    proportions: Mapping[str, float],
    seed: int,
) -> QuerySplitManifest:
    """Build and atomically write query splits tied to hashed source tables."""

    queries_path = queries_path.resolve()
    judgments_path = judgments_path.resolve()
    output_path = output_path.resolve()
    queries = pd.read_parquet(queries_path)
    judgments = pd.read_parquet(judgments_path, columns=["query_id"])

    unknown_query_ids = set(judgments["query_id"].astype(str)) - set(
        queries["query_id"].astype(str)
    )
    if unknown_query_ids:
        preview = sorted(unknown_query_ids)[:5]
        raise ValueError(f"canonical judgments reference unknown query IDs: {preview}")

    manifest = build_query_split_manifest(
        queries,
        proportions=proportions,
        seed=seed,
        source_hashes={
            queries_path.name: sha256_file(queries_path),
            judgments_path.name: sha256_file(judgments_path),
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_path, manifest)
    return manifest


def _allocate_counts(total: int, proportions: Mapping[str, float]) -> dict[str, int]:
    exact_counts = {name: total * float(proportions[name]) for name in SPLIT_NAMES}
    counts = {name: math.floor(exact_counts[name]) for name in SPLIT_NAMES}
    remainder = total - sum(counts.values())
    priority = sorted(
        SPLIT_NAMES,
        key=lambda name: (-(exact_counts[name] - counts[name]), SPLIT_NAMES.index(name)),
    )
    for name in priority[:remainder]:
        counts[name] += 1
    return counts


def _validate_proportions(proportions: Mapping[str, float]) -> None:
    if set(proportions) != set(SPLIT_NAMES):
        raise ValueError(f"split proportions must have exactly these keys: {list(SPLIT_NAMES)}")
    values = [float(proportions[name]) for name in SPLIT_NAMES]
    if any(value <= 0.0 or value >= 1.0 for value in values):
        raise ValueError("each split proportion must be between zero and one")
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("split proportions must sum to 1.0")


def _write_json_atomic(path: Path, payload: QuerySplitManifest) -> None:
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
    parser.add_argument("--queries", type=Path, help="Processed queries Parquet input.")
    parser.add_argument("--judgments", type=Path, help="Canonical judgments Parquet input.")
    parser.add_argument("--output", type=Path, help="Query split JSON output.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run deterministic query splitting from the command line."""

    arguments = _build_parser().parse_args(argv)
    settings = load_settings()
    manifest = write_query_splits(
        queries_path=arguments.queries or settings.paths.processed_data / "queries.parquet",
        judgments_path=arguments.judgments
        or settings.paths.processed_data / "evaluation_judgments.parquet",
        output_path=arguments.output or settings.paths.processed_data / "query_splits.json",
        proportions=settings.splits.model_dump(),
        seed=settings.random_seed,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
