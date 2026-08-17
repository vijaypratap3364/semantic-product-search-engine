"""Reusable, bounded performance measurements for the local search stack."""

from __future__ import annotations

import ctypes
import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from ctypes import wintypes
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from pandas import DataFrame


@dataclass(frozen=True, slots=True)
class LatencySummary:
    """Observed warm-process latency distribution in milliseconds."""

    sample_count: int
    mean: float
    p50: float
    p95: float
    p99: float | None


@dataclass(frozen=True, slots=True)
class BenchmarkOperations:
    """Injected component boundaries used by the benchmark loop."""

    lexical_search: Callable[[str, int], object]
    semantic_embed: Callable[[str], object]
    semantic_retrieve: Callable[[object, int], object]
    hybrid_search: Callable[[str, int], object]
    prepare_rerank_candidates: Callable[[str, int], object]
    rerank: Callable[[str, object, int], object]
    api_search: Callable[[str, int], object]


@dataclass(frozen=True, slots=True)
class PerformanceRun:
    """Component timings and practical process-memory observations."""

    latencies_ms: dict[str, LatencySummary]
    observed_rss_bytes: tuple[int, ...]


def select_representative_queries(
    queries: DataFrame,
    *,
    sample_size: int,
    seed: int,
) -> DataFrame:
    """Select a deterministic sample stratified by simple query-length buckets."""

    missing = {"query_id", "query"} - set(queries.columns)
    if missing:
        raise ValueError(f"queries are missing required columns: {sorted(missing)}")
    if queries[["query_id", "query"]].isna().any(axis=None):
        raise ValueError("query IDs and text must not be missing")
    if queries["query_id"].astype(str).duplicated().any():
        raise ValueError("query IDs must be unique")
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size <= 0:
        raise ValueError("sample_size must be a positive integer")
    if sample_size > len(queries):
        raise ValueError("sample_size must not exceed the query population")

    frame = queries.reset_index(drop=True).copy()
    frame["query_id"] = frame["query_id"].astype(str)
    frame["query"] = frame["query"].astype(str)
    frame["query_length_tokens"] = frame["query"].str.split().str.len().astype(int)
    frame["query_length_bucket"] = frame["query_length_tokens"].map(_query_length_bucket)
    bucket_sizes = frame["query_length_bucket"].value_counts().sort_index().to_dict()
    quotas = _proportional_quotas(
        {str(bucket): int(count) for bucket, count in bucket_sizes.items()},
        sample_size,
    )
    rng = np.random.default_rng(seed)
    selected_positions: list[int] = []
    for bucket in sorted(quotas):
        positions = frame.index[frame["query_length_bucket"] == bucket].to_numpy(dtype=np.int64)
        chosen = rng.choice(positions, size=quotas[bucket], replace=False)
        selected_positions.extend(int(position) for position in chosen)
    rng.shuffle(selected_positions)
    return frame.loc[selected_positions].reset_index(drop=True)


def summarize_latencies(values: Sequence[float]) -> LatencySummary:
    """Return p50/p95 and p99 only when at least 100 observations support it."""

    if not values:
        raise ValueError("at least one latency observation is required")
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all() or (array < 0.0).any():
        raise ValueError("latencies must be finite and non-negative")
    return LatencySummary(
        sample_count=int(array.size),
        mean=float(np.mean(array)),
        p50=float(np.percentile(array, 50)),
        p95=float(np.percentile(array, 95)),
        p99=float(np.percentile(array, 99)) if array.size >= 100 else None,
    )


def run_performance_benchmark(
    queries: Sequence[str],
    operations: BenchmarkOperations,
    *,
    top_k: int,
    repeats: int,
    warmup_query_count: int,
    clock: Callable[[], float] = time.perf_counter,
    memory_probe: Callable[[], int | None] = lambda: current_process_rss_bytes(),
) -> PerformanceRun:
    """Measure fixed component boundaries with a bounded warm-process workload."""

    if not queries or any(not query.strip() for query in queries):
        raise ValueError("benchmark queries must be non-empty")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("repeats must be a positive integer")
    if warmup_query_count < 0:
        raise ValueError("warmup_query_count must not be negative")
    if len(queries) * repeats > 200:
        raise ValueError("benchmark workload is capped at 200 timed query repetitions")

    for query in queries[:warmup_query_count]:
        _exercise_operations(query, operations, top_k)

    timings: dict[str, list[float]] = {
        "lexical_query": [],
        "semantic_embedding": [],
        "semantic_retrieval": [],
        "hybrid_search": [],
        "reranking_stage": [],
        "api_end_to_end": [],
    }
    rss_observations: list[int] = []
    for _ in range(repeats):
        for query in queries:
            _, lexical_ms = _timed(operations.lexical_search, query, top_k, clock=clock)
            encoded, embedding_ms = _timed(operations.semantic_embed, query, clock=clock)
            _, retrieval_ms = _timed(
                operations.semantic_retrieve,
                encoded,
                top_k,
                clock=clock,
            )
            _, hybrid_ms = _timed(operations.hybrid_search, query, top_k, clock=clock)
            candidates = operations.prepare_rerank_candidates(query, top_k)
            _, reranking_ms = _timed(
                operations.rerank,
                query,
                candidates,
                top_k,
                clock=clock,
            )
            _, api_ms = _timed(operations.api_search, query, top_k, clock=clock)
            timings["lexical_query"].append(lexical_ms)
            timings["semantic_embedding"].append(embedding_ms)
            timings["semantic_retrieval"].append(retrieval_ms)
            timings["hybrid_search"].append(hybrid_ms)
            timings["reranking_stage"].append(reranking_ms)
            timings["api_end_to_end"].append(api_ms)
            rss = memory_probe()
            if rss is not None:
                rss_observations.append(rss)
    return PerformanceRun(
        latencies_ms={name: summarize_latencies(values) for name, values in timings.items()},
        observed_rss_bytes=tuple(rss_observations),
    )


def latency_payload(run: PerformanceRun) -> dict[str, dict[str, float | int | None]]:
    """Return JSON-compatible latency summaries."""

    return {name: asdict(summary) for name, summary in run.latencies_ms.items()}


def collect_artifact_sizes(families: Mapping[str, Path]) -> dict[str, object]:
    """Record deterministic per-file and total disk usage for artifact families."""

    payload: dict[str, object] = {}
    for family, root in sorted(families.items()):
        resolved = root.resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"artifact directory does not exist ({family}): {resolved}")
        files = {
            path.relative_to(resolved).as_posix(): path.stat().st_size
            for path in sorted(resolved.rglob("*"))
            if path.is_file()
        }
        payload[family] = {
            "file_count": len(files),
            "total_bytes": sum(files.values()),
            "files": files,
        }
    return payload


if sys.platform == "win32":

    class _ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    def _platform_process_rss_bytes() -> int | None:
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        process = kernel32.GetCurrentProcess()
        succeeded = psapi.GetProcessMemoryInfo(
            process,
            ctypes.byref(counters),
            counters.cb,
        )
        return int(counters.WorkingSetSize) if succeeded else None

else:

    def _platform_process_rss_bytes() -> int | None:
        return None


def current_process_rss_bytes() -> int | None:
    """Return the Windows working set, or explicitly report unavailable elsewhere."""

    return _platform_process_rss_bytes()


def _exercise_operations(
    query: str,
    operations: BenchmarkOperations,
    top_k: int,
) -> None:
    operations.lexical_search(query, top_k)
    encoded = operations.semantic_embed(query)
    operations.semantic_retrieve(encoded, top_k)
    operations.hybrid_search(query, top_k)
    candidates = operations.prepare_rerank_candidates(query, top_k)
    operations.rerank(query, candidates, top_k)
    operations.api_search(query, top_k)


def _timed(
    operation: Callable[..., object],
    *args: object,
    clock: Callable[[], float],
) -> tuple[object, float]:
    started_at = clock()
    result = operation(*args)
    return result, max(0.0, (clock() - started_at) * 1000.0)


def _query_length_bucket(length: int) -> str:
    if length <= 1:
        return "one_token"
    if length == 2:
        return "two_tokens"
    if length <= 4:
        return "three_to_four_tokens"
    return "five_or_more_tokens"


def _proportional_quotas(bucket_sizes: Mapping[str, int], sample_size: int) -> dict[str, int]:
    population = sum(bucket_sizes.values())
    raw = {bucket: sample_size * size / population for bucket, size in bucket_sizes.items()}
    quotas = {bucket: min(size, math.floor(raw[bucket])) for bucket, size in bucket_sizes.items()}
    remaining = sample_size - sum(quotas.values())
    order = sorted(
        bucket_sizes,
        key=lambda bucket: (-(raw[bucket] - quotas[bucket]), bucket),
    )
    for bucket in order:
        if remaining == 0:
            break
        if quotas[bucket] < bucket_sizes[bucket]:
            quotas[bucket] += 1
            remaining -= 1
    if remaining:
        raise ValueError("unable to allocate the requested stratified sample")
    return quotas
