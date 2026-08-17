"""Tests for bounded deterministic performance measurements."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pandas import DataFrame

from product_search.evaluation.performance import (
    BenchmarkOperations,
    collect_artifact_sizes,
    current_process_rss_bytes,
    latency_payload,
    run_performance_benchmark,
    select_representative_queries,
    summarize_latencies,
)


class StepClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.001
        return self.value


def test_representative_query_sample_is_stratified_and_deterministic() -> None:
    queries = DataFrame(
        {
            "query_id": [str(index) for index in range(8)],
            "query": [
                "lamp",
                "desk",
                "blue rug",
                "round table",
                "small black lamp",
                "modern wood coffee table",
                "very small blue outdoor area rug",
                "large modern black metal office desk",
            ],
        },
        index=range(10, 18),
    )

    first = select_representative_queries(queries, sample_size=4, seed=42)
    second = select_representative_queries(queries, sample_size=4, seed=42)

    assert first.equals(second)
    assert len(first) == 4
    assert first["query_id"].is_unique
    assert set(first["query_length_bucket"]).issubset(
        {"one_token", "two_tokens", "three_to_four_tokens", "five_or_more_tokens"}
    )


@pytest.mark.parametrize(
    ("queries", "sample_size", "message"),
    [
        (DataFrame({"query_id": ["1"]}), 1, "missing"),
        (DataFrame({"query_id": ["1"], "query": ["lamp"]}), 0, "positive"),
        (DataFrame({"query_id": ["1"], "query": ["lamp"]}), 2, "population"),
        (
            DataFrame({"query_id": ["1", "1"], "query": ["lamp", "desk"]}),
            1,
            "unique",
        ),
    ],
)
def test_invalid_query_samples_are_rejected(
    queries: DataFrame,
    sample_size: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        select_representative_queries(queries, sample_size=sample_size, seed=42)


def test_latency_summary_reports_p99_only_with_supported_sample_size() -> None:
    below_threshold = summarize_latencies([float(value) for value in range(1, 100)])
    supported = summarize_latencies([float(value) for value in range(1, 101)])

    assert below_threshold.sample_count == 99
    assert below_threshold.p50 == pytest.approx(50.0)
    assert below_threshold.p95 == pytest.approx(94.1)
    assert below_threshold.p99 is None
    assert supported.sample_count == 100
    assert supported.mean == pytest.approx(50.5)
    assert supported.p99 == pytest.approx(99.01)


@pytest.mark.parametrize("values", [[], [-1.0], [float("nan")]])
def test_latency_summary_rejects_invalid_observations(values: list[float]) -> None:
    with pytest.raises(ValueError):
        summarize_latencies(values)


def test_benchmark_measures_each_boundary_after_warmup() -> None:
    calls: list[tuple[str, str, int]] = []

    def operation(name: str):  # type: ignore[no-untyped-def]
        def call(query: str, top_k: int) -> object:
            calls.append((name, query, top_k))
            return f"{name}:{query}"

        return call

    def embed(query: str) -> object:
        calls.append(("embed", query, 0))
        return f"encoded:{query}"

    def rerank(query: str, candidates: object, top_k: int) -> object:
        assert candidates == f"prepare:{query}"
        calls.append(("rerank", query, top_k))
        return candidates

    operations = BenchmarkOperations(
        lexical_search=operation("lexical"),
        semantic_embed=embed,
        semantic_retrieve=lambda encoded, top_k: calls.append(("retrieve", str(encoded), top_k)),
        hybrid_search=operation("hybrid"),
        prepare_rerank_candidates=operation("prepare"),
        rerank=rerank,
        api_search=operation("api"),
    )

    run = run_performance_benchmark(
        ["lamp", "table"],
        operations,
        top_k=10,
        repeats=2,
        warmup_query_count=1,
        clock=StepClock(),
        memory_probe=lambda: 123,
    )

    assert set(run.latencies_ms) == {
        "lexical_query",
        "semantic_embedding",
        "semantic_retrieval",
        "hybrid_search",
        "reranking_stage",
        "api_end_to_end",
    }
    assert all(summary.sample_count == 4 for summary in run.latencies_ms.values())
    assert all(summary.p50 == pytest.approx(1.0) for summary in run.latencies_ms.values())
    assert run.observed_rss_bytes == (123, 123, 123, 123)
    assert latency_payload(run)["lexical_query"]["p99"] is None
    assert calls.count(("lexical", "lamp", 10)) == 3


@pytest.mark.parametrize(
    ("queries", "top_k", "repeats", "warmups", "message"),
    [
        ([], 10, 1, 0, "non-empty"),
        (["lamp"], 0, 1, 0, "top_k"),
        (["lamp"], 10, 0, 0, "repeats"),
        (["lamp"], 10, 1, -1, "warmup"),
        (["lamp"] * 21, 10, 10, 0, "capped"),
    ],
)
def test_benchmark_rejects_unbounded_or_invalid_workloads(
    queries: list[str],
    top_k: int,
    repeats: int,
    warmups: int,
    message: str,
) -> None:
    def unused(*args: object) -> None:
        del args

    operations = BenchmarkOperations(
        lexical_search=unused,
        semantic_embed=unused,
        semantic_retrieve=unused,
        hybrid_search=unused,
        prepare_rerank_candidates=unused,
        rerank=unused,
        api_search=unused,
    )
    with pytest.raises(ValueError, match=message):
        run_performance_benchmark(
            queries,
            operations,
            top_k=top_k,
            repeats=repeats,
            warmup_query_count=warmups,
        )


def test_artifact_sizes_are_deterministic_and_missing_directories_fail(tmp_path: Path) -> None:
    index = tmp_path / "index"
    index.mkdir()
    (index / "a.bin").write_bytes(b"123")
    nested = index / "nested"
    nested.mkdir()
    (nested / "b.bin").write_bytes(b"12345")

    sizes = collect_artifact_sizes({"index": index})

    assert sizes["index"] == {
        "file_count": 2,
        "total_bytes": 8,
        "files": {"a.bin": 3, "nested/b.bin": 5},
    }
    with pytest.raises(FileNotFoundError, match="missing"):
        collect_artifact_sizes({"missing": tmp_path / "missing"})


def test_process_memory_probe_is_supported_or_explicitly_unavailable() -> None:
    value = current_process_rss_bytes()

    if os.name == "nt":
        assert value is not None
        assert value > 0
    else:
        assert value is None
