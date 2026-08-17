"""Tests for performance benchmark orchestration helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from product_search.evaluation.benchmark_performance import (
    _benchmark_operations,
    _counts,
    _read_json,
    _recorded_or_unavailable_build_duration,
    _sample_hash,
    write_benchmark_report,
)
from product_search.ranking.reranker import RerankingSearchEngine
from product_search.retrieval.base import SearchResult
from product_search.retrieval.hybrid import HybridSearchEngine
from product_search.retrieval.lexical import LexicalSearchEngine
from product_search.retrieval.semantic import SemanticSearchEngine


class FakeLexical:
    def search(self, query: str, top_k: int) -> list[SearchResult]:
        return [SearchResult("p1", 1, 1.0, {"lexical": 1.0})]


class FakeSemantic:
    def embed_query(self, query: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)

    def search_embedding(self, vector: np.ndarray, top_k: int) -> list[SearchResult]:
        assert vector.tolist() == [1.0, 0.0]
        return [SearchResult("p1", 1, 1.0, {"semantic": 1.0})]


class FakeHybrid:
    def __init__(self) -> None:
        self.top_ks: list[int] = []

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        self.top_ks.append(top_k)
        return [SearchResult("p1", 1, 1.0, {"hybrid": 1.0})]


class FakeReranker:
    candidate_depth = 100

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        return candidates[:top_k]


def test_benchmark_operations_preserve_component_boundaries() -> None:
    hybrid = FakeHybrid()
    api_calls: list[tuple[str, int]] = []
    operations = _benchmark_operations(
        cast(LexicalSearchEngine, FakeLexical()),
        cast(SemanticSearchEngine, FakeSemantic()),
        cast(HybridSearchEngine, hybrid),
        cast(RerankingSearchEngine, FakeReranker()),
        api_search=lambda query, top_k: api_calls.append((query, top_k)),
    )

    assert operations.lexical_search("lamp", 10)[0].product_id == "p1"  # type: ignore[index]
    encoded = operations.semantic_embed("lamp")
    assert operations.semantic_retrieve(encoded, 10)[0].product_id == "p1"  # type: ignore[index]
    candidates = operations.prepare_rerank_candidates("lamp", 10)
    assert operations.rerank("lamp", candidates, 10) == candidates
    operations.api_search("lamp", 10)

    assert hybrid.top_ks == [100]
    assert api_calls == [("lamp", 10)]


def test_invalid_benchmark_operation_outputs_are_rejected() -> None:
    operations = _benchmark_operations(
        cast(LexicalSearchEngine, FakeLexical()),
        cast(SemanticSearchEngine, FakeSemantic()),
        cast(HybridSearchEngine, FakeHybrid()),
        cast(RerankingSearchEngine, FakeReranker()),
        api_search=lambda query, top_k: None,
    )

    with pytest.raises(ValueError, match="NumPy"):
        operations.semantic_retrieve("not an array", 10)
    with pytest.raises(ValueError, match="invalid results"):
        operations.rerank("lamp", "not candidates", 10)


def test_recorded_and_unavailable_build_durations(tmp_path: Path) -> None:
    recorded = tmp_path / "recorded.json"
    recorded.write_text(json.dumps({"build_duration_seconds": 12.5}), encoding="utf-8")
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"created_at": "now"}), encoding="utf-8")

    assert _recorded_or_unavailable_build_duration(recorded) == {
        "status": "recorded_in_artifact_metadata",
        "duration_seconds": 12.5,
    }
    unavailable = _recorded_or_unavailable_build_duration(legacy)
    assert unavailable["status"] == "unavailable_prior_build_not_instrumented"
    assert "skipped" in str(unavailable["reason"])


def test_report_write_hash_counts_and_json_validation(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.json"
    write_benchmark_report(path, {"schema_version": 1, "latency": 2.5})

    assert json.loads(path.read_text(encoding="utf-8"))["latency"] == 2.5
    assert not path.with_name("benchmark.json.part").exists()
    assert _sample_hash(["1", "2"]) == _sample_hash(["1", "2"])
    assert _sample_hash(["1", "2"]) != _sample_hash(["2", "1"])
    assert _counts(["lamp", "desk lamp", "modern black lamp", "very large blue area rug"]) == {
        "five_or_more_tokens": 1,
        "one_token": 1,
        "three_to_four_tokens": 1,
        "two_tokens": 1,
    }
    assert _read_json(path)["schema_version"] == 1

    array_path = tmp_path / "array.json"
    array_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="object"):
        _read_json(array_path)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _read_json(tmp_path / "missing.json")
