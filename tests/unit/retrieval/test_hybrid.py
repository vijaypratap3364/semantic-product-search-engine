"""Tests for normalized weighted fusion and reciprocal-rank fusion."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from product_search.retrieval import hybrid as hybrid_module
from product_search.retrieval.base import SearchResult
from product_search.retrieval.hybrid import (
    HybridSearchEngine,
    fuse_rankings,
    min_max_normalize,
)


class StaticSearchEngine:
    """Deterministic tiny engine that can omit modality scores."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.full_top_k_calls: list[int] = []
        self.candidate_queries: list[str] = []

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        self.full_top_k_calls.append(top_k)
        return self._rank(self.scores, top_k)

    def search_candidates(
        self,
        query: str,
        candidate_product_ids: Sequence[str],
        top_k: int,
    ) -> list[SearchResult]:
        self.candidate_queries.append(query)
        allowed = set(candidate_product_ids)
        return self._rank(
            {
                product_id: score
                for product_id, score in self.scores.items()
                if product_id in allowed
            },
            top_k,
        )

    @staticmethod
    def _rank(scores: dict[str, float], top_k: int) -> list[SearchResult]:
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        return [
            SearchResult(product_id=product_id, rank=rank, score=score)
            for rank, (product_id, score) in enumerate(ordered, start=1)
        ]


def _result(product_id: str, rank: int, score: float) -> SearchResult:
    return SearchResult(product_id=product_id, rank=rank, score=score)


def test_min_max_normalization_handles_ranges_constants_and_empty_scores() -> None:
    assert min_max_normalize({"p1": 2.0, "p2": 4.0, "p3": 3.0}) == {
        "p1": 0.0,
        "p2": 1.0,
        "p3": 0.5,
    }
    assert min_max_normalize({"p1": 0.4, "p2": 0.4}) == {"p1": 1.0, "p2": 1.0}
    assert min_max_normalize({}) == {}
    with pytest.raises(ValueError, match="finite"):
        min_max_normalize({"p1": float("nan")})


def test_weighted_fusion_combines_normalized_scores_and_components() -> None:
    results = fuse_rankings(
        [_result("p1", 1, 10.0), _result("p2", 2, 0.0)],
        [_result("p2", 1, 1.0), _result("p1", 2, 0.0)],
        candidate_product_ids=["p1", "p2"],
        top_k=2,
        strategy="weighted_normalized",
        semantic_weight=0.75,
    )

    assert [result.product_id for result in results] == ["p2", "p1"]
    assert results[0].score == pytest.approx(0.75)
    assert results[1].score == pytest.approx(0.25)
    assert results[0].score_components == {
        "lexical_raw": 0.0,
        "semantic_raw": 1.0,
        "lexical_normalized": 0.0,
        "semantic_normalized": 1.0,
        "lexical_rrf": pytest.approx(1.0 / 62.0),
        "semantic_rrf": pytest.approx(1.0 / 61.0),
        "lexical_present": 1.0,
        "semantic_present": 1.0,
        "hybrid": pytest.approx(0.75),
    }


def test_rrf_uses_ranks_and_zero_contribution_for_missing_scores() -> None:
    results = fuse_rankings(
        [_result("p1", 1, 0.9), _result("p2", 2, 0.8)],
        [_result("p2", 1, 0.7)],
        candidate_product_ids=["p1", "p2", "p3"],
        top_k=3,
        strategy="rrf",
        rrf_k=60,
    )

    assert [result.product_id for result in results] == ["p2", "p1", "p3"]
    assert results[0].score == pytest.approx(0.5 / 62.0 + 0.5 / 61.0)
    assert results[1].score == pytest.approx(0.5 / 61.0)
    assert results[2].score == 0.0
    assert results[2].score_components["lexical_present"] == 0.0  # type: ignore[index]
    assert results[2].score_components["semantic_present"] == 0.0  # type: ignore[index]


def test_fusion_ties_are_broken_by_product_id_not_input_rank() -> None:
    results = fuse_rankings(
        [_result("p2", 1, 0.5), _result("p1", 2, 0.5)],
        [],
        candidate_product_ids=["p2", "p1"],
        top_k=2,
        strategy="weighted_normalized",
        semantic_weight=0.0,
    )

    assert [result.product_id for result in results] == ["p1", "p2"]
    assert [result.rank for result in results] == [1, 2]


def test_hybrid_candidate_search_handles_missing_scores_and_is_deterministic() -> None:
    lexical = StaticSearchEngine({"p1": 0.9, "p2": 0.4})
    semantic = StaticSearchEngine({"p2": 0.8})
    engine = HybridSearchEngine(
        lexical,
        semantic,
        strategy="weighted_normalized",
        semantic_weight=0.6,
    )

    first = engine.search_candidates("query", ["p3", "p2", "p1"], top_k=3)
    second = engine.search_candidates("query", ["p3", "p2", "p1"], top_k=3)

    assert first == second
    assert [result.product_id for result in first] == ["p2", "p1", "p3"]
    assert first[2].score == 0.0
    assert first[2].score_components["lexical_present"] == 0.0  # type: ignore[index]
    assert first[2].score_components["semantic_present"] == 0.0  # type: ignore[index]
    assert lexical.candidate_queries == ["query", "query"]
    assert semantic.candidate_queries == ["query", "query"]


def test_hybrid_full_search_uses_candidate_union_depth_and_top_k() -> None:
    lexical = StaticSearchEngine({"p1": 0.9, "p3": 0.2})
    semantic = StaticSearchEngine({"p2": 0.8, "p3": 0.3})
    engine = HybridSearchEngine(lexical, semantic, candidate_depth=5)

    results = engine.search("query", top_k=2)

    assert len(results) == 2
    assert set(result.product_id for result in results) <= {"p1", "p2", "p3"}
    assert lexical.full_top_k_calls == [5]
    assert semantic.full_top_k_calls == [5]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"strategy": "unknown"}, "unsupported fusion strategy"),
        ({"semantic_weight": -0.1}, "between 0.0 and 1.0"),
        ({"candidate_depth": 0}, "positive integer"),
        ({"rrf_k": 0}, "positive integer"),
    ],
)
def test_hybrid_rejects_invalid_settings(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        HybridSearchEngine(StaticSearchEngine({}), StaticSearchEngine({}), **kwargs)  # type: ignore[arg-type]


def test_hybrid_rejects_duplicate_candidates_and_invalid_component_results() -> None:
    engine = HybridSearchEngine(StaticSearchEngine({}), StaticSearchEngine({}))
    with pytest.raises(ValueError, match="must be unique"):
        engine.search_candidates("query", ["p1", "p1"], top_k=2)
    with pytest.raises(ValueError, match="unexpected products"):
        fuse_rankings(
            [_result("outside", 1, 1.0)],
            [],
            candidate_product_ids=["p1"],
            top_k=1,
            strategy="weighted_normalized",
        )

    with pytest.raises(ValueError, match="duplicate product IDs"):
        fuse_rankings(
            [_result("p1", 1, 1.0), _result("p1", 2, 0.5)],
            [],
            candidate_product_ids=["p1"],
            top_k=1,
            strategy="weighted_normalized",
        )
    with pytest.raises(ValueError, match="contiguous from one"):
        fuse_rankings(
            [_result("p1", 2, 1.0)],
            [],
            candidate_product_ids=["p1"],
            top_k=1,
            strategy="weighted_normalized",
        )


def test_hybrid_empty_candidates_and_invalid_top_k() -> None:
    engine = HybridSearchEngine(StaticSearchEngine({}), StaticSearchEngine({}))
    assert engine.search_candidates("query", [], top_k=2) == []
    assert (
        fuse_rankings(
            [],
            [],
            candidate_product_ids=[],
            top_k=1,
            strategy="weighted_normalized",
        )
        == []
    )
    with pytest.raises(ValueError, match="positive integer"):
        engine.search("query", top_k=0)


def test_hybrid_cli_loads_engines_and_prints_components(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    lexical = StaticSearchEngine({"p1": 0.9})
    semantic = StaticSearchEngine({"p1": 0.8})
    provider = object()

    class LexicalLoader:
        @classmethod
        def from_index_dir(cls, index_dir: Path) -> StaticSearchEngine:
            return lexical

    class SemanticLoader:
        @classmethod
        def from_index_dir(
            cls,
            index_dir: Path,
            *,
            provider: object,
            expected_dimension: int,
        ) -> StaticSearchEngine:
            return semantic

    monkeypatch.setattr(hybrid_module, "FastEmbedProvider", lambda *args, **kwargs: provider)
    monkeypatch.setattr(hybrid_module, "LexicalSearchEngine", LexicalLoader)
    monkeypatch.setattr(hybrid_module, "SemanticSearchEngine", SemanticLoader)

    exit_code = hybrid_module.main(
        [
            "query",
            "--top-k",
            "1",
            "--strategy",
            "rrf",
            "--semantic-weight",
            "0.7",
            "--candidate-depth",
            "3",
            "--rrf-k",
            "10",
            "--lexical-index-dir",
            str(tmp_path / "tfidf"),
            "--dense-index-dir",
            str(tmp_path / "dense"),
            "--local-files-only",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert '"product_id": "p1"' in output
    assert '"hybrid"' in output
