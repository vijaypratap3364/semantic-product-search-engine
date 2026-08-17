"""Tests for factual judged-candidate error analysis."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest
from pandas import DataFrame

from product_search.evaluation.error_analysis import (
    QualityRecord,
    categorize_quality_records,
    compute_quality_records,
    error_analysis_payload,
    render_error_analysis_markdown,
    write_error_analysis,
)
from product_search.ranking.reranker import RerankingSearchEngine
from product_search.retrieval.base import SearchResult
from product_search.retrieval.lexical import LexicalSearchEngine
from product_search.retrieval.semantic import SemanticSearchEngine


def _record(
    query_id: str,
    *,
    lexical: float,
    semantic: float,
    hybrid: float,
    reranked: float,
    top_grade: int = 2,
    exact_count: int = 1,
) -> QualityRecord:
    return QualityRecord(
        query_id=query_id,
        query=f"query {query_id}",
        lexical_ndcg_at_10=lexical,
        semantic_ndcg_at_10=semantic,
        hybrid_ndcg_at_10=hybrid,
        reranked_ndcg_at_10=reranked,
        lexical_top_grade=2,
        semantic_top_grade=1,
        hybrid_top_grade=2,
        reranked_top_grade=top_grade,
        reranked_top_product_id="p1",
        reranked_top_product_name=f"Product {query_id}",
        exact_judgment_count=exact_count,
        first_reranked_exact_rank=2 if top_grade == 1 else 1,
    )


def test_error_categories_use_strict_independent_comparisons() -> None:
    records = (
        _record("lexical", lexical=0.9, semantic=0.3, hybrid=0.8, reranked=0.85),
        _record("semantic", lexical=0.2, semantic=0.9, hybrid=0.8, reranked=0.7),
        _record("hybrid", lexical=0.4, semantic=0.5, hybrid=0.9, reranked=0.95),
        _record(
            "partial",
            lexical=0.5,
            semantic=0.5,
            hybrid=0.6,
            reranked=0.4,
            top_grade=1,
        ),
        _record("tie", lexical=0.5, semantic=0.5, hybrid=0.5, reranked=0.5),
    )

    analysis = categorize_quality_records(records, example_limit=2, tail_limit=2)

    assert analysis.query_count == 5
    assert analysis.category_counts == {
        "lexical_better_than_semantic": 1,
        "semantic_better_than_lexical": 2,
        "hybrid_better_than_both": 2,
        "failed_tail_queries": 2,
        "partial_vs_exact_confusion": 1,
        "reranking_helps": 2,
        "reranking_hurts": 2,
    }
    assert analysis.examples["lexical_better_than_semantic"][0].query_id == "lexical"
    assert analysis.examples["failed_tail_queries"][0].query_id == "partial"
    assert error_analysis_payload(analysis)["query_count"] == 5


def test_error_analysis_markdown_contains_only_recorded_fields() -> None:
    analysis = categorize_quality_records(
        [_record("1", lexical=0.9, semantic=0.2, hybrid=0.8, reranked=0.7)],
        example_limit=1,
        tail_limit=1,
    )

    markdown = render_error_analysis_markdown(analysis, split="test")

    assert "Analyzed split: `test`" in markdown
    assert "query 1" in markdown
    assert "Product 1" in markdown
    assert "No qualifying query was observed." in markdown


class StaticCandidateEngine:
    def __init__(self, order: Sequence[str], component: str) -> None:
        self.order = order
        self.component = component

    def search_candidates(
        self,
        query: str,
        candidate_product_ids: Sequence[str],
        top_k: int,
    ) -> list[SearchResult]:
        allowed = set(candidate_product_ids)
        return [
            SearchResult(
                product_id=product_id,
                rank=rank,
                score=float(len(self.order) - rank + 1),
                score_components={self.component: float(len(self.order) - rank + 1)},
            )
            for rank, product_id in enumerate(
                (product_id for product_id in self.order if product_id in allowed), start=1
            )
        ][:top_k]


class StaticReranker:
    candidate_depth = 3

    def rerank(
        self,
        query: str,
        candidates: Sequence[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        return [
            SearchResult(result.product_id, rank, result.score, result.score_components)
            for rank, result in enumerate(reversed(candidates), start=1)
        ][:top_k]


def test_quality_records_are_computed_from_actual_rankings() -> None:
    queries = DataFrame({"query_id": ["q1"], "query": ["lamp"]})
    judgments = DataFrame(
        {
            "query_id": ["q1", "q1", "q1"],
            "product_id": ["p1", "p2", "p3"],
            "relevance_grade": [2, 1, 0],
        }
    )
    lexical = cast(LexicalSearchEngine, StaticCandidateEngine(["p1", "p2", "p3"], "lexical"))
    semantic = cast(
        SemanticSearchEngine,
        StaticCandidateEngine(["p2", "p1", "p3"], "semantic"),
    )
    reranker = cast(RerankingSearchEngine, StaticReranker())

    records = compute_quality_records(
        lexical=lexical,
        semantic=semantic,
        reranker=reranker,
        queries=queries,
        judgments=judgments,
        product_names={"p1": "Exact Lamp", "p2": "Partial Lamp", "p3": "Other"},
        strategy="weighted_normalized",
        semantic_weight=0.5,
        candidate_depth=3,
        rrf_k=60,
    )

    assert len(records) == 1
    assert records[0].lexical_ndcg_at_10 == pytest.approx(1.0)
    assert records[0].semantic_ndcg_at_10 < 1.0
    assert records[0].reranked_top_product_id == "p3"
    assert records[0].reranked_top_product_name == "Other"
    assert records[0].exact_judgment_count == 1


def test_duplicate_judgments_and_invalid_limits_are_rejected() -> None:
    queries = DataFrame({"query_id": ["q1"], "query": ["lamp"]})
    judgments = DataFrame(
        {
            "query_id": ["q1", "q1"],
            "product_id": ["p1", "p1"],
            "relevance_grade": [2, 1],
        }
    )
    engine = cast(LexicalSearchEngine, StaticCandidateEngine(["p1"], "lexical"))
    semantic = cast(SemanticSearchEngine, StaticCandidateEngine(["p1"], "semantic"))
    reranker = cast(RerankingSearchEngine, StaticReranker())

    with pytest.raises(ValueError, match="canonical"):
        compute_quality_records(
            lexical=engine,
            semantic=semantic,
            reranker=reranker,
            queries=queries,
            judgments=judgments,
            product_names={},
            strategy="weighted_normalized",
            semantic_weight=0.5,
            candidate_depth=3,
            rrf_k=60,
        )
    with pytest.raises(ValueError, match="positive"):
        categorize_quality_records([], example_limit=0)


def test_error_analysis_is_written_atomically(tmp_path: Path) -> None:
    analysis = categorize_quality_records(
        [_record("1", lexical=0.9, semantic=0.2, hybrid=0.8, reranked=0.7)],
        example_limit=1,
        tail_limit=1,
    )
    path = tmp_path / "error_analysis.md"

    write_error_analysis(path, analysis, split="test")

    assert path.read_text(encoding="utf-8").startswith("# Search error analysis")
    assert not path.with_name("error_analysis.md.part").exists()
