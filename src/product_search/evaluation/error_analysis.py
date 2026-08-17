"""Deterministic, judgment-grounded search error analysis."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from pandas import DataFrame

from product_search.evaluation.metrics import ndcg_at_k
from product_search.ranking.reranker import RerankingSearchEngine
from product_search.retrieval.base import SearchResult
from product_search.retrieval.hybrid import FusionStrategy, fuse_rankings
from product_search.retrieval.lexical import LexicalSearchEngine
from product_search.retrieval.semantic import SemanticSearchEngine


@dataclass(frozen=True, slots=True)
class QualityRecord:
    """One query's controlled judged-candidate comparison."""

    query_id: str
    query: str
    lexical_ndcg_at_10: float
    semantic_ndcg_at_10: float
    hybrid_ndcg_at_10: float
    reranked_ndcg_at_10: float
    lexical_top_grade: int
    semantic_top_grade: int
    hybrid_top_grade: int
    reranked_top_grade: int
    reranked_top_product_id: str | None
    reranked_top_product_name: str | None
    exact_judgment_count: int
    first_reranked_exact_rank: int | None


@dataclass(frozen=True, slots=True)
class ErrorAnalysis:
    """Strict category counts and selected factual examples."""

    query_count: int
    category_counts: dict[str, int]
    examples: dict[str, tuple[QualityRecord, ...]]


def compute_quality_records(
    *,
    lexical: LexicalSearchEngine,
    semantic: SemanticSearchEngine,
    reranker: RerankingSearchEngine,
    queries: DataFrame,
    judgments: DataFrame,
    product_names: Mapping[str, str],
    strategy: FusionStrategy,
    semantic_weight: float,
    candidate_depth: int,
    rrf_k: int,
    top_k: int = 10,
) -> tuple[QualityRecord, ...]:
    """Score each query once per base modality and reuse those scores for fusion/reranking."""

    _validate_inputs(queries, judgments, top_k=top_k)
    judgments_by_query = {
        str(query_id): {
            str(product_id): int(grade)
            for product_id, grade in zip(
                group["product_id"],
                group["relevance_grade"],
                strict=True,
            )
        }
        for query_id, group in judgments.groupby("query_id", sort=True, observed=True)
    }
    query_records = cast(
        list[dict[str, object]],
        queries.loc[:, ["query_id", "query"]].to_dict(orient="records"),
    )
    records: list[QualityRecord] = []
    for query_record in query_records:
        query_id = str(query_record["query_id"])
        query = str(query_record["query"])
        grades = judgments_by_query.get(query_id, {})
        candidate_ids = sorted(grades)
        if not candidate_ids:
            continue
        lexical_all = lexical.search_candidates(query, candidate_ids, len(candidate_ids))
        semantic_all = semantic.search_candidates(query, candidate_ids, len(candidate_ids))
        hybrid_depth = min(max(top_k, candidate_depth), len(candidate_ids))
        hybrid_pool = fuse_rankings(
            lexical_all,
            semantic_all,
            candidate_product_ids=candidate_ids,
            top_k=hybrid_depth,
            strategy=strategy,
            semantic_weight=semantic_weight,
            rrf_k=rrf_k,
        )
        reranked = reranker.rerank(query, hybrid_pool, top_k)
        lexical_top = lexical_all[:top_k]
        semantic_top = semantic_all[:top_k]
        hybrid_top = hybrid_pool[:top_k]
        exact_count = sum(grade == 2 for grade in grades.values())
        records.append(
            QualityRecord(
                query_id=query_id,
                query=query,
                lexical_ndcg_at_10=_ndcg(lexical_top, grades, top_k),
                semantic_ndcg_at_10=_ndcg(semantic_top, grades, top_k),
                hybrid_ndcg_at_10=_ndcg(hybrid_top, grades, top_k),
                reranked_ndcg_at_10=_ndcg(reranked, grades, top_k),
                lexical_top_grade=_top_grade(lexical_top, grades),
                semantic_top_grade=_top_grade(semantic_top, grades),
                hybrid_top_grade=_top_grade(hybrid_top, grades),
                reranked_top_grade=_top_grade(reranked, grades),
                reranked_top_product_id=reranked[0].product_id if reranked else None,
                reranked_top_product_name=(
                    product_names.get(reranked[0].product_id) if reranked else None
                ),
                exact_judgment_count=exact_count,
                first_reranked_exact_rank=next(
                    (result.rank for result in reranked if grades[result.product_id] == 2),
                    None,
                ),
            )
        )
    return tuple(records)


def categorize_quality_records(
    records: Sequence[QualityRecord],
    *,
    example_limit: int = 3,
    tail_limit: int = 5,
) -> ErrorAnalysis:
    """Apply strict, independently evaluated comparison definitions."""

    if example_limit <= 0 or tail_limit <= 0:
        raise ValueError("example limits must be positive")
    lexical_better = sorted(
        (record for record in records if record.lexical_ndcg_at_10 > record.semantic_ndcg_at_10),
        key=lambda record: (
            -(record.lexical_ndcg_at_10 - record.semantic_ndcg_at_10),
            record.query_id,
        ),
    )
    semantic_better = sorted(
        (record for record in records if record.semantic_ndcg_at_10 > record.lexical_ndcg_at_10),
        key=lambda record: (
            -(record.semantic_ndcg_at_10 - record.lexical_ndcg_at_10),
            record.query_id,
        ),
    )
    hybrid_better = sorted(
        (
            record
            for record in records
            if record.hybrid_ndcg_at_10 > record.lexical_ndcg_at_10
            and record.hybrid_ndcg_at_10 > record.semantic_ndcg_at_10
        ),
        key=lambda record: (
            -(
                record.hybrid_ndcg_at_10
                - max(record.lexical_ndcg_at_10, record.semantic_ndcg_at_10)
            ),
            record.query_id,
        ),
    )
    reranking_helps = sorted(
        (record for record in records if record.reranked_ndcg_at_10 > record.hybrid_ndcg_at_10),
        key=lambda record: (
            -(record.reranked_ndcg_at_10 - record.hybrid_ndcg_at_10),
            record.query_id,
        ),
    )
    reranking_hurts = sorted(
        (record for record in records if record.reranked_ndcg_at_10 < record.hybrid_ndcg_at_10),
        key=lambda record: (
            -(record.hybrid_ndcg_at_10 - record.reranked_ndcg_at_10),
            record.query_id,
        ),
    )
    partial_exact = sorted(
        (
            record
            for record in records
            if record.exact_judgment_count > 0 and record.reranked_top_grade == 1
        ),
        key=lambda record: (record.first_reranked_exact_rank is None, record.query_id),
    )
    failed_tail = sorted(
        records,
        key=lambda record: (record.reranked_ndcg_at_10, record.query_id),
    )
    full_categories = {
        "lexical_better_than_semantic": lexical_better,
        "semantic_better_than_lexical": semantic_better,
        "hybrid_better_than_both": hybrid_better,
        "failed_tail_queries": failed_tail[:tail_limit],
        "partial_vs_exact_confusion": partial_exact,
        "reranking_helps": reranking_helps,
        "reranking_hurts": reranking_hurts,
    }
    return ErrorAnalysis(
        query_count=len(records),
        category_counts={name: len(values) for name, values in full_categories.items()},
        examples={
            name: tuple(values[: tail_limit if name == "failed_tail_queries" else example_limit])
            for name, values in full_categories.items()
        },
    )


def error_analysis_payload(analysis: ErrorAnalysis) -> dict[str, object]:
    """Return JSON-compatible category metadata for benchmark.json."""

    return {
        "query_count": analysis.query_count,
        "category_counts": analysis.category_counts,
        "example_query_ids": {
            category: [record.query_id for record in examples]
            for category, examples in analysis.examples.items()
        },
    }


def render_error_analysis_markdown(analysis: ErrorAnalysis, *, split: str) -> str:
    """Render only measured comparisons and catalog-sourced fields."""

    definitions = {
        "lexical_better_than_semantic": "Lexical nDCG@10 is strictly greater than semantic.",
        "semantic_better_than_lexical": "Semantic nDCG@10 is strictly greater than lexical.",
        "hybrid_better_than_both": "Hybrid nDCG@10 is strictly greater than both base systems.",
        "failed_tail_queries": "The five lowest reranked-hybrid nDCG@10 queries.",
        "partial_vs_exact_confusion": (
            "Reranked hybrid places a Partial judgment first while an Exact judgment exists."
        ),
        "reranking_helps": "Reranked-hybrid nDCG@10 is strictly greater than hybrid.",
        "reranking_hurts": "Reranked-hybrid nDCG@10 is strictly lower than hybrid.",
    }
    lines = [
        "# Search error analysis",
        "",
        f"Analyzed split: `{split}`. Query count: **{analysis.query_count}**.",
        "All comparisons use canonical judged-candidate relevance and nDCG@10. Categories are "
        "evaluated independently and may overlap. Product names and labels come from processed "
        "WANDS data.",
        "",
    ]
    for category, examples in analysis.examples.items():
        title = category.replace("_", " ").title()
        lines.extend(
            [
                f"## {title}",
                "",
                definitions[category],
                f"Qualifying query count: **{analysis.category_counts[category]}**.",
                "",
            ]
        )
        if not examples:
            lines.extend(["No qualifying query was observed.", ""])
            continue
        lines.extend(
            [
                (
                    "| Query ID | Query | L | S | H | R | Top grades L/S/H/R | "
                    "Reranked top product | First Exact rank |"
                ),
                "|---|---|---:|---:|---:|---:|---|---|---:|",
            ]
        )
        for record in examples:
            product = record.reranked_top_product_name or record.reranked_top_product_id or "none"
            first_exact = (
                str(record.first_reranked_exact_rank)
                if record.first_reranked_exact_rank is not None
                else "not in top 10"
            )
            lines.append(
                "| "
                + " | ".join(
                    (
                        _markdown(record.query_id),
                        _markdown(record.query),
                        f"{record.lexical_ndcg_at_10:.4f}",
                        f"{record.semantic_ndcg_at_10:.4f}",
                        f"{record.hybrid_ndcg_at_10:.4f}",
                        f"{record.reranked_ndcg_at_10:.4f}",
                        (
                            f"{record.lexical_top_grade}/{record.semantic_top_grade}/"
                            f"{record.hybrid_top_grade}/{record.reranked_top_grade}"
                        ),
                        _markdown(product),
                        first_exact,
                    )
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_error_analysis(path: Path, analysis: ErrorAnalysis, *, split: str) -> None:
    """Atomically write the generated Markdown report."""

    temporary = path.with_name(f"{path.name}.part")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary.write_text(
            render_error_analysis_markdown(analysis, split=split),
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def _validate_inputs(queries: DataFrame, judgments: DataFrame, *, top_k: int) -> None:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    missing_queries = {"query_id", "query"} - set(queries.columns)
    missing_judgments = {"query_id", "product_id", "relevance_grade"} - set(judgments.columns)
    if missing_queries or missing_judgments:
        raise ValueError(f"missing analysis columns: {sorted(missing_queries | missing_judgments)}")
    if judgments.duplicated(["query_id", "product_id"]).any():
        raise ValueError("analysis judgments must be canonical and unique")


def _ndcg(results: Sequence[SearchResult], grades: Mapping[str, int], top_k: int) -> float:
    ranked_grades = [grades[result.product_id] for result in results]
    return ndcg_at_k(ranked_grades, top_k, ideal_relevance=list(grades.values()))


def _top_grade(results: Sequence[SearchResult], grades: Mapping[str, int]) -> int:
    return grades[results[0].product_id] if results else 0


def _markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\r", " ").replace("\n", " ")
