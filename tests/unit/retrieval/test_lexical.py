"""Tests for sparse TF-IDF full-catalog and candidate retrieval."""

from __future__ import annotations

from pathlib import Path

import pytest
from pandas import DataFrame

from product_search.config import LexicalSettings
from product_search.indexing.tfidf import build_tfidf_index
from product_search.retrieval.lexical import LexicalSearchEngine
from product_search.retrieval.lexical import main as lexical_main


@pytest.fixture
def lexical_index_dir(tmp_path: Path) -> Path:
    products_path = tmp_path / "products.parquet"
    DataFrame(
        {
            "product_id": ["p3", "p1", "p2", "p0"],
            "product_text": [
                "blue outdoor rug",
                "round coffee table wood",
                "rectangular dining table",
                "round accent mirror",
            ],
        }
    ).to_parquet(products_path, index=False)
    index_dir = tmp_path / "tfidf"
    build_tfidf_index(
        products_path,
        index_dir,
        settings=LexicalSettings(min_df=1, max_features=None),
    )
    return index_dir


@pytest.fixture
def lexical_engine(lexical_index_dir: Path) -> LexicalSearchEngine:
    return LexicalSearchEngine.from_index_dir(lexical_index_dir)


def test_search_orders_results_by_sparse_similarity(lexical_engine: LexicalSearchEngine) -> None:
    results = lexical_engine.search("round coffee table", top_k=3)

    assert [result.product_id for result in results] == ["p1", "p0", "p2"]
    assert [result.rank for result in results] == [1, 2, 3]
    assert results[0].score > results[1].score
    assert results[1].score == results[2].score
    assert results[0].score_components == {"lexical": results[0].score}
    assert lexical_engine.product_ids == ("p0", "p1", "p2", "p3")


def test_candidate_search_ranks_only_supplied_ids_and_retains_zero_scores(
    lexical_engine: LexicalSearchEngine,
) -> None:
    results = lexical_engine.search_candidates("round", ["p3", "p2", "p1"], top_k=3)

    assert [result.product_id for result in results] == ["p1", "p2", "p3"]
    assert results[0].score > 0.0
    assert results[1].score == 0.0
    assert results[2].score == 0.0


def test_top_k_limits_results_and_breaks_score_ties_by_product_id(
    lexical_engine: LexicalSearchEngine,
) -> None:
    results = lexical_engine.search_candidates("round", ["p3", "p2"], top_k=1)

    assert [result.product_id for result in results] == ["p2"]
    assert len(lexical_engine.search("table", top_k=1)) == 1


@pytest.mark.parametrize("query", ["", "   ", "unseenquuxword"])
def test_empty_or_unknown_query_returns_no_results(
    lexical_engine: LexicalSearchEngine, query: str
) -> None:
    assert lexical_engine.search(query, top_k=10) == []
    assert lexical_engine.search_candidates(query, ["p1"], top_k=10) == []


@pytest.mark.parametrize("top_k", [0, -1, True])
def test_invalid_top_k_fails_clearly(lexical_engine: LexicalSearchEngine, top_k: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        lexical_engine.search("table", top_k)


def test_unknown_or_duplicate_candidate_ids_fail_clearly(
    lexical_engine: LexicalSearchEngine,
) -> None:
    with pytest.raises(ValueError, match="absent from the TF-IDF index"):
        lexical_engine.search_candidates("table", ["missing"], 10)
    with pytest.raises(ValueError, match="must be unique"):
        lexical_engine.search_candidates("table", ["p1", "p1"], 10)


def test_vocabulary_analysis_uses_fitted_vectorizer(
    lexical_engine: LexicalSearchEngine,
) -> None:
    analysis = lexical_engine.vocabulary_analysis("round quux")

    assert analysis["tokens"] == ("round", "quux", "round quux")
    assert analysis["indexed_tokens"] == ("round",)
    assert analysis["out_of_vocabulary_tokens"] == ("quux", "round quux")


def test_search_cli_prints_ranked_json(
    lexical_index_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = lexical_main(
        ["round coffee table", "--top-k", "2", "--index-dir", str(lexical_index_dir)]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert '"product_id": "p1"' in output
    assert '"rank": 1' in output
