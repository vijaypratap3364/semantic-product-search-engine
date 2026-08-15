"""Tests for shared search result contracts."""

from __future__ import annotations

import math

import pytest

from product_search.retrieval.base import SearchResult


def test_search_result_accepts_optional_score_components() -> None:
    result = SearchResult(
        product_id="p1",
        rank=1,
        score=0.75,
        score_components={"lexical": 0.5, "dense": 1.0},
    )

    assert result.product_id == "p1"
    assert result.score_components == {"lexical": 0.5, "dense": 1.0}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"product_id": " ", "rank": 1, "score": 1.0}, "product_id"),
        ({"product_id": "p1", "rank": 0, "score": 1.0}, "rank"),
        ({"product_id": "p1", "rank": True, "score": 1.0}, "rank"),
        ({"product_id": "p1", "rank": 1, "score": math.inf}, "score"),
        (
            {
                "product_id": "p1",
                "rank": 1,
                "score": 1.0,
                "score_components": {"dense": math.nan},
            },
            "components",
        ),
    ],
)
def test_search_result_rejects_invalid_fields(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SearchResult(**kwargs)  # type: ignore[arg-type]
