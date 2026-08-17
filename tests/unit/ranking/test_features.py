"""Tests for leakage-safe query-product reranking features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pandas import DataFrame

from product_search.ranking.features import (
    FEATURE_NAMES,
    ProductFeatureStore,
    extract_query_product_features,
    feature_matrix,
    feature_schema_sha256,
    validate_feature_schema,
)
from product_search.retrieval.base import SearchResult


def _result(**overrides: float) -> SearchResult:
    components = {
        "lexical_raw": 0.8,
        "semantic_raw": 0.6,
        "lexical_rank": 2.0,
        "semantic_rank": 4.0,
        "lexical_present": 1.0,
        "semantic_present": 1.0,
        "hybrid": 0.7,
        **overrides,
    }
    return SearchResult(product_id="p1", rank=1, score=0.7, score_components=components)


def _store() -> ProductFeatureStore:
    return ProductFeatureStore.from_frame(
        DataFrame(
            {
                "product_id": ["p1"],
                "product_name": ["Round Oak Coffee Table"],
                "product_description": ["A circular solid wood living room table"],
                "product_text": ["name round oak coffee table description circular solid wood"],
            }
        )
    )


def test_extract_features_uses_only_inference_available_text_and_scores() -> None:
    features = extract_query_product_features(
        "round coffee table",
        _store().get("p1"),
        _result(),
        candidate_depth=100,
    )

    assert tuple(features) == FEATURE_NAMES
    assert features["lexical_similarity"] == 0.8
    assert features["semantic_similarity"] == 0.6
    assert features["lexical_rank"] == 2.0
    assert features["semantic_rank"] == 4.0
    assert features["exact_phrase_in_title"] == 0.0
    assert features["query_title_token_coverage"] == 1.0
    assert features["query_product_text_token_coverage"] == 1.0
    assert features["query_length"] == 3.0
    assert features["title_length"] == 4.0
    assert features["query_title_token_overlap"] == pytest.approx(0.75)


def test_extract_features_handles_missing_modality_and_missing_description() -> None:
    store = ProductFeatureStore.from_frame(
        DataFrame(
            {
                "product_id": ["p1"],
                "product_name": ["Coffee Table"],
                "product_description": [None],
                "product_text": ["coffee table"],
            }
        )
    )
    features = extract_query_product_features(
        "coffee table",
        store.get("p1"),
        _result(semantic_present=0.0, semantic_rank=0.0),
        candidate_depth=50,
    )

    assert features["semantic_rank"] == 51.0
    assert features["query_description_token_overlap"] == 0.0
    assert features["exact_phrase_in_title"] == 1.0


def test_feature_schema_excludes_ids_classes_and_labels() -> None:
    assert "query_id" not in FEATURE_NAMES
    assert "product_id" not in FEATURE_NAMES
    assert "query_class" not in FEATURE_NAMES
    assert "relevance_grade" not in FEATURE_NAMES
    with pytest.raises(ValueError, match="forbidden predictive features"):
        validate_feature_schema((*FEATURE_NAMES[:-1], "query_id"))
    with pytest.raises(ValueError, match="does not match"):
        validate_feature_schema(FEATURE_NAMES[:-1])


def test_feature_matrix_is_ordered_finite_and_schema_hash_is_stable() -> None:
    features = extract_query_product_features(
        "round table", _store().get("p1"), _result(), candidate_depth=100
    )
    matrix = feature_matrix([features, features])

    assert matrix.shape == (2, len(FEATURE_NAMES))
    assert np.isfinite(matrix).all()
    assert feature_schema_sha256() == feature_schema_sha256()
    with pytest.raises(ValueError, match="finite"):
        feature_matrix([{**features, "hybrid_score": np.nan}])


def test_product_feature_store_loads_parquet_and_rejects_incompatible_products(
    tmp_path: Path,
) -> None:
    products_path = tmp_path / "products.parquet"
    DataFrame(
        {
            "product_id": ["p1"],
            "product_name": ["Table"],
            "product_description": ["Wood"],
            "product_text": ["table wood"],
        }
    ).to_parquet(products_path, index=False)
    store = ProductFeatureStore.from_parquet(products_path)

    assert len(store.dataset_sha256) == 64
    with pytest.raises(ValueError, match="absent"):
        store.get("missing")
    with pytest.raises(ValueError, match="missing feature columns"):
        ProductFeatureStore.from_frame(DataFrame({"product_id": ["p1"]}))
    duplicate = DataFrame(
        {
            "product_id": ["p1", "p1"],
            "product_name": ["A", "B"],
            "product_description": ["", ""],
            "product_text": ["a", "b"],
        }
    )
    with pytest.raises(ValueError, match="must be unique"):
        ProductFeatureStore.from_frame(duplicate)


@pytest.mark.parametrize(
    ("query", "result", "candidate_depth", "message"),
    [
        ("", _result(), 100, "must not be blank"),
        ("table", SearchResult("other", 1, 0.2, {}), 100, "IDs do not match"),
        ("table", SearchResult("p1", 1, 0.2, None), 100, "score components"),
        ("table", _result(), 0, "positive integer"),
    ],
)
def test_extract_features_rejects_invalid_inputs(
    query: str,
    result: SearchResult,
    candidate_depth: int,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        extract_query_product_features(
            query,
            _store().get("p1"),
            result,
            candidate_depth=candidate_depth,
        )
