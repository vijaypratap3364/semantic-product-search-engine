"""Shared leakage-safe query-product features for reranker training and inference."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame

from product_search.data.download import sha256_file
from product_search.retrieval.base import SearchResult

FEATURE_SCHEMA_VERSION = 1
PRODUCT_COLUMNS = (
    "product_id",
    "product_name",
    "product_description",
    "product_text",
)
FEATURE_NAMES = (
    "lexical_similarity",
    "semantic_similarity",
    "hybrid_score",
    "lexical_rank",
    "semantic_rank",
    "query_title_token_overlap",
    "query_description_token_overlap",
    "exact_phrase_in_title",
    "query_title_token_coverage",
    "query_product_text_token_coverage",
    "query_length",
    "title_length",
    "product_text_length",
)
FORBIDDEN_FEATURE_NAMES = frozenset(
    {
        "query_id",
        "product_id",
        "query_class",
        "label",
        "relevance_grade",
        "selected_label_count",
        "judgment_count",
        "distinct_label_count",
        "observed_labels",
        "resolution",
    }
)
FEATURE_DEFINITIONS: dict[str, str] = {
    "lexical_similarity": "Raw TF-IDF cosine-equivalent score.",
    "semantic_similarity": "Raw normalized dense dot-product score.",
    "hybrid_score": "Validation-selected normalized hybrid fusion score.",
    "lexical_rank": "One-based lexical rank; candidate_depth + 1 when absent.",
    "semantic_rank": "One-based semantic rank; candidate_depth + 1 when absent.",
    "query_title_token_overlap": "Jaccard overlap of unique query and title tokens.",
    "query_description_token_overlap": ("Jaccard overlap of unique query and description tokens."),
    "exact_phrase_in_title": "One when the normalized query token phrase occurs in the title.",
    "query_title_token_coverage": "Proportion of unique query tokens present in the title.",
    "query_product_text_token_coverage": (
        "Proportion of unique query tokens present anywhere in product_text."
    ),
    "query_length": "Query token count including repeated tokens.",
    "title_length": "Product-title token count including repeated tokens.",
    "product_text_length": "Searchable product-text token count including repeated tokens.",
}
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

FloatMatrix = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ProductFeatureRecord:
    """Only catalog fields available at online reranking time."""

    product_id: str
    product_name: str
    product_description: str
    product_text: str


@dataclass(frozen=True, slots=True)
class ProductFeatureStore:
    """Validated in-memory product metadata keyed by product ID."""

    records: Mapping[str, ProductFeatureRecord]
    dataset_sha256: str

    @classmethod
    def from_parquet(cls, products_path: Path) -> ProductFeatureStore:
        """Load only the product fields required for online-compatible features."""

        resolved = products_path.resolve()
        frame = pd.read_parquet(resolved, columns=list(PRODUCT_COLUMNS))
        return cls.from_frame(frame, dataset_sha256=sha256_file(resolved))

    @classmethod
    def from_frame(
        cls,
        products: DataFrame,
        *,
        dataset_sha256: str = "fixture",
    ) -> ProductFeatureStore:
        """Validate a product frame and create a deterministic feature store."""

        missing = set(PRODUCT_COLUMNS) - set(products.columns)
        if missing:
            raise ValueError(f"products are missing feature columns: {sorted(missing)}")
        normalized = products.loc[:, PRODUCT_COLUMNS].copy()
        if normalized["product_id"].isna().any():
            raise ValueError("product IDs must not be missing")
        normalized["product_id"] = normalized["product_id"].astype(str)
        if normalized["product_id"].str.strip().eq("").any():
            raise ValueError("product IDs must not be blank")
        if normalized["product_id"].duplicated().any():
            raise ValueError("product IDs must be unique")
        for column in PRODUCT_COLUMNS[1:]:
            normalized[column] = normalized[column].fillna("").astype(str)
        records = {
            str(row["product_id"]): ProductFeatureRecord(
                product_id=str(row["product_id"]),
                product_name=str(row["product_name"]),
                product_description=str(row["product_description"]),
                product_text=str(row["product_text"]),
            )
            for row in cast(list[dict[str, object]], normalized.to_dict(orient="records"))
        }
        if not records:
            raise ValueError("products must not be empty")
        return cls(records=records, dataset_sha256=dataset_sha256)

    def get(self, product_id: str) -> ProductFeatureRecord:
        """Return one record or fail explicitly for an incompatible product mapping."""

        try:
            return self.records[str(product_id)]
        except KeyError as error:
            raise ValueError(f"product is absent from feature store: {product_id}") from error


def extract_query_product_features(
    query: str,
    product: ProductFeatureRecord,
    retrieval_result: SearchResult,
    *,
    candidate_depth: int,
) -> dict[str, float]:
    """Extract the fixed numeric feature schema from inference-available values."""

    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if not query.strip():
        raise ValueError("query must not be blank")
    if retrieval_result.product_id != product.product_id:
        raise ValueError("retrieval result and product feature record IDs do not match")
    if (
        isinstance(candidate_depth, bool)
        or not isinstance(candidate_depth, int)
        or candidate_depth <= 0
    ):
        raise ValueError("candidate_depth must be a positive integer")
    components = retrieval_result.score_components
    if components is None:
        raise ValueError("hybrid retrieval result must include score components")
    required_components = {
        "lexical_raw",
        "semantic_raw",
        "lexical_rank",
        "semantic_rank",
        "lexical_present",
        "semantic_present",
        "hybrid",
    }
    missing = required_components - set(components)
    if missing:
        raise ValueError(f"hybrid score components are missing: {sorted(missing)}")

    query_tokens = _tokens(query)
    title_tokens = _tokens(product.product_name)
    description_tokens = _tokens(product.product_description)
    product_text_tokens = _tokens(product.product_text)
    query_set = set(query_tokens)
    title_set = set(title_tokens)
    description_set = set(description_tokens)
    product_text_set = set(product_text_tokens)
    missing_rank = float(candidate_depth + 1)
    lexical_rank = (
        float(components["lexical_rank"]) if components["lexical_present"] >= 0.5 else missing_rank
    )
    semantic_rank = (
        float(components["semantic_rank"])
        if components["semantic_present"] >= 0.5
        else missing_rank
    )
    features = {
        "lexical_similarity": float(components["lexical_raw"]),
        "semantic_similarity": float(components["semantic_raw"]),
        "hybrid_score": float(components["hybrid"]),
        "lexical_rank": lexical_rank,
        "semantic_rank": semantic_rank,
        "query_title_token_overlap": _jaccard(query_set, title_set),
        "query_description_token_overlap": _jaccard(query_set, description_set),
        "exact_phrase_in_title": float(_contains_token_phrase(query_tokens, title_tokens)),
        "query_title_token_coverage": _coverage(query_set, title_set),
        "query_product_text_token_coverage": _coverage(query_set, product_text_set),
        "query_length": float(len(query_tokens)),
        "title_length": float(len(title_tokens)),
        "product_text_length": float(len(product_text_tokens)),
    }
    validate_feature_schema(tuple(features))
    if not np.isfinite(np.fromiter(features.values(), dtype=np.float64)).all():
        raise ValueError("reranker features must be finite")
    return features


def feature_matrix(rows: Sequence[Mapping[str, object]]) -> FloatMatrix:
    """Convert feature mappings to a finite matrix in the persisted schema order."""

    matrix = np.asarray(
        [[float(cast(Any, row[name])) for name in FEATURE_NAMES] for row in rows],
        dtype=np.float64,
    )
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
        raise ValueError("feature rows must produce the complete two-dimensional schema")
    if not np.isfinite(matrix).all():
        raise ValueError("feature matrix must contain only finite values")
    return matrix


def validate_feature_schema(feature_names: Sequence[str]) -> None:
    """Reject forbidden, missing, reordered, or inference-unavailable model inputs."""

    normalized = tuple(feature_names)
    forbidden = sorted(set(normalized) & FORBIDDEN_FEATURE_NAMES)
    if forbidden:
        raise ValueError(f"forbidden predictive features: {forbidden}")
    if normalized != FEATURE_NAMES:
        raise ValueError("feature schema does not match the supported ordered feature names")


def feature_schema_sha256() -> str:
    """Return a deterministic hash of feature names, definitions, and tokenization."""

    payload = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": FEATURE_NAMES,
        "definitions": FEATURE_DEFINITIONS,
        "token_pattern": _TOKEN_PATTERN.pattern,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN_PATTERN.findall(text.lower()))


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _coverage(query_tokens: set[str], document_tokens: set[str]) -> float:
    return len(query_tokens & document_tokens) / len(query_tokens) if query_tokens else 0.0


def _contains_token_phrase(query_tokens: Sequence[str], title_tokens: Sequence[str]) -> bool:
    if not query_tokens or len(query_tokens) > len(title_tokens):
        return False
    width = len(query_tokens)
    return any(
        tuple(title_tokens[start : start + width]) == tuple(query_tokens)
        for start in range(len(title_tokens) - width + 1)
    )
