"""Build, persist, validate, and load the sparse TF-IDF product index."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict, cast

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import scipy
import sklearn
from pandas import DataFrame
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from product_search.config import LexicalSettings
from product_search.data.download import sha256_file

SCHEMA_VERSION = 1
VECTORIZER_FILENAME = "vectorizer.joblib"
MATRIX_FILENAME = "product_matrix.npz"
PRODUCT_IDS_FILENAME = "product_ids.json"
METADATA_FILENAME = "metadata.json"
REQUIRED_ARTIFACTS = (VECTORIZER_FILENAME, MATRIX_FILENAME, PRODUCT_IDS_FILENAME)


class IndexArtifactError(RuntimeError):
    """Raised when a TF-IDF index is missing, corrupt, or internally incompatible."""


class ArtifactMetadata(TypedDict):
    """Integrity metadata for one persisted index artifact."""

    sha256: str
    byte_size: int


class TfidfIndexMetadata(TypedDict):
    """Serializable build provenance and compatibility metadata."""

    schema_version: int
    created_at: str
    dataset_filename: str
    dataset_sha256: str
    product_count: int
    vectorizer_parameters: dict[str, object]
    vocabulary_size: int
    matrix_shape: list[int]
    matrix_dtype: str
    package_versions: dict[str, str]
    artifacts: dict[str, ArtifactMetadata]


@dataclass(frozen=True, slots=True)
class LoadedTfidfIndex:
    """Validated in-memory TF-IDF artifacts."""

    vectorizer: TfidfVectorizer
    product_matrix: csr_matrix
    product_ids: tuple[str, ...]
    metadata: TfidfIndexMetadata


def vectorizer_parameters(settings: LexicalSettings) -> dict[str, object]:
    """Return the reviewed settings in JSON-serializable form."""

    return {
        "lowercase": settings.lowercase,
        "analyzer": settings.analyzer,
        "ngram_range": [settings.ngram_min, settings.ngram_max],
        "sublinear_tf": settings.sublinear_tf,
        "min_df": settings.min_df,
        "max_features": settings.max_features,
        "norm": settings.norm,
        "dtype": "float32",
    }


def build_tfidf_index(
    products_path: Path,
    output_dir: Path,
    *,
    settings: LexicalSettings,
    force: bool = False,
    timestamp: datetime | None = None,
) -> TfidfIndexMetadata:
    """Fit only ``product_text`` and atomically persist reusable sparse artifacts."""

    products_path = products_path.resolve()
    output_dir = output_dir.resolve()
    products = pd.read_parquet(products_path)
    ordered_products = _validate_and_order_products(products)

    parameters = vectorizer_parameters(settings)
    vectorizer = TfidfVectorizer(
        lowercase=settings.lowercase,
        analyzer=settings.analyzer,
        ngram_range=(settings.ngram_min, settings.ngram_max),
        sublinear_tf=settings.sublinear_tf,
        min_df=settings.min_df,
        max_features=settings.max_features,
        norm=settings.norm,
        dtype=np.float32,
    )
    product_matrix = cast(
        csr_matrix,
        vectorizer.fit_transform(ordered_products["product_text"].astype(str)),
    )
    product_ids = tuple(ordered_products["product_id"].astype(str))

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = {filename: output_dir / filename for filename in REQUIRED_ARTIFACTS}
    metadata_path = output_dir / METADATA_FILENAME
    existing = [path.name for path in (*artifact_paths.values(), metadata_path) if path.exists()]
    if existing and not force:
        raise FileExistsError(
            f"TF-IDF index artifacts already exist in {output_dir}: {sorted(existing)}; "
            "pass force=True to replace them"
        )

    temporary_paths = {
        VECTORIZER_FILENAME: output_dir / f"{VECTORIZER_FILENAME}.part",
        MATRIX_FILENAME: output_dir / "product_matrix.part.npz",
        PRODUCT_IDS_FILENAME: output_dir / f"{PRODUCT_IDS_FILENAME}.part",
    }
    metadata_temporary = output_dir / f"{METADATA_FILENAME}.part"
    metadata_path.unlink(missing_ok=True)
    try:
        joblib.dump(vectorizer, temporary_paths[VECTORIZER_FILENAME], compress=3)
        sparse.save_npz(temporary_paths[MATRIX_FILENAME], product_matrix, compressed=True)
        _write_json(temporary_paths[PRODUCT_IDS_FILENAME], {"product_ids": product_ids})

        artifacts = {
            filename: ArtifactMetadata(
                sha256=sha256_file(temporary_paths[filename]),
                byte_size=temporary_paths[filename].stat().st_size,
            )
            for filename in REQUIRED_ARTIFACTS
        }
        metadata = TfidfIndexMetadata(
            schema_version=SCHEMA_VERSION,
            created_at=(timestamp or datetime.now(UTC)).astimezone(UTC).isoformat(),
            dataset_filename=products_path.name,
            dataset_sha256=sha256_file(products_path),
            product_count=len(product_ids),
            vectorizer_parameters=parameters,
            vocabulary_size=len(vectorizer.vocabulary_),
            matrix_shape=[int(dimension) for dimension in product_matrix.shape],
            matrix_dtype=str(product_matrix.dtype),
            package_versions={
                "joblib": joblib.__version__,
                "numpy": np.__version__,
                "scikit-learn": sklearn.__version__,
                "scipy": scipy.__version__,
            },
            artifacts=artifacts,
        )
        _write_json(metadata_temporary, metadata)

        for filename, destination in artifact_paths.items():
            os.replace(temporary_paths[filename], destination)
        os.replace(metadata_temporary, metadata_path)
    except Exception:
        for temporary_path in (*temporary_paths.values(), metadata_temporary):
            temporary_path.unlink(missing_ok=True)
        raise
    return metadata


def load_tfidf_index(index_dir: Path) -> LoadedTfidfIndex:
    """Load a TF-IDF index only after verifying every persisted artifact hash."""

    index_dir = index_dir.resolve()
    metadata = _load_metadata(index_dir / METADATA_FILENAME)
    _verify_artifacts(index_dir, metadata)

    try:
        vectorizer_object = joblib.load(index_dir / VECTORIZER_FILENAME)
        product_matrix = sparse.load_npz(index_dir / MATRIX_FILENAME)
        product_ids_payload = _read_json(index_dir / PRODUCT_IDS_FILENAME)
    except Exception as error:
        raise IndexArtifactError(
            f"unable to deserialize TF-IDF index in {index_dir}: {error}"
        ) from error

    if not isinstance(vectorizer_object, TfidfVectorizer):
        raise IndexArtifactError("vectorizer artifact is not a TfidfVectorizer")
    if not sparse.isspmatrix_csr(product_matrix):
        raise IndexArtifactError("product matrix must use CSR sparse format")
    product_ids_value = product_ids_payload.get("product_ids")
    if not isinstance(product_ids_value, list) or not all(
        isinstance(product_id, str) for product_id in product_ids_value
    ):
        raise IndexArtifactError("product ID artifact must contain a string list")

    product_ids = tuple(product_ids_value)
    _verify_compatibility(vectorizer_object, product_matrix, product_ids, metadata)
    return LoadedTfidfIndex(
        vectorizer=vectorizer_object,
        product_matrix=product_matrix,
        product_ids=product_ids,
        metadata=metadata,
    )


def _validate_and_order_products(products: DataFrame) -> DataFrame:
    missing_columns = {"product_id", "product_text"} - set(products.columns)
    if missing_columns:
        raise ValueError(f"products are missing required columns: {sorted(missing_columns)}")
    if products[["product_id", "product_text"]].isna().any(axis=None):
        raise ValueError("product IDs and product_text must not be missing")
    product_ids = products["product_id"].astype(str)
    product_text = products["product_text"].astype(str)
    if product_ids.str.strip().eq("").any() or product_text.str.strip().eq("").any():
        raise ValueError("product IDs and product_text must not be blank")
    if product_ids.duplicated().any():
        raise ValueError("product IDs must be unique")
    ordered = products.assign(product_id=product_ids, product_text=product_text).sort_values(
        "product_id", kind="stable", ignore_index=True
    )
    if ordered.empty:
        raise ValueError("products must not be empty")
    return ordered


def _load_metadata(path: Path) -> TfidfIndexMetadata:
    try:
        payload = _read_json(path)
    except (OSError, json.JSONDecodeError) as error:
        raise IndexArtifactError(f"unable to read TF-IDF metadata at {path}: {error}") from error
    required = {
        "schema_version",
        "product_count",
        "vocabulary_size",
        "matrix_shape",
        "matrix_dtype",
        "artifacts",
    }
    missing = required - set(payload)
    if missing:
        raise IndexArtifactError(f"TF-IDF metadata is missing fields: {sorted(missing)}")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise IndexArtifactError(f"unsupported TF-IDF metadata schema: {payload['schema_version']}")
    return cast(TfidfIndexMetadata, payload)


def _verify_artifacts(index_dir: Path, metadata: TfidfIndexMetadata) -> None:
    recorded = metadata["artifacts"]
    for filename in REQUIRED_ARTIFACTS:
        path = index_dir / filename
        artifact = recorded.get(filename)
        if artifact is None:
            raise IndexArtifactError(f"metadata does not describe required artifact: {filename}")
        if not path.is_file():
            raise IndexArtifactError(f"required TF-IDF artifact is missing: {path}")
        actual_size = path.stat().st_size
        if actual_size != artifact["byte_size"]:
            raise IndexArtifactError(
                f"artifact byte-size mismatch for {filename}: expected {artifact['byte_size']}, "
                f"got {actual_size}"
            )
        actual_hash = sha256_file(path)
        if actual_hash != artifact["sha256"]:
            raise IndexArtifactError(f"artifact SHA-256 mismatch for {filename}")


def _verify_compatibility(
    vectorizer: TfidfVectorizer,
    product_matrix: csr_matrix,
    product_ids: tuple[str, ...],
    metadata: TfidfIndexMetadata,
) -> None:
    expected_shape = tuple(metadata["matrix_shape"])
    if product_matrix.shape != expected_shape:
        raise IndexArtifactError(
            f"product matrix shape mismatch: expected {expected_shape}, got {product_matrix.shape}"
        )
    if str(product_matrix.dtype) != metadata["matrix_dtype"]:
        raise IndexArtifactError("product matrix dtype does not match metadata")
    if len(product_ids) != metadata["product_count"] or len(set(product_ids)) != len(product_ids):
        raise IndexArtifactError("product ID ordering is incompatible with metadata")
    vocabulary_size = len(vectorizer.vocabulary_)
    if vocabulary_size != metadata["vocabulary_size"] or product_matrix.shape[1] != vocabulary_size:
        raise IndexArtifactError("vectorizer vocabulary is incompatible with product matrix")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        return cast(dict[str, Any], json.load(input_file))


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
