"""Build, persist, validate, and load a memory-mapped dense product index."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict, cast, runtime_checkable

import fastembed
import numpy as np
import pandas as pd
from fastembed import TextEmbedding
from numpy.typing import NDArray
from pandas import DataFrame

from product_search.config import DenseSettings
from product_search.data.download import sha256_file

SCHEMA_VERSION = 1
EMBEDDINGS_FILENAME = "embeddings.npy"
PRODUCT_IDS_FILENAME = "product_ids.json"
METADATA_FILENAME = "metadata.json"
REQUIRED_ARTIFACTS = (EMBEDDINGS_FILENAME, PRODUCT_IDS_FILENAME)
NORMALIZATION_DESCRIPTION = "project_applied_l2_unit_normalization"

Float32Array = NDArray[np.float32]


class DenseIndexArtifactError(RuntimeError):
    """Raised when a dense index is missing, corrupt, or incompatible."""


class ArtifactMetadata(TypedDict):
    """Integrity metadata for one persisted artifact."""

    sha256: str
    byte_size: int


class DenseIndexMetadata(TypedDict):
    """Serializable dense-index provenance and compatibility metadata."""

    schema_version: int
    created_at: str
    dataset_filename: str
    dataset_sha256: str
    product_count: int
    model_name: str
    embedding_dimension: int
    embedding_dtype: str
    embedding_normalization: str
    matrix_shape: list[int]
    batch_size: int
    provider: str
    package_versions: dict[str, str]
    artifacts: dict[str, ArtifactMetadata]


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Small provider boundary used to keep unit tests model-free."""

    @property
    def model_name(self) -> str:
        """Return the exact model identifier used for embedding."""
        ...

    @property
    def provider_name(self) -> str:
        """Return the embedding implementation name recorded in metadata."""
        ...

    @property
    def provider_version(self) -> str:
        """Return the embedding implementation version recorded in metadata."""
        ...

    def embed_documents(self, texts: Sequence[str], *, batch_size: int) -> Iterable[Float32Array]:
        """Yield one product vector for each input text, in input order."""
        ...

    def embed_queries(self, texts: Sequence[str], *, batch_size: int) -> Iterable[Float32Array]:
        """Yield one query vector for each input text, in input order."""
        ...


class FastEmbedProvider:
    """One reusable CPU FastEmbed model with separate passage/query methods."""

    def __init__(
        self,
        model_name: str,
        *,
        cache_dir: Path,
        local_files_only: bool = False,
    ) -> None:
        self._model_name = model_name
        self._model = TextEmbedding(
            model_name=model_name,
            cache_dir=str(cache_dir.resolve()),
            cuda=False,
            lazy_load=True,
            local_files_only=local_files_only,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider_name(self) -> str:
        return "fastembed"

    @property
    def provider_version(self) -> str:
        return str(fastembed.__version__)

    def embed_documents(self, texts: Sequence[str], *, batch_size: int) -> Iterable[Float32Array]:
        vectors = self._model.passage_embed(texts, batch_size=batch_size)
        return (np.asarray(vector, dtype=np.float32) for vector in vectors)

    def embed_queries(self, texts: Sequence[str], *, batch_size: int) -> Iterable[Float32Array]:
        vectors = self._model.query_embed(texts, batch_size=batch_size)
        return (np.asarray(vector, dtype=np.float32) for vector in vectors)


@dataclass(frozen=True, slots=True)
class LoadedDenseIndex:
    """Validated memory-mapped dense artifacts."""

    embeddings: Float32Array
    product_ids: tuple[str, ...]
    metadata: DenseIndexMetadata


def build_dense_index(
    products_path: Path,
    output_dir: Path,
    *,
    provider: EmbeddingProvider,
    settings: DenseSettings,
    force: bool = False,
    timestamp: datetime | None = None,
) -> DenseIndexMetadata:
    """Embed sorted product text in batches and atomically persist a float32 NPY index."""

    products_path = products_path.resolve()
    output_dir = output_dir.resolve()
    if provider.model_name != settings.model_name:
        raise ValueError(
            f"embedding provider model {provider.model_name!r} does not match configured model "
            f"{settings.model_name!r}"
        )

    products = pd.read_parquet(products_path)
    ordered_products = _validate_and_order_products(products)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = {filename: output_dir / filename for filename in REQUIRED_ARTIFACTS}
    metadata_path = output_dir / METADATA_FILENAME
    existing = [path.name for path in (*artifact_paths.values(), metadata_path) if path.exists()]
    if existing and not force:
        raise FileExistsError(
            f"dense index artifacts already exist in {output_dir}: {sorted(existing)}; "
            "pass force=True to replace them"
        )

    embedding_temporary = output_dir / f"{EMBEDDINGS_FILENAME}.part"
    product_ids_temporary = output_dir / f"{PRODUCT_IDS_FILENAME}.part"
    metadata_temporary = output_dir / f"{METADATA_FILENAME}.part"
    temporary_paths = (embedding_temporary, product_ids_temporary, metadata_temporary)
    for temporary_path in temporary_paths:
        temporary_path.unlink(missing_ok=True)

    product_ids = tuple(ordered_products["product_id"].astype(str))
    embedding_dimension: int | None = None
    embeddings: np.memmap[Any, np.dtype[np.float32]] | None = None
    try:
        for start in range(0, len(ordered_products), settings.batch_size):
            stop = min(start + settings.batch_size, len(ordered_products))
            texts = ordered_products.iloc[start:stop]["product_text"].astype(str).tolist()
            batch = _collect_and_normalize(
                provider.embed_documents(texts, batch_size=settings.batch_size),
                expected_count=len(texts),
                expected_dimension=settings.expected_dimension,
            )
            if embedding_dimension is None:
                embedding_dimension = int(batch.shape[1])
                embeddings = np.lib.format.open_memmap(
                    embedding_temporary,
                    mode="w+",
                    dtype=np.float32,
                    shape=(len(product_ids), embedding_dimension),
                )
            if embeddings is None or batch.shape[1] != embedding_dimension:
                raise ValueError("embedding provider returned inconsistent dimensions")
            embeddings[start:stop] = batch

        if embeddings is None or embedding_dimension is None:
            raise ValueError("products must not be empty")
        embeddings.flush()
        embeddings = None

        _write_json(product_ids_temporary, {"product_ids": product_ids})
        artifacts = {
            filename: ArtifactMetadata(
                sha256=sha256_file(path),
                byte_size=path.stat().st_size,
            )
            for filename, path in (
                (EMBEDDINGS_FILENAME, embedding_temporary),
                (PRODUCT_IDS_FILENAME, product_ids_temporary),
            )
        }
        metadata = DenseIndexMetadata(
            schema_version=SCHEMA_VERSION,
            created_at=(timestamp or datetime.now(UTC)).astimezone(UTC).isoformat(),
            dataset_filename=products_path.name,
            dataset_sha256=sha256_file(products_path),
            product_count=len(product_ids),
            model_name=provider.model_name,
            embedding_dimension=embedding_dimension,
            embedding_dtype="float32",
            embedding_normalization=NORMALIZATION_DESCRIPTION,
            matrix_shape=[len(product_ids), embedding_dimension],
            batch_size=settings.batch_size,
            provider=provider.provider_name,
            package_versions={
                "numpy": np.__version__,
                provider.provider_name: provider.provider_version,
            },
            artifacts=artifacts,
        )
        _write_json(metadata_temporary, metadata)
        os.replace(embedding_temporary, artifact_paths[EMBEDDINGS_FILENAME])
        os.replace(product_ids_temporary, artifact_paths[PRODUCT_IDS_FILENAME])
        os.replace(metadata_temporary, metadata_path)
    except Exception:
        if embeddings is not None:
            embeddings.flush()
            del embeddings
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)
        raise
    return metadata


def load_dense_index(
    index_dir: Path,
    *,
    expected_model_name: str | None = None,
    expected_dimension: int | None = None,
    mmap_mode: Literal["r", "r+"] | None = "r",
) -> LoadedDenseIndex:
    """Hash-check and compatibility-check a dense index before returning it."""

    index_dir = index_dir.resolve()
    metadata = _load_metadata(index_dir / METADATA_FILENAME)
    _verify_artifacts(index_dir, metadata)
    try:
        embeddings = np.load(
            index_dir / EMBEDDINGS_FILENAME,
            allow_pickle=False,
            mmap_mode=mmap_mode,
        )
        product_ids_payload = _read_json(index_dir / PRODUCT_IDS_FILENAME)
    except Exception as error:
        raise DenseIndexArtifactError(
            f"unable to deserialize dense index in {index_dir}: {error}"
        ) from error

    product_ids_value = product_ids_payload.get("product_ids")
    if not isinstance(product_ids_value, list) or not all(
        isinstance(product_id, str) for product_id in product_ids_value
    ):
        raise DenseIndexArtifactError("product ID artifact must contain a string list")
    product_ids = tuple(product_ids_value)
    typed_embeddings = cast(Float32Array, embeddings)
    _verify_compatibility(
        typed_embeddings,
        product_ids,
        metadata,
        expected_model_name=expected_model_name,
        expected_dimension=expected_dimension,
    )
    return LoadedDenseIndex(
        embeddings=typed_embeddings,
        product_ids=product_ids,
        metadata=metadata,
    )


def normalize_embeddings(
    embeddings: Float32Array,
    *,
    expected_dimension: int | None = None,
) -> Float32Array:
    """Return finite, non-zero, L2-normalized float32 row vectors."""

    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional matrix")
    if vectors.shape[0] == 0 or vectors.shape[1] == 0:
        raise ValueError("embeddings must not be empty")
    if expected_dimension is not None and vectors.shape[1] != expected_dimension:
        raise ValueError(
            f"embedding dimension mismatch: expected {expected_dimension}, got {vectors.shape[1]}"
        )
    if not np.isfinite(vectors).all():
        raise ValueError("embeddings must contain only finite values")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("embeddings must not contain zero vectors")
    return np.asarray(vectors / norms, dtype=np.float32)


def _collect_and_normalize(
    vectors: Iterable[Float32Array],
    *,
    expected_count: int,
    expected_dimension: int,
) -> Float32Array:
    collected = [np.asarray(vector, dtype=np.float32) for vector in vectors]
    if len(collected) != expected_count:
        raise ValueError(
            f"embedding provider returned {len(collected)} vectors for {expected_count} texts"
        )
    try:
        matrix = np.stack(collected)
    except ValueError as error:
        raise ValueError("embedding provider returned inconsistent vector shapes") from error
    return normalize_embeddings(matrix, expected_dimension=expected_dimension)


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


def _load_metadata(path: Path) -> DenseIndexMetadata:
    try:
        payload = _read_json(path)
    except (OSError, json.JSONDecodeError) as error:
        raise DenseIndexArtifactError(
            f"unable to read dense metadata at {path}: {error}"
        ) from error
    required = {
        "schema_version",
        "product_count",
        "model_name",
        "embedding_dimension",
        "embedding_dtype",
        "embedding_normalization",
        "matrix_shape",
        "artifacts",
    }
    missing = required - set(payload)
    if missing:
        raise DenseIndexArtifactError(f"dense metadata is missing fields: {sorted(missing)}")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise DenseIndexArtifactError(
            f"unsupported dense metadata schema: {payload['schema_version']}"
        )
    return cast(DenseIndexMetadata, payload)


def _verify_artifacts(index_dir: Path, metadata: DenseIndexMetadata) -> None:
    recorded = metadata["artifacts"]
    for filename in REQUIRED_ARTIFACTS:
        path = index_dir / filename
        artifact = recorded.get(filename)
        if artifact is None:
            raise DenseIndexArtifactError(
                f"metadata does not describe required dense artifact: {filename}"
            )
        if not path.is_file():
            raise DenseIndexArtifactError(f"required dense artifact is missing: {path}")
        actual_size = path.stat().st_size
        if actual_size != artifact["byte_size"]:
            raise DenseIndexArtifactError(
                f"artifact byte-size mismatch for {filename}: expected {artifact['byte_size']}, "
                f"got {actual_size}"
            )
        if sha256_file(path) != artifact["sha256"]:
            raise DenseIndexArtifactError(f"artifact SHA-256 mismatch for {filename}")


def _verify_compatibility(
    embeddings: Float32Array,
    product_ids: tuple[str, ...],
    metadata: DenseIndexMetadata,
    *,
    expected_model_name: str | None,
    expected_dimension: int | None,
) -> None:
    expected_shape = tuple(metadata["matrix_shape"])
    if embeddings.ndim != 2 or embeddings.shape != expected_shape:
        raise DenseIndexArtifactError(
            f"embedding matrix shape mismatch: expected {expected_shape}, got {embeddings.shape}"
        )
    if str(embeddings.dtype) != metadata["embedding_dtype"] or embeddings.dtype != np.float32:
        raise DenseIndexArtifactError("embedding matrix dtype does not match float32 metadata")
    if embeddings.shape[1] != metadata["embedding_dimension"]:
        raise DenseIndexArtifactError("embedding dimension is incompatible with matrix shape")
    if len(product_ids) != metadata["product_count"] or len(set(product_ids)) != len(product_ids):
        raise DenseIndexArtifactError("product ID ordering is incompatible with metadata")
    if embeddings.shape[0] != len(product_ids):
        raise DenseIndexArtifactError("embedding rows are incompatible with product ID ordering")
    if metadata["embedding_normalization"] != NORMALIZATION_DESCRIPTION:
        raise DenseIndexArtifactError("unsupported embedding normalization metadata")
    if expected_model_name is not None and metadata["model_name"] != expected_model_name:
        raise DenseIndexArtifactError(
            f"dense index model mismatch: expected {expected_model_name!r}, "
            f"got {metadata['model_name']!r}"
        )
    if expected_dimension is not None and embeddings.shape[1] != expected_dimension:
        raise DenseIndexArtifactError(
            f"dense index dimension mismatch: expected {expected_dimension}, "
            f"got {embeddings.shape[1]}"
        )
    if not np.isfinite(embeddings).all():
        raise DenseIndexArtifactError("embedding matrix contains non-finite values")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-5):
        raise DenseIndexArtifactError("embedding matrix is not L2 normalized")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        return cast(dict[str, Any], json.load(input_file))


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
