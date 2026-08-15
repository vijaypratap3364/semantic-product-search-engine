"""Tests for batched dense index persistence and compatibility checks."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from pandas import DataFrame

from product_search.config import DenseSettings
from product_search.indexing.dense import (
    EMBEDDINGS_FILENAME,
    DenseIndexArtifactError,
    build_dense_index,
    load_dense_index,
    normalize_embeddings,
)


class FakeEmbeddingProvider:
    """Deterministic local provider that records bounded document batches."""

    model_name = "fake/dense"
    provider_name = "fake"
    provider_version = "1.0"

    def __init__(self, vectors: dict[str, list[float]] | None = None) -> None:
        self.vectors = vectors or {
            "alpha": [3.0, 0.0, 0.0],
            "beta": [0.0, 4.0, 0.0],
            "gamma": [0.0, 0.0, 5.0],
        }
        self.document_batches: list[tuple[str, ...]] = []

    def embed_documents(
        self, texts: Sequence[str], *, batch_size: int
    ) -> Iterable[NDArray[np.float32]]:
        assert len(texts) <= batch_size
        self.document_batches.append(tuple(texts))
        return (np.asarray(self.vectors[text], dtype=np.float32) for text in texts)

    def embed_queries(
        self, texts: Sequence[str], *, batch_size: int
    ) -> Iterable[NDArray[np.float32]]:
        return (np.asarray(self.vectors[text], dtype=np.float32) for text in texts)


def _settings(*, expected_dimension: int = 3) -> DenseSettings:
    return DenseSettings(
        model_name="fake/dense",
        expected_dimension=expected_dimension,
        batch_size=2,
    )


def _build(tmp_path: Path) -> tuple[Path, Path, FakeEmbeddingProvider]:
    products_path = tmp_path / "products.parquet"
    DataFrame(
        {
            "product_id": ["p3", "p1", "p2"],
            "product_text": ["gamma", "alpha", "beta"],
        }
    ).to_parquet(products_path, index=False)
    output_dir = tmp_path / "dense"
    provider = FakeEmbeddingProvider()
    build_dense_index(
        products_path,
        output_dir,
        provider=provider,
        settings=_settings(),
        timestamp=datetime(2026, 8, 14, 12, 0, tzinfo=UTC),
    )
    return products_path, output_dir, provider


def test_build_and_load_dense_index_with_deterministic_order(tmp_path: Path) -> None:
    products_path, output_dir, provider = _build(tmp_path)

    loaded = load_dense_index(
        output_dir,
        expected_model_name="fake/dense",
        expected_dimension=3,
    )

    assert loaded.product_ids == ("p1", "p2", "p3")
    assert isinstance(loaded.embeddings, np.memmap)
    assert loaded.embeddings.dtype == np.float32
    assert loaded.embeddings.shape == (3, 3)
    np.testing.assert_allclose(loaded.embeddings, np.eye(3, dtype=np.float32))
    assert provider.document_batches == [("alpha", "beta"), ("gamma",)]
    assert loaded.metadata["dataset_filename"] == products_path.name
    assert loaded.metadata["model_name"] == "fake/dense"
    assert loaded.metadata["embedding_dimension"] == 3
    assert loaded.metadata["embedding_normalization"] == ("project_applied_l2_unit_normalization")
    assert loaded.metadata["created_at"] == "2026-08-14T12:00:00+00:00"
    assert all(len(value["sha256"]) == 64 for value in loaded.metadata["artifacts"].values())


def test_dense_build_refuses_overwrite_without_force(tmp_path: Path) -> None:
    products_path, output_dir, _ = _build(tmp_path)

    with pytest.raises(FileExistsError, match="force=True"):
        build_dense_index(
            products_path,
            output_dir,
            provider=FakeEmbeddingProvider(),
            settings=_settings(),
        )

    metadata = build_dense_index(
        products_path,
        output_dir,
        provider=FakeEmbeddingProvider(),
        settings=_settings(),
        force=True,
    )
    assert metadata["product_count"] == 3


def test_dense_build_rejects_provider_model_or_dimension_mismatch(tmp_path: Path) -> None:
    products_path = tmp_path / "products.parquet"
    DataFrame({"product_id": ["p1"], "product_text": ["alpha"]}).to_parquet(
        products_path, index=False
    )
    wrong_model = FakeEmbeddingProvider()
    wrong_model.model_name = "fake/other"

    with pytest.raises(ValueError, match="does not match configured model"):
        build_dense_index(
            products_path,
            tmp_path / "wrong-model",
            provider=wrong_model,
            settings=_settings(),
        )
    with pytest.raises(ValueError, match="dimension mismatch"):
        build_dense_index(
            products_path,
            tmp_path / "wrong-dimension",
            provider=FakeEmbeddingProvider(),
            settings=_settings(expected_dimension=2),
        )


def test_dense_load_rejects_corrupted_embedding_artifact(tmp_path: Path) -> None:
    _, output_dir, _ = _build(tmp_path)
    with (output_dir / EMBEDDINGS_FILENAME).open("ab") as embedding_file:
        embedding_file.write(b"corruption")

    with pytest.raises(DenseIndexArtifactError, match="byte-size mismatch"):
        load_dense_index(output_dir)


def test_dense_load_rejects_incompatible_model_and_metadata(tmp_path: Path) -> None:
    _, output_dir, _ = _build(tmp_path)

    with pytest.raises(DenseIndexArtifactError, match="model mismatch"):
        load_dense_index(output_dir, expected_model_name="fake/other")

    metadata_path = output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["matrix_shape"] = [99, 3]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(DenseIndexArtifactError, match="matrix shape mismatch"):
        load_dense_index(output_dir)


@pytest.mark.parametrize(
    ("vectors", "message"),
    [
        (np.asarray([[0.0, 0.0]], dtype=np.float32), "zero vectors"),
        (np.asarray([[np.nan, 1.0]], dtype=np.float32), "finite"),
        (np.asarray([1.0, 2.0], dtype=np.float32), "two-dimensional"),
    ],
)
def test_normalize_embeddings_rejects_invalid_vectors(
    vectors: NDArray[np.float32], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_embeddings(vectors)


@pytest.mark.parametrize(
    ("products", "message"),
    [
        (DataFrame({"product_id": ["p1"]}), "missing required columns"),
        (
            DataFrame({"product_id": ["p1", "p1"], "product_text": ["alpha", "beta"]}),
            "must be unique",
        ),
        (DataFrame({"product_id": ["p1"], "product_text": [""]}), "must not be blank"),
        (DataFrame({"product_id": [], "product_text": []}), "must not be empty"),
    ],
)
def test_dense_build_rejects_invalid_products(
    tmp_path: Path, products: DataFrame, message: str
) -> None:
    products_path = tmp_path / "products.parquet"
    products.to_parquet(products_path, index=False)

    with pytest.raises(ValueError, match=message):
        build_dense_index(
            products_path,
            tmp_path / "dense",
            provider=FakeEmbeddingProvider(),
            settings=_settings(),
        )
