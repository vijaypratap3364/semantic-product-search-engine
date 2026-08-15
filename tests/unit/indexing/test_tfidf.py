"""Tests for deterministic TF-IDF index persistence and validation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pandas import DataFrame

from product_search.config import LexicalSettings
from product_search.indexing.build_lexical import main as build_main
from product_search.indexing.tfidf import (
    MATRIX_FILENAME,
    IndexArtifactError,
    build_tfidf_index,
    load_tfidf_index,
)


def _products() -> DataFrame:
    return DataFrame(
        {
            "product_id": ["p3", "p1", "p2"],
            "product_name": ["secretword", "secretword", "secretword"],
            "product_text": [
                "blue outdoor rug",
                "round coffee table wood",
                "rectangular dining table",
            ],
        }
    )


def _settings() -> LexicalSettings:
    return LexicalSettings(min_df=1, max_features=None)


def _build(tmp_path: Path) -> tuple[Path, Path]:
    products_path = tmp_path / "products.parquet"
    index_dir = tmp_path / "tfidf"
    _products().to_parquet(products_path, index=False)
    build_tfidf_index(
        products_path,
        index_dir,
        settings=_settings(),
        timestamp=datetime(2026, 8, 14, 12, 0, tzinfo=UTC),
    )
    return products_path, index_dir


def test_build_and_load_index_with_deterministic_product_order(tmp_path: Path) -> None:
    products_path, index_dir = _build(tmp_path)

    loaded = load_tfidf_index(index_dir)
    metadata = loaded.metadata

    assert loaded.product_ids == ("p1", "p2", "p3")
    assert loaded.product_matrix.shape[0] == 3
    assert loaded.product_matrix.format == "csr"
    assert loaded.product_matrix.dtype.name == "float32"
    assert metadata["product_count"] == 3
    assert metadata["dataset_filename"] == products_path.name
    assert len(metadata["dataset_sha256"]) == 64
    assert metadata["created_at"] == "2026-08-14T12:00:00+00:00"
    assert metadata["vectorizer_parameters"]["ngram_range"] == [1, 2]
    assert metadata["vocabulary_size"] == len(loaded.vectorizer.vocabulary_)
    assert "secretword" not in loaded.vectorizer.vocabulary_
    assert all(len(artifact["sha256"]) == 64 for artifact in metadata["artifacts"].values())


def test_build_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    products_path, index_dir = _build(tmp_path)

    with pytest.raises(FileExistsError, match="force=True"):
        build_tfidf_index(products_path, index_dir, settings=_settings())

    metadata = build_tfidf_index(
        products_path,
        index_dir,
        settings=_settings(),
        force=True,
    )
    assert metadata["product_count"] == 3


def test_build_cli_uses_explicit_paths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    products_path = tmp_path / "products.parquet"
    index_dir = tmp_path / "cli-index"
    _products().to_parquet(products_path, index=False)

    exit_code = build_main(["--products", str(products_path), "--index-dir", str(index_dir)])

    assert exit_code == 0
    assert '"product_count": 3' in capsys.readouterr().out
    assert (index_dir / "metadata.json").is_file()


def test_load_rejects_corrupted_artifact(tmp_path: Path) -> None:
    _, index_dir = _build(tmp_path)
    matrix_path = index_dir / MATRIX_FILENAME
    with matrix_path.open("ab") as matrix_file:
        matrix_file.write(b"corruption")

    with pytest.raises(IndexArtifactError, match="byte-size mismatch"):
        load_tfidf_index(index_dir)


def test_load_rejects_incompatible_metadata(tmp_path: Path) -> None:
    _, index_dir = _build(tmp_path)
    metadata_path = index_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["matrix_shape"] = [99, metadata["matrix_shape"][1]]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(IndexArtifactError, match="matrix shape mismatch"):
        load_tfidf_index(index_dir)


@pytest.mark.parametrize(
    ("products", "message"),
    [
        (DataFrame({"product_id": ["p1"]}), "missing required columns"),
        (
            DataFrame({"product_id": ["p1", "p1"], "product_text": ["one", "two"]}),
            "must be unique",
        ),
        (DataFrame({"product_id": ["p1"], "product_text": [""]}), "must not be blank"),
        (DataFrame({"product_id": [], "product_text": []}), "must not be empty"),
    ],
)
def test_build_rejects_invalid_products(tmp_path: Path, products: DataFrame, message: str) -> None:
    products_path = tmp_path / "products.parquet"
    products.to_parquet(products_path, index=False)

    with pytest.raises(ValueError, match=message):
        build_tfidf_index(products_path, tmp_path / "index", settings=_settings())
