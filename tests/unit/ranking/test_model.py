"""Tests for logistic relevance training and verified persistence."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from product_search.ranking.features import FEATURE_NAMES
from product_search.ranking.model import (
    MODEL_FILENAME,
    RelevanceModel,
    RerankerArtifactError,
    load_relevance_model,
    probability_to_expected_relevance,
    save_relevance_model,
    train_relevance_model,
)


def _training_data() -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    labels: list[int] = []
    for grade in (0, 1, 2):
        for offset in (0.0, 0.05, 0.1, 0.15):
            rows.append([grade + offset] * len(FEATURE_NAMES))
            labels.append(grade)
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def _model() -> RelevanceModel:
    features, labels = _training_data()
    return train_relevance_model(
        features,
        labels,
        c_value=1.0,
        class_weight="none",
        max_iter=500,
        random_seed=42,
        candidate_depth=100,
    )


def test_train_multinomial_model_and_expose_standardized_coefficients() -> None:
    features, labels = _training_data()
    model = train_relevance_model(
        features,
        labels,
        c_value=1.0,
        class_weight="balanced",
        max_iter=500,
        random_seed=42,
        candidate_depth=100,
    )

    assert model.classes == (0, 1, 2)
    scores = model.predict_expected_relevance(features[[0, 4, 8]])
    assert scores[0] < scores[1] < scores[2]
    coefficients = model.standardized_coefficients()
    assert set(coefficients) == {"0", "1", "2"}
    assert tuple(coefficients["2"]) == FEATURE_NAMES


def test_probability_to_expected_relevance_uses_partial_plus_twice_exact() -> None:
    probabilities = np.asarray([[0.7, 0.2, 0.1], [0.0, 0.25, 0.75]], dtype=np.float64)

    np.testing.assert_allclose(
        probability_to_expected_relevance(probabilities, (0, 1, 2)),
        [0.4, 1.75],
    )
    with pytest.raises(ValueError, match="ordered"):
        probability_to_expected_relevance(probabilities, (2, 1, 0))
    with pytest.raises(ValueError, match="sum to one"):
        probability_to_expected_relevance(
            np.asarray([[0.2, 0.2, 0.2]], dtype=np.float64), (0, 1, 2)
        )


def test_model_serialization_round_trip_and_metadata(tmp_path: Path) -> None:
    model = _model()
    model_dir = tmp_path / "reranker"
    metadata = save_relevance_model(
        model,
        model_dir,
        product_dataset_sha256="products-hash",
        source_hashes={"query_splits.json": "splits-hash"},
        train_query_count=3,
        training_row_count=12,
        class_distribution={0: 4, 1: 4, 2: 4},
        timestamp=datetime(2026, 8, 16, 12, 0, tzinfo=UTC),
    )
    loaded = load_relevance_model(
        model_dir,
        expected_product_dataset_sha256="products-hash",
        expected_candidate_depth=100,
    )
    features, _ = _training_data()

    np.testing.assert_allclose(
        loaded.predict_expected_relevance(features),
        model.predict_expected_relevance(features),
    )
    assert metadata["created_at"] == "2026-08-16T12:00:00+00:00"
    assert metadata["feature_names"] == list(FEATURE_NAMES)
    assert metadata["training"]["class_distribution"] == {"0": 4, "1": 4, "2": 4}
    assert len(metadata["artifacts"][MODEL_FILENAME]["sha256"]) == 64
    with pytest.raises(FileExistsError, match="force=True"):
        save_relevance_model(
            model,
            model_dir,
            product_dataset_sha256="products-hash",
            source_hashes={},
            train_query_count=3,
            training_row_count=12,
            class_distribution={0: 4, 1: 4, 2: 4},
        )


def test_model_load_rejects_corruption_and_incompatible_metadata(tmp_path: Path) -> None:
    model = _model()
    model_dir = tmp_path / "reranker"
    save_relevance_model(
        model,
        model_dir,
        product_dataset_sha256="products-hash",
        source_hashes={},
        train_query_count=3,
        training_row_count=12,
        class_distribution={0: 4, 1: 4, 2: 4},
    )

    with pytest.raises(RerankerArtifactError, match="dataset hash"):
        load_relevance_model(model_dir, expected_product_dataset_sha256="other")
    with pytest.raises(RerankerArtifactError, match="candidate depth"):
        load_relevance_model(model_dir, expected_candidate_depth=50)
    with (model_dir / MODEL_FILENAME).open("ab") as model_file:
        model_file.write(b"corruption")
    with pytest.raises(RerankerArtifactError, match="byte-size mismatch"):
        load_relevance_model(model_dir)

    schema_dir = tmp_path / "schema-mismatch"
    save_relevance_model(
        model,
        schema_dir,
        product_dataset_sha256="products-hash",
        source_hashes={},
        train_query_count=3,
        training_row_count=12,
        class_distribution={0: 4, 1: 4, 2: 4},
    )
    metadata_path = schema_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["feature_names"][-1] = "query_id"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(RerankerArtifactError, match="forbidden predictive"):
        load_relevance_model(schema_dir)


@pytest.mark.parametrize(
    ("labels", "kwargs", "message"),
    [
        ([0] * 12, {}, "must contain"),
        ([0, 1, 2] * 4, {"c_value": 0.0}, "positive and finite"),
        ([0, 1, 2] * 4, {"class_weight": "invalid"}, "unsupported class weight"),
        ([0, 1, 2] * 4, {"max_iter": 0}, "max_iter must be"),
        ([0, 1, 2] * 4, {"candidate_depth": 0}, "candidate_depth must be"),
    ],
)
def test_model_training_rejects_invalid_inputs(
    labels: list[int], kwargs: dict[str, object], message: str
) -> None:
    features, _ = _training_data()
    parameters: dict[str, object] = {
        "c_value": 1.0,
        "class_weight": "none",
        "max_iter": 500,
        "random_seed": 42,
        "candidate_depth": 100,
        **kwargs,
    }
    with pytest.raises(ValueError, match=message):
        train_relevance_model(features, labels, **parameters)  # type: ignore[arg-type]


def test_model_rejects_misaligned_labels_and_invalid_feature_matrices() -> None:
    features, labels = _training_data()
    with pytest.raises(ValueError, match="one-to-one"):
        train_relevance_model(
            features,
            labels[:-1],
            c_value=1.0,
            class_weight="none",
            max_iter=500,
            random_seed=42,
            candidate_depth=100,
        )
    model = _model()
    with pytest.raises(ValueError, match="shape"):
        model.predict_probabilities(np.ones((2, 2), dtype=np.float64))
    invalid = np.ones((1, len(FEATURE_NAMES)), dtype=np.float64)
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        model.predict_probabilities(invalid)
    with pytest.raises(ValueError, match="one column"):
        probability_to_expected_relevance(np.asarray([[0.5, 0.5]], dtype=np.float64), (0, 1, 2))
    with pytest.raises(ValueError, match="non-negative"):
        probability_to_expected_relevance(
            np.asarray([[-0.1, 0.5, 0.6]], dtype=np.float64), (0, 1, 2)
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 99, "unsupported reranker metadata schema"),
        ("feature_schema_version", 99, "feature schema version"),
        ("feature_schema_sha256", "wrong", "feature schema hash"),
        ("model_type", "other", "unsupported reranker model type"),
    ],
)
def test_model_load_rejects_incompatible_metadata_contracts(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    model_dir = tmp_path / field
    save_relevance_model(
        _model(),
        model_dir,
        product_dataset_sha256="products-hash",
        source_hashes={},
        train_query_count=3,
        training_row_count=12,
        class_distribution={0: 4, 1: 4, 2: 4},
    )
    metadata_path = model_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[field] = value
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(RerankerArtifactError, match=message):
        load_relevance_model(model_dir)


def test_model_load_rejects_unreadable_missing_and_mismatched_artifacts(tmp_path: Path) -> None:
    invalid_dir = tmp_path / "invalid-json"
    invalid_dir.mkdir()
    (invalid_dir / "metadata.json").write_text("{invalid", encoding="utf-8")
    with pytest.raises(RerankerArtifactError, match="unable to read"):
        load_relevance_model(invalid_dir)

    model_dir = tmp_path / "missing-field"
    save_relevance_model(
        _model(),
        model_dir,
        product_dataset_sha256="products-hash",
        source_hashes={},
        train_query_count=3,
        training_row_count=12,
        class_distribution={0: 4, 1: 4, 2: 4},
    )
    metadata_path = model_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    del metadata["classes"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(RerankerArtifactError, match="missing fields"):
        load_relevance_model(model_dir)

    hash_dir = tmp_path / "hash"
    save_relevance_model(
        _model(),
        hash_dir,
        product_dataset_sha256="products-hash",
        source_hashes={},
        train_query_count=3,
        training_row_count=12,
        class_distribution={0: 4, 1: 4, 2: 4},
    )
    hash_metadata_path = hash_dir / "metadata.json"
    hash_metadata = json.loads(hash_metadata_path.read_text(encoding="utf-8"))
    hash_metadata["artifacts"][MODEL_FILENAME]["sha256"] = "0" * 64
    hash_metadata_path.write_text(json.dumps(hash_metadata), encoding="utf-8")
    with pytest.raises(RerankerArtifactError, match="SHA-256 mismatch"):
        load_relevance_model(hash_dir)
