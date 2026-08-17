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


def _model() -> object:
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
        model,  # type: ignore[arg-type]
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
        model.predict_expected_relevance(features),  # type: ignore[union-attr]
    )
    assert metadata["created_at"] == "2026-08-16T12:00:00+00:00"
    assert metadata["feature_names"] == list(FEATURE_NAMES)
    assert metadata["training"]["class_distribution"] == {"0": 4, "1": 4, "2": 4}
    assert len(metadata["artifacts"][MODEL_FILENAME]["sha256"]) == 64
    with pytest.raises(FileExistsError, match="force=True"):
        save_relevance_model(
            model,  # type: ignore[arg-type]
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
        model,  # type: ignore[arg-type]
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
        model,  # type: ignore[arg-type]
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
