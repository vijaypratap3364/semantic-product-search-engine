"""Interpretable multinomial relevance model with verified local persistence."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict, cast, runtime_checkable

import joblib  # type: ignore[import-untyped]
import numpy as np
import sklearn
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from product_search.data.download import sha256_file
from product_search.ranking.features import (
    FEATURE_DEFINITIONS,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    FloatMatrix,
    feature_schema_sha256,
    validate_feature_schema,
)

SCHEMA_VERSION = 1
MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"
RELEVANCE_CLASSES = (0, 1, 2)
ClassWeightName = Literal["none", "balanced"]
FloatArray = NDArray[np.float64]


class RerankerArtifactError(RuntimeError):
    """Raised when a persisted reranker is missing, corrupt, or incompatible."""


class ArtifactMetadata(TypedDict):
    """Integrity information for one persisted artifact."""

    sha256: str
    byte_size: int


class RerankerMetadata(TypedDict):
    """Persisted model provenance and inference compatibility contract."""

    schema_version: int
    created_at: str
    model_type: str
    feature_schema_version: int
    feature_names: list[str]
    feature_definitions: dict[str, str]
    feature_schema_sha256: str
    classes: list[int]
    expected_relevance_formula: str
    candidate_depth: int
    hyperparameters: dict[str, object]
    product_dataset_sha256: str
    source_hashes: dict[str, str]
    training: dict[str, object]
    package_versions: dict[str, str]
    artifacts: dict[str, ArtifactMetadata]


@runtime_checkable
class RelevanceScorer(Protocol):
    """Small inference boundary used by the reranking search engine and tests."""

    @property
    def classes(self) -> tuple[int, ...]:
        """Return numeric relevance classes in probability-column order."""
        ...

    def predict_probabilities(self, features: FloatMatrix) -> FloatMatrix:
        """Return one probability distribution per feature row."""
        ...

    def predict_expected_relevance(self, features: FloatMatrix) -> FloatArray:
        """Return expected graded relevance for each feature row."""
        ...


@dataclass(frozen=True, slots=True)
class RelevanceModel:
    """Validated sklearn pipeline and metadata used for inference."""

    pipeline: Pipeline
    candidate_depth: int
    c_value: float
    class_weight: ClassWeightName
    metadata: RerankerMetadata | None = None

    @property
    def classes(self) -> tuple[int, ...]:
        classifier = _classifier(self.pipeline)
        return tuple(int(value) for value in classifier.classes_)

    def predict_probabilities(self, features: FloatMatrix) -> FloatMatrix:
        matrix = _validate_feature_matrix(features)
        probabilities = np.asarray(self.pipeline.predict_proba(matrix), dtype=np.float64)
        _validate_probabilities(probabilities, self.classes)
        return cast(FloatMatrix, probabilities)

    def predict_expected_relevance(self, features: FloatMatrix) -> FloatArray:
        probabilities = self.predict_probabilities(features)
        return probability_to_expected_relevance(probabilities, self.classes)

    def standardized_coefficients(self) -> dict[str, dict[str, float]]:
        """Return classifier coefficients in standardized feature space."""

        classifier = _classifier(self.pipeline)
        return {
            str(int(class_value)): {
                feature_name: float(coefficient)
                for feature_name, coefficient in zip(
                    FEATURE_NAMES,
                    classifier.coef_[class_index],
                    strict=True,
                )
            }
            for class_index, class_value in enumerate(classifier.classes_)
        }


def train_relevance_model(
    features: FloatMatrix,
    relevance_grades: Sequence[int] | NDArray[np.int64],
    *,
    c_value: float,
    class_weight: ClassWeightName,
    max_iter: int,
    random_seed: int,
    candidate_depth: int,
) -> RelevanceModel:
    """Fit a scaled multinomial logistic regression on train-query feature rows."""

    matrix = _validate_feature_matrix(features)
    labels = np.asarray(relevance_grades, dtype=np.int64)
    if labels.ndim != 1 or labels.shape[0] != matrix.shape[0]:
        raise ValueError("relevance grades must align one-to-one with feature rows")
    if tuple(sorted(int(value) for value in np.unique(labels))) != RELEVANCE_CLASSES:
        raise ValueError("training data must contain Irrelevant, Partial, and Exact grades")
    if not math.isfinite(c_value) or c_value <= 0:
        raise ValueError("logistic-regression C must be positive and finite")
    if class_weight not in {"none", "balanced"}:
        raise ValueError(f"unsupported class weight: {class_weight}")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    if (
        isinstance(candidate_depth, bool)
        or not isinstance(candidate_depth, int)
        or candidate_depth <= 0
    ):
        raise ValueError("candidate_depth must be a positive integer")
    pipeline = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight=None if class_weight == "none" else "balanced",
                    max_iter=max_iter,
                    random_state=random_seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipeline.fit(matrix, labels)
    model = RelevanceModel(
        pipeline=pipeline,
        candidate_depth=candidate_depth,
        c_value=c_value,
        class_weight=class_weight,
    )
    if model.classes != RELEVANCE_CLASSES:
        raise ValueError(f"trained class order is incompatible: {model.classes}")
    return model


def probability_to_expected_relevance(
    probabilities: FloatMatrix,
    classes: Sequence[int],
) -> FloatArray:
    """Compute P(Partial) + 2 * P(Exact) using the recorded class order."""

    matrix = np.asarray(probabilities, dtype=np.float64)
    normalized_classes = tuple(int(value) for value in classes)
    _validate_probabilities(matrix, normalized_classes)
    return np.asarray(matrix @ np.asarray(normalized_classes, dtype=np.float64), dtype=np.float64)


def save_relevance_model(
    model: RelevanceModel,
    output_dir: Path,
    *,
    product_dataset_sha256: str,
    source_hashes: Mapping[str, str],
    train_query_count: int,
    training_row_count: int,
    class_distribution: Mapping[int, int],
    force: bool = False,
    timestamp: datetime | None = None,
) -> RerankerMetadata:
    """Persist a joblib pipeline and atomic integrity metadata."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / MODEL_FILENAME
    metadata_path = output_dir / METADATA_FILENAME
    existing = [path.name for path in (model_path, metadata_path) if path.exists()]
    if existing and not force:
        raise FileExistsError(
            f"reranker artifacts already exist in {output_dir}: {sorted(existing)}; "
            "pass force=True to replace them"
        )
    model_temporary = output_dir / f"{MODEL_FILENAME}.part"
    metadata_temporary = output_dir / f"{METADATA_FILENAME}.part"
    for path in (model_temporary, metadata_temporary):
        path.unlink(missing_ok=True)
    try:
        joblib.dump(model.pipeline, model_temporary)
        artifact = ArtifactMetadata(
            sha256=sha256_file(model_temporary),
            byte_size=model_temporary.stat().st_size,
        )
        metadata = RerankerMetadata(
            schema_version=SCHEMA_VERSION,
            created_at=(timestamp or datetime.now(UTC)).astimezone(UTC).isoformat(),
            model_type="standard_scaler_multinomial_logistic_regression",
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            feature_names=list(FEATURE_NAMES),
            feature_definitions=dict(FEATURE_DEFINITIONS),
            feature_schema_sha256=feature_schema_sha256(),
            classes=list(model.classes),
            expected_relevance_formula="P(Partial) * 1 + P(Exact) * 2",
            candidate_depth=model.candidate_depth,
            hyperparameters={
                "C": model.c_value,
                "class_weight": model.class_weight,
                "solver": "lbfgs",
            },
            product_dataset_sha256=product_dataset_sha256,
            source_hashes=dict(source_hashes),
            training={
                "train_query_count": train_query_count,
                "training_row_count": training_row_count,
                "class_distribution": {
                    str(grade): int(count) for grade, count in sorted(class_distribution.items())
                },
            },
            package_versions={
                "joblib": joblib.__version__,
                "numpy": np.__version__,
                "scikit-learn": sklearn.__version__,
            },
            artifacts={MODEL_FILENAME: artifact},
        )
        _write_json(metadata_temporary, metadata)
        os.replace(model_temporary, model_path)
        os.replace(metadata_temporary, metadata_path)
    except BaseException:
        model_temporary.unlink(missing_ok=True)
        metadata_temporary.unlink(missing_ok=True)
        raise
    return metadata


def load_relevance_model(
    model_dir: Path,
    *,
    expected_product_dataset_sha256: str | None = None,
    expected_candidate_depth: int | None = None,
) -> RelevanceModel:
    """Load only a hash-verified, schema-compatible reranker artifact."""

    resolved = model_dir.resolve()
    metadata = _load_metadata(resolved / METADATA_FILENAME)
    _verify_artifact(resolved, metadata)
    _verify_metadata_compatibility(
        metadata,
        expected_product_dataset_sha256=expected_product_dataset_sha256,
        expected_candidate_depth=expected_candidate_depth,
    )
    try:
        loaded = joblib.load(resolved / MODEL_FILENAME)
    except Exception as error:
        raise RerankerArtifactError(f"unable to deserialize reranker model: {error}") from error
    if not isinstance(loaded, Pipeline):
        raise RerankerArtifactError("reranker artifact must contain an sklearn Pipeline")
    try:
        classifier = _classifier(loaded)
    except (TypeError, ValueError) as error:
        raise RerankerArtifactError(str(error)) from error
    classes = tuple(int(value) for value in classifier.classes_)
    if classes != tuple(metadata["classes"]) or classes != RELEVANCE_CLASSES:
        raise RerankerArtifactError("model classes are incompatible with relevance metadata")
    if int(classifier.n_features_in_) != len(FEATURE_NAMES):
        raise RerankerArtifactError("model feature dimension is incompatible with feature schema")
    hyperparameters = metadata["hyperparameters"]
    return RelevanceModel(
        pipeline=loaded,
        candidate_depth=metadata["candidate_depth"],
        c_value=float(cast(Any, hyperparameters["C"])),
        class_weight=cast(ClassWeightName, hyperparameters["class_weight"]),
        metadata=metadata,
    )


def _validate_feature_matrix(features: FloatMatrix) -> FloatMatrix:
    validate_feature_schema(FEATURE_NAMES)
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"features must have shape (rows, {len(FEATURE_NAMES)})")
    if not np.isfinite(matrix).all():
        raise ValueError("features must contain only finite values")
    return matrix


def _validate_probabilities(probabilities: FloatMatrix, classes: Sequence[int]) -> None:
    if tuple(int(value) for value in classes) != RELEVANCE_CLASSES:
        raise ValueError("probability classes must be ordered as Irrelevant, Partial, Exact")
    if probabilities.ndim != 2 or probabilities.shape[1] != len(RELEVANCE_CLASSES):
        raise ValueError("probabilities must have one column per relevance class")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-7, atol=1e-7):
        raise ValueError("each probability row must sum to one")


def _classifier(pipeline: Pipeline) -> LogisticRegression:
    scale = pipeline.named_steps.get("scale")
    classifier = pipeline.named_steps.get("classifier")
    if not isinstance(scale, StandardScaler) or not isinstance(classifier, LogisticRegression):
        raise TypeError("reranker pipeline must contain StandardScaler and LogisticRegression")
    if not hasattr(classifier, "classes_"):
        raise ValueError("reranker classifier is not fitted")
    return classifier


def _load_metadata(path: Path) -> RerankerMetadata:
    try:
        with path.open("r", encoding="utf-8") as input_file:
            payload = cast(dict[str, Any], json.load(input_file))
    except (OSError, json.JSONDecodeError) as error:
        raise RerankerArtifactError(
            f"unable to read reranker metadata at {path}: {error}"
        ) from error
    required = {
        "schema_version",
        "model_type",
        "feature_schema_version",
        "feature_names",
        "feature_schema_sha256",
        "classes",
        "candidate_depth",
        "hyperparameters",
        "product_dataset_sha256",
        "artifacts",
    }
    missing = required - set(payload)
    if missing:
        raise RerankerArtifactError(f"reranker metadata is missing fields: {sorted(missing)}")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise RerankerArtifactError(
            f"unsupported reranker metadata schema: {payload['schema_version']}"
        )
    return cast(RerankerMetadata, payload)


def _verify_artifact(model_dir: Path, metadata: RerankerMetadata) -> None:
    artifact = metadata["artifacts"].get(MODEL_FILENAME)
    path = model_dir / MODEL_FILENAME
    if artifact is None:
        raise RerankerArtifactError("metadata does not describe required model artifact")
    if not path.is_file():
        raise RerankerArtifactError(f"required reranker artifact is missing: {path}")
    if path.stat().st_size != artifact["byte_size"]:
        raise RerankerArtifactError("reranker artifact byte-size mismatch")
    if sha256_file(path) != artifact["sha256"]:
        raise RerankerArtifactError("reranker artifact SHA-256 mismatch")


def _verify_metadata_compatibility(
    metadata: RerankerMetadata,
    *,
    expected_product_dataset_sha256: str | None,
    expected_candidate_depth: int | None,
) -> None:
    try:
        validate_feature_schema(metadata["feature_names"])
    except ValueError as error:
        raise RerankerArtifactError(str(error)) from error
    if metadata["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
        raise RerankerArtifactError("reranker feature schema version is incompatible")
    if metadata["feature_schema_sha256"] != feature_schema_sha256():
        raise RerankerArtifactError("reranker feature schema hash is incompatible")
    if metadata["model_type"] != "standard_scaler_multinomial_logistic_regression":
        raise RerankerArtifactError("unsupported reranker model type")
    if expected_product_dataset_sha256 is not None and (
        metadata["product_dataset_sha256"] != expected_product_dataset_sha256
    ):
        raise RerankerArtifactError("reranker product dataset hash is incompatible")
    if expected_candidate_depth is not None and (
        metadata["candidate_depth"] != expected_candidate_depth
    ):
        raise RerankerArtifactError("reranker candidate depth is incompatible")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
