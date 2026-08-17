"""Tests for the one-time validation-frozen held-out benchmark."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pandas import DataFrame

from product_search.config import load_settings
from product_search.data.download import sha256_file
from product_search.evaluation import benchmark_final as benchmark_module
from product_search.evaluation.benchmark_final import (
    freeze_final_configurations,
    run_final_test_benchmark,
    select_default_engine,
)
from product_search.retrieval.base import SearchResult


class RecordingEngine:
    def __init__(self, rankings: dict[str, list[str]]) -> None:
        self.rankings = rankings
        self.seen_queries: list[str] = []

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        self.seen_queries.append(query)
        return self._results(self.rankings[query][:top_k])

    def search_candidates(
        self,
        query: str,
        candidate_product_ids: Sequence[str],
        top_k: int,
    ) -> list[SearchResult]:
        self.seen_queries.append(query)
        allowed = set(candidate_product_ids)
        ordered = [product_id for product_id in self.rankings[query] if product_id in allowed]
        return self._results(ordered[:top_k])

    @staticmethod
    def _results(product_ids: Sequence[str]) -> list[SearchResult]:
        return [
            SearchResult(product_id=product_id, rank=rank, score=1.0 / rank)
            for rank, product_id in enumerate(product_ids, start=1)
        ]


class StepClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.001
        return self.value


def _evaluation_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    queries_path = tmp_path / "queries.parquet"
    judgments_path = tmp_path / "evaluation_judgments.parquet"
    splits_path = tmp_path / "query_splits.json"
    DataFrame(
        {
            "query_id": ["train", "validation", "test-1", "test-2"],
            "query": ["train query", "validation query", "test lamp", "test table"],
        }
    ).to_parquet(queries_path, index=False)
    DataFrame(
        {
            "query_id": ["train", "validation"] + ["test-1"] * 3 + ["test-2"] * 3,
            "product_id": ["p1", "p1", "p1", "p2", "p3", "p1", "p2", "p3"],
            "relevance_grade": [2, 2, 2, 1, 0, 2, 1, 0],
        }
    ).to_parquet(judgments_path, index=False)
    splits_path.write_text(
        json.dumps(
            {
                "query_ids": {
                    "train": ["train"],
                    "validation": ["validation"],
                    "test": ["test-1", "test-2"],
                }
            }
        ),
        encoding="utf-8",
    )
    return queries_path, judgments_path, splits_path


def _frozen(*, reranker_eligible: bool = True) -> dict[str, object]:
    return {
        "schema_version": 1,
        "frozen_before_test_evaluation": True,
        "source_hashes": {"query_splits.json": "split-hash"},
        "lexical": {"parameters": "frozen"},
        "semantic": {"model": "frozen"},
        "hybrid": {"semantic_weight": 0.9},
        "reranked_hybrid": {"eligible": reranker_eligible, "model": "frozen"},
    }


def _engines() -> dict[str, RecordingEngine]:
    rankings = {
        "lexical": ["p3", "p2", "p1"],
        "semantic": ["p2", "p1", "p3"],
        "hybrid": ["p2", "p1", "p3"],
        "reranked_hybrid": ["p1", "p2", "p3"],
    }
    return {
        system: RecordingEngine({"test lamp": ordering, "test table": ordering})
        for system, ordering in rankings.items()
    }


def test_final_benchmark_uses_only_test_queries_and_writes_reports_and_charts(
    tmp_path: Path,
) -> None:
    queries_path, judgments_path, splits_path = _evaluation_fixture(tmp_path)
    engines = _engines()
    output_dir = tmp_path / "reports"

    report = run_final_test_benchmark(
        engines=engines,
        frozen_configurations=_frozen(),
        queries_path=queries_path,
        judgments_path=judgments_path,
        splits_path=splits_path,
        output_dir=output_dir,
        timestamp=datetime(2026, 8, 16, 12, 0, tzinfo=UTC),
        clock=StepClock(),
        warmup_query_count=1,
        hardware={
            "processor": "fixture CPU",
            "logical_cpu_count": 4,
            "platform": "fixture platform",
        },
    )

    assert report["split"] == "test"
    assert report["test_query_count"] == 2
    assert report["train_query_count_evaluated"] == 0
    assert report["validation_query_count_evaluated"] == 0
    assert report["final_engine"]["system"] == "reranked_hybrid"  # type: ignore[index]
    assert report["reranker_included"] is True
    assert all(
        set(engine.seen_queries) <= {"test lamp", "test table"} for engine in engines.values()
    )
    assert (output_dir / "final_test_metrics.json").is_file()
    assert (output_dir / "final_test_metrics.csv").is_file()
    assert (output_dir / "final_test_per_query_metrics.csv").is_file()
    assert "Selected default" in (output_dir / "final_comparison.md").read_text(encoding="utf-8")
    for chart in (
        "final_system_comparison.svg",
        "final_ndcg_distribution.svg",
        "final_latency_comparison.svg",
    ):
        content = (output_dir / chart).read_text(encoding="utf-8")
        assert content.startswith("<svg")
    with (output_dir / "final_test_metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["system"] for row in rows] == [
        "lexical",
        "semantic",
        "hybrid",
        "reranked_hybrid",
    ]
    assert (
        json.loads((output_dir / "final_engine.json").read_text(encoding="utf-8"))[
            "selected_search_mode"
        ]
        == "reranker"
    )


def test_final_benchmark_enforces_reranker_eligibility_and_immutable_outputs(
    tmp_path: Path,
) -> None:
    queries_path, judgments_path, splits_path = _evaluation_fixture(tmp_path)
    engines = _engines()
    with pytest.raises(ValueError, match="validation eligibility"):
        run_final_test_benchmark(
            engines=engines,
            frozen_configurations=_frozen(reranker_eligible=False),
            queries_path=queries_path,
            judgments_path=judgments_path,
            splits_path=splits_path,
            output_dir=tmp_path / "reports",
        )

    eligible_engines = {
        name: engine for name, engine in engines.items() if name != "reranked_hybrid"
    }
    output_dir = tmp_path / "eligible-reports"
    run_final_test_benchmark(
        engines=eligible_engines,
        frozen_configurations=_frozen(reranker_eligible=False),
        queries_path=queries_path,
        judgments_path=judgments_path,
        splits_path=splits_path,
        output_dir=output_dir,
        clock=StepClock(),
        warmup_query_count=0,
        hardware={
            "processor": "fixture CPU",
            "logical_cpu_count": 4,
            "platform": "fixture platform",
        },
    )
    with pytest.raises(FileExistsError, match="immutable"):
        run_final_test_benchmark(
            engines=eligible_engines,
            frozen_configurations=_frozen(reranker_eligible=False),
            queries_path=queries_path,
            judgments_path=judgments_path,
            splits_path=splits_path,
            output_dir=output_dir,
        )


def test_final_benchmark_rejects_query_split_overlap(tmp_path: Path) -> None:
    queries_path, judgments_path, splits_path = _evaluation_fixture(tmp_path)
    splits_path.write_text(
        json.dumps(
            {
                "query_ids": {
                    "train": ["train"],
                    "validation": ["test-1"],
                    "test": ["test-1", "test-2"],
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must be disjoint"):
        run_final_test_benchmark(
            engines=_engines(),
            frozen_configurations=_frozen(),
            queries_path=queries_path,
            judgments_path=judgments_path,
            splits_path=splits_path,
            output_dir=tmp_path / "reports",
        )


def test_default_selection_prioritizes_ndcg_and_uses_latency_only_for_ties() -> None:
    rows: list[dict[str, object]] = [
        {"system": "lexical", "ndcg_at_10": 0.8, "median_latency_ms": 1.0},
        {"system": "semantic", "ndcg_at_10": 0.81, "median_latency_ms": 100.0},
    ]
    assert select_default_engine(rows)["system"] == "semantic"
    rows[0]["ndcg_at_10"] = 0.81
    assert select_default_engine(rows)["system"] == "lexical"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _frozen_artifact_fixture(tmp_path: Path) -> tuple[dict[str, Path], object]:
    settings = load_settings()
    paths = {
        "products": tmp_path / "products.parquet",
        "queries": tmp_path / "queries.parquet",
        "judgments": tmp_path / "evaluation_judgments.parquet",
        "splits": tmp_path / "query_splits.json",
        "lexical": tmp_path / "lexical_metadata.json",
        "dense": tmp_path / "dense_metadata.json",
        "reranker": tmp_path / "reranker_metadata.json",
        "hybrid_report": tmp_path / "hybrid_validation_metrics.json",
        "reranker_report": tmp_path / "reranker_validation_metrics.json",
    }
    for key in ("products", "queries", "judgments", "splits"):
        paths[key].write_bytes(f"fixture-{key}".encode())
    hashes = {
        "products.parquet": sha256_file(paths["products"]),
        "queries.parquet": sha256_file(paths["queries"]),
        "evaluation_judgments.parquet": sha256_file(paths["judgments"]),
        "query_splits.json": sha256_file(paths["splits"]),
    }
    source_hashes = {
        name: hashes[name]
        for name in ("queries.parquet", "evaluation_judgments.parquet", "query_splits.json")
    }
    _write_json(
        paths["lexical"],
        {
            "dataset_sha256": hashes["products.parquet"],
            "product_count": 3,
            "vocabulary_size": 10,
            "vectorizer_parameters": {
                "analyzer": "word",
                "dtype": "float32",
                "lowercase": True,
                "max_features": 100000,
                "min_df": 2,
                "ngram_range": [1, 2],
                "norm": "l2",
                "sublinear_tf": True,
            },
            "artifacts": {"matrix": {"sha256": "lexical-artifact"}},
        },
    )
    _write_json(
        paths["dense"],
        {
            "dataset_sha256": hashes["products.parquet"],
            "model_name": "BAAI/bge-small-en-v1.5",
            "embedding_dimension": 384,
            "embedding_normalization": "project_applied_l2_unit_normalization",
            "embedding_dtype": "float32",
            "product_count": 3,
            "artifacts": {"embeddings": {"sha256": "dense-artifact"}},
        },
    )
    _write_json(
        paths["reranker"],
        {
            "product_dataset_sha256": hashes["products.parquet"],
            "candidate_depth": 100,
            "classes": [0, 1, 2],
            "expected_relevance_formula": "P(Partial) * 1 + P(Exact) * 2",
            "model_type": "standard_scaler_multinomial_logistic_regression",
            "feature_schema_sha256": "feature-schema",
            "hyperparameters": {"C": 1.0, "class_weight": "balanced", "solver": "lbfgs"},
            "source_hashes": source_hashes,
            "artifacts": {"model.joblib": {"sha256": "model-artifact"}},
        },
    )
    _write_json(
        paths["hybrid_report"],
        {
            "split": "validation",
            "test_query_count_evaluated": 0,
            "source_hashes": source_hashes,
            "selection_metric": "validation_judged_candidate_ndcg_at_10",
            "selected_configuration": {
                "strategy": "weighted_normalized",
                "semantic_weight": 0.9,
                "candidate_depth": 100,
                "rrf_k": 60,
            },
        },
    )
    _write_json(
        paths["reranker_report"],
        {
            "selection_split": "validation",
            "test_query_count_evaluated": 0,
            "source_hashes": source_hashes,
            "production_decision": {
                "reranker_improves_validation_ndcg_at_10": True,
                "recommended_default_search_mode": "reranker",
                "policy": "strict validation improvement",
            },
            "model_search": {"selected_configuration": {"C": 1.0, "class_weight": "balanced"}},
        },
    )
    return paths, settings


def test_frozen_configuration_accepts_validation_artifacts_and_rejects_parameter_drift(
    tmp_path: Path,
) -> None:
    paths, settings = _frozen_artifact_fixture(tmp_path)
    kwargs = {
        "products_path": paths["products"],
        "queries_path": paths["queries"],
        "judgments_path": paths["judgments"],
        "splits_path": paths["splits"],
        "lexical_metadata_path": paths["lexical"],
        "dense_metadata_path": paths["dense"],
        "reranker_metadata_path": paths["reranker"],
        "hybrid_validation_report_path": paths["hybrid_report"],
        "reranker_validation_report_path": paths["reranker_report"],
    }
    frozen = freeze_final_configurations(settings, **kwargs)  # type: ignore[arg-type]
    assert frozen["frozen_before_test_evaluation"] is True
    assert frozen["reranked_hybrid"]["eligible"] is True  # type: ignore[index]

    hybrid_report = json.loads(paths["hybrid_report"].read_text(encoding="utf-8"))
    hybrid_report["selected_configuration"]["semantic_weight"] = 0.8
    _write_json(paths["hybrid_report"], hybrid_report)
    with pytest.raises(ValueError, match="semantic weight mismatch"):
        freeze_final_configurations(settings, **kwargs)  # type: ignore[arg-type]


def test_frozen_configuration_rejects_prior_test_evaluation(tmp_path: Path) -> None:
    paths, settings = _frozen_artifact_fixture(tmp_path)
    report = json.loads(paths["reranker_report"].read_text(encoding="utf-8"))
    report["test_query_count_evaluated"] = 1
    _write_json(paths["reranker_report"], report)
    with pytest.raises(ValueError, match="test query count mismatch"):
        freeze_final_configurations(
            settings,
            products_path=paths["products"],
            queries_path=paths["queries"],
            judgments_path=paths["judgments"],
            splits_path=paths["splits"],
            lexical_metadata_path=paths["lexical"],
            dense_metadata_path=paths["dense"],
            reranker_metadata_path=paths["reranker"],
            hybrid_validation_report_path=paths["hybrid_report"],
            reranker_validation_report_path=paths["reranker_report"],
        )


def test_final_benchmark_cli_verifies_without_loading_search_engines(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        benchmark_module, "freeze_final_configurations", lambda *args, **kwargs: _frozen()
    )

    assert benchmark_module.main(["--verify-only"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["frozen_before_test_evaluation"] is True


def test_final_benchmark_cli_wires_validation_frozen_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from product_search.indexing import dense as dense_module
    from product_search.ranking import features as features_module
    from product_search.ranking import model as model_module
    from product_search.ranking import reranker as reranker_module
    from product_search.retrieval import hybrid as hybrid_module
    from product_search.retrieval import lexical as lexical_module
    from product_search.retrieval import semantic as semantic_module

    provider = object()
    lexical = object()
    semantic = object()
    hybrid = object()
    model = object()
    reranker = object()
    store = type("Store", (), {"dataset_sha256": "products-hash"})()
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        benchmark_module, "freeze_final_configurations", lambda *args, **kwargs: _frozen()
    )
    monkeypatch.setattr(benchmark_module, "_assert_outputs_absent", lambda path: None)
    monkeypatch.setattr(dense_module, "FastEmbedProvider", lambda *args, **kwargs: provider)
    monkeypatch.setattr(lexical_module.LexicalSearchEngine, "from_index_dir", lambda *args: lexical)
    monkeypatch.setattr(
        semantic_module.SemanticSearchEngine,
        "from_index_dir",
        lambda *args, **kwargs: semantic,
    )
    monkeypatch.setattr(hybrid_module, "HybridSearchEngine", lambda *args, **kwargs: hybrid)
    monkeypatch.setattr(
        features_module.ProductFeatureStore,
        "from_parquet",
        lambda *args: store,
    )
    monkeypatch.setattr(model_module, "load_relevance_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(reranker_module, "RerankingSearchEngine", lambda *args, **kwargs: reranker)

    def fake_run(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"split": "test", "test_query_count": 2}

    monkeypatch.setattr(benchmark_module, "run_final_test_benchmark", fake_run)

    assert benchmark_module.main(["--local-files-only"]) == 0

    assert captured["engines"] == {
        "lexical": lexical,
        "semantic": semantic,
        "hybrid": hybrid,
        "reranked_hybrid": reranker,
    }
    assert json.loads(capsys.readouterr().out)["split"] == "test"
