"""Tests for train-only fitting and validation-only reranker selection."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pandas import DataFrame

from product_search.evaluation import benchmark_reranker as benchmark_module
from product_search.evaluation.benchmark_reranker import (
    _configurations,
    _precomputed_engine,
    _PrecomputedRankingEngine,
    run_reranker_validation_experiment,
)
from product_search.indexing import dense as dense_module
from product_search.ranking.features import ProductFeatureStore
from product_search.retrieval import hybrid as hybrid_module
from product_search.retrieval import lexical as lexical_module
from product_search.retrieval import semantic as semantic_module
from product_search.retrieval.base import SearchResult


class RecordingHybridEngine:
    def __init__(self) -> None:
        self.seen_queries: list[str] = []
        self.scores = {
            "alpha table": {"p1": 0.9, "p2": 0.5, "p3": 0.1},
            "beta chair": {"p1": 0.4, "p2": 0.8, "p3": 0.2},
            "gamma rug": {"p4": 0.8, "p5": 0.4, "p6": 0.1},
        }

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        self.seen_queries.append(query)
        return self._rank(query, list(self.scores[query]), top_k)

    def search_candidates(
        self,
        query: str,
        candidate_product_ids: Sequence[str],
        top_k: int,
    ) -> list[SearchResult]:
        self.seen_queries.append(query)
        return self._rank(query, candidate_product_ids, top_k)

    def _rank(self, query: str, product_ids: Sequence[str], top_k: int) -> list[SearchResult]:
        ordered = sorted(
            product_ids,
            key=lambda product_id: (-self.scores[query][product_id], product_id),
        )[:top_k]
        return [
            SearchResult(
                product_id=product_id,
                rank=rank,
                score=self.scores[query][product_id],
                score_components={
                    "lexical_raw": self.scores[query][product_id],
                    "semantic_raw": self.scores[query][product_id] * 0.9,
                    "lexical_rank": float(rank),
                    "semantic_rank": float(rank),
                    "lexical_present": 1.0,
                    "semantic_present": 1.0,
                    "hybrid": self.scores[query][product_id],
                },
            )
            for rank, product_id in enumerate(ordered, start=1)
        ]


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, ProductFeatureStore]:
    queries_path = tmp_path / "queries.parquet"
    judgments_path = tmp_path / "evaluation_judgments.parquet"
    splits_path = tmp_path / "query_splits.json"
    DataFrame(
        {
            "query_id": ["tr1", "tr2", "v1", "t1"],
            "query": ["alpha table", "beta chair", "gamma rug", "held out lamp"],
        }
    ).to_parquet(queries_path, index=False)
    DataFrame(
        {
            "query_id": ["tr1"] * 3 + ["tr2"] * 3 + ["v1"] * 3 + ["t1"],
            "product_id": ["p1", "p2", "p3"] * 2 + ["p4", "p5", "p6", "p6"],
            "relevance_grade": [2, 1, 0, 1, 2, 0, 2, 1, 0, 2],
        }
    ).to_parquet(judgments_path, index=False)
    splits_path.write_text(
        json.dumps(
            {
                "query_ids": {
                    "train": ["tr1", "tr2"],
                    "validation": ["v1"],
                    "test": ["t1"],
                }
            }
        ),
        encoding="utf-8",
    )
    store = ProductFeatureStore.from_frame(
        DataFrame(
            {
                "product_id": [f"p{index}" for index in range(1, 7)],
                "product_name": [
                    "Alpha table",
                    "Beta chair",
                    "Other",
                    "Gamma rug",
                    "Rug pad",
                    "Lamp",
                ],
                "product_description": ["wood", "seat", "other", "floor", "soft", "light"],
                "product_text": [
                    "alpha table wood",
                    "beta chair seat",
                    "other",
                    "gamma rug floor",
                    "rug pad soft",
                    "lamp light",
                ],
            }
        ),
        dataset_sha256="fixture-products-hash",
    )
    return queries_path, judgments_path, splits_path, store


def test_reranker_experiment_fits_train_selects_validation_and_never_accesses_test(
    tmp_path: Path,
) -> None:
    queries_path, judgments_path, splits_path, store = _fixture(tmp_path)
    engine = RecordingHybridEngine()
    model_dir = tmp_path / "models" / "reranker"
    search_path = tmp_path / "reports" / "reranker_model_search.csv"
    report_path = tmp_path / "reports" / "reranker_validation_metrics.json"

    report = run_reranker_validation_experiment(
        hybrid_engine=engine,
        product_store=store,
        queries_path=queries_path,
        judgments_path=judgments_path,
        splits_path=splits_path,
        model_dir=model_dir,
        model_search_path=search_path,
        report_path=report_path,
        c_grid=[0.1, 1.0],
        class_weight_options=["none", "balanced"],
        max_iter=500,
        random_seed=42,
        candidate_depth=3,
        timestamp=datetime(2026, 8, 16, 12, 0, tzinfo=UTC),
    )

    assert report["training_split"] == "train"
    assert report["selection_split"] == "validation"
    assert report["train_query_count"] == 2
    assert report["validation_query_count"] == 1
    assert report["test_query_count_evaluated"] == 0
    assert "held out lamp" not in engine.seen_queries
    assert report["model_search"]["configuration_count"] == 4  # type: ignore[index]
    assert report["validation_classification"]["confusion_matrix"]  # type: ignore[index]
    assert report["feature_coefficients"]["by_relevance_grade"]  # type: ignore[index]
    assert report["production_decision"]["recommended_default_search_mode"] in {  # type: ignore[index]
        "hybrid",
        "reranker",
    }
    assert (model_dir / "model.joblib").is_file()
    assert (model_dir / "metadata.json").is_file()
    with search_path.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    assert len(rows) == 4
    assert {row["test_query_count_evaluated"] for row in rows} == {"0"}
    assert json.loads(report_path.read_text(encoding="utf-8"))["selection_split"] == "validation"


def test_reranker_experiment_rejects_split_overlap(tmp_path: Path) -> None:
    queries_path, judgments_path, splits_path, store = _fixture(tmp_path)
    splits_path.write_text(
        json.dumps(
            {
                "query_ids": {
                    "train": ["tr1"],
                    "validation": ["v1"],
                    "test": ["v1"],
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be disjoint"):
        run_reranker_validation_experiment(
            hybrid_engine=RecordingHybridEngine(),
            product_store=store,
            queries_path=queries_path,
            judgments_path=judgments_path,
            splits_path=splits_path,
            model_dir=tmp_path / "model",
            model_search_path=tmp_path / "search.csv",
            report_path=tmp_path / "report.json",
            c_grid=[1.0],
            class_weight_options=["none"],
            max_iter=500,
            random_seed=42,
            candidate_depth=3,
        )


def test_reranker_benchmark_cli_wires_configured_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    provider = object()
    lexical = object()
    semantic = object()
    hybrid = object()
    store = object()
    captured: dict[str, object] = {}

    monkeypatch.setattr(dense_module, "FastEmbedProvider", lambda *args, **kwargs: provider)
    monkeypatch.setattr(
        lexical_module.LexicalSearchEngine,
        "from_index_dir",
        lambda *args, **kwargs: lexical,
    )
    monkeypatch.setattr(
        semantic_module.SemanticSearchEngine,
        "from_index_dir",
        lambda *args, **kwargs: semantic,
    )
    monkeypatch.setattr(hybrid_module, "HybridSearchEngine", lambda *args, **kwargs: hybrid)
    monkeypatch.setattr(
        benchmark_module.ProductFeatureStore,
        "from_parquet",
        lambda *args, **kwargs: store,
    )

    def fake_run(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"selection_split": "validation", "test_query_count_evaluated": 0}

    monkeypatch.setattr(benchmark_module, "run_reranker_validation_experiment", fake_run)

    assert benchmark_module.main(["--local-files-only"]) == 0
    assert captured["hybrid_engine"] is hybrid
    assert captured["product_store"] is store
    assert '"test_query_count_evaluated": 0' in capsys.readouterr().out


@pytest.mark.parametrize(
    ("c_grid", "weights", "message"),
    [
        ([], ["none"], "positive finite"),
        ([1.0, 0.1], ["none"], "unique and ascending"),
        ([1.0], [], "non-empty and unique"),
        ([1.0], ["invalid"], "unsupported value"),
    ],
)
def test_reranker_benchmark_rejects_invalid_model_search_grids(
    c_grid: list[float], weights: list[str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _configurations(c_grid, weights)  # type: ignore[arg-type]


def test_precomputed_ranking_engine_validates_queries_and_score_alignment() -> None:
    result = SearchResult("p1", 1, 1.0)
    engine = _PrecomputedRankingEngine({"query": [result]})

    assert engine.search("query", 1) == [result]
    with pytest.raises(ValueError, match="no precomputed"):
        engine.search("missing", 1)
    rows = DataFrame({"query": ["query"], "product_id": ["p1"], "hybrid_rank": [1]})
    with pytest.raises(ValueError, match="align"):
        _precomputed_engine(rows, [])
