"""Tests for deterministic NumPy semantic retrieval."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from pandas import DataFrame

from product_search.config import DenseSettings
from product_search.indexing.dense import build_dense_index
from product_search.retrieval import semantic as semantic_module
from product_search.retrieval.semantic import SemanticSearchEngine


class FakeEmbeddingProvider:
    model_name = "fake/semantic"
    provider_name = "fake"
    provider_version = "1.0"

    def __init__(self) -> None:
        self.vectors = {
            "same-a": [1.0, 0.0, 0.0],
            "same-b": [1.0, 0.0, 0.0],
            "middle": [0.5, 0.5, 0.0],
            "other": [0.0, 1.0, 0.0],
            "query-a": [2.0, 0.0, 0.0],
        }

    def embed_documents(
        self, texts: Sequence[str], *, batch_size: int
    ) -> Iterable[NDArray[np.float32]]:
        return (np.asarray(self.vectors[text], dtype=np.float32) for text in texts)

    def embed_queries(
        self, texts: Sequence[str], *, batch_size: int
    ) -> Iterable[NDArray[np.float32]]:
        return (np.asarray(self.vectors[text], dtype=np.float32) for text in texts)


@pytest.fixture
def semantic_engine(tmp_path: Path) -> SemanticSearchEngine:
    products_path = tmp_path / "products.parquet"
    DataFrame(
        {
            "product_id": ["p3", "p1", "p2", "p0"],
            "product_text": ["other", "same-a", "middle", "same-b"],
        }
    ).to_parquet(products_path, index=False)
    index_dir = tmp_path / "dense"
    provider = FakeEmbeddingProvider()
    build_dense_index(
        products_path,
        index_dir,
        provider=provider,
        settings=DenseSettings(model_name="fake/semantic", expected_dimension=3, batch_size=2),
    )
    return SemanticSearchEngine.from_index_dir(
        index_dir,
        provider=provider,
        expected_dimension=3,
    )


def test_semantic_search_orders_results_and_breaks_ties_by_id(
    semantic_engine: SemanticSearchEngine,
) -> None:
    results = semantic_engine.search("query-a", top_k=3)

    assert [result.product_id for result in results] == ["p0", "p1", "p2"]
    assert [result.rank for result in results] == [1, 2, 3]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(1.0)
    assert results[2].score < results[1].score
    assert results[0].score_components == {"semantic": results[0].score}
    assert semantic_engine.product_ids == ("p0", "p1", "p2", "p3")


def test_semantic_candidate_search_and_top_k(semantic_engine: SemanticSearchEngine) -> None:
    results = semantic_engine.search_candidates("query-a", ["p3", "p2", "p1"], top_k=2)

    assert [result.product_id for result in results] == ["p1", "p2"]
    assert len(semantic_engine.search("query-a", top_k=99)) == 4


def test_empty_query_returns_no_results(semantic_engine: SemanticSearchEngine) -> None:
    assert semantic_engine.search("   ", top_k=10) == []
    assert semantic_engine.search_candidates("", ["p1"], top_k=10) == []


@pytest.mark.parametrize("top_k", [0, -1, True])
def test_semantic_invalid_top_k_fails(semantic_engine: SemanticSearchEngine, top_k: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        semantic_engine.search("query-a", top_k)


def test_semantic_unknown_or_duplicate_candidates_fail(
    semantic_engine: SemanticSearchEngine,
) -> None:
    with pytest.raises(ValueError, match="absent from the dense index"):
        semantic_engine.search_candidates("query-a", ["missing"], 10)
    with pytest.raises(ValueError, match="must be unique"):
        semantic_engine.search_candidates("query-a", ["p1", "p1"], 10)


def test_semantic_rejects_provider_for_different_model(
    semantic_engine: SemanticSearchEngine,
) -> None:
    different = FakeEmbeddingProvider()
    different.model_name = "fake/different"

    with pytest.raises(ValueError, match="incompatible with index model"):
        SemanticSearchEngine(
            type(
                "Index",
                (),
                {
                    "embeddings": semantic_engine._embeddings,
                    "product_ids": semantic_engine.product_ids,
                    "metadata": semantic_engine.metadata,
                },
            )(),
            different,
        )


def test_semantic_search_cli_prints_ranked_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    products_path = tmp_path / "products.parquet"
    DataFrame({"product_id": ["p1"], "product_text": ["document"]}).to_parquet(
        products_path, index=False
    )
    vector = [1.0, *([0.0] * 383)]
    provider = FakeEmbeddingProvider()
    provider.model_name = "BAAI/bge-small-en-v1.5"
    provider.vectors = {"document": vector, "query": vector}
    index_dir = tmp_path / "dense"
    build_dense_index(
        products_path,
        index_dir,
        provider=provider,
        settings=DenseSettings(
            model_name="BAAI/bge-small-en-v1.5",
            expected_dimension=384,
            batch_size=1,
        ),
    )
    monkeypatch.setattr(
        semantic_module,
        "FastEmbedProvider",
        lambda *args, **kwargs: provider,
    )

    exit_code = semantic_module.main(
        ["query", "--top-k", "1", "--index-dir", str(index_dir), "--local-files-only"]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert '"product_id": "p1"' in output
    assert '"rank": 1' in output
