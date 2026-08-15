"""Memory-mapped NumPy semantic retrieval over normalized product embeddings."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from product_search.config import load_settings
from product_search.indexing.dense import (
    EmbeddingProvider,
    FastEmbedProvider,
    LoadedDenseIndex,
    load_dense_index,
    normalize_embeddings,
)
from product_search.retrieval.base import SearchResult


class SemanticSearchEngine:
    """Brute-force cosine search suitable for the approximately 43K-product catalog."""

    def __init__(self, index: LoadedDenseIndex, provider: EmbeddingProvider) -> None:
        if provider.model_name != index.metadata["model_name"]:
            raise ValueError(
                f"embedding provider model {provider.model_name!r} is incompatible with index "
                f"model {index.metadata['model_name']!r}"
            )
        self._embeddings = index.embeddings
        self._product_ids = index.product_ids
        self._row_by_product_id = {
            product_id: row_index for row_index, product_id in enumerate(self._product_ids)
        }
        self._provider = provider
        self.metadata = index.metadata

    @classmethod
    def from_index_dir(
        cls,
        index_dir: Path,
        *,
        provider: EmbeddingProvider,
        expected_dimension: int | None = None,
    ) -> SemanticSearchEngine:
        """Load and verify artifacts against the supplied provider before searching."""

        return cls(
            load_dense_index(
                index_dir,
                expected_model_name=provider.model_name,
                expected_dimension=expected_dimension,
            ),
            provider,
        )

    @property
    def product_ids(self) -> tuple[str, ...]:
        """Return stable product row ordering."""

        return self._product_ids

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Embed one query and partially select its highest full-catalog dot products."""

        _validate_top_k(top_k)
        query_vector = self._embed_query(query)
        if query_vector is None:
            return []
        scores = np.asarray(self._embeddings @ query_vector, dtype=np.float32)
        limit = min(top_k, len(self._product_ids))
        selected_positions = _select_top_positions(scores, self._product_ids, limit)
        return [
            _search_result(
                product_id=self._product_ids[position],
                rank=rank,
                score=float(scores[position]),
            )
            for rank, position in enumerate(selected_positions, start=1)
        ]

    def search_candidates(
        self,
        query: str,
        candidate_product_ids: Sequence[str],
        top_k: int,
    ) -> list[SearchResult]:
        """Rank only an explicit judged product set using the same dense query vector."""

        _validate_top_k(top_k)
        normalized_ids = [str(product_id) for product_id in candidate_product_ids]
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("candidate product IDs must be unique")
        unknown = sorted(set(normalized_ids) - set(self._row_by_product_id))
        if unknown:
            raise ValueError(f"candidate products are absent from the dense index: {unknown[:5]}")
        if not normalized_ids:
            return []
        query_vector = self._embed_query(query)
        if query_vector is None:
            return []
        rows = np.asarray(
            [self._row_by_product_id[product_id] for product_id in normalized_ids],
            dtype=np.int64,
        )
        scores = np.asarray(self._embeddings[rows] @ query_vector, dtype=np.float32)
        limit = min(top_k, len(normalized_ids))
        selected_positions = _select_top_positions(scores, normalized_ids, limit)
        return [
            _search_result(
                product_id=normalized_ids[position],
                rank=rank,
                score=float(scores[position]),
            )
            for rank, position in enumerate(selected_positions, start=1)
        ]

    def _embed_query(self, query: str) -> NDArray[np.float32] | None:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            return None
        vectors = list(self._provider.embed_queries([query], batch_size=1))
        if len(vectors) != 1:
            raise ValueError(f"embedding provider returned {len(vectors)} vectors for one query")
        vector = np.asarray(vectors[0], dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError("query embedding must be one-dimensional")
        normalized = normalize_embeddings(
            vector.reshape(1, -1),
            expected_dimension=self.metadata["embedding_dimension"],
        )
        return cast(NDArray[np.float32], normalized[0])


def _select_top_positions(
    scores: NDArray[np.float32],
    product_ids: Sequence[str],
    limit: int,
) -> list[int]:
    """Partially select highest scores and deterministically break ties by product ID."""

    if limit <= 0:
        return []
    if scores.size <= limit:
        candidates = list(range(scores.size))
    else:
        cutoff = float(np.partition(scores, scores.size - limit)[scores.size - limit])
        above = np.flatnonzero(scores > cutoff).tolist()
        tied = np.flatnonzero(scores == cutoff).tolist()
        tied.sort(key=lambda position: product_ids[position])
        candidates = above + tied[: limit - len(above)]
    return sorted(candidates, key=lambda position: (-scores[position], product_ids[position]))


def _search_result(*, product_id: str, rank: int, score: float) -> SearchResult:
    return SearchResult(
        product_id=product_id,
        rank=rank,
        score=score,
        score_components={"semantic": score},
    )


def _validate_top_k(top_k: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Free-text product query.")
    parser.add_argument("--top-k", type=int, help="Maximum result count.")
    parser.add_argument("--index-dir", type=Path, help="Persisted dense index directory.")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Refuse model downloads and use only the configured local model cache.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load the configured model/index and print semantic results as JSON."""

    arguments = _build_parser().parse_args(argv)
    settings = load_settings()
    provider = FastEmbedProvider(
        settings.dense.model_name,
        cache_dir=settings.paths.embeddings / "model_cache",
        local_files_only=arguments.local_files_only,
    )
    engine = SemanticSearchEngine.from_index_dir(
        arguments.index_dir or settings.paths.embeddings / "dense",
        provider=provider,
        expected_dimension=settings.dense.expected_dimension,
    )
    results = engine.search(arguments.query, arguments.top_k or settings.default_top_k)
    print(
        json.dumps(
            [
                {
                    "product_id": result.product_id,
                    "rank": result.rank,
                    "score": result.score,
                    "score_components": result.score_components,
                }
                for result in results
            ],
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
