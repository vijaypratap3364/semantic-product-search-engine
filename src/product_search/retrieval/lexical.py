"""Sparse TF-IDF lexical retrieval for full-catalog and judged-candidate search."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.sparse import csr_matrix

from product_search.config import load_settings
from product_search.indexing.tfidf import LoadedTfidfIndex, load_tfidf_index
from product_search.retrieval.base import SearchResult


class LexicalSearchEngine:
    """Query an already-fitted, L2-normalized sparse product index."""

    def __init__(self, index: LoadedTfidfIndex) -> None:
        self._vectorizer = index.vectorizer
        self._product_matrix = index.product_matrix
        self._product_ids = index.product_ids
        self._row_by_product_id = {
            product_id: row_index for row_index, product_id in enumerate(self._product_ids)
        }
        self.metadata = index.metadata

    @classmethod
    def from_index_dir(cls, index_dir: Path) -> LexicalSearchEngine:
        """Load and verify persisted artifacts before constructing the engine."""

        return cls(load_tfidf_index(index_dir))

    @property
    def product_ids(self) -> tuple[str, ...]:
        """Return the stable product row ordering."""

        return self._product_ids

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Search the full catalog using sparse cosine-equivalent dot products."""

        _validate_top_k(top_k)
        query_matrix = self._transform_query(query)
        if query_matrix is None:
            return []

        # The product matrix is already CSR. Multiplying it by a dense query vector avoids
        # converting its large transpose on every request while keeping the product matrix sparse.
        query_vector = query_matrix.toarray().reshape(-1).astype(np.float32, copy=False)
        similarities = np.asarray(self._product_matrix @ query_vector, dtype=np.float32)
        rows = np.flatnonzero(similarities).astype(np.int64, copy=False)
        scores = similarities[rows].astype(np.float64, copy=False)
        if scores.size == 0:
            return []
        selected_positions = _select_top_positions(
            scores,
            [self._product_ids[int(row)] for row in rows],
            min(top_k, scores.size),
        )
        return [
            _search_result(
                product_id=self._product_ids[int(rows[position])],
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
        """Rank only an explicit product set, retaining zero-score candidates for evaluation."""

        _validate_top_k(top_k)
        normalized_ids = [str(product_id) for product_id in candidate_product_ids]
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("candidate product IDs must be unique")
        unknown = sorted(set(normalized_ids) - set(self._row_by_product_id))
        if unknown:
            raise ValueError(f"candidate products are absent from the TF-IDF index: {unknown[:5]}")
        if not normalized_ids:
            return []

        query_matrix = self._transform_query(query)
        if query_matrix is None:
            return []
        candidate_rows = np.asarray(
            [self._row_by_product_id[product_id] for product_id in normalized_ids],
            dtype=np.int64,
        )
        candidate_matrix = self._product_matrix[candidate_rows]
        # Candidate sets are small WANDS judgment groups. Densifying this one score vector does
        # not densify the product-feature matrix or the full catalog score vector.
        candidate_scores = cast(csr_matrix, (candidate_matrix @ query_matrix.T).tocsr())
        scores = candidate_scores.toarray().reshape(-1).astype(np.float64)
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

    def vocabulary_analysis(self, query: str) -> dict[str, tuple[str, ...]]:
        """Return analyzer tokens split into indexed and out-of-vocabulary groups."""

        tokens = tuple(self._vectorizer.build_analyzer()(query))
        vocabulary = self._vectorizer.vocabulary_
        return {
            "tokens": tokens,
            "indexed_tokens": tuple(token for token in tokens if token in vocabulary),
            "out_of_vocabulary_tokens": tuple(token for token in tokens if token not in vocabulary),
        }

    def _transform_query(self, query: str) -> csr_matrix | None:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            return None
        transformed = cast(csr_matrix, self._vectorizer.transform([query]))
        return transformed if transformed.nnz else None


def _select_top_positions(
    scores: np.ndarray[Any, np.dtype[np.float64]],
    product_ids: Sequence[str],
    limit: int,
) -> list[int]:
    """Partially select top scores, then apply stable product-ID tie-breaking."""

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
        score_components={"lexical": score},
    )


def _validate_top_k(top_k: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Free-text product query.")
    parser.add_argument("--top-k", type=int, help="Maximum result count.")
    parser.add_argument("--index-dir", type=Path, help="Persisted TF-IDF index directory.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load the configured index and print ranked product IDs and scores."""

    arguments = _build_parser().parse_args(argv)
    settings = load_settings()
    engine = LexicalSearchEngine.from_index_dir(
        arguments.index_dir or settings.paths.indexes / "tfidf"
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
