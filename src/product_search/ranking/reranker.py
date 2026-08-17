"""Second-stage expected-relevance reranking over bounded hybrid candidates."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from product_search.config import load_settings
from product_search.indexing.dense import FastEmbedProvider
from product_search.ranking.features import (
    ProductFeatureStore,
    extract_query_product_features,
    feature_matrix,
)
from product_search.ranking.model import RelevanceScorer, load_relevance_model
from product_search.retrieval.base import JudgedCandidateSearchEngine, SearchResult
from product_search.retrieval.hybrid import HybridSearchEngine
from product_search.retrieval.lexical import LexicalSearchEngine
from product_search.retrieval.semantic import SemanticSearchEngine


class RerankingSearchEngine:
    """Generate bounded hybrid candidates, then rank by expected relevance."""

    def __init__(
        self,
        candidate_engine: JudgedCandidateSearchEngine,
        scorer: RelevanceScorer,
        product_store: ProductFeatureStore,
        *,
        candidate_depth: int,
    ) -> None:
        if (
            isinstance(candidate_depth, bool)
            or not isinstance(candidate_depth, int)
            or candidate_depth <= 0
        ):
            raise ValueError("candidate_depth must be a positive integer")
        self._candidate_engine = candidate_engine
        self._scorer = scorer
        self._product_store = product_store
        self.candidate_depth = candidate_depth

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Rerank a bounded full-catalog hybrid candidate set."""

        _validate_top_k(top_k)
        candidates = self._candidate_engine.search(query, max(top_k, self.candidate_depth))
        return self.rerank(query, candidates, top_k)

    def search_candidates(
        self,
        query: str,
        candidate_product_ids: Sequence[str],
        top_k: int,
    ) -> list[SearchResult]:
        """Rerank up to candidate_depth products from an explicit judged set."""

        _validate_top_k(top_k)
        normalized_ids = [str(product_id) for product_id in candidate_product_ids]
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("candidate product IDs must be unique")
        if not normalized_ids:
            return []
        depth = min(max(top_k, self.candidate_depth), len(normalized_ids))
        candidates = self._candidate_engine.search_candidates(query, normalized_ids, depth)
        return self.rerank(query, candidates, top_k)

    def rerank(
        self,
        query: str,
        candidates: Sequence[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """Rerank an existing candidate pool, isolating second-stage work for profiling."""

        _validate_top_k(top_k)
        if not candidates:
            return []
        _validate_candidates(candidates)
        rows = [
            extract_query_product_features(
                query,
                self._product_store.get(result.product_id),
                result,
                candidate_depth=self.candidate_depth,
            )
            for result in candidates
        ]
        matrix = feature_matrix(rows)
        probabilities = self._scorer.predict_probabilities(matrix)
        expected_scores = self._scorer.predict_expected_relevance(matrix)
        scored = sorted(
            zip(candidates, expected_scores, probabilities, strict=True),
            key=lambda item: (-float(item[1]), item[0].rank, item[0].product_id),
        )
        return [
            SearchResult(
                product_id=candidate.product_id,
                rank=rank,
                score=float(expected_score),
                score_components={
                    **(candidate.score_components or {}),
                    "hybrid_original_rank": float(candidate.rank),
                    "reranker_probability_irrelevant": float(probability[0]),
                    "reranker_probability_partial": float(probability[1]),
                    "reranker_probability_exact": float(probability[2]),
                    "reranker_expected_relevance": float(expected_score),
                },
            )
            for rank, (candidate, expected_score, probability) in enumerate(scored[:top_k], start=1)
        ]


def _validate_candidates(candidates: Sequence[SearchResult]) -> None:
    product_ids = [result.product_id for result in candidates]
    if len(set(product_ids)) != len(product_ids):
        raise ValueError("hybrid candidates contain duplicate product IDs")
    if [result.rank for result in candidates] != list(range(1, len(candidates) + 1)):
        raise ValueError("hybrid candidate ranks must be contiguous from one")


def _validate_top_k(top_k: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Free-text product query.")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--products", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--lexical-index-dir", type=Path)
    parser.add_argument("--dense-index-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load verified local artifacts and print reranked search results."""

    arguments = _build_parser().parse_args(argv)
    settings = load_settings()
    products_path = arguments.products or settings.paths.processed_data / "products.parquet"
    product_store = ProductFeatureStore.from_parquet(products_path)
    model = load_relevance_model(
        arguments.model_dir or settings.paths.models / "reranker",
        expected_product_dataset_sha256=product_store.dataset_sha256,
        expected_candidate_depth=settings.reranker.candidate_depth,
    )
    provider = FastEmbedProvider(
        settings.dense.model_name,
        cache_dir=settings.paths.embeddings / "model_cache",
        local_files_only=arguments.local_files_only,
    )
    lexical = LexicalSearchEngine.from_index_dir(
        arguments.lexical_index_dir or settings.paths.indexes / "tfidf"
    )
    semantic = SemanticSearchEngine.from_index_dir(
        arguments.dense_index_dir or settings.paths.embeddings / "dense",
        provider=provider,
        expected_dimension=settings.dense.expected_dimension,
    )
    hybrid = HybridSearchEngine(
        lexical,
        semantic,
        strategy=settings.hybrid.strategy,
        semantic_weight=settings.hybrid.semantic_weight,
        candidate_depth=settings.hybrid.candidate_depth,
        rrf_k=settings.hybrid.rrf_k,
    )
    engine = RerankingSearchEngine(
        hybrid,
        model,
        product_store,
        candidate_depth=settings.reranker.candidate_depth,
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
