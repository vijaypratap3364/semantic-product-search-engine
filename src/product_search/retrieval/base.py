"""Shared result and engine contracts for every retrieval implementation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One scored product returned by a search engine."""

    product_id: str
    rank: int
    score: float
    score_components: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if not self.product_id.strip():
            raise ValueError("product_id must not be blank")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank <= 0:
            raise ValueError("rank must be a positive integer")
        if not math.isfinite(self.score):
            raise ValueError("score must be finite")
        if self.score_components is not None and any(
            not math.isfinite(value) for value in self.score_components.values()
        ):
            raise ValueError("score components must be finite")


@runtime_checkable
class SearchEngine(Protocol):
    """Minimum full-catalog search interface used by evaluation and serving."""

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return at most ``top_k`` full-catalog results ordered by rank."""
        ...


@runtime_checkable
class JudgedCandidateSearchEngine(SearchEngine, Protocol):
    """Optional extension for controlled ranking of an explicit candidate set."""

    def search_candidates(
        self,
        query: str,
        candidate_product_ids: Sequence[str],
        top_k: int,
    ) -> list[SearchResult]:
        """Score and rank only the supplied product IDs."""
        ...
