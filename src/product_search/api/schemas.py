"""Validated request and response contracts for the product search API."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from product_search.service import ProductSearchResult, SearchResponse

ApiSearchMode = Literal["default", "lexical", "semantic", "hybrid", "rerank"]
ResolvedApiSearchMode = Literal["lexical", "semantic", "hybrid", "rerank"]

MAX_QUERY_LENGTH = 500
MIN_TOP_K = 1
MAX_TOP_K = 100

QueryText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUERY_LENGTH),
]


class ApiModel(BaseModel):
    """Strict API model shared by every JSON contract."""

    model_config = ConfigDict(extra="forbid")


class SearchRequest(ApiModel):
    """One validated product search request."""

    query: QueryText
    top_k: int = Field(default=10, ge=MIN_TOP_K, le=MAX_TOP_K, strict=True)
    mode: ApiSearchMode = "default"


class SearchExplanationResponse(ApiModel):
    """Non-generative explanation fields safe for clients."""

    matched_query_terms_in_title: tuple[str, ...]
    lexical_contribution: float | None
    semantic_contribution: float | None


class SearchResultResponse(ApiModel):
    """Display-safe ranked product result."""

    product_id: str
    product_name: str
    product_class: str | None
    category_hierarchy: str | None
    short_description: str | None
    rank: int
    final_score: float
    lexical_score: float | None
    semantic_score: float | None
    explanation: SearchExplanationResponse

    @classmethod
    def from_service_result(cls, result: ProductSearchResult) -> SearchResultResponse:
        """Translate a service result without exposing internal product features."""

        return cls.model_validate(
            {
                "product_id": result.product_id,
                "product_name": result.product_name,
                "product_class": result.product_class,
                "category_hierarchy": result.category_hierarchy,
                "short_description": result.short_description,
                "rank": result.rank,
                "final_score": result.final_score,
                "lexical_score": result.lexical_score,
                "semantic_score": result.semantic_score,
                "explanation": {
                    "matched_query_terms_in_title": (
                        result.explanation.matched_query_terms_in_title
                    ),
                    "lexical_contribution": result.explanation.lexical_contribution,
                    "semantic_contribution": result.explanation.semantic_contribution,
                },
            }
        )


class SearchApiResponse(ApiModel):
    """Complete response for one successful search."""

    query: str
    mode: ResolvedApiSearchMode
    latency_ms: float
    result_count: int
    results: tuple[SearchResultResponse, ...]

    @classmethod
    def from_service_response(cls, response: SearchResponse) -> SearchApiResponse:
        """Translate the transport-independent service response."""

        results = tuple(SearchResultResponse.from_service_result(item) for item in response.results)
        return cls(
            query=response.query,
            mode=response.resolved_mode,
            latency_ms=response.latency_ms,
            result_count=len(results),
            results=results,
        )


class ProcessMetricsResponse(ApiModel):
    """Small in-process request counters without external infrastructure."""

    request_count: int
    error_count: int
    average_latency_ms: float


class HealthResponse(ApiModel):
    """Process liveness and lightweight counters."""

    status: Literal["ok"] = "ok"
    metrics: ProcessMetricsResponse


class ReadinessResponse(ApiModel):
    """Search artifact readiness independent of process liveness."""

    ready: bool
    status: Literal["ready", "not_ready"]


class ModelResponse(ApiModel):
    """Safe immutable model metadata."""

    default_search_mode: ResolvedApiSearchMode
    embedding_model: str
    product_count: int
    artifact_version: str
    build_timestamp: str


class ModesResponse(ApiModel):
    """Search modes loaded by the service."""

    modes: tuple[ApiSearchMode, ...]


class ErrorDetail(ApiModel):
    """One sanitized validation issue."""

    location: tuple[str | int, ...]
    message: str
    error_type: str


class ErrorBody(ApiModel):
    """Stable machine-readable error payload."""

    code: str
    message: str
    details: tuple[ErrorDetail, ...] = ()


class ErrorResponse(ApiModel):
    """Envelope used by every API error."""

    error: ErrorBody
