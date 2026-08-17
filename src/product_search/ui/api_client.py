"""Typed HTTP client used by the Streamlit presentation layer."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Literal, Protocol, cast

import requests

SearchMode = Literal["default", "lexical", "semantic", "hybrid", "rerank"]
ResolvedSearchMode = Literal["lexical", "semantic", "hybrid", "rerank"]
FeedbackType = Literal["relevant", "not_relevant", "clicked"]


class HttpResponse(Protocol):
    """Small response surface needed from the HTTP transport."""

    status_code: int

    def json(self) -> object:
        """Decode the response body."""


class HttpTransport(Protocol):
    """Injectable requests-compatible transport for deterministic tests."""

    def request(
        self,
        method: str,
        url: str,
        *,
        json: object | None,
        timeout: float,
    ) -> HttpResponse:
        """Execute one HTTP request."""


class ApiClientError(RuntimeError):
    """Safe API, transport, or response-contract failure for UI display."""

    def __init__(self, message: str, *, code: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class ApiUnavailableError(ApiClientError):
    """The configured FastAPI process could not be reached."""


class ApiContractError(ApiClientError):
    """FastAPI returned a payload that does not match the UI contract."""


@dataclass(frozen=True, slots=True)
class ProcessMetrics:
    """Process-local counters returned with health status."""

    request_count: int
    error_count: int
    average_latency_ms: float


@dataclass(frozen=True, slots=True)
class HealthStatus:
    """FastAPI process liveness response."""

    status: str
    metrics: ProcessMetrics


@dataclass(frozen=True, slots=True)
class ReadinessStatus:
    """Search artifact readiness response."""

    ready: bool
    status: str


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Safe selected-engine metadata shown by the system page."""

    default_search_mode: ResolvedSearchMode
    embedding_model: str
    product_count: int
    artifact_version: str
    build_timestamp: str


@dataclass(frozen=True, slots=True)
class ResultExplanation:
    """Non-LLM evidence supplied by the search service."""

    matched_query_terms_in_title: tuple[str, ...]
    lexical_contribution: float | None
    semantic_contribution: float | None


@dataclass(frozen=True, slots=True)
class ProductResult:
    """One display-safe API search result."""

    product_id: str
    product_name: str
    product_class: str | None
    category_hierarchy: str | None
    short_description: str | None
    rank: int
    final_score: float
    lexical_score: float | None
    semantic_score: float | None
    explanation: ResultExplanation


@dataclass(frozen=True, slots=True)
class SearchResponse:
    """One transformed search response for dashboard rendering."""

    query: str
    mode: ResolvedSearchMode
    latency_ms: float
    search_id: str | None
    result_count: int
    results: tuple[ProductResult, ...]


@dataclass(frozen=True, slots=True)
class FeedbackReceipt:
    """Confirmation that local feedback was persisted by FastAPI."""

    feedback_id: str
    search_id: str
    timestamp: str
    product_id: str
    feedback_type: FeedbackType


class ApiClient:
    """Reusable, thread-safe client for the dashboard's FastAPI dependency."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 30.0,
        transport: HttpTransport | None = None,
    ) -> None:
        normalized_url = base_url.strip().rstrip("/")
        if not normalized_url.startswith(("http://", "https://")):
            raise ValueError("base_url must use http:// or https://")
        if timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")
        self.base_url = normalized_url
        self.timeout_seconds = timeout_seconds
        self._transport = transport or cast(HttpTransport, requests.Session())
        self._request_lock = Lock()

    def health(self) -> HealthStatus:
        """Return FastAPI process health and lightweight counters."""

        payload = self._request("GET", "/health")
        metrics = _required_object(payload, "metrics")
        return HealthStatus(
            status=_required_string(payload, "status"),
            metrics=ProcessMetrics(
                request_count=_required_integer(metrics, "request_count"),
                error_count=_required_integer(metrics, "error_count"),
                average_latency_ms=_required_number(metrics, "average_latency_ms"),
            ),
        )

    def readiness(self) -> ReadinessStatus:
        """Return whether generated search artifacts are loaded."""

        payload = self._request("GET", "/ready")
        return ReadinessStatus(
            ready=_required_boolean(payload, "ready"),
            status=_required_string(payload, "status"),
        )

    def model(self) -> ModelMetadata:
        """Return selected engine and artifact metadata."""

        payload = self._request("GET", "/model")
        default_mode = _required_string(payload, "default_search_mode")
        if default_mode not in {"lexical", "semantic", "hybrid", "rerank"}:
            raise _contract_error("default_search_mode is not supported")
        return ModelMetadata(
            default_search_mode=cast(ResolvedSearchMode, default_mode),
            embedding_model=_required_string(payload, "embedding_model"),
            product_count=_required_integer(payload, "product_count"),
            artifact_version=_required_string(payload, "artifact_version"),
            build_timestamp=_required_string(payload, "build_timestamp"),
        )

    def modes(self) -> tuple[SearchMode, ...]:
        """Return search modes currently exposed by FastAPI."""

        payload = self._request("GET", "/modes")
        values = payload.get("modes")
        if not isinstance(values, list):
            raise _contract_error("modes must be a list")
        supported = {"default", "lexical", "semantic", "hybrid", "rerank"}
        if not all(isinstance(value, str) and value in supported for value in values):
            raise _contract_error("modes contains an unsupported value")
        return tuple(cast(SearchMode, value) for value in values)

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        mode: SearchMode = "default",
        session_id: str | None = None,
    ) -> SearchResponse:
        """Search through FastAPI and transform the display-safe response."""

        request_payload: dict[str, object] = {
            "query": query,
            "top_k": top_k,
            "mode": mode,
        }
        if session_id is not None:
            request_payload["session_id"] = session_id
        payload = self._request("POST", "/search", json=request_payload)
        raw_results = payload.get("results")
        if not isinstance(raw_results, list):
            raise _contract_error("results must be a list")
        results = tuple(_parse_product_result(value) for value in raw_results)
        result_count = _required_integer(payload, "result_count")
        if result_count != len(results):
            raise _contract_error("result_count does not match results")
        resolved_mode = _required_string(payload, "mode")
        if resolved_mode not in {"lexical", "semantic", "hybrid", "rerank"}:
            raise _contract_error("search mode is not supported")
        return SearchResponse(
            query=_required_string(payload, "query"),
            mode=cast(ResolvedSearchMode, resolved_mode),
            latency_ms=_required_number(payload, "latency_ms"),
            search_id=_optional_string(payload, "search_id"),
            result_count=result_count,
            results=results,
        )

    def send_feedback(
        self,
        *,
        search_id: str,
        product_id: str,
        feedback_type: FeedbackType,
    ) -> FeedbackReceipt:
        """Persist one relevance signal through FastAPI, never SQLite directly."""

        payload = self._request(
            "POST",
            "/feedback",
            json={
                "search_id": search_id,
                "product_id": product_id,
                "feedback_type": feedback_type,
            },
        )
        returned_type = _required_string(payload, "feedback_type")
        if returned_type not in {"relevant", "not_relevant", "clicked"}:
            raise _contract_error("feedback_type is not supported")
        return FeedbackReceipt(
            feedback_id=_required_string(payload, "feedback_id"),
            search_id=_required_string(payload, "search_id"),
            timestamp=_required_string(payload, "timestamp"),
            product_id=_required_string(payload, "product_id"),
            feedback_type=cast(FeedbackType, returned_type),
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: object | None = None,
    ) -> dict[str, Any]:
        try:
            with self._request_lock:
                response = self._transport.request(
                    method,
                    f"{self.base_url}{path}",
                    json=json,
                    timeout=self.timeout_seconds,
                )
        except requests.RequestException as error:
            raise ApiUnavailableError(
                "The search API is unavailable. Start FastAPI and try again.",
                code="api_unavailable",
            ) from error

        try:
            decoded = response.json()
        except ValueError as error:
            raise _contract_error("API response was not valid JSON") from error
        payload = _as_object(decoded, context="API response")
        if response.status_code >= 400:
            error_payload = payload.get("error")
            if isinstance(error_payload, dict):
                safe_error = cast(dict[str, Any], error_payload)
                code = safe_error.get("code")
                message = safe_error.get("message")
                if isinstance(code, str) and isinstance(message, str):
                    raise ApiClientError(
                        message,
                        code=code,
                        status_code=response.status_code,
                    )
            raise ApiClientError(
                f"The search API returned HTTP {response.status_code}.",
                code="api_error",
                status_code=response.status_code,
            )
        return payload


def _parse_product_result(value: object) -> ProductResult:
    payload = _as_object(value, context="search result")
    explanation = _required_object(payload, "explanation")
    raw_terms = explanation.get("matched_query_terms_in_title")
    if not isinstance(raw_terms, list) or not all(isinstance(term, str) for term in raw_terms):
        raise _contract_error("matched query terms must be a list of strings")
    return ProductResult(
        product_id=_required_string(payload, "product_id"),
        product_name=_required_string(payload, "product_name"),
        product_class=_optional_string(payload, "product_class"),
        category_hierarchy=_optional_string(payload, "category_hierarchy"),
        short_description=_optional_string(payload, "short_description"),
        rank=_required_integer(payload, "rank"),
        final_score=_required_number(payload, "final_score"),
        lexical_score=_optional_number(payload, "lexical_score"),
        semantic_score=_optional_number(payload, "semantic_score"),
        explanation=ResultExplanation(
            matched_query_terms_in_title=tuple(cast(list[str], raw_terms)),
            lexical_contribution=_optional_number(explanation, "lexical_contribution"),
            semantic_contribution=_optional_number(explanation, "semantic_contribution"),
        ),
    )


def _as_object(value: object, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _contract_error(f"{context} must be an object")
    return cast(dict[str, Any], value)


def _required_object(payload: dict[str, Any], key: str) -> dict[str, Any]:
    return _as_object(payload.get(key), context=key)


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise _contract_error(f"{key} must be a string")
    return value


def _optional_string(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise _contract_error(f"{key} must be a string or null")
    return value


def _required_integer(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise _contract_error(f"{key} must be an integer")
    return value


def _required_boolean(payload: dict[str, Any], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise _contract_error(f"{key} must be a boolean")
    return value


def _required_number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _contract_error(f"{key} must be a number")
    return float(value)


def _optional_number(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _contract_error(f"{key} must be a number or null")
    return float(value)


def _contract_error(message: str) -> ApiContractError:
    return ApiContractError(
        f"The search API returned an incompatible response: {message}.",
        code="api_contract_error",
    )
