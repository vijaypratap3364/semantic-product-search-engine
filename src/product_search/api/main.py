"""FastAPI application exposing the artifact-backed product search service."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import cast

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from product_search.analytics.database import SQLiteAnalyticsDatabase
from product_search.analytics.repository import (
    AnalyticsRepository,
    ProductNotInSearchError,
    QueryLoggingDisabledError,
    SearchEventNotFoundError,
)
from product_search.api.schemas import (
    AnalyticsSummaryResponse,
    ApiSearchMode,
    ErrorBody,
    ErrorDetail,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    ModelResponse,
    ModesResponse,
    ProcessMetricsResponse,
    ReadinessResponse,
    SearchApiResponse,
    SearchRequest,
)
from product_search.config import load_settings
from product_search.service import (
    SearchModeUnavailableError,
    SearchService,
    SearchServiceStartupError,
)

LOGGER = logging.getLogger(__name__)
ServiceLoader = Callable[[], SearchService]
AnalyticsLoader = Callable[[], AnalyticsRepository]


@dataclass(slots=True)
class RequestMetrics:
    """Concurrency-safe process-local request counters."""

    request_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def record(self, *, latency_ms: float, is_error: bool) -> None:
        with self._lock:
            self.request_count += 1
            self.error_count += int(is_error)
            self.total_latency_ms += max(0.0, latency_ms)

    def snapshot(self) -> ProcessMetricsResponse:
        with self._lock:
            average = self.total_latency_ms / self.request_count if self.request_count else 0.0
            return ProcessMetricsResponse(
                request_count=self.request_count,
                error_count=self.error_count,
                average_latency_ms=average,
            )


@dataclass(slots=True)
class ApiRuntime:
    """Application-owned service and health state."""

    service: SearchService | None = None
    analytics: AnalyticsRepository | None = None
    startup_attempted: bool = False
    startup_failed: bool = False
    analytics_failed: bool = False
    metrics: RequestMetrics = field(default_factory=RequestMetrics)


def _default_service_loader() -> SearchService:
    return SearchService.load(local_files_only=True)


def _default_analytics_loader() -> AnalyticsRepository:
    settings = load_settings()
    return AnalyticsRepository(
        SQLiteAnalyticsDatabase(settings.analytics.database_path),
        query_logging_enabled=settings.analytics.query_logging_enabled,
    )


def create_app(
    *,
    service_loader: ServiceLoader = _default_service_loader,
    analytics_loader: AnalyticsLoader | None = _default_analytics_loader,
) -> FastAPI:
    """Create an app whose lifespan loads the static search service exactly once."""

    runtime = ApiRuntime()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        runtime.startup_attempted = True
        try:
            runtime.service = service_loader()
        except SearchServiceStartupError:
            runtime.startup_failed = True
            LOGGER.exception("Search service artifacts failed startup validation")
        if analytics_loader is not None:
            try:
                runtime.analytics = analytics_loader()
                runtime.analytics.initialize()
            except Exception:
                runtime.analytics_failed = True
                runtime.analytics = None
                LOGGER.exception("Local analytics failed startup initialization")
        app.state.runtime = runtime
        yield

    app = FastAPI(
        title="Semantic Product Search API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.runtime = runtime

    @app.middleware("http")
    async def record_request_metrics(
        request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        started_at = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            runtime.metrics.record(
                latency_ms=(time.perf_counter() - started_at) * 1000.0,
                is_error=True,
            )
            raise
        runtime.metrics.record(
            latency_ms=(time.perf_counter() - started_at) * 1000.0,
            is_error=response.status_code >= 400,
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        del request
        details = tuple(
            ErrorDetail(
                location=tuple(issue.get("loc", ())),
                message=str(issue.get("msg", "Invalid value")),
                error_type=str(issue.get("type", "validation_error")),
            )
            for issue in error.errors()
        )
        return _error_response(
            status_code=422,
            code="request_validation_error",
            message="Request validation failed.",
            details=details,
        )

    @app.exception_handler(SearchModeUnavailableError)
    async def unavailable_mode_error(
        request: Request, error: SearchModeUnavailableError
    ) -> JSONResponse:
        del request, error
        return _error_response(
            status_code=400,
            code="search_mode_unavailable",
            message="The requested search mode is not available.",
        )

    @app.exception_handler(Exception)
    async def internal_error(request: Request, error: Exception) -> JSONResponse:
        del request
        LOGGER.exception("Unhandled API error", exc_info=error)
        return _error_response(
            status_code=500,
            code="internal_error",
            message="The request could not be completed.",
        )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(metrics=runtime.metrics.snapshot())

    @app.get("/ready", response_model=ReadinessResponse)
    def ready() -> ReadinessResponse:
        is_ready = runtime.service is not None and not runtime.startup_failed
        return ReadinessResponse(
            ready=is_ready,
            status="ready" if is_ready else "not_ready",
        )

    @app.get("/model", response_model=ModelResponse)
    def model() -> ModelResponse | JSONResponse:
        service = _ready_service(runtime)
        if service is None:
            return _service_unavailable_response()
        metadata = service.metadata
        return ModelResponse(
            default_search_mode=metadata.default_search_mode,
            embedding_model=metadata.embedding_model,
            product_count=metadata.product_count,
            artifact_version=metadata.artifact_version,
            build_timestamp=metadata.build_timestamp,
        )

    @app.get("/modes", response_model=ModesResponse)
    def modes() -> ModesResponse | JSONResponse:
        service = _ready_service(runtime)
        if service is None:
            return _service_unavailable_response()
        return ModesResponse(
            modes=tuple(cast(ApiSearchMode, mode) for mode in service.available_modes)
        )

    @app.post("/search", response_model=SearchApiResponse)
    def search(payload: SearchRequest) -> SearchApiResponse | JSONResponse:
        service = _ready_service(runtime)
        if service is None:
            return _service_unavailable_response()
        response = service.search(payload.query, top_k=payload.top_k, mode=payload.mode)
        search_id: str | None = None
        repository = _ready_analytics(runtime)
        if repository is not None:
            try:
                event = repository.log_search(
                    query=response.query,
                    mode=response.resolved_mode,
                    top_k=payload.top_k,
                    latency_ms=response.latency_ms,
                    returned_product_ids=tuple(result.product_id for result in response.results),
                    session_id=payload.session_id,
                )
                search_id = None if event is None else event.search_id
            except Exception:
                LOGGER.exception("Local search analytics logging failed")
        return SearchApiResponse.from_service_response(response, search_id=search_id)

    @app.post("/feedback", response_model=FeedbackResponse, status_code=201)
    def feedback(payload: FeedbackRequest) -> FeedbackResponse | JSONResponse:
        repository = _ready_analytics(runtime)
        if repository is None:
            return _analytics_unavailable_response()
        try:
            event = repository.log_feedback(
                search_id=payload.search_id,
                product_id=payload.product_id,
                feedback_type=payload.feedback_type,
            )
        except SearchEventNotFoundError:
            return _error_response(
                status_code=404,
                code="search_event_not_found",
                message="The referenced search event does not exist.",
            )
        except ProductNotInSearchError:
            return _error_response(
                status_code=400,
                code="product_not_in_search",
                message="The product was not returned by the referenced search.",
            )
        except QueryLoggingDisabledError:
            return _error_response(
                status_code=409,
                code="query_logging_disabled",
                message="Local query logging is disabled.",
            )
        except Exception:
            LOGGER.exception("Local feedback analytics logging failed")
            return _analytics_unavailable_response()
        return FeedbackResponse(
            feedback_id=event.feedback_id,
            search_id=event.search_id,
            timestamp=event.timestamp,
            product_id=event.product_id,
            feedback_type=event.feedback_type,
        )

    @app.get("/analytics/summary", response_model=AnalyticsSummaryResponse)
    def analytics_summary() -> AnalyticsSummaryResponse | JSONResponse:
        repository = _ready_analytics(runtime)
        if repository is None:
            return _analytics_unavailable_response()
        try:
            summary = repository.summary()
        except Exception:
            LOGGER.exception("Local analytics summary failed")
            return _analytics_unavailable_response()
        return AnalyticsSummaryResponse(
            query_logging_enabled=summary.query_logging_enabled,
            search_count=summary.search_count,
            feedback_count=summary.feedback_count,
            average_latency_ms=summary.average_latency_ms,
            searches_by_mode=summary.searches_by_mode,
            feedback_by_type=summary.feedback_by_type,
        )

    return app


def _ready_service(runtime: ApiRuntime) -> SearchService | None:
    if not runtime.startup_attempted or runtime.startup_failed:
        return None
    return runtime.service


def _ready_analytics(runtime: ApiRuntime) -> AnalyticsRepository | None:
    if runtime.analytics_failed:
        return None
    return runtime.analytics


def _service_unavailable_response() -> JSONResponse:
    return _error_response(
        status_code=503,
        code="search_service_unavailable",
        message="Search artifacts are not ready.",
    )


def _analytics_unavailable_response() -> JSONResponse:
    return _error_response(
        status_code=503,
        code="analytics_unavailable",
        message="Local analytics are not available.",
    )


def _error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    details: tuple[ErrorDetail, ...] = (),
) -> JSONResponse:
    payload = ErrorResponse(error=ErrorBody(code=code, message=message, details=details))
    return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))


app = create_app()
