"""Contract tests for the FastAPI search transport."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from fastapi.testclient import TestClient

from product_search.analytics.database import SQLiteAnalyticsDatabase
from product_search.analytics.repository import AnalyticsRepository
from product_search.api.main import create_app
from product_search.api.schemas import MAX_QUERY_LENGTH
from product_search.service import (
    ProductSearchResult,
    ResolvedSearchMode,
    SearchExplanation,
    SearchModeUnavailableError,
    SearchResponse,
    SearchService,
    SearchServiceMetadata,
    SearchServiceStartupError,
)


class FakeSearchService:
    """Small deterministic transport dependency with no generated artifacts."""

    default_mode = "rerank"
    available_modes = ("default", "lexical", "semantic", "hybrid", "rerank")
    product_count = 3
    metadata = SearchServiceMetadata(
        default_search_mode="rerank",
        embedding_model="fake/english-embedding",
        product_count=3,
        artifact_version="artifact-v1",
        build_timestamp="2026-08-16T12:00:00+00:00",
    )

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, str]] = []

    def search(self, query: str, top_k: int = 10, mode: str = "default") -> SearchResponse:
        self.calls.append((query, top_k, mode))
        result = ProductSearchResult(
            product_id="p1",
            product_name="Modern Black Desk Lamp",
            product_class="Desk Lamps",
            category_hierarchy="Lighting / Lamps / Desk Lamps",
            short_description="A compact matte-black task lamp.",
            rank=1,
            final_score=0.91,
            lexical_score=0.72,
            semantic_score=0.88,
            explanation=SearchExplanation(
                matched_query_terms_in_title=("modern", "black", "desk", "lamp"),
                lexical_contribution=0.072,
                semantic_contribution=0.792,
            ),
        )
        resolved_mode = "rerank" if mode == "default" else mode
        return SearchResponse(
            query=query,
            requested_mode=mode,
            resolved_mode=cast(ResolvedSearchMode, resolved_mode),
            top_k=top_k,
            latency_ms=4.25,
            results=(result,),
        )


def _loader(service: FakeSearchService, calls: list[int]) -> Callable[[], SearchService]:
    def load() -> SearchService:
        calls.append(1)
        return cast(SearchService, service)

    return load


def test_health_readiness_modes_model_and_lifespan_load_once() -> None:
    service = FakeSearchService()
    load_calls: list[int] = []
    app = create_app(service_loader=_loader(service, load_calls), analytics_loader=None)

    with TestClient(app) as client:
        health = client.get("/health")
        ready = client.get("/ready")
        modes = client.get("/modes")
        model = client.get("/model")
        client.get("/ready")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert set(health.json()["metrics"]) == {
        "request_count",
        "error_count",
        "average_latency_ms",
    }
    assert ready.json() == {"ready": True, "status": "ready"}
    assert modes.json()["modes"] == list(service.available_modes)
    assert model.json() == {
        "default_search_mode": "rerank",
        "embedding_model": "fake/english-embedding",
        "product_count": 3,
        "artifact_version": "artifact-v1",
        "build_timestamp": "2026-08-16T12:00:00+00:00",
    }
    assert load_calls == [1]


def test_valid_search_enriches_response_without_raw_feature_text() -> None:
    service = FakeSearchService()
    app = create_app(service_loader=lambda: cast(SearchService, service), analytics_loader=None)

    with TestClient(app) as client:
        response = client.post(
            "/search",
            json={"query": "  modern black desk lamp  ", "top_k": 5, "mode": "hybrid"},
        )

    assert response.status_code == 200
    assert service.calls == [("modern black desk lamp", 5, "hybrid")]
    body = response.json()
    assert body["query"] == "modern black desk lamp"
    assert body["mode"] == "hybrid"
    assert body["latency_ms"] == 4.25
    assert body["search_id"] is None
    assert body["result_count"] == 1
    assert body["results"] == [
        {
            "product_id": "p1",
            "product_name": "Modern Black Desk Lamp",
            "product_class": "Desk Lamps",
            "category_hierarchy": "Lighting / Lamps / Desk Lamps",
            "short_description": "A compact matte-black task lamp.",
            "rank": 1,
            "final_score": 0.91,
            "lexical_score": 0.72,
            "semantic_score": 0.88,
            "explanation": {
                "matched_query_terms_in_title": ["modern", "black", "desk", "lamp"],
                "lexical_contribution": 0.072,
                "semantic_contribution": 0.792,
            },
        }
    ]
    assert "product_text" not in response.text
    assert "product_features" not in response.text


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({"query": "   "}, "query"),
        ({"query": "x" * (MAX_QUERY_LENGTH + 1)}, "query"),
        ({"query": "lamp", "mode": "unsupported"}, "mode"),
        ({"query": "lamp", "top_k": 0}, "top_k"),
        ({"query": "lamp", "top_k": 101}, "top_k"),
        ({"query": "lamp", "top_k": True}, "top_k"),
    ],
)
def test_search_validation_returns_structured_errors(
    payload: dict[str, object], field: str
) -> None:
    app = create_app(
        service_loader=lambda: cast(SearchService, FakeSearchService()),
        analytics_loader=None,
    )

    with TestClient(app) as client:
        response = client.post("/search", json=payload)

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "request_validation_error"
    assert body["error"]["message"] == "Request validation failed."
    assert body["error"]["details"][0]["location"] == ["body", field]
    assert "input" not in body["error"]["details"][0]


def test_missing_artifacts_leave_process_live_but_not_ready() -> None:
    local_path = "C:\\private\\artifacts\\dense\\metadata.json"

    def missing_loader() -> SearchService:
        raise SearchServiceStartupError(f"missing dense index: {local_path}")

    app = create_app(service_loader=missing_loader, analytics_loader=None)
    with TestClient(app) as client:
        health = client.get("/health")
        ready = client.get("/ready")
        search = client.post("/search", json={"query": "lamp"})
        model = client.get("/model")
        modes = client.get("/modes")

    assert health.status_code == 200
    assert ready.json() == {"ready": False, "status": "not_ready"}
    for response in (search, model, modes):
        assert response.status_code == 503
        assert response.json() == {
            "error": {
                "code": "search_service_unavailable",
                "message": "Search artifacts are not ready.",
                "details": [],
            }
        }
        assert local_path not in response.text
        assert "traceback" not in response.text.lower()


def test_unavailable_loaded_mode_returns_sanitized_error() -> None:
    class UnavailableModeService(FakeSearchService):
        def search(self, query: str, top_k: int = 10, mode: str = "default") -> SearchResponse:
            raise SearchModeUnavailableError("private internal mode details")

    app = create_app(
        service_loader=lambda: cast(SearchService, UnavailableModeService()),
        analytics_loader=None,
    )
    with TestClient(app) as client:
        response = client.post("/search", json={"query": "lamp", "mode": "rerank"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "search_mode_unavailable"
    assert "private" not in response.text


def test_unexpected_search_error_does_not_expose_details() -> None:
    class ExplodingSearchService(FakeSearchService):
        def search(self, query: str, top_k: int = 10, mode: str = "default") -> SearchResponse:
            raise RuntimeError("C:\\private\\model.joblib failed")

    app = create_app(
        service_loader=lambda: cast(SearchService, ExplodingSearchService()),
        analytics_loader=None,
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/search", json={"query": "lamp"})

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "private" not in response.text
    assert "traceback" not in response.text.lower()


def test_process_metrics_count_requests_and_errors() -> None:
    app = create_app(
        service_loader=lambda: cast(SearchService, FakeSearchService()),
        analytics_loader=None,
    )

    with TestClient(app) as client:
        client.get("/ready")
        client.post("/search", json={"query": ""})
        health = client.get("/health")

    metrics = health.json()["metrics"]
    assert metrics["request_count"] == 2
    assert metrics["error_count"] == 1
    assert metrics["average_latency_ms"] >= 0.0


def _repository(
    tmp_path: Path,
    *,
    ids: tuple[str, ...] = ("search-1", "feedback-1"),
    query_logging_enabled: bool = True,
) -> AnalyticsRepository:
    generated_ids = iter(ids)
    return AnalyticsRepository(
        SQLiteAnalyticsDatabase(tmp_path / "api_analytics.sqlite"),
        query_logging_enabled=query_logging_enabled,
        clock=lambda: datetime(2026, 8, 16, 12, 0, tzinfo=UTC),
        id_factory=lambda: next(generated_ids),
    )


def test_search_logging_feedback_and_aggregate_summary(tmp_path: Path) -> None:
    service = FakeSearchService()
    repository = _repository(tmp_path)
    app = create_app(
        service_loader=lambda: cast(SearchService, service),
        analytics_loader=lambda: repository,
    )

    with TestClient(app) as client:
        search_response = client.post(
            "/search",
            json={
                "query": "modern black desk lamp",
                "top_k": 5,
                "mode": "hybrid",
                "session_id": "local-demo-session",
            },
        )
        feedback_response = client.post(
            "/feedback",
            json={
                "search_id": "search-1",
                "product_id": "p1",
                "feedback_type": "clicked",
            },
        )
        summary_response = client.get("/analytics/summary")

    assert search_response.status_code == 200
    assert search_response.json()["search_id"] == "search-1"
    stored = repository.get_search("search-1")
    assert stored is not None
    assert stored.query == "modern black desk lamp"
    assert stored.mode == "hybrid"
    assert stored.top_k == 5
    assert stored.returned_product_ids == ("p1",)
    assert stored.session_id == "local-demo-session"
    assert feedback_response.status_code == 201
    assert feedback_response.json() == {
        "feedback_id": "feedback-1",
        "search_id": "search-1",
        "timestamp": "2026-08-16T12:00:00+00:00",
        "product_id": "p1",
        "feedback_type": "clicked",
    }
    assert summary_response.status_code == 200
    assert summary_response.json() == {
        "query_logging_enabled": True,
        "search_count": 1,
        "feedback_count": 1,
        "average_latency_ms": 4.25,
        "searches_by_mode": {"hybrid": 1},
        "feedback_by_type": {"clicked": 1},
    }
    assert "modern black desk lamp" not in summary_response.text
    assert '"p1"' not in summary_response.text


def test_feedback_rejects_unknown_search_and_unreturned_product(tmp_path: Path) -> None:
    repository = _repository(tmp_path, ids=("search-1",))
    app = create_app(
        service_loader=lambda: cast(SearchService, FakeSearchService()),
        analytics_loader=lambda: repository,
    )

    with TestClient(app) as client:
        missing = client.post(
            "/feedback",
            json={
                "search_id": "missing-search",
                "product_id": "p1",
                "feedback_type": "relevant",
            },
        )
        client.post("/search", json={"query": "lamp"})
        unreturned = client.post(
            "/feedback",
            json={
                "search_id": "search-1",
                "product_id": "p2",
                "feedback_type": "not_relevant",
            },
        )

    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "search_event_not_found"
    assert unreturned.status_code == 400
    assert unreturned.json()["error"]["code"] == "product_not_in_search"


def test_query_logging_can_be_disabled_without_failing_search(tmp_path: Path) -> None:
    repository = _repository(tmp_path, query_logging_enabled=False)
    app = create_app(
        service_loader=lambda: cast(SearchService, FakeSearchService()),
        analytics_loader=lambda: repository,
    )

    with TestClient(app) as client:
        search = client.post("/search", json={"query": "private query"})
        feedback = client.post(
            "/feedback",
            json={
                "search_id": "search-1",
                "product_id": "p1",
                "feedback_type": "clicked",
            },
        )
        summary = client.get("/analytics/summary")

    assert search.status_code == 200
    assert search.json()["search_id"] is None
    assert feedback.status_code == 409
    assert feedback.json()["error"]["code"] == "query_logging_disabled"
    assert summary.json()["query_logging_enabled"] is False
    assert summary.json()["search_count"] == 0


def test_search_logging_failure_does_not_fail_search(tmp_path: Path) -> None:
    class FailingAnalyticsRepository(AnalyticsRepository):
        def log_search(self, **kwargs: object) -> None:
            raise RuntimeError("C:\\private\\search_analytics.sqlite is unavailable")

    repository = FailingAnalyticsRepository(SQLiteAnalyticsDatabase(tmp_path / "failing.sqlite"))
    app = create_app(
        service_loader=lambda: cast(SearchService, FakeSearchService()),
        analytics_loader=lambda: repository,
    )

    with TestClient(app) as client:
        response = client.post("/search", json={"query": "lamp"})

    assert response.status_code == 200
    assert response.json()["search_id"] is None
    assert "private" not in response.text
