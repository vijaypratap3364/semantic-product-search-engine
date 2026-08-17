"""Contract tests for the FastAPI search transport."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest
from fastapi.testclient import TestClient

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
    app = create_app(service_loader=_loader(service, load_calls))

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
    app = create_app(service_loader=lambda: cast(SearchService, service))

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
    app = create_app(service_loader=lambda: cast(SearchService, FakeSearchService()))

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

    app = create_app(service_loader=missing_loader)
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

    app = create_app(service_loader=lambda: cast(SearchService, UnavailableModeService()))
    with TestClient(app) as client:
        response = client.post("/search", json={"query": "lamp", "mode": "rerank"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "search_mode_unavailable"
    assert "private" not in response.text


def test_unexpected_search_error_does_not_expose_details() -> None:
    class ExplodingSearchService(FakeSearchService):
        def search(self, query: str, top_k: int = 10, mode: str = "default") -> SearchResponse:
            raise RuntimeError("C:\\private\\model.joblib failed")

    app = create_app(service_loader=lambda: cast(SearchService, ExplodingSearchService()))
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/search", json={"query": "lamp"})

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "private" not in response.text
    assert "traceback" not in response.text.lower()


def test_process_metrics_count_requests_and_errors() -> None:
    app = create_app(service_loader=lambda: cast(SearchService, FakeSearchService()))

    with TestClient(app) as client:
        client.get("/ready")
        client.post("/search", json={"query": ""})
        health = client.get("/health")

    metrics = health.json()["metrics"]
    assert metrics["request_count"] == 2
    assert metrics["error_count"] == 1
    assert metrics["average_latency_ms"] >= 0.0
