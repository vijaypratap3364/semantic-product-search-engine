"""Tests for the reusable Streamlit-facing FastAPI client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest
import requests

from product_search.ui.api_client import (
    ApiClient,
    ApiClientError,
    ApiContractError,
    ApiUnavailableError,
    HttpResponse,
)


@dataclass(slots=True)
class FakeResponse:
    status_code: int
    payload: object = None
    json_error: ValueError | None = None

    def json(self) -> object:
        if self.json_error is not None:
            raise self.json_error
        return self.payload


class FakeTransport:
    def __init__(self, *responses: FakeResponse, error: Exception | None = None) -> None:
        self.responses = list(responses)
        self.error = error
        self.calls: list[tuple[str, str, object | None, float]] = []

    def request(
        self,
        method: str,
        url: str,
        *,
        json: object | None,
        timeout: float,
    ) -> HttpResponse:
        self.calls.append((method, url, json, timeout))
        if self.error is not None:
            raise self.error
        return cast(HttpResponse, self.responses.pop(0))


def _client(transport: FakeTransport) -> ApiClient:
    return ApiClient(
        " http://127.0.0.1:8000/ ",
        timeout_seconds=4.5,
        transport=transport,
    )


def _search_payload() -> dict[str, object]:
    return {
        "query": "modern black desk lamp",
        "mode": "hybrid",
        "latency_ms": 4.25,
        "search_id": "search-1",
        "result_count": 1,
        "results": [
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
        ],
    }


def test_health_readiness_model_and_modes_are_transformed() -> None:
    transport = FakeTransport(
        FakeResponse(
            200,
            {
                "status": "ok",
                "metrics": {
                    "request_count": 7,
                    "error_count": 1,
                    "average_latency_ms": 3.5,
                },
            },
        ),
        FakeResponse(200, {"ready": True, "status": "ready"}),
        FakeResponse(
            200,
            {
                "default_search_mode": "rerank",
                "embedding_model": "BAAI/bge-small-en-v1.5",
                "product_count": 42_994,
                "artifact_version": "artifact-v1",
                "build_timestamp": "2026-08-17T02:48:14+00:00",
            },
        ),
        FakeResponse(200, {"modes": ["default", "lexical", "semantic", "hybrid", "rerank"]}),
    )
    client = _client(transport)

    health = client.health()
    readiness = client.readiness()
    model = client.model()
    modes = client.modes()

    assert health.status == "ok"
    assert health.metrics.request_count == 7
    assert health.metrics.error_count == 1
    assert health.metrics.average_latency_ms == 3.5
    assert readiness.ready is True
    assert model.default_search_mode == "rerank"
    assert model.embedding_model == "BAAI/bge-small-en-v1.5"
    assert model.product_count == 42_994
    assert model.artifact_version == "artifact-v1"
    assert model.build_timestamp == "2026-08-17T02:48:14+00:00"
    assert modes == ("default", "lexical", "semantic", "hybrid", "rerank")
    assert [call[:2] for call in transport.calls] == [
        ("GET", "http://127.0.0.1:8000/health"),
        ("GET", "http://127.0.0.1:8000/ready"),
        ("GET", "http://127.0.0.1:8000/model"),
        ("GET", "http://127.0.0.1:8000/modes"),
    ]
    assert all(call[3] == 4.5 for call in transport.calls)


def test_search_response_is_transformed_without_internal_fields() -> None:
    transport = FakeTransport(FakeResponse(200, _search_payload()))
    response = _client(transport).search(
        "modern black desk lamp",
        top_k=5,
        mode="hybrid",
        session_id="demo-session",
    )

    assert response.query == "modern black desk lamp"
    assert response.mode == "hybrid"
    assert response.latency_ms == 4.25
    assert response.search_id == "search-1"
    assert response.result_count == 1
    result = response.results[0]
    assert result.product_id == "p1"
    assert result.product_name == "Modern Black Desk Lamp"
    assert result.product_class == "Desk Lamps"
    assert result.category_hierarchy == "Lighting / Lamps / Desk Lamps"
    assert result.short_description == "A compact matte-black task lamp."
    assert result.rank == 1
    assert result.final_score == 0.91
    assert result.lexical_score == 0.72
    assert result.semantic_score == 0.88
    assert result.explanation.matched_query_terms_in_title == (
        "modern",
        "black",
        "desk",
        "lamp",
    )
    assert result.explanation.lexical_contribution == 0.072
    assert result.explanation.semantic_contribution == 0.792
    assert transport.calls == [
        (
            "POST",
            "http://127.0.0.1:8000/search",
            {
                "query": "modern black desk lamp",
                "top_k": 5,
                "mode": "hybrid",
                "session_id": "demo-session",
            },
            4.5,
        )
    ]


def test_search_supports_empty_results_and_nullable_display_fields() -> None:
    payload = _search_payload()
    payload.update(
        {
            "mode": "semantic",
            "search_id": None,
            "result_count": 0,
            "results": [],
        }
    )
    transport = FakeTransport(FakeResponse(200, payload))

    response = _client(transport).search("unknown", mode="semantic")

    assert response.results == ()
    assert response.search_id is None
    assert "session_id" not in cast(dict[str, object], transport.calls[0][2])


def test_feedback_is_sent_to_fastapi_and_transformed() -> None:
    transport = FakeTransport(
        FakeResponse(
            201,
            {
                "feedback_id": "feedback-1",
                "search_id": "search-1",
                "timestamp": "2026-08-17T03:00:00+00:00",
                "product_id": "p1",
                "feedback_type": "relevant",
            },
        )
    )

    receipt = _client(transport).send_feedback(
        search_id="search-1",
        product_id="p1",
        feedback_type="relevant",
    )

    assert receipt.feedback_id == "feedback-1"
    assert receipt.search_id == "search-1"
    assert receipt.product_id == "p1"
    assert receipt.feedback_type == "relevant"
    assert transport.calls[0][0:2] == ("POST", "http://127.0.0.1:8000/feedback")
    assert transport.calls[0][2] == {
        "search_id": "search-1",
        "product_id": "p1",
        "feedback_type": "relevant",
    }


def test_structured_api_error_is_preserved_without_internal_payload_details() -> None:
    transport = FakeTransport(
        FakeResponse(
            503,
            {
                "error": {
                    "code": "search_service_unavailable",
                    "message": "Search artifacts are not ready.",
                    "details": [],
                }
            },
        )
    )

    with pytest.raises(ApiClientError, match="Search artifacts are not ready") as raised:
        _client(transport).search("lamp")

    assert raised.value.code == "search_service_unavailable"
    assert raised.value.status_code == 503


def test_unstructured_api_error_uses_safe_http_message() -> None:
    transport = FakeTransport(FakeResponse(500, {"private": "C:\\models\\secret.joblib"}))

    with pytest.raises(ApiClientError, match="HTTP 500") as raised:
        _client(transport).health()

    assert raised.value.code == "api_error"
    assert "secret" not in str(raised.value)


def test_network_failure_becomes_actionable_unavailable_error() -> None:
    transport = FakeTransport(error=requests.ConnectionError("private connection details"))

    with pytest.raises(ApiUnavailableError, match="Start FastAPI") as raised:
        _client(transport).readiness()

    assert raised.value.code == "api_unavailable"
    assert "private" not in str(raised.value)


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (FakeResponse(200, ["not", "an", "object"]), "must be an object"),
        (FakeResponse(200, json_error=ValueError("invalid")), "not valid JSON"),
        (FakeResponse(200, {"modes": "lexical"}), "modes must be a list"),
        (FakeResponse(200, {"modes": ["magic"]}), "unsupported value"),
    ],
)
def test_incompatible_top_level_responses_are_rejected(
    response: FakeResponse,
    message: str,
) -> None:
    client = _client(FakeTransport(response))

    with pytest.raises(ApiContractError, match=message):
        if "modes" in cast(dict[str, object], response.payload or {}):
            client.modes()
        else:
            client.health()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"results": "invalid"}, "results must be a list"),
        ({"result_count": 2}, "does not match"),
        ({"mode": "magic"}, "search mode is not supported"),
    ],
)
def test_incompatible_search_envelopes_are_rejected(
    mutation: dict[str, object],
    message: str,
) -> None:
    payload = _search_payload()
    payload.update(mutation)

    with pytest.raises(ApiContractError, match=message):
        _client(FakeTransport(FakeResponse(200, payload))).search("lamp")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("rank", True, "rank must be an integer"),
        ("final_score", "high", "final_score must be a number"),
        ("product_class", 3, "product_class must be a string or null"),
        ("lexical_score", "high", "lexical_score must be a number or null"),
    ],
)
def test_incompatible_product_result_fields_are_rejected(
    field: str,
    value: object,
    message: str,
) -> None:
    payload = _search_payload()
    result = cast(list[dict[str, object]], payload["results"])[0]
    result[field] = value

    with pytest.raises(ApiContractError, match=message):
        _client(FakeTransport(FakeResponse(200, payload))).search("lamp")


def test_invalid_model_and_feedback_enums_are_rejected() -> None:
    client = _client(
        FakeTransport(
            FakeResponse(
                200,
                {
                    "default_search_mode": "magic",
                    "embedding_model": "fake",
                    "product_count": 1,
                    "artifact_version": "v1",
                    "build_timestamp": "now",
                },
            ),
            FakeResponse(
                201,
                {
                    "feedback_id": "f1",
                    "search_id": "s1",
                    "timestamp": "now",
                    "product_id": "p1",
                    "feedback_type": "magic",
                },
            ),
        )
    )

    with pytest.raises(ApiContractError, match="default_search_mode"):
        client.model()
    with pytest.raises(ApiContractError, match="feedback_type"):
        client.send_feedback(search_id="s1", product_id="p1", feedback_type="clicked")


@pytest.mark.parametrize(
    ("base_url", "timeout", "message"),
    [
        ("localhost:8000", 1.0, "http"),
        ("http://localhost:8000", 0.0, "positive"),
    ],
)
def test_client_configuration_is_validated(base_url: str, timeout: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ApiClient(base_url, timeout_seconds=timeout)
