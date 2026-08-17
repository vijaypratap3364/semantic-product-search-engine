"""Streamlit dashboard that consumes the local FastAPI search contract."""

from __future__ import annotations

from typing import cast

import streamlit as st

from product_search.config import ProjectSettings, load_settings
from product_search.ui.api_client import (
    ApiClient,
    ApiClientError,
    FeedbackType,
    ProductResult,
    SearchMode,
    SearchResponse,
)
from product_search.ui.benchmarks import BenchmarkReportError, load_benchmark_report

MODE_LABELS = {
    "default": "Default (selected engine)",
    "lexical": "Lexical (TF-IDF)",
    "semantic": "Semantic (dense)",
    "hybrid": "Hybrid",
    "rerank": "Reranked hybrid",
}
COMPARISON_MODES: tuple[SearchMode, ...] = ("lexical", "semantic", "hybrid")


@st.cache_resource
def _api_client(base_url: str, timeout_seconds: float) -> ApiClient:  # pragma: no cover
    """Reuse one thread-safe local HTTP client across Streamlit reruns."""

    return ApiClient(base_url, timeout_seconds=timeout_seconds)


def main() -> None:  # pragma: no cover
    """Render the complete dashboard without loading search artifacts locally."""

    st.set_page_config(
        page_title="Semantic Product Search",
        page_icon="🔎",
        layout="wide",
    )
    settings = load_settings()
    client = _api_client(
        settings.ui.api_base_url,
        settings.ui.request_timeout_seconds,
    )

    st.sidebar.title("Explore")
    page = st.sidebar.radio("Page", ("Search", "Benchmarks", "System"))
    st.sidebar.caption(f"FastAPI: {settings.ui.api_base_url}")

    if page == "Search":
        _render_search_page(client)
    elif page == "Benchmarks":
        _render_benchmark_page(settings)
    else:
        _render_system_page(client)


def _render_search_page(client: ApiClient) -> None:  # pragma: no cover
    st.title("Semantic Product Search")
    st.caption(
        "Search the WANDS catalog with lexical, semantic, hybrid, or reranked retrieval. "
        "All results come from the local FastAPI service."
    )

    try:
        available_modes = client.modes()
    except ApiClientError as error:
        _show_api_error(error)
        return

    with st.form("product_search_form"):
        query = st.text_input(
            "What are you looking for?",
            placeholder="modern black desk lamp",
            max_chars=500,
        )
        controls = st.columns((2, 1, 2))
        with controls[0]:
            mode = st.selectbox(
                "Retrieval mode",
                options=available_modes,
                format_func=lambda value: MODE_LABELS[value],
            )
        with controls[1]:
            top_k = st.number_input("Top K", min_value=1, max_value=100, value=10, step=1)
        with controls[2]:
            comparison = st.checkbox(
                "Compare lexical, semantic, and hybrid",
                help="Runs the same query through three API search modes and shows tabs.",
            )
        submitted = st.form_submit_button("Search", type="primary", use_container_width=True)

    if submitted:
        normalized_query = query.strip()
        if not normalized_query:
            st.warning("Enter a product query before searching.")
        else:
            targets = COMPARISON_MODES if comparison else (mode,)
            responses: dict[str, SearchResponse] = {}
            failures: dict[str, str] = {}
            with st.spinner("Searching the catalog..."):
                for target in targets:
                    try:
                        responses[target] = client.search(
                            normalized_query,
                            top_k=int(top_k),
                            mode=target,
                        )
                    except ApiClientError as error:
                        failures[target] = str(error)
            st.session_state["search_responses"] = responses
            st.session_state["search_failures"] = failures
            st.session_state["search_is_comparison"] = comparison

    stored = st.session_state.get("search_responses")
    if not isinstance(stored, dict):
        return
    responses = cast(dict[str, SearchResponse], stored)
    failures_value = st.session_state.get("search_failures", {})
    failures = cast(dict[str, str], failures_value) if isinstance(failures_value, dict) else {}
    if not responses and not failures:
        return

    for failed_mode, message in failures.items():
        st.error(f"{MODE_LABELS.get(failed_mode, failed_mode)}: {message}")

    if bool(st.session_state.get("search_is_comparison")) and responses:
        ordered_responses = [
            responses[search_mode] for search_mode in COMPARISON_MODES if search_mode in responses
        ]
        tabs = st.tabs([MODE_LABELS[response.mode] for response in ordered_responses])
        for tab, response in zip(tabs, ordered_responses, strict=True):
            with tab:
                _render_search_response(client, response)
    else:
        for response in responses.values():
            _render_search_response(client, response)


def _render_search_response(
    client: ApiClient, response: SearchResponse
) -> None:  # pragma: no cover
    st.subheader(f"{MODE_LABELS[response.mode]} results")
    st.caption(f"{response.result_count} results · {response.latency_ms:.3f} ms service latency")
    if not response.results:
        st.info("No products matched this query.")
        return
    if response.search_id is None:
        st.caption("Feedback is unavailable because this search was not logged by the API.")
    for result in response.results:
        _render_product_card(client, response, result)


def _render_product_card(
    client: ApiClient,
    response: SearchResponse,
    result: ProductResult,
) -> None:  # pragma: no cover
    with st.container(border=True):
        heading, score = st.columns((5, 1))
        with heading:
            st.markdown(f"### {result.rank}. {result.product_name}")
            category = result.category_hierarchy or result.product_class or "Category unavailable"
            st.caption(category)
        with score:
            st.metric("Final score", f"{result.final_score:.4f}")

        st.write(result.short_description or "No description is available for this product.")
        score_parts = []
        if result.lexical_score is not None:
            score_parts.append(f"Lexical: {result.lexical_score:.4f}")
        if result.semantic_score is not None:
            score_parts.append(f"Semantic: {result.semantic_score:.4f}")
        if score_parts:
            st.caption(" · ".join(score_parts))

        with st.expander("Why this result?"):
            terms = result.explanation.matched_query_terms_in_title
            st.write(
                "Title terms: " + (", ".join(terms) if terms else "No direct title-term overlap")
            )
            if result.explanation.lexical_contribution is not None:
                st.write(f"Lexical contribution: {result.explanation.lexical_contribution:.4f}")
            if result.explanation.semantic_contribution is not None:
                st.write(f"Semantic contribution: {result.explanation.semantic_contribution:.4f}")

        relevant, not_relevant, status = st.columns((1, 1, 4))
        disabled = response.search_id is None
        key_root = f"{response.mode}_{response.search_id}_{result.product_id}_{result.rank}"
        with relevant:
            if st.button(
                "Relevant",
                key=f"relevant_{key_root}",
                disabled=disabled,
                use_container_width=True,
            ):
                _submit_feedback(client, response, result, "relevant", key_root)
        with not_relevant:
            if st.button(
                "Not relevant",
                key=f"not_relevant_{key_root}",
                disabled=disabled,
                use_container_width=True,
            ):
                _submit_feedback(client, response, result, "not_relevant", key_root)
        with status:
            feedback_status = st.session_state.get(f"feedback_status_{key_root}")
            if isinstance(feedback_status, str):
                st.success(feedback_status)


def _submit_feedback(
    client: ApiClient,
    response: SearchResponse,
    result: ProductResult,
    feedback_type: FeedbackType,
    key_root: str,
) -> None:  # pragma: no cover
    if response.search_id is None:
        return
    try:
        client.send_feedback(
            search_id=response.search_id,
            product_id=result.product_id,
            feedback_type=feedback_type,
        )
    except ApiClientError as error:
        st.error(str(error))
    else:
        label = "Relevant" if feedback_type == "relevant" else "Not relevant"
        st.session_state[f"feedback_status_{key_root}"] = f"Recorded: {label}"


def _render_benchmark_page(settings: ProjectSettings) -> None:  # pragma: no cover
    st.title("Verified held-out benchmarks")
    report_path = settings.paths.reports / "final_test_metrics.json"
    try:
        report = load_benchmark_report(report_path)
    except BenchmarkReportError as error:
        st.error(str(error))
        return

    st.caption(
        f"Generated {report.created_at} from {report.test_query_count} held-out test queries. "
        "Ranking metrics use judged candidates; latency is end-to-end full-catalog search at K=10."
    )
    table = [
        {
            "Search method": row.display_name,
            "nDCG@10": row.ndcg_at_10,
            "Recall@10": row.recall_at_10,
            "MRR@10": row.mrr_at_10,
            "Median latency (ms)": row.median_latency_ms,
        }
        for row in report.rows
    ]
    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        column_config={
            "nDCG@10": st.column_config.NumberColumn(format="%.4f"),
            "Recall@10": st.column_config.NumberColumn(format="%.4f"),
            "MRR@10": st.column_config.NumberColumn(format="%.4f"),
            "Median latency (ms)": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    st.info(
        "These values are read at runtime from artifacts/reports/final_test_metrics.json; "
        "the dashboard does not contain fallback benchmark numbers."
    )


def _render_system_page(client: ApiClient) -> None:  # pragma: no cover
    st.title("System status")
    try:
        health = client.health()
        readiness = client.readiness()
    except ApiClientError as error:
        _show_api_error(error)
        return

    status_columns = st.columns(3)
    status_columns[0].metric("API health", health.status)
    status_columns[1].metric("Search readiness", readiness.status)
    status_columns[2].metric("API requests", health.metrics.request_count)
    st.caption(
        f"Process errors: {health.metrics.error_count} · "
        f"Average HTTP latency: {health.metrics.average_latency_ms:.3f} ms"
    )
    if not readiness.ready:
        st.warning("FastAPI is running, but required search artifacts are not loaded.")
        return

    try:
        model = client.model()
    except ApiClientError as error:
        _show_api_error(error)
        return

    st.subheader("Selected search system")
    st.metric("Default engine", MODE_LABELS[model.default_search_mode])
    details = st.columns(2)
    details[0].metric("Indexed products", f"{model.product_count:,}")
    details[1].metric("Embedding model", model.embedding_model)
    st.write(f"Index/model build timestamp: `{model.build_timestamp}`")
    st.write(f"Artifact version: `{model.artifact_version}`")


def _show_api_error(error: ApiClientError) -> None:  # pragma: no cover
    st.error(str(error))
    if error.code == "api_unavailable":
        st.code("uv run uvicorn product_search.api.main:app --reload", language="powershell")


if __name__ == "__main__":  # pragma: no cover
    main()
