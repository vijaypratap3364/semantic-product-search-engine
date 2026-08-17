"""Benchmark local search components and generate factual search error analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import Thread
from typing import Any, cast

import numpy as np
import pandas as pd
import requests
import uvicorn
from numpy.typing import NDArray

from product_search.api.main import create_app
from product_search.config import ProjectSettings, load_settings
from product_search.data.download import sha256_file
from product_search.evaluation.benchmark_final import collect_hardware_metadata
from product_search.evaluation.error_analysis import (
    ErrorAnalysis,
    categorize_quality_records,
    compute_quality_records,
    error_analysis_payload,
    write_error_analysis,
)
from product_search.evaluation.performance import (
    BenchmarkOperations,
    collect_artifact_sizes,
    current_process_rss_bytes,
    latency_payload,
    run_performance_benchmark,
    select_representative_queries,
)
from product_search.indexing.tfidf import build_tfidf_index
from product_search.ranking.reranker import RerankingSearchEngine
from product_search.retrieval.base import SearchResult
from product_search.retrieval.hybrid import HybridSearchEngine
from product_search.retrieval.lexical import LexicalSearchEngine
from product_search.retrieval.semantic import SemanticSearchEngine
from product_search.service import SearchService

DEFAULT_SAMPLE_SIZE = 20
DEFAULT_REPEATS = 5
DEFAULT_WARMUP_QUERIES = 2
MAX_TIMED_REPETITIONS = 200


def run_benchmark(
    settings: ProjectSettings,
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    repeats: int = DEFAULT_REPEATS,
    warmup_query_count: int = DEFAULT_WARMUP_QUERIES,
    top_k: int = 10,
    seed: int = 42,
    local_files_only: bool = True,
    measure_lexical_build: bool = True,
) -> tuple[dict[str, object], ErrorAnalysis]:  # pragma: no cover
    """Run the bounded real-artifact benchmark and return JSON/Markdown source data."""

    if sample_size * repeats > MAX_TIMED_REPETITIONS:
        raise ValueError(f"sample_size * repeats must not exceed {MAX_TIMED_REPETITIONS}")
    queries_path = settings.paths.processed_data / "queries.parquet"
    judgments_path = settings.paths.processed_data / "evaluation_judgments.parquet"
    products_path = settings.paths.processed_data / "products.parquet"
    splits_path = settings.paths.processed_data / "query_splits.json"
    initial_rss = current_process_rss_bytes()
    service = SearchService.load(settings, local_files_only=local_files_only)
    loaded_rss = current_process_rss_bytes()
    lexical, semantic, hybrid, reranker = _loaded_engines(service)

    queries = pd.read_parquet(queries_path)
    sample = select_representative_queries(queries, sample_size=sample_size, seed=seed)
    sample_queries = sample["query"].astype(str).tolist()
    with _local_api_search(service) as api_search:
        operations = _benchmark_operations(
            lexical,
            semantic,
            hybrid,
            reranker,
            api_search=api_search,
        )
        performance = run_performance_benchmark(
            sample_queries,
            operations,
            top_k=top_k,
            repeats=repeats,
            warmup_query_count=warmup_query_count,
        )
    after_benchmark_rss = current_process_rss_bytes()

    split_manifest = _read_json(splits_path)
    test_ids = set(_string_list(_mapping(_mapping(split_manifest, "query_ids"), "test")))
    test_queries = queries[queries["query_id"].astype(str).isin(test_ids)].copy()
    judgments = pd.read_parquet(judgments_path)
    test_judgments = judgments[judgments["query_id"].astype(str).isin(test_ids)].copy()
    product_frame = pd.read_parquet(products_path, columns=["product_id", "product_name"])
    product_names = dict(
        zip(
            product_frame["product_id"].astype(str),
            product_frame["product_name"].astype(str),
            strict=True,
        )
    )
    records = compute_quality_records(
        lexical=lexical,
        semantic=semantic,
        reranker=reranker,
        queries=test_queries,
        judgments=test_judgments,
        product_names=product_names,
        strategy=settings.hybrid.strategy,
        semantic_weight=settings.hybrid.semantic_weight,
        candidate_depth=settings.hybrid.candidate_depth,
        rrf_k=settings.hybrid.rrf_k,
        top_k=top_k,
    )
    error_analysis = categorize_quality_records(records)

    lexical_dir = settings.paths.indexes / "tfidf"
    dense_dir = settings.paths.embeddings / "dense"
    model_dir = settings.paths.models / "reranker"
    artifact_sizes = collect_artifact_sizes(
        {"lexical_index": lexical_dir, "dense_index": dense_dir, "reranker_model": model_dir}
    )
    build_durations = {
        "lexical_index": (
            _measure_full_lexical_build(settings, products_path)
            if measure_lexical_build
            else {"status": "skipped_by_user"}
        ),
        "dense_index": _recorded_or_unavailable_build_duration(dense_dir / "metadata.json"),
    }
    observed_rss = performance.observed_rss_bytes
    report: dict[str, object] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "hardware": collect_hardware_metadata(),
        "catalog_product_count": service.product_count,
        "embedding_model": service.metadata.embedding_model,
        "default_search_mode": service.default_mode,
        "configuration": {
            "seed": seed,
            "sample_size": sample_size,
            "repeats": repeats,
            "timed_observation_count_per_component": sample_size * repeats,
            "warmup_query_count": min(warmup_query_count, sample_size),
            "top_k": top_k,
            "maximum_timed_query_repetitions": MAX_TIMED_REPETITIONS,
            "p99_minimum_sample_count": 100,
            "query_sample_method": "proportional_query_length_stratification",
        },
        "query_sample": {
            "query_ids": sample["query_id"].astype(str).tolist(),
            "query_sample_sha256": _sample_hash(sample["query_id"].astype(str).tolist()),
            "population_query_count": len(queries),
            "population_length_buckets": _counts(queries["query"].astype(str)),
            "sample_length_buckets": sample["query_length_bucket"].value_counts().to_dict(),
        },
        "latency_boundaries": {
            "lexical_query": (
                "LexicalSearchEngine.search including TF-IDF query transform, sparse scoring, "
                "and top-K."
            ),
            "semantic_embedding": "FastEmbed query encoding plus project L2 normalization.",
            "semantic_retrieval": (
                "Exact dot products against all products plus top-K, using a precomputed query "
                "embedding."
            ),
            "hybrid_search": "HybridSearchEngine.search including both base retrievals and fusion.",
            "reranking_stage": (
                "Feature extraction, logistic probabilities, expected relevance, and reranking "
                "of an already generated hybrid pool."
            ),
            "api_end_to_end": (
                "Loopback HTTP POST /search through Uvicorn/FastAPI, schema serialization, "
                "SearchService default mode, and response decoding; analytics disabled."
            ),
            "initialization_excluded": True,
        },
        "latencies_ms": latency_payload(performance),
        "process_memory": {
            "measurement": "Windows process working set sampled with GetProcessMemoryInfo",
            "before_service_load_bytes": initial_rss,
            "after_service_load_bytes": loaded_rss,
            "after_timed_benchmark_bytes": after_benchmark_rss,
            "maximum_observed_during_timed_benchmark_bytes": (
                max(observed_rss) if observed_rss else None
            ),
            "service_initialization_ms": service.initialization_time_ms,
        },
        "artifact_disk_sizes": artifact_sizes,
        "index_build_duration": build_durations,
        "source_hashes": {
            "products.parquet": sha256_file(products_path),
            "queries.parquet": sha256_file(queries_path),
            "evaluation_judgments.parquet": sha256_file(judgments_path),
            "query_splits.json": sha256_file(splits_path),
            "lexical_metadata.json": sha256_file(lexical_dir / "metadata.json"),
            "dense_metadata.json": sha256_file(dense_dir / "metadata.json"),
            "reranker_metadata.json": sha256_file(model_dir / "metadata.json"),
        },
        "error_analysis": {
            "split": "test",
            "method": "controlled_judged_candidate_ndcg_at_10",
            **error_analysis_payload(error_analysis),
        },
        "scaling_note": (
            "Exact NumPy scoring is appropriate for this 42,994-product catalog. Reassess ANN "
            "only after measured latency or memory misses the service target at a materially "
            "larger catalog; benchmark candidate approaches before adding infrastructure."
        ),
    }
    return report, error_analysis


def write_benchmark_report(path: Path, report: Mapping[str, object]) -> None:
    """Atomically write benchmark.json."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.part")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            json.dump(report, output, indent=2, sort_keys=True)
            output.write("\n")
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def _benchmark_operations(
    lexical: LexicalSearchEngine,
    semantic: SemanticSearchEngine,
    hybrid: HybridSearchEngine,
    reranker: RerankingSearchEngine,
    *,
    api_search: Callable[[str, int], object],
) -> BenchmarkOperations:
    def semantic_retrieve(encoded: object, top_k: int) -> object:
        if not isinstance(encoded, np.ndarray):
            raise ValueError("semantic embedding did not return a NumPy vector")
        return semantic.search_embedding(cast(NDArray[np.float32], encoded), top_k)

    def prepare_candidates(query: str, top_k: int) -> object:
        return hybrid.search(query, max(top_k, reranker.candidate_depth))

    def rerank(query: str, candidates: object, top_k: int) -> object:
        if not isinstance(candidates, list) or not all(
            isinstance(candidate, SearchResult) for candidate in candidates
        ):
            raise ValueError("reranker candidate preparation returned invalid results")
        return reranker.rerank(query, cast(list[SearchResult], candidates), top_k)

    return BenchmarkOperations(
        lexical_search=lexical.search,
        semantic_embed=semantic.embed_query,
        semantic_retrieve=semantic_retrieve,
        hybrid_search=hybrid.search,
        prepare_rerank_candidates=prepare_candidates,
        rerank=rerank,
        api_search=api_search,
    )


def _loaded_engines(
    service: SearchService,
) -> tuple[
    LexicalSearchEngine,
    SemanticSearchEngine,
    HybridSearchEngine,
    RerankingSearchEngine,
]:  # pragma: no cover
    engines = (
        service.loaded_engine("lexical"),
        service.loaded_engine("semantic"),
        service.loaded_engine("hybrid"),
        service.loaded_engine("rerank"),
    )
    expected_types = (
        LexicalSearchEngine,
        SemanticSearchEngine,
        HybridSearchEngine,
        RerankingSearchEngine,
    )
    if not all(
        isinstance(engine, expected)
        for engine, expected in zip(engines, expected_types, strict=True)
    ):
        raise TypeError("loaded service engines do not match the performance benchmark contract")
    return cast(
        tuple[
            LexicalSearchEngine,
            SemanticSearchEngine,
            HybridSearchEngine,
            RerankingSearchEngine,
        ],
        engines,
    )


@contextmanager
def _local_api_search(
    service: SearchService,
) -> Iterator[Callable[[str, int], object]]:  # pragma: no cover
    """Serve the injected loaded service over a real loopback HTTP socket."""

    app = create_app(service_loader=lambda: service, analytics_loader=None)
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("127.0.0.1", 0))
    server_socket.listen(128)
    port = int(server_socket.getsockname()[1])
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", lifespan="on")
    )
    thread = Thread(
        target=server.run,
        kwargs={"sockets": [server_socket]},
        name="performance-benchmark-api",
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + 10.0
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=5.0)
        server_socket.close()
        raise RuntimeError("local benchmark API failed to start")

    session = requests.Session()

    def search(query: str, top_k: int) -> object:
        response = session.post(
            f"http://127.0.0.1:{port}/search",
            json={"query": query, "top_k": top_k, "mode": "default"},
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()

    try:
        yield search
    finally:
        session.close()
        server.should_exit = True
        thread.join(timeout=10.0)
        server_socket.close()
        if thread.is_alive():
            raise RuntimeError("local benchmark API did not stop cleanly")


def _measure_full_lexical_build(
    settings: ProjectSettings,
    products_path: Path,
) -> dict[str, object]:  # pragma: no cover
    with tempfile.TemporaryDirectory(prefix="product-search-lexical-build-") as temporary:
        started_at = time.perf_counter()
        metadata = build_tfidf_index(
            products_path,
            Path(temporary) / "tfidf",
            settings=settings.lexical,
        )
        duration = time.perf_counter() - started_at
    return {
        "status": "measured_full_catalog_temporary_build",
        "duration_seconds": duration,
        "product_count": metadata["product_count"],
        "vocabulary_size": metadata["vocabulary_size"],
        "includes": "parquet read, fit/transform, serialization, hashing, and atomic publication",
    }


def _recorded_or_unavailable_build_duration(metadata_path: Path) -> dict[str, object]:
    metadata = _read_json(metadata_path)
    value = metadata.get("build_duration_seconds")
    if isinstance(value, int | float) and not isinstance(value, bool) and value >= 0:
        return {"status": "recorded_in_artifact_metadata", "duration_seconds": float(value)}
    return {
        "status": "unavailable_prior_build_not_instrumented",
        "reason": (
            "The existing dense metadata records creation time but not elapsed build time. "
            "A full 42,994-product re-embedding was intentionally skipped to bound CPU load."
        ),
    }


def _counts(queries: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for query in queries:
        length = len(query.split())
        bucket = (
            "one_token"
            if length <= 1
            else "two_tokens"
            if length == 2
            else "three_to_four_tokens"
            if length <= 4
            else "five_or_more_tokens"
        )
        counts[bucket] = counts.get(bucket, 0) + 1
    return dict(sorted(counts.items()))


def _sample_hash(query_ids: Sequence[str]) -> str:
    encoded = json.dumps(list(query_ids), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"required benchmark input does not exist: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"benchmark JSON input must be an object: {path}")
    return cast(dict[str, Any], value)


def _mapping(value: object, key: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"query split field {key!r} must be an object")
    if key not in value:
        raise ValueError(f"query split field {key!r} is missing")
    nested = cast(dict[str, Any], value)[key]
    if key == "test":
        return {"values": nested}
    if not isinstance(nested, dict):
        raise ValueError(f"query split field {key!r} must be an object")
    return cast(dict[str, Any], nested)


def _string_list(value: Mapping[str, Any]) -> list[str]:
    raw = value.get("values")
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ValueError("test query IDs must be a string list")
    return cast(list[str], raw)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--warmup-query-count", type=int, default=DEFAULT_WARMUP_QUERIES)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-lexical-build-measurement", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Run the benchmark and write ignored generated reports."""

    arguments = _build_parser().parse_args(argv)
    settings = load_settings()
    benchmark_path = settings.paths.reports / "benchmark.json"
    errors_path = settings.paths.reports / "error_analysis.md"
    existing = [path.name for path in (benchmark_path, errors_path) if path.exists()]
    if existing and not arguments.force:
        raise FileExistsError(
            f"benchmark outputs already exist: {sorted(existing)}; pass --force to replace them"
        )
    report, analysis = run_benchmark(
        settings,
        sample_size=arguments.sample_size,
        repeats=arguments.repeats,
        warmup_query_count=arguments.warmup_query_count,
        top_k=arguments.top_k,
        seed=arguments.seed if arguments.seed is not None else settings.random_seed,
        local_files_only=arguments.local_files_only,
        measure_lexical_build=not arguments.skip_lexical_build_measurement,
    )
    write_benchmark_report(benchmark_path, report)
    write_error_analysis(errors_path, analysis, split="test")
    print(
        json.dumps(
            {
                "benchmark_path": str(benchmark_path),
                "error_analysis_path": str(errors_path),
                "latencies_ms": report["latencies_ms"],
                "error_analysis": report["error_analysis"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
