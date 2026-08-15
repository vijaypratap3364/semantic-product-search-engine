"""Command-line builder for the persisted FastEmbed product index."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from product_search.config import load_settings
from product_search.indexing.dense import FastEmbedProvider, build_dense_index


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products", type=Path, help="Processed products Parquet input.")
    parser.add_argument("--index-dir", type=Path, help="Dense artifact output directory.")
    parser.add_argument("--batch-size", type=int, help="Products embedded per bounded batch.")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Refuse model downloads and use only the configured local model cache.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing dense artifacts. Existing files are otherwise protected.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build the configured dense index with one lazily loaded model instance."""

    arguments = _build_parser().parse_args(argv)
    settings = load_settings()
    dense_settings = settings.dense
    if arguments.batch_size is not None:
        dense_settings = dense_settings.model_copy(update={"batch_size": arguments.batch_size})
        dense_settings = type(settings.dense).model_validate(dense_settings.model_dump())
    provider = FastEmbedProvider(
        dense_settings.model_name,
        cache_dir=settings.paths.embeddings / "model_cache",
        local_files_only=arguments.local_files_only,
    )
    metadata = build_dense_index(
        products_path=arguments.products or settings.paths.processed_data / "products.parquet",
        output_dir=arguments.index_dir or settings.paths.embeddings / "dense",
        provider=provider,
        settings=dense_settings,
        force=arguments.force,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
