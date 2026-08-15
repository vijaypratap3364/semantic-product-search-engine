"""Command-line builder for the persisted TF-IDF product index."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from product_search.config import load_settings
from product_search.indexing.tfidf import build_tfidf_index


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products", type=Path, help="Processed products Parquet input.")
    parser.add_argument("--index-dir", type=Path, help="TF-IDF artifact output directory.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing TF-IDF artifacts. Existing files are otherwise protected.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build the configured TF-IDF index."""

    arguments = _build_parser().parse_args(argv)
    settings = load_settings()
    metadata = build_tfidf_index(
        products_path=arguments.products or settings.paths.processed_data / "products.parquet",
        output_dir=arguments.index_dir or settings.paths.indexes / "tfidf",
        settings=settings.lexical,
        force=arguments.force,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
