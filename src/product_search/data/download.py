"""Reproducible downloader for the three required WANDS dataset files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypedDict

import requests

from product_search.config import load_settings

WANDS_REPOSITORY = "wayfair/WANDS"
WANDS_REVISION = "3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5"
WANDS_DATASET_DIRECTORY = "dataset"
WANDS_DELIMITER = "\t"
REQUIRED_FILES = ("product.csv", "query.csv", "label.csv")
RAW_CONTENT_BASE_URL = "https://raw.githubusercontent.com"


class DownloadError(RuntimeError):
    """Raised when required WANDS data cannot be acquired safely."""


class SourceMetadata(TypedDict):
    """Pinned upstream source recorded in a download manifest."""

    repository: str
    revision: str
    dataset_directory: str


class FileMetadata(TypedDict):
    """Integrity and schema metadata for one downloaded CSV."""

    source_url: str
    cache_status: Literal["downloaded", "reused"]
    sha256: str
    byte_size: int
    row_count: int
    column_names: list[str]


class DownloadManifest(TypedDict):
    """Serializable metadata for one WANDS acquisition run."""

    source: SourceMetadata
    download_timestamp: str
    files: dict[str, FileMetadata]


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Calculate the SHA-256 digest of ``path`` without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        while chunk := source_file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_csv(path: Path) -> tuple[int, list[str]]:
    """Return logical data-row count and column names for a non-empty CSV."""

    _require_non_empty(path)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.reader(csv_file, delimiter=WANDS_DELIMITER)
            try:
                column_names = next(reader)
            except StopIteration as error:
                message = f"CSV file has no header: {path}"
                raise DownloadError(message) from error
            row_count = sum(1 for _ in reader)
    except UnicodeDecodeError as error:
        message = f"CSV file is not valid UTF-8: {path}"
        raise DownloadError(message) from error

    if not column_names or any(not name.strip() for name in column_names):
        message = f"CSV file has an invalid header: {path}"
        raise DownloadError(message)
    if row_count == 0:
        message = f"CSV file contains no data rows: {path}"
        raise DownloadError(message)
    return row_count, column_names


def build_source_url(
    filename: str,
    *,
    repository: str = WANDS_REPOSITORY,
    revision: str = WANDS_REVISION,
) -> str:
    """Build a raw GitHub URL for one allow-listed WANDS dataset file."""

    if filename not in REQUIRED_FILES:
        message = f"unsupported WANDS file requested: {filename}"
        raise ValueError(message)
    return f"{RAW_CONTENT_BASE_URL}/{repository}/{revision}/{WANDS_DATASET_DIRECTORY}/{filename}"


def download_wands_files(
    output_dir: Path,
    *,
    manifest_path: Path | None = None,
    force: bool = False,
    repository: str = WANDS_REPOSITORY,
    revision: str = WANDS_REVISION,
    session: requests.Session | None = None,
    timestamp_factory: Callable[[], datetime] | None = None,
) -> DownloadManifest:
    """Download or reuse only the required WANDS CSVs and write their manifest."""

    output_dir = output_dir.resolve()
    selected_manifest = (manifest_path or output_dir / "manifest.json").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_manifest.parent.mkdir(parents=True, exist_ok=True)

    client = session or requests.Session()
    owns_session = session is None
    files: dict[str, FileMetadata] = {}
    try:
        for filename in REQUIRED_FILES:
            destination = output_dir / filename
            source_url = build_source_url(
                filename,
                repository=repository,
                revision=revision,
            )
            downloaded = _acquire_file(
                source_url,
                destination,
                force=force,
                session=client,
            )
            row_count, column_names = inspect_csv(destination)
            files[filename] = FileMetadata(
                source_url=source_url,
                cache_status="downloaded" if downloaded else "reused",
                sha256=sha256_file(destination),
                byte_size=destination.stat().st_size,
                row_count=row_count,
                column_names=column_names,
            )
    finally:
        if owns_session:
            client.close()

    now = timestamp_factory() if timestamp_factory is not None else datetime.now(UTC)
    manifest = DownloadManifest(
        source=SourceMetadata(
            repository=repository,
            revision=revision,
            dataset_directory=WANDS_DATASET_DIRECTORY,
        ),
        download_timestamp=now.astimezone(UTC).isoformat(),
        files=files,
    )
    _write_json_atomic(selected_manifest, manifest)
    return manifest


def _acquire_file(
    source_url: str,
    destination: Path,
    *,
    force: bool,
    session: requests.Session,
) -> bool:
    if destination.exists() and not force:
        _require_non_empty(destination, force_hint=True)
        return False

    temporary_path = destination.with_name(f"{destination.name}.part")
    response: requests.Response | None = None
    try:
        response = session.get(source_url, stream=True, timeout=(10, 120))
        response.raise_for_status()
        with temporary_path.open("wb") as output_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output_file.write(chunk)
        _require_non_empty(temporary_path)
        os.replace(temporary_path, destination)
    except requests.RequestException as error:
        temporary_path.unlink(missing_ok=True)
        message = f"unable to download {source_url}: {error}"
        raise DownloadError(message) from error
    except OSError as error:
        temporary_path.unlink(missing_ok=True)
        message = f"unable to store WANDS file at {destination}: {error}"
        raise DownloadError(message) from error
    finally:
        if response is not None:
            response.close()
    return True


def _require_non_empty(path: Path, *, force_hint: bool = False) -> None:
    if path.is_file() and path.stat().st_size > 0:
        return
    hint = "; pass --force to replace it" if force_hint else ""
    message = f"required file is missing or empty: {path}{hint}"
    raise DownloadError(message)


def _write_json_atomic(path: Path, payload: DownloadManifest) -> None:
    temporary_path = path.with_name(f"{path.name}.part")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as output_file:
            json.dump(payload, output_file, indent=2, sort_keys=True)
            output_file.write("\n")
        os.replace(temporary_path, path)
    except OSError as error:
        temporary_path.unlink(missing_ok=True)
        message = f"unable to write download manifest at {path}: {error}"
        raise DownloadError(message) from error


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Raw data directory; defaults to the configured project path.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Manifest path; defaults to <output-dir>/manifest.json.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing CSV files. Without this flag cached files are reused.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the WANDS downloader command-line interface."""

    parser = _build_parser()
    arguments = parser.parse_args(argv)
    settings = load_settings()
    output_dir = arguments.output_dir or settings.paths.raw_data
    try:
        manifest = download_wands_files(
            output_dir,
            manifest_path=arguments.manifest,
            force=arguments.force,
            repository=settings.wands_repository,
        )
    except DownloadError as error:
        parser.exit(1, f"error: {error}\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
