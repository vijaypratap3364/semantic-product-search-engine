"""Tests for reproducible WANDS downloads and manifests."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import requests

from product_search.data.download import (
    REQUIRED_FILES,
    DownloadError,
    build_source_url,
    download_wands_files,
    inspect_csv,
    sha256_file,
)
from product_search.data.download import main as download_main


class FakeResponse:
    """Small requests response stand-in backed by fixture bytes."""

    def __init__(self, content: bytes) -> None:
        self.content = content
        self.closed = False

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, *, chunk_size: int) -> list[bytes]:
        assert chunk_size > 0
        return [self.content]

    def close(self) -> None:
        self.closed = True


class FixtureSession(requests.Session):
    """Serve requested WANDS filenames from committed fixture files."""

    def __init__(self, fixture_dir: Path) -> None:
        super().__init__()
        self.fixture_dir = fixture_dir
        self.requested_urls: list[str] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:  # type: ignore[override]
        assert kwargs["stream"] is True
        self.requested_urls.append(url)
        return FakeResponse((self.fixture_dir / Path(url).name).read_bytes())


class OfflineSession(requests.Session):
    """Fail every network request with a requests-level connection error."""

    def get(self, url: str, **kwargs: Any) -> requests.Response:  # type: ignore[override]
        raise requests.ConnectionError(f"offline while requesting {url}")


def test_sha256_file(tmp_path: Path) -> None:
    target = tmp_path / "payload.txt"
    target.write_bytes(b"abc")

    assert sha256_file(target) == hashlib.sha256(b"abc").hexdigest()


def test_downloader_fetches_only_required_files(
    tmp_path: Path,
    raw_wands_dir: Path,
) -> None:
    output_dir = tmp_path / "downloaded"
    session = FixtureSession(raw_wands_dir)
    timestamp = datetime(2026, 1, 2, 3, 4, tzinfo=UTC)

    manifest = download_wands_files(
        output_dir,
        session=session,
        timestamp_factory=lambda: timestamp,
    )

    assert tuple(manifest["files"]) == REQUIRED_FILES
    assert len(session.requested_urls) == len(REQUIRED_FILES)
    assert all("/dataset/" in url for url in session.requested_urls)
    assert manifest["download_timestamp"] == timestamp.isoformat()
    assert manifest["files"]["product.csv"]["row_count"] == 3
    assert manifest["files"]["query.csv"]["column_names"] == [
        "query_id",
        "query",
        "query_class",
    ]
    assert (output_dir / "manifest.json").is_file()


def test_cached_files_are_not_overwritten_without_force(raw_wands_dir: Path) -> None:
    before = {name: (raw_wands_dir / name).read_bytes() for name in REQUIRED_FILES}
    session = OfflineSession()

    manifest = download_wands_files(raw_wands_dir, session=session)

    after = {name: (raw_wands_dir / name).read_bytes() for name in REQUIRED_FILES}
    assert after == before
    assert all(metadata["cache_status"] == "reused" for metadata in manifest["files"].values())


def test_force_replaces_cached_files(tmp_path: Path, raw_wands_dir: Path) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    for name in REQUIRED_FILES:
        (output_dir / name).write_text("old content", encoding="utf-8")

    manifest = download_wands_files(
        output_dir,
        force=True,
        session=FixtureSession(raw_wands_dir),
    )

    assert all(metadata["cache_status"] == "downloaded" for metadata in manifest["files"].values())
    assert (output_dir / "product.csv").read_bytes() == (raw_wands_dir / "product.csv").read_bytes()


def test_network_failure_has_clear_context(tmp_path: Path) -> None:
    with pytest.raises(DownloadError, match="unable to download"):
        download_wands_files(tmp_path / "raw", session=OfflineSession())


def test_empty_cached_file_requires_force(tmp_path: Path) -> None:
    output_dir = tmp_path / "raw"
    output_dir.mkdir()
    (output_dir / "product.csv").touch()

    with pytest.raises(DownloadError, match="pass --force"):
        download_wands_files(output_dir, session=OfflineSession())


def test_source_url_rejects_unrelated_files() -> None:
    with pytest.raises(ValueError, match="unsupported WANDS file"):
        build_source_url("README.md")


def test_csv_inspection_rejects_header_without_rows(tmp_path: Path) -> None:
    target = tmp_path / "header-only.csv"
    target.write_text("id\tname\n", encoding="utf-8")

    with pytest.raises(DownloadError, match="contains no data rows"):
        inspect_csv(target)


def test_download_cli_reuses_cached_files(
    raw_wands_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = download_main(["--output-dir", str(raw_wands_dir)])

    assert exit_code == 0
    assert '"cache_status": "reused"' in capsys.readouterr().out
    assert (raw_wands_dir / "manifest.json").is_file()
