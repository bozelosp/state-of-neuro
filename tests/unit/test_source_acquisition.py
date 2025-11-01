# state_of_neuro/tests/unit/test_source_acquisition.py
"""Unit tests for source acquisition manifest parsing."""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
import sys
import textwrap

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import csv

import pytest

from scripts import source_acquisition


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "source_acquisition"
GOLDEN = ROOT / "tests" / "golden" / "source_acquisition"


def test_parse_listing_generates_entries() -> None:
    html = (FIXTURES / "ftp_listing.html").read_text(encoding="utf-8")
    rows = list(source_acquisition.parse_listing(html, "ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline"))
    assert len(rows) == 3
    first = rows[0]
    assert first["name"] == "pubmed25n0001.xml.gz"
    assert first["md5_url"].endswith("pubmed25n0001.xml.gz.md5")
    second = rows[1]
    assert second["name"] == "pubmed25n0002.xml.gz"
    assert second["md5_url"] == ""


def test_run_writes_manifest(tmp_path: Path) -> None:
    html_path = tmp_path / "listing.html"
    html_path.write_text((FIXTURES / "ftp_listing.html").read_text(encoding="utf-8"), encoding="utf-8")
    output_dir = tmp_path / "manifests"
    config = source_acquisition.SourceAcquisitionConfig(
        listing_path=html_path,
        manifest_directory=output_dir,
        ftp_base_url="ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/",
        default_year=2025,
        manifest_name_template="pubmed_baseline_files_{year}.csv",
    )
    result = source_acquisition.run(config)
    manifest_path = result.manifest_path
    assert manifest_path.exists()
    assert result.checksum_path.exists()
    checksum_content = result.checksum_path.read_text(encoding="utf-8").strip()
    assert result.checksum in checksum_content
    with manifest_path.open() as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert len(rows) == 3
    assert rows[0]["name"] == "pubmed25n0001.xml.gz"
    assert rows[1]["md5_url"] == ""
    assert rows[2]["name"] == "pubmed25n0003.xml.gz"
    assert manifest_path.read_text(encoding="utf-8") == (GOLDEN / "pubmed_baseline_files_2025.csv").read_text(encoding="utf-8")
    assert result.downloaded_files == 0


def test_run_downloads_archives_with_md5(tmp_path: Path) -> None:
    ftp_root = tmp_path / "ftp"
    ftp_root.mkdir()

    gz_payload = b"<PubMed baseline sample>"
    gz_path = ftp_root / "pubmed25n0001.xml.gz"
    with gzip.open(gz_path, "wb") as handle:
        handle.write(gz_payload)
    expected_md5 = hashlib.md5(gz_path.read_bytes()).hexdigest()
    (ftp_root / "pubmed25n0001.xml.gz.md5").write_text(
        f"{expected_md5}  pubmed25n0001.xml.gz\n",
        encoding="utf-8",
    )

    listing_html = textwrap.dedent(
        """
        <html><body><pre>
        <a href="pubmed25n0001.xml.gz">pubmed25n0001.xml.gz</a>  01-Jan-2025 01:00  123M
        <a href="pubmed25n0001.xml.gz.md5">pubmed25n0001.xml.gz.md5</a>  01-Jan-2025 01:01  60
        </pre></body></html>
        """
    ).strip()

    listing_path = tmp_path / "listing.html"
    listing_path.write_text(listing_html, encoding="utf-8")

    download_dir = tmp_path / "downloads"
    config = source_acquisition.SourceAcquisitionConfig(
        listing_path=listing_path,
        manifest_directory=tmp_path / "manifests",
        ftp_base_url=ftp_root.as_uri(),
        default_year=2025,
        manifest_name_template="pubmed_baseline_files_2025.csv",
        download_dir=download_dir,
        max_files=5,
        verify_md5=True,
    )

    result = source_acquisition.run(config)

    downloaded_file = download_dir / "pubmed25n0001.xml.gz"
    assert downloaded_file.exists()
    assert hashlib.md5(downloaded_file.read_bytes()).hexdigest() == expected_md5
    md5_sidecar = download_dir / "pubmed25n0001.xml.gz.md5"
    assert md5_sidecar.exists()
    assert expected_md5 in md5_sidecar.read_text(encoding="utf-8")
    assert result.downloaded_files == 1
    assert result.download_dir == download_dir
