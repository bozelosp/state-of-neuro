# state_of_neuro/scripts/source_acquisition.py
"""Step 1 of the pipeline - curate PubMed baseline manifests and archives.

This module reads an FTP directory listing (from disk or the network), emits a
deterministic CSV manifest with MD5 metadata, and can optionally download the
referenced ``.xml.gz`` files with checksum verification. It mirrors the first
stage driven by ``scripts/run_step.py source_acquisition`` so downstream steps
always receive reproducible PubMed inputs.

Typical usage
-------------
```python
from pathlib import Path
from scripts.source_acquisition import SourceAcquisitionConfig, run

config = SourceAcquisitionConfig(
    listing_path=Path("data/pubmed/listing.html"),
    manifest_directory=Path("artifacts/manifests"),
    ftp_base_url="https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/",
    default_year=2024,
)
result = run(config)
print(result.manifest_path)
```

Key inputs
----------
- `listing_path`: cached HTML from the PubMed baseline FTP listing. When absent,
  the module falls back to ``listing_url`` and writes the snapshot to disk.
- `manifest_directory`: directory that receives the generated manifest and
  checksum files.
- `ftp_base_url`: base URL used to resolve relative download links in the listing.

Outputs
-------
- `<manifest>.csv` plus a matching `.sha256` checksum.
- Optional `.xml.gz` downloads and `.md5` mirrors when ``download_dir`` and
  ``verify_md5`` are enabled.
"""

from __future__ import annotations

import csv
import hashlib
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from bs4 import BeautifulSoup


@dataclass
class SourceAcquisitionConfig:
    """Typed view of manifest-related configuration."""

    listing_path: Path
    manifest_directory: Path
    ftp_base_url: str
    default_year: int
    manifest_name_template: str = "pubmed_baseline_files_{year}.csv"
    listing_url: str | None = None
    download_dir: Path | None = None
    max_files: int | None = None
    verify_md5: bool = True


@dataclass
class SourceAcquisitionResult:
    """Result bundle reporting manifest location and checksum."""

    manifest_path: Path
    checksum_path: Path
    checksum: str
    download_dir: Path | None = None
    downloaded_files: int = 0
    listing_path: Path | None = None


def run(config: SourceAcquisitionConfig) -> SourceAcquisitionResult:
    """Generate a manifest CSV and optionally download baseline archives."""
    html = _load_or_fetch_listing(config)
    rows = list(parse_listing(html, config.ftp_base_url))
    if not rows:
        raise ValueError("No downloadable entries found in listing.")

    config.manifest_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = config.manifest_directory / config.manifest_name_template.format(
        year=config.default_year
    )
    write_manifest(manifest_path, rows)
    checksum = compute_checksum(manifest_path)
    checksum_path = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    checksum_path.write_text(f"{checksum}  {manifest_path.name}\n", encoding="utf-8")

    downloaded_files = 0
    if config.download_dir:
        config.download_dir.mkdir(parents=True, exist_ok=True)
        for index, entry in enumerate(rows):
            if config.max_files is not None and index >= config.max_files:
                break
            if not entry["name"].endswith(".gz"):
                continue
            try:
                downloaded = _download_entry(entry, config.download_dir, verify=config.verify_md5)
            except Exception as exc:  # pragma: no cover - bubbled to caller for visibility.
                raise RuntimeError(f"Failed to download {entry['name']} from {entry['url']}: {exc}") from exc
            if downloaded:
                downloaded_files += 1

    return SourceAcquisitionResult(
        manifest_path=manifest_path,
        checksum_path=checksum_path,
        checksum=checksum,
        download_dir=config.download_dir,
        downloaded_files=downloaded_files,
        listing_path=config.listing_path,
    )


def parse_listing(html: str, base_url: str) -> Iterable[dict[str, str]]:
    """Parse FTP HTML listing and yield manifest rows."""
    soup = BeautifulSoup(html, "html.parser")
    pre = soup.find("pre")
    if pre is None:
        raise ValueError("Listing does not contain <pre> block with directory contents.")

    entries: dict[str, dict[str, str]] = {}
    md5_entries: dict[str, dict[str, str]] = {}
    base_url = base_url.rstrip("/") + "/"

    for anchor in pre.find_all("a"):
        name = (anchor.text or "").strip()
        if name in {"Parent Directory", "README.txt"}:
            continue
        href = anchor.get("href") or ""
        info = _extract_line_info(anchor, name)
        if name.endswith(".md5"):
            key = name[:-4]
            md5_entries[key] = {
                "md5_url": base_url + href,
                "md5_last_modified": info["last_modified"],
                "md5_size": info["size"],
            }
        else:
            entries[name] = {
                "name": name,
                "url": base_url + href,
                "last_modified": info["last_modified"],
                "size": info["size"],
            }

    for name, entry in entries.items():
        entry.update(md5_entries.get(name, {"md5_url": "", "md5_last_modified": "", "md5_size": ""}))
        yield entry


def _extract_line_info(anchor, name: str) -> dict[str, str]:
    """Extract last-modified timestamp and size from listing row."""
    line = anchor.parent.get_text(" ", strip=True)
    parts = line.split()
    if name not in parts:
        raise ValueError(f"Unable to parse listing row for {name!r}.")
    idx = parts.index(name)
    last_modified = ""
    size = ""
    if idx + 2 < len(parts):
        last_modified = f"{parts[idx + 1]} {parts[idx + 2]}"
    if idx + 3 < len(parts):
        size = parts[idx + 3]
    return {"last_modified": last_modified, "size": size}


def write_manifest(path: Path, rows: Iterable[dict[str, str]]) -> None:
    """Write manifest CSV to disk."""
    fieldnames = [
        "name",
        "url",
        "last_modified",
        "size",
        "md5_url",
        "md5_last_modified",
        "md5_size",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def compute_checksum(path: Path) -> str:
    """Compute the SHA-256 checksum for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_or_fetch_listing(config: SourceAcquisitionConfig) -> str:
    if config.listing_path.exists():
        return config.listing_path.read_text(encoding="utf-8")
    if not config.listing_url:
        raise FileNotFoundError(f"Listing file not found: {config.listing_path}")
    listing_html = _read_text_from_url(config.listing_url)
    config.listing_path.parent.mkdir(parents=True, exist_ok=True)
    config.listing_path.write_text(listing_html, encoding="utf-8")
    return listing_html


def _download_entry(entry: dict[str, str], destination_dir: Path, *, verify: bool) -> bool:
    target = destination_dir / entry["name"]
    md5_text: str | None = None
    expected_md5: str | None = None
    md5_url = entry.get("md5_url") or ""
    if verify and md5_url:
        md5_text = _read_text_from_url(md5_url)
        expected_md5 = _parse_md5_from_text(md5_text)
    if target.exists():
        if verify and expected_md5 and _compute_md5(target) == expected_md5:
            if md5_text:
                _write_md5_file(target, md5_text)
            return False
        target.unlink()

    actual_md5 = _stream_download(entry["url"], target)
    if verify and expected_md5 and actual_md5 != expected_md5:
        with suppress(FileNotFoundError):
            target.unlink()
        raise ValueError(f"MD5 mismatch (expected {expected_md5}, got {actual_md5})")
    if md5_text:
        _write_md5_file(target, md5_text)
    return True


def _stream_download(url: str, destination: Path) -> str:
    digest = hashlib.md5()
    try:
        with urlopen(url) as response, destination.open("wb") as handle:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
    except (HTTPError, URLError) as exc:  # pragma: no cover - network failures bubble up.
        with suppress(FileNotFoundError):
            destination.unlink()
        raise RuntimeError(exc) from exc
    return digest.hexdigest()


def _read_text_from_url(url: str) -> str:
    try:
        with urlopen(url) as response:
            data = response.read()
            try:
                encoding = response.headers.get_content_charset("utf-8")
            except AttributeError:
                encoding = "utf-8"
    except (HTTPError, URLError) as exc:  # pragma: no cover - network failures bubble up.
        raise RuntimeError(exc) from exc
    return data.decode(encoding or "utf-8")


def _parse_md5_from_text(md5_payload: str) -> str:
    first_line = md5_payload.strip().splitlines()[0]
    token = first_line.strip().split()[0]
    if not token:
        raise ValueError("MD5 file is empty or malformed.")
    return token


def _write_md5_file(target: Path, payload: str) -> None:
    md5_path = target.parent / f"{target.name}.md5"
    md5_path.write_text(payload, encoding="utf-8")


def _compute_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()
