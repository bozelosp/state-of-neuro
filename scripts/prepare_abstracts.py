# state_of_neuro/scripts/prepare_abstracts.py
"""Step 3 of the pipeline - normalise, hash, and batch abstract records.

Reads raw PubMed JSON exports, enforces minimum content rules, assigns a
deterministic ``hash_id`` fingerprint, and persists both batched abstracts and
supporting manifests. The outputs are reused by keyword and embedding stages,
so the hashing logic must remain stable across runs.

The module is typically orchestrated through ``scripts/run_step.py
abstract_ingestion`` but can be invoked directly for fixture generation.

Input expectations
------------------
- JSON files in ``input_dir`` should contain lists of PubMed citation dicts.
- Records must expose the fields listed in ``hash_fields`` (defaults to title,
  abstract, doi, and journal) for deterministic hashing.

Outputs
-------
- Batched abstract JSON files stored under ``output_dir``.
- ``hash_id_manifest.csv`` summarising file level statistics plus a `.sha256`.
- Pickled metadata cache keyed by ``hash_id`` for downstream joins.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, MutableMapping, Sequence


DEFAULT_HASH_FIELDS: Sequence[str] = ("title", "abstract", "doi", "journal")


@dataclass
class AbstractIngestionConfig:
    """Configuration for filtering and hashing abstract records."""

    input_dir: Path
    output_dir: Path
    metadata_output: Path
    batch_size: int
    min_abstract_length: int
    manifest_path: Path | None = None
    hash_fields: Sequence[str] = field(default_factory=lambda: DEFAULT_HASH_FIELDS)


@dataclass
class AbstractIngestionResult:
    """Summary statistics from processing abstract files."""

    total_abstracts: int
    retained_abstracts: int
    dropped_short: int
    dropped_invalid: int
    duplicates: int
    batches_written: int
    manifest_path: Path
    manifest_checksum: str
    manifest_changed: bool
    manifest_checksum_path: Path
    metadata_path: Path
    metadata_checksum: str
    metadata_checksum_path: Path


@dataclass
class _FileStats:
    """Per-source file statistics for the manifest."""

    source_file: str
    total: int = 0
    retained: int = 0
    dropped_short: int = 0
    dropped_invalid: int = 0
    duplicates: int = 0


def run(config: AbstractIngestionConfig) -> AbstractIngestionResult:
    """Normalise abstract records, assign hash IDs, and write batched outputs."""
    if not config.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {config.input_dir}")
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_output = config.metadata_output
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = config.manifest_path or (output_dir / "hash_id_manifest.csv")

    seen_hashes: set[str] = set()
    metadata_by_hash: Dict[str, MutableMapping[str, object]] = {}
    manifest_rows: List[_FileStats] = []
    current_batch: List[dict[str, object]] = []
    batch_index = 1

    total_abstracts = retained_abstracts = dropped_short = dropped_invalid = duplicates = 0

    for abstract_file in sorted(config.input_dir.glob("*.json")):
        file_stats = _FileStats(source_file=abstract_file.name)
        manifest_rows.append(file_stats)
        for abstract in _iter_abstracts(abstract_file):
            file_stats.total += 1
            total_abstracts += 1

            normalised = _normalise_abstract(abstract)
            if normalised is None:
                file_stats.dropped_invalid += 1
                dropped_invalid += 1
                continue
            if len(normalised.get("abstract", "")) < config.min_abstract_length:
                file_stats.dropped_short += 1
                dropped_short += 1
                continue

            hash_id = _create_hash_id(normalised, config.hash_fields)
            if hash_id in seen_hashes:
                file_stats.duplicates += 1
                duplicates += 1
                continue

            normalised["hash_id"] = hash_id
            seen_hashes.add(hash_id)
            metadata_by_hash[hash_id] = _extract_metadata(normalised)

            current_batch.append(normalised)
            file_stats.retained += 1
            retained_abstracts += 1

            if len(current_batch) >= config.batch_size:
                _write_batch(output_dir, batch_index, current_batch)
                current_batch.clear()
                batch_index += 1

    if current_batch:
        _write_batch(output_dir, batch_index, current_batch)
        current_batch.clear()
        batch_index += 1

    previous_manifest_checksum = _compute_checksum(manifest_path) if manifest_path.exists() else None
    aggregate_totals = _Totals(
        total=total_abstracts,
        retained=retained_abstracts,
        dropped_short=dropped_short,
        dropped_invalid=dropped_invalid,
        duplicates=duplicates,
    )
    _write_manifest(manifest_path, manifest_rows, aggregate_totals)
    with metadata_output.open("wb") as handle:
        pickle.dump(metadata_by_hash, handle)
    manifest_checksum = _compute_checksum(manifest_path)
    manifest_checksum_path = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    manifest_checksum_path.write_text(f"{manifest_checksum}  {manifest_path.name}\n", encoding="utf-8")

    metadata_checksum = _compute_checksum(metadata_output)
    metadata_checksum_path = metadata_output.with_suffix(metadata_output.suffix + ".sha256")
    metadata_checksum_path.write_text(f"{metadata_checksum}  {metadata_output.name}\n", encoding="utf-8")
    manifest_changed = previous_manifest_checksum != manifest_checksum if previous_manifest_checksum else True

    return AbstractIngestionResult(
        total_abstracts=total_abstracts,
        retained_abstracts=retained_abstracts,
        dropped_short=dropped_short,
        dropped_invalid=dropped_invalid,
        duplicates=duplicates,
        batches_written=batch_index - 1,
        manifest_path=manifest_path,
        manifest_checksum=manifest_checksum,
        manifest_changed=manifest_changed,
        manifest_checksum_path=manifest_checksum_path,
        metadata_path=metadata_output,
        metadata_checksum=metadata_checksum,
        metadata_checksum_path=metadata_checksum_path,
    )


def _iter_abstracts(path: Path) -> Iterator[dict]:
    """Yield abstract dictionaries from a JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        json_payload = json.load(handle)
    if isinstance(json_payload, dict):
        yield json_payload
    elif isinstance(json_payload, list):
        for abstract in json_payload:
            if isinstance(abstract, dict):
                yield abstract
    else:
        raise ValueError(f"Unexpected JSON payload in {path}")


def _normalise_abstract(abstract: dict) -> dict[str, object] | None:
    """Clean and coerce abstract fields into a consistent shape."""
    if not isinstance(abstract, dict):
        return None
    return dict(abstract)


def _create_hash_id(abstract: dict[str, object], fields: Sequence[str]) -> str:
    """Create a deterministic SHA-256 hash based on selected fields."""
    concatenated = "||".join(str(abstract.get(field, "None")) for field in fields)
    digest = hashlib.sha256()
    digest.update(concatenated.encode("utf-8"))
    return digest.hexdigest()


def _extract_metadata(abstract: dict[str, object]) -> dict[str, object]:
    """Select a subset of fields for metadata pickle output."""
    return {
        key: abstract.get(key)
        for key in ("title", "abstract", "pubdate", "journal", "doi", "authors", "hash_id")
    }


def _write_batch(output_dir: Path, index: int, abstracts: Iterable[dict[str, object]]) -> None:
    """Write a batch of abstracts to JSON."""
    batch_path = output_dir / f"abstracts_batch_{index:04d}.json"
    with batch_path.open("w", encoding="utf-8") as handle:
        json.dump(list(abstracts), handle, ensure_ascii=False, indent=2)


def _write_manifest(path: Path, rows: Iterable[_FileStats], totals: _Totals) -> None:
    """Write per-file statistics to manifest CSV."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_file",
                "total",
                "retained",
                "dropped_short",
                "dropped_invalid",
                "duplicates",
            ],
        )
        writer.writeheader()
        for stats in rows:
            writer.writerow(
                {
                    "source_file": stats.source_file,
                    "total": stats.total,
                    "retained": stats.retained,
                    "dropped_short": stats.dropped_short,
                    "dropped_invalid": stats.dropped_invalid,
                    "duplicates": stats.duplicates,
                }
            )
        writer.writerow(
            {
                "source_file": "__TOTAL__",
                "total": totals.total,
                "retained": totals.retained,
                "dropped_short": totals.dropped_short,
                "dropped_invalid": totals.dropped_invalid,
                "duplicates": totals.duplicates,
            }
        )


def _compute_checksum(path: Path) -> str:
    """Compute SHA-256 checksum for the given file if it exists."""
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class _Totals:
    """Aggregate totals written to the manifest."""

    total: int
    retained: int
    dropped_short: int
    dropped_invalid: int
    duplicates: int
