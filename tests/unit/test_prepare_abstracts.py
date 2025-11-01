# state_of_neuro/tests/unit/test_prepare_abstracts.py
"""Unit tests for Step 3 abstract ingestion."""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts import prepare_abstracts

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "golden" / "prepare_abstracts"


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_prepare_abstracts_filters_and_batches(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    file_one = input_dir / "part1.json"
    file_two = input_dir / "part2.json"

    _write_json(
        file_one,
        [
            {
                "title": "Alpha",
                "abstract": "A" * 500,
                "doi": "10.1000/alpha",
                "journal": "Neuro",
                "pubdate": "2024",
            },
            {
                "title": "Beta",
                "abstract": "short",
                "doi": "10.1000/beta",
                "journal": "Neuro",
                "pubdate": "2024",
            },
        ],
    )
    _write_json(
        file_two,
        [
            {
                "title": "Alpha",  # duplicate by hash
                "abstract": "A" * 500,
                "doi": "10.1000/alpha",
                "journal": "Neuro",
                "pubdate": "2024",
            },
            {
                "title": "Gamma",
                "abstract": "G" * 600,
                "doi": "10.1000/gamma",
                "journal": "Neuro",
                "pubdate": "2025",
            },
        ],
    )

    output_dir = tmp_path / "processed"
    metadata_output = tmp_path / "meta" / "hash_id_to_metadata.pkl"
    config = prepare_abstracts.AbstractIngestionConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        metadata_output=metadata_output,
        batch_size=1,
        min_abstract_length=100,
    )

    result = prepare_abstracts.run(config)
    assert result.total_abstracts == 4
    assert result.retained_abstracts == 2
    assert result.dropped_short == 1
    assert result.duplicates == 1
    assert result.batches_written == 2
    assert result.manifest_path.exists()
    assert metadata_output.exists()
    assert result.manifest_checksum
    assert result.metadata_checksum
    assert result.manifest_checksum_path.exists()
    assert result.metadata_checksum_path.exists()
    assert result.manifest_changed is True
    assert result.manifest_path.read_text(encoding="utf-8") == GOLDEN.joinpath("hash_id_manifest.csv").read_text(encoding="utf-8")
    assert result.metadata_checksum == GOLDEN.joinpath("metadata_checksum.txt").read_text(encoding="utf-8").strip()

    batch_files = sorted(output_dir.glob("abstracts_batch_*.json"))
    assert len(batch_files) == 2
    first_batch = json.loads(batch_files[0].read_text())
    assert first_batch[0]["hash_id"]

    with metadata_output.open("rb") as handle:
        metadata = pickle.load(handle)
    assert len(metadata) == 2

    with result.manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert rows[0]["source_file"] == "part1.json"
    assert rows[0]["retained"] == "1"
    assert rows[1]["duplicates"] == "1"
    assert rows[2]["source_file"] == "__TOTAL__"
    assert rows[2]["retained"] == "2"

    second_result = prepare_abstracts.run(config)
    assert second_result.manifest_changed is False
    assert second_result.manifest_checksum == result.manifest_checksum
    assert second_result.manifest_checksum_path == result.manifest_checksum_path
    assert second_result.metadata_checksum_path == result.metadata_checksum_path
