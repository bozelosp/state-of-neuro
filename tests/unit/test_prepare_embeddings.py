# state_of_neuro/tests/unit/test_prepare_embeddings.py
"""Unit tests for Step C embedding preparation."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts import prepare_embeddings

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "golden" / "prepare_embeddings"
FIXTURES = ROOT / "tests" / "fixtures" / "prepare_embeddings"


def test_prepare_embeddings_batches_and_truncates(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    (input_dir / "sample.json").write_bytes((FIXTURES / "sample.json").read_bytes())

    config = prepare_embeddings.EmbeddingPrepConfig(
        input_dir=input_dir,
        output_dir=tmp_path / "embeddings",
        embed_dim=3,
        block_size=1,
    )

    result = prepare_embeddings.run(config)
    assert result.total_vectors == 2
    assert result.batches_written == 2

    files = sorted(config.output_dir.glob("batch_*.pkl"))
    assert len(files) == 2
    with files[0].open("rb") as handle:
        batch = pickle.load(handle)
    assert batch[0]["embedding"] == [0, 1, 2]
    assert batch[0]["metadata"]["title"] == "Doc 1"
    expected_first = json.loads(GOLDEN.joinpath("batch_0001.json").read_text(encoding="utf-8"))
    assert batch == expected_first

    with files[1].open("rb") as handle:
        second_batch = pickle.load(handle)
    expected_second = json.loads(GOLDEN.joinpath("batch_0002.json").read_text(encoding="utf-8"))
    assert second_batch == expected_second
