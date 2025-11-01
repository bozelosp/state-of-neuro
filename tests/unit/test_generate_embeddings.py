# state_of_neuro/tests/unit/test_generate_embeddings.py
"""Unit tests for asynchronous embedding generation using precomputed vectors."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts import generate_embeddings


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
GOLDEN = ROOT / "tests" / "golden" / "embeddings"


def test_generate_embeddings_produces_expected_batches(tmp_path: Path) -> None:
    input_dir = tmp_path / "abstracts"
    input_dir.mkdir()
    shutil.copy(FIXTURES / "abstracts" / "abstracts_batch_0001.json", input_dir)

    output_dir = tmp_path / "embeddings"
    config = generate_embeddings.EmbeddingGenerationConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        responses_path=FIXTURES / "embeddings" / "responses.json",
        max_concurrency=4,
    )

    result = generate_embeddings.run(config)

    assert result.total_abstracts == 2
    assert result.embeddings_written == 2
    assert result.batches_processed == 1
    assert len(result.output_files) == 1

    output_path = result.output_files[0]
    records = json.loads(output_path.read_text(encoding="utf-8"))
    golden = json.loads(GOLDEN.joinpath("embeddings_batch_0001.json").read_text(encoding="utf-8"))

    def _simplify(batch: list[dict]) -> list[tuple[str, list[float]]]:
        return sorted(
            [(item["hash_id"], item["embedding"]) for item in batch],
            key=lambda pair: pair[0],
        )

    assert _simplify(records) == _simplify(golden)
