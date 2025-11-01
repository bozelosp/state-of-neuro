# state_of_neuro/scripts/generate_embeddings.py
"""Step 5 of the pipeline - replay cached embedding vectors asynchronously.

This module recreates the concurrent embedding run by loading precomputed
``text-embedding-3-large`` responses from disk, enforcing the original semaphore
limits, and emitting batches that :mod:`scripts.prepare_embeddings` consumes in
the next step. No live OpenAI calls are made, which keeps reruns reproducible.

Prerequisites
-------------
- Abstract batches ``abstracts_batch_*.json`` inside ``input_dir``.
- Response cache mapping ``hash_id`` to embedding vectors supplied via
  ``responses_path``.

Outputs
-------
- Deterministic ``embeddings_batch_*.json`` files under ``output_dir``.
- Structured counts summarised by :class:`EmbeddingGenerationResult`.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


@dataclass
class EmbeddingGenerationConfig:
    """Configuration for embedding generation using precomputed vectors."""

    input_dir: Path
    output_dir: Path
    responses_path: Path
    max_concurrency: int = 50


@dataclass
class EmbeddingGenerationResult:
    """Summary of embedding generation."""

    total_abstracts: int
    embeddings_written: int
    batches_processed: int
    output_files: Sequence[Path]


def run(config: EmbeddingGenerationConfig) -> EmbeddingGenerationResult:
    """Emit embedding batches using precomputed vectors with asynchronous concurrency."""

    if not config.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {config.input_dir}")
    if not config.responses_path.exists():
        raise FileNotFoundError(f"Embedding responses file not found: {config.responses_path}")
    if config.max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")

    abstract_files = sorted(config.input_dir.glob("abstracts_batch_*.json"))
    if not abstract_files:
        raise FileNotFoundError(f"No abstract batches found in {config.input_dir}")

    responses = _load_responses(config.responses_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    total_abstracts = 0
    embeddings_written = 0
    output_files: List[Path] = []

    for index, batch_path in enumerate(abstract_files, start=1):
        abstracts = _load_abstracts(batch_path)
        total_abstracts += len(abstracts)
        embedding_records = asyncio.run(
            _generate_embeddings_for_batch(abstracts, responses, config.max_concurrency)
        )
        embeddings_written += len(embedding_records)
        output_path = config.output_dir / f"embeddings_batch_{index:04d}.json"
        output_path.write_text(
            json.dumps(embedding_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        output_files.append(output_path)

    return EmbeddingGenerationResult(
        total_abstracts=total_abstracts,
        embeddings_written=embeddings_written,
        batches_processed=len(abstract_files),
        output_files=tuple(output_files),
    )


def _load_responses(path: Path) -> Mapping[str, Sequence[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Embedding responses must be a JSON object.")
    normalised: Dict[str, Sequence[float]] = {}
    for key, value in payload.items():
        if not isinstance(value, list):
            raise ValueError(f"Embedding response for {key!r} must be a list.")
        normalised[str(key)] = [float(component) for component in value]
    return normalised


def _load_abstracts(path: Path) -> List[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Abstract batch must be a list: {path}")
    return payload


async def _generate_embeddings_for_batch(
    abstracts: Iterable[dict],
    responses: Mapping[str, Sequence[float]],
    max_concurrency: int,
) -> List[dict]:
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _produce(index: int, record: dict) -> tuple[int, dict]:
        async with semaphore:
            hash_id = str(record.get("hash_id") or "")
            if not hash_id:
                raise ValueError("Abstract record missing 'hash_id'.")
            if hash_id not in responses:
                raise KeyError(f"No embedding response recorded for hash_id {hash_id}")
            await asyncio.sleep(0)
            payload = {
                "hash_id": hash_id,
                "embedding": responses[hash_id],
                "title": record.get("title"),
                "abstract": record.get("abstract"),
            }
            pubdate = record.get("pubdate")
            if pubdate is not None:
                payload["pubdate"] = pubdate
            return index, payload

    tasks = [_produce(idx, record) for idx, record in enumerate(abstracts)]
    results = await asyncio.gather(*tasks)
    results.sort(key=lambda item: item[0])
    return [payload for _, payload in results]


__all__ = [
    "EmbeddingGenerationConfig",
    "EmbeddingGenerationResult",
    "run",
]
