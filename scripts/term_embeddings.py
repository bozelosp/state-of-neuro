# state_of_neuro/scripts/term_embeddings.py
"""Generate real cached term embeddings for the recovered canonicalization stage.

Uses the historical embedding_utils.generate_embedding request (float vectors,
text-embedding-3-large), batching requests without carrying over its random
vector fallback. Abstract embeddings from Step 6 are not term embeddings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .canonical_clustering import load_categories


@dataclass
class TermEmbeddingsConfig:
    output_path: Path
    categories_dir: Path | None = None
    records_path: Path | None = None
    top_n: int = 15000
    model: str = "text-embedding-3-large"
    batch_size: int = 64


def run(config: TermEmbeddingsConfig, client=None) -> dict:
    """Reuse existing vectors and request only missing category terms."""
    terms = set()
    if config.categories_dir:
        categories = load_categories(config.categories_dir)
        terms.update(tag for entries in categories.values() for tag in entries)
    if config.records_path:
        records = json.loads(config.records_path.read_text())
        ranked = sorted(records, key=lambda record: -record["frequency"])[:config.top_n]
        terms.update(record["label"] for record in ranked)
    if not terms:
        raise ValueError("Supply category CSVs or keyword frequency records")
    terms = sorted(terms)
    cache = json.loads(config.output_path.read_text()) if config.output_path.exists() else {}
    missing = [tag for tag in terms if tag not in cache]
    if config.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if missing and client is None:
        from openai import OpenAI
        client = OpenAI()
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    for offset in range(0, len(missing), config.batch_size):
        batch = missing[offset:offset + config.batch_size]
        response = client.embeddings.create(model=config.model, input=batch, encoding_format="float")
        returned = {entry.index: entry.embedding for entry in response.data}
        if set(returned) != set(range(len(batch))):
            raise ValueError("Embedding response omitted or misindexed requested terms")
        for index, tag in enumerate(batch):
            cache[tag] = returned[index]
        temporary = config.output_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(cache), encoding="utf-8")
        temporary.replace(config.output_path)
    return {"terms": len(terms), "new_vectors": len(missing), "output_path": str(config.output_path)}
