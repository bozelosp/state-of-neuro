# state_of_neuro/scripts/attach_keywords.py
"""Join keyword batches to article metadata for specificity and topic export."""

from __future__ import annotations

import json
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AttachKeywordsConfig:
    metadata_path: Path
    keywords_dir: Path
    output_path: Path
    records_path: Path


def run(config: AttachKeywordsConfig) -> dict:
    """Join by hash_id without rehashing, retagging or changing source metadata."""
    with config.metadata_path.open("rb") as handle:
        metadata = pickle.load(handle)
    files = sorted(config.keywords_dir.glob("keywords_batch_*.json"))
    if not files:
        raise FileNotFoundError(f"No keyword batches in {config.keywords_dir}")
    assignments = {}
    for file in files:
        for article, tags in json.loads(file.read_text()).items():
            if article not in metadata:
                raise ValueError(f"Keyword hash_id {article!r} has no metadata record")
            if article in assignments and assignments[article] != tags:
                raise ValueError(f"Conflicting keyword batches for {article!r}")
            if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
                raise ValueError(f"Invalid keyword list for {article!r}")
            assignments[article] = tags
    missing = metadata.keys() - assignments.keys()
    if missing:
        raise ValueError(f"Keyword batches omit {len(missing)} metadata records")
    counts = Counter()
    for article, record in metadata.items():
        record["tags"] = assignments[article]
        # Historical neuroscience_tags_specificity.count_tags counts once/paper.
        counts.update(set(record["tags"]))
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    with config.output_path.open("wb") as handle:
        pickle.dump(metadata, handle, protocol=4)
    config.records_path.parent.mkdir(parents=True, exist_ok=True)
    config.records_path.write_text(json.dumps([
        {"label": tag, "frequency": count}
        for tag, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ], indent=2, ensure_ascii=False), encoding="utf-8")
    return {"articles": len(metadata), "tags": len(counts),
            "output_path": str(config.output_path), "records_path": str(config.records_path)}
