# state_of_neuro/scripts/category_assignment.py
"""Assign and rank terms using the historical nearest-category cosine scan.

Recovered from neuroscience_tags_specificity.py at private source revision
61358677e2ce862dd465c99c61541782d8549a70 (similarity scan and composite score).
These CSVs are automated assignments; they do not replace the original team's
manual review decisions when reproducing the published dataset.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .canonical_clustering import cosine_similarity
from .category_resolutions import CATEGORY_TO_GROUP
from .specificity_scoring import validate_flags


@dataclass
class CategoryAssignmentConfig:
    records_path: Path
    embeddings_path: Path
    category_embeddings_path: Path
    specificity_path: Path
    output_dir: Path
    embedding_dimensions: int = 800
    top_n: int = 15000
    min_similarity: float = 0.01
    max_similarity: float = 0.70


def run(config: CategoryAssignmentConfig) -> dict:
    """Select the maximum-cosine category and recover the original ranking."""
    records = json.loads(config.records_path.read_text())
    embeddings = json.loads(config.embeddings_path.read_text())
    category_vectors = json.loads(config.category_embeddings_path.read_text())
    flags = json.loads(config.specificity_path.read_text())
    if not category_vectors or category_vectors.keys() - CATEGORY_TO_GROUP.keys():
        raise ValueError("Category vectors must use the documented historical category slugs")
    candidates = sorted(records, key=lambda record: -record["frequency"])[:config.top_n]
    categories = defaultdict(list)
    for record in candidates:
        tag, count = record["label"], record["frequency"]
        vector = embeddings[tag][:config.embedding_dimensions]
        best_similarity, best_category = -1.0, None
        for category, category_vector in category_vectors.items():
            similarity = cosine_similarity(vector, category_vector[:config.embedding_dimensions])
            if similarity > best_similarity:
                best_similarity, best_category = similarity, category
        if config.min_similarity <= best_similarity <= config.max_similarity:
            decision = flags[tag]
            validate_flags(decision, tag)
            flag_value = next(score for flag, score in [
                ("too_generic", -10), ("generic", -1),
                ("specific_enough", 5), ("very_specific", 20),
            ] if decision[flag])
            score = 0.025 * count + 5.0 * flag_value + 0.25 * best_similarity
            categories[best_category].append({"tag": tag, "count": count,
                                             "cosine_similarity": best_similarity,
                                             "composite_score": score})
    config.output_dir.mkdir(parents=True, exist_ok=True)
    # Write all supplied categories, including empty ones, so reruns do not
    # accidentally leave an earlier assignment file in the current result.
    for category in category_vectors:
        rows = sorted(categories[category], key=lambda row: -row["composite_score"])
        with (config.output_dir / f"{category}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["tag", "count", "cosine_similarity", "composite_score"])
            writer.writeheader()
            writer.writerows(rows)
    return {"terms_assigned": sum(map(len, categories.values())), "output_dir": str(config.output_dir)}
