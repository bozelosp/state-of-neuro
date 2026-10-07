# state_of_neuro/scripts/run_category_pipeline.py
"""Compile recovered canonical clusters into the authoritative Step 10 input.

Recovered from bozelosp/state-of-neuroscience at
61358677e2ce862dd465c99c61541782d8549a70: all_categories/
process_canonical_names.py and compile_master_clusters.py. Category membership
comes from the editorial category CSVs used by canonical_clustering, not from
newly invented taxonomy assignments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .category_resolutions import CATEGORY_TO_GROUP, apply_conflict_resolutions


@dataclass
class CategoryPipelineConfig:
    clusters_path: Path
    output_dir: Path
    preserved_clusters_path: Path | None = None


def merge_preserved(clusters: list[dict], preserved: list[dict]) -> list[dict]:
    """Keep editorial labels and memberships when recovered clusters overlap."""
    final = {}
    owners = {}
    for cluster in preserved:
        canonical = cluster["canonical"]
        members = {canonical, *cluster["variants"]}
        final.setdefault(canonical, set()).update(members)
        for tag in members:
            owners[tag.lower()] = canonical
    for cluster in clusters:
        members = {cluster["canonical"], *cluster["variants"]}
        matches = {owners[tag.lower()] for tag in members if tag.lower() in owners}
        if matches:
            # Existing editorial clusters remain separate. The historical
            # loop attached unassigned variants to the first overlap; sort
            # here so Python hash order cannot change that choice.
            canonical = sorted(matches)[0]
            additions = {tag for tag in members if tag.lower() not in owners}
        else:
            canonical = cluster["canonical"]
            additions = members
        final.setdefault(canonical, set()).update(additions)
        for tag in additions:
            owners[tag.lower()] = canonical
    return [{"canonical": canonical, "variants": sorted(members)}
            for canonical, members in sorted(final.items())]


def run(config: CategoryPipelineConfig) -> dict:
    """Apply historical editorial resolutions and emit grouped label/gid/members."""
    clusters = json.loads(config.clusters_path.read_text(encoding="utf-8"))
    preserved = (json.loads(config.preserved_clusters_path.read_text())
                 if config.preserved_clusters_path else {})
    merged = {}
    for category in sorted(clusters.keys() | preserved.keys()):
        if category not in CATEGORY_TO_GROUP:
            raise ValueError(f"Unknown category {category!r}; use historical category slugs")
        merged[category] = merge_preserved(clusters.get(category, []), preserved.get(category, []))
    resolved = apply_conflict_resolutions(merged)
    master = {}
    for category, entries in resolved.items():
        gid = CATEGORY_TO_GROUP[category]["id"]
        master[gid] = [{"label": entry["canonical"], "gid": gid,
                        "members": sorted(set(entry["variants"]))}
                       for entry in entries]
    config.output_dir.mkdir(parents=True, exist_ok=True)
    path = config.output_dir / "MASTER_TAG_CLUSTERS.json"
    path.write_text(json.dumps(master, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"categories": len(master), "clusters": sum(map(len, master.values())),
            "output_path": str(path)}
