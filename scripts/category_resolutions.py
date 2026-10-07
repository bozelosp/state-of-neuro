# state_of_neuro/scripts/category_resolutions.py
"""Apply the recovered within-category and cross-category editorial decisions.

Recovered from bozelosp/state-of-neuroscience at 61358677e2ce862dd465c99c61541782d8549a70.
Sources: all_categories/conflict_resolutions.py and compile_master_clusters.py.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List

WITHIN_CATEGORY_RESOLUTIONS = {
    "animal_models": {
        "rhesus monkeys": "rhesus macaque",  # Merge both into "rhesus macaque"
    },
    "brain_regions": {
        "lateral prefrontal cortex": "lateral prefrontal cortex",  # Keep as separate from DLPFC and OFC
    }
}

CROSS_CATEGORY_RESOLUTIONS = {
    # fMRI should stay in methods_metrics_technology only
    "fMRI": "methods_metrics_technology",
    "functional MRI": "methods_metrics_technology",
    "functional magnetic resonance imaging": "methods_metrics_technology",
    "functional Magnetic Resonance Imaging": "methods_metrics_technology",
    "functional brain imaging": "methods_metrics_technology",
    "functional imaging": "methods_metrics_technology",
    "fMRI data": "methods_metrics_technology",
    "BOLD signal": "methods_metrics_technology",
    "BOLD signals": "methods_metrics_technology",
    "BOLD activation": "methods_metrics_technology",
    "BOLD responses": "methods_metrics_technology",
    "BOLD fMRI": "methods_metrics_technology",
    "BOLD-fMRI": "methods_metrics_technology",

    # Anxiety should stay in disease_condition only
    "anxiety": "disease_condition",
    "anxiety behavior": "disease_condition",
    "anxiety disorders": "disease_condition",
    "anxiety-like behaviour": "disease_condition",
    "social anxiety disorder": "disease_condition",
    "state anxiety": "disease_condition",
    "trait anxiety": "disease_condition",

    # Retina as anatomy should be in brain_regions
    "retina": "brain_regions",
    "cat retina": "brain_regions",
    "rat retina": "brain_regions",

    # Theta burst stimulation is a method
    "theta burst stimulation": "methods_metrics_technology",
    "theta-burst stimulation": "methods_metrics_technology",
    "intermittent theta burst stimulation": "methods_metrics_technology",
    "intermittent theta-burst stimulation": "methods_metrics_technology",

    # Neuromodulation as a technique stays in methods
    "neuromodulation": "methods_metrics_technology",
    "neurostimulation": "methods_metrics_technology",
    # But neuromodulator/neuromodulators as molecules go to molecules
    "neuromodulator": "molecules",
    "neuromodulators": "molecules",

    # Drugs should be in molecules
    "citalopram": "molecules",
    "escitalopram": "molecules",

    # Congenital blindness is a disease condition
    "congenital blindness": "disease_condition",
    "congenitally blind": "disease_condition",
}

def apply_conflict_resolutions(clusters_by_category: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    """Apply hardcoded conflict resolutions before creating master file."""

    resolved_clusters = {}
    removed_tags = defaultdict(list)
    merged_clusters = defaultdict(list)

    for category, clusters in clusters_by_category.items():
        resolved_clusters[category] = []

        # Track which tags belong to which clusters for merging
        tag_to_clusters = defaultdict(list)
        for cluster in clusters:
            for variant in cluster["variants"]:
                tag_to_clusters[variant].append(cluster)

        # Apply within-category resolutions
        clusters_to_merge = defaultdict(set)
        if category in WITHIN_CATEGORY_RESOLUTIONS:
            for tag, preferred_canonical in WITHIN_CATEGORY_RESOLUTIONS[category].items():
                if tag in tag_to_clusters:
                    # Find all clusters containing this tag
                    for cluster in tag_to_clusters[tag]:
                        clusters_to_merge[preferred_canonical].add(cluster["canonical"])

        # Process clusters, merging where needed
        processed_canonicals = set()
        for cluster in clusters:
            canonical = cluster["canonical"]

            # Skip if already processed as part of a merge
            if canonical in processed_canonicals:
                continue

            # Check if this cluster should be merged with others
            if canonical in clusters_to_merge:
                # Merge all specified clusters
                merged_variants = set()
                merge_targets = clusters_to_merge[canonical]
                merge_targets.add(canonical)  # Include self

                for target_canonical in merge_targets:
                    for c in clusters:
                        if c["canonical"] == target_canonical:
                            merged_variants.update(c["variants"])
                            processed_canonicals.add(target_canonical)
                            if target_canonical != canonical:
                                merged_clusters[category].append(f"{target_canonical} -> {canonical}")

                cluster["variants"] = sorted(merged_variants)
                logging.info(f"[{category}] Merged clusters: {merge_targets} -> {canonical}")

            # Apply cross-category resolutions - remove tags that belong elsewhere
            filtered_variants = []
            for variant in cluster["variants"]:
                if variant in CROSS_CATEGORY_RESOLUTIONS:
                    correct_category = CROSS_CATEGORY_RESOLUTIONS[variant]
                    if correct_category != category:
                        removed_tags[category].append(f"{variant} -> {correct_category}")
                        continue
                filtered_variants.append(variant)

            # Only keep cluster if it still has variants
            if filtered_variants:
                cluster["variants"] = filtered_variants
                resolved_clusters[category].append(cluster)
                processed_canonicals.add(canonical)

    # Log resolution summary
    if removed_tags:
        logging.info("\nCross-category resolutions applied:")
        for cat, moves in removed_tags.items():
            logging.info(f"  {cat}: removed {len(moves)} tags")
            for move in moves[:3]:
                logging.info(f"    - {move}")
            if len(moves) > 3:
                logging.info(f"    ... and {len(moves)-3} more")

    if merged_clusters:
        logging.info("\nWithin-category merges applied:")
        for cat, merges in merged_clusters.items():
            logging.info(f"  {cat}: {len(merges)} merges")
            for merge in merges:
                logging.info(f"    - {merge}")

    return resolved_clusters

CATEGORY_TO_GROUP: Dict[str, Dict[str, str]] = {
    "disease_condition": {"id": "G01", "group_name": "Disease & Condition"},
    "methods_metrics_technology": {"id": "G02", "group_name": "Methods & Technology"},
    "brain_regions": {"id": "G03", "group_name": "Brain Regions & Structures"},
    "molecules": {"id": "G04", "group_name": "Molecules"},
    "animal_models": {"id": "G05", "group_name": "Animal Models & Experimental Systems"},
    "biological_and_cognitive_processes": {"id": "G06", "group_name": "Biological & Cognitive Processes"},
    "miscellaneous": {"id": "G07", "group_name": "Miscellaneous"},
}
