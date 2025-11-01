# state_of_neuro/scripts/topic_trends_export.py
"""Step 10 – compile topic-trend exports from canonical clusters and metadata.

Adapted from the historical standalone script that produced Moritz Stefaner’s
visualisation bundle. The implementation mirrors the original logic so derived
counts remain identical, while exposing a typed configuration for
``run_step.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Set

import yaml

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TopicTrendsExportConfig:
    authoritative_clusters_path: Path
    metadata_path: Path
    output_dir: Path
    master_tag_clusters_path: Path | None = None
    excluded_tags_path: Path | None = None
    apply_excluded_tags: bool = False
    case_variants_path: Path | None = None
    recreate_case_variants_on_miss: bool = True
    use_cluster_key_as_label: bool = True
    compute_case_variants: bool = False
    miscellaneous_limit: int = 300
    min_frequency: int = 24
    excluded_group_ids: Sequence[str] = field(default_factory=lambda: ("G05",))
    slug_to_group: Mapping[str, str] = field(
        default_factory=lambda: {
            "disease_condition": "G01",
            "methods_metrics_technology": "G02",
            "brain_regions": "G03",
            "molecules": "G04",
            "animal_models": "G05",
            "biological_and_cognitive_processes": "G06",
            "miscellaneous": "G07",
        }
    )
    group_labels: Mapping[str, str] = field(
        default_factory=lambda: {
            "G01": "Disease & Condition",
            "G02": "Methods & Technology",
            "G03": "Brain Regions & Structures",
            "G04": "Molecules",
            "G05": "Animal Models & Experimental Systems",
            "G06": "Biological & Cognitive Processes",
            "G07": "Miscellaneous",
        }
    )
    full_output_name: str = "topic_trends_full.json"
    lean_output_name: str = "topic_trends_lean.json"
    year_totals_full_name: str = "year_totals_full.json"
    year_totals_used_name: str = "year_totals_used.json"
    category_summary_dir: str = "category_summaries"
    print_summary: bool = True


@dataclass
class TopicTrendsExportResult:
    clusters_processed: int
    entries_retained: int
    excluded_clusters: int
    excluded_by_group: int
    full_output_path: Path
    lean_output_path: Path
    year_totals_full_path: Path
    year_totals_used_path: Path
    category_summary_dir: Path


# ---------------------------------------------------------------------------
# Module-level knobs populated from config (mirrors historical globals)
# ---------------------------------------------------------------------------

AUTHORITATIVE_CLUSTERS_PATH: Path = Path()
EXCLUDED_TAGS_PATH: Path | None = None
APPLY_EXCLUDED_TAGS: bool = False
META_PATH: Path = Path()
CASE_VARIANTS_PATH: Path | None = None
MISCELLANEOUS_LIMIT: int = 300
USE_CLUSTER_KEY_AS_LABEL: bool = True
RECREATE_CASE_VARIANTS_ON_MISS: bool = True
COMPUTE_CASE_VARIANTS: bool = False
PRINT_SUMMARY: bool = True

CATEGORY_SLUG_TO_GID: Dict[str, str] = {}
STATIC_GROUP_LABELS: Dict[str, str] = {}
EXCLUDED_GROUP_IDS: Set[str] = set()
ALLOWED_GROUP_IDS: Set[str] = set()
GROUP_ID_TO_NAME: Dict[str, str] = {}

OUTPUT_PATH_FULL: Path = Path()
OUTPUT_PATH_LEAN: Path = Path()
YEAR_TOTALS_FULL_OUTPUT_PATH: Path = Path()
YEAR_TOTALS_USED_OUTPUT_PATH: Path = Path()
CATEGORY_SUMMARY_DIR: Path = Path()
MIN_FREQUENCY: int = 24

_CASE_COUNTS: Dict[str, Counter] = {}
_METADATA_RECORDS: Dict[str, Any] = {}

# ---------------------------------------------------------------------------
# Helper functions (lifted from the historical script)
# ---------------------------------------------------------------------------


def get_europe_date_str() -> str:
    now = datetime.now()
    return now.strftime("%d_%B_%Y")


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_metadata(path: Path) -> Dict[str, Any]:
    with path.open("rb") as fh:
        return pickle.load(fh)


def load_excluded_tags() -> Set[str]:
    excluded_tags: Set[str] = set()
    if not EXCLUDED_TAGS_PATH or not EXCLUDED_TAGS_PATH.exists():
        logging.info("Excluded tags file not found at %s - proceeding without exclusions.", EXCLUDED_TAGS_PATH)
        return excluded_tags
    with EXCLUDED_TAGS_PATH.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            tag = row.get("tag", "").strip()
            if tag:
                excluded_tags.add(tag.lower())
    logging.info("Loaded %d excluded tags from %s", len(excluded_tags), EXCLUDED_TAGS_PATH)
    return excluded_tags


def should_exclude_cluster(canonical: str, variants: List[str], excluded_tags: Set[str]) -> bool:
    if canonical.lower() in excluded_tags:
        return True
    return any(variant.lower() in excluded_tags for variant in variants)


def load_case_variants() -> Dict[str, Dict[str, Any]]:
    if CASE_VARIANTS_PATH and CASE_VARIANTS_PATH.exists():
        try:
            return load_json(CASE_VARIANTS_PATH)
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning("Failed to read %s – starting fresh (%s)", CASE_VARIANTS_PATH, exc)
    return {}


def save_case_variants(mapping: Dict[str, Dict[str, Any]], used_keys: Set[str]) -> None:
    if not CASE_VARIANTS_PATH:
        return
    payload = {k: mapping[k] for k in used_keys if k in mapping}
    CASE_VARIANTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CASE_VARIANTS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
    logging.info("Saved case-variant stats for %d tags → %s", len(payload), CASE_VARIANTS_PATH)


def build_case_counts_from_metadata(metadata: Dict[str, Any]) -> Dict[str, Counter]:
    counts: Dict[str, Counter] = defaultdict(Counter)
    for rec in metadata.values():
        for raw_tag in rec.get("tags") or []:
            if isinstance(raw_tag, str) and raw_tag.strip():
                counts[raw_tag.lower()][raw_tag] += 1
    logging.info("Built case-variant counts from metadata for %d lowercase tags", len(counts))
    return counts


def compute_case_variants_from_text_single(tag_lower: str, default: str) -> Optional[Dict[str, int]]:
    if not _METADATA_RECORDS:
        return None
    anchor = re.escape(default)
    patt = re.compile(rf"\b{anchor}\b", re.IGNORECASE)
    counter: Counter[str] = Counter()
    for rec in _METADATA_RECORDS.values():
        text = f"{rec.get('title','')} {rec.get('abstract','')}"
        for m in patt.finditer(text):
            counter[m.group(0)] += 1
    return dict(counter) if counter else None


def consensus_casing(tag_lower: str, default: str, cv_map: Dict[str, Dict[str, Any]]) -> str:
    rec = cv_map.get(tag_lower)
    if rec and rec.get("preferred_casing"):
        return rec["preferred_casing"]

    cc = _CASE_COUNTS.get(tag_lower)
    if cc:
        preferred = sorted(cc.items(), key=lambda kv: (-kv[1], len(kv[0]), kv[0]))[0][0]
        cv_map[tag_lower] = {"case_variants": dict(cc), "preferred_casing": preferred}
        return preferred

    if RECREATE_CASE_VARIANTS_ON_MISS:
        cv = compute_case_variants_from_text_single(tag_lower, default)
        if cv:
            preferred = sorted(cv.items(), key=lambda kv: (-kv[1], len(kv[0]), kv[0]))[0][0]
            cv_map[tag_lower] = {"case_variants": cv, "preferred_casing": preferred}
            return preferred

    return default


def deduplicate_variants_by_case(variants: List[str], cv_map: Dict[str, Dict[str, Any]]) -> List[str]:
    seen_lower = set()
    unique_variants = []
    for variant in variants:
        variant_lower = variant.lower()
        if variant_lower not in seen_lower:
            variant_cased = consensus_casing(variant_lower, variant, cv_map)
            unique_variants.append(variant_cased)
            seen_lower.add(variant_lower)
    return unique_variants


def load_authoritative_clusters() -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, Set[str]]]:
    if not AUTHORITATIVE_CLUSTERS_PATH.exists():
        raise FileNotFoundError(f"Authoritative clusters file not found at {AUTHORITATIVE_CLUSTERS_PATH}")
    data = load_json(AUTHORITATIVE_CLUSTERS_PATH)
    clusters: Dict[str, List[str]] = {}
    cluster_gid_map: Dict[str, str] = {}
    tag_to_canons: Dict[str, Set[str]] = defaultdict(set)

    entries: List[Tuple[int, Optional[str], Any]] = []
    if isinstance(data, list):
        entries = [(idx, None, entry) for idx, entry in enumerate(data)]
    elif isinstance(data, dict):
        running_idx = 0
        for gid_key, group_entries in data.items():
            if not isinstance(group_entries, list):
                logging.error(
                    "Malformed group '%s': expected list but found %s → %r",
                    gid_key,
                    type(group_entries).__name__,
                    group_entries,
                )
                continue
            for entry in group_entries:
                entries.append((running_idx, gid_key, entry))
                running_idx += 1
    else:
        logging.error(
            "Authoritative clusters format unsupported: expected list or dict but found %s",
            type(data).__name__,
        )
        return clusters, cluster_gid_map, tag_to_canons

    for idx, fallback_gid, entry in entries:
        if not isinstance(entry, dict):
            logging.error(
                "Malformed cluster entry at index %d: expected dict but found %s → %r",
                idx,
                type(entry).__name__,
                entry,
            )
            continue
        label = entry.get("label", "").strip()
        gid = entry.get("gid", fallback_gid or "").strip()
        members = entry.get("members", []) or []
        if not label or not gid:
            logging.warning("Skipping malformed cluster entry (missing label or gid): %s", entry)
            continue

        label_lc = label.lower()
        variant_pool = []
        seen_variants = {label_lc}
        for m in members:
            if isinstance(m, str) and m.strip():
                m_lc = m.lower()
                if m_lc not in seen_variants:
                    variant_pool.append(m)
                    seen_variants.add(m_lc)

        clusters[label] = variant_pool
        cluster_gid_map[label_lc] = gid

        for tag_in_cluster in seen_variants:
            tag_to_canons[tag_in_cluster].add(label)

    logging.info("Loaded %d clusters from %s", len(clusters), AUTHORITATIVE_CLUSTERS_PATH)
    return clusters, cluster_gid_map, tag_to_canons


def build_tag_index_from_metadata(metadata: Dict[str, Any], cv_map: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    tag_articles: Dict[str, List[str]] = defaultdict(list)
    tag_years: Dict[str, Counter] = defaultdict(Counter)
    first_seen_label: Dict[str, str] = {}

    for art_id, rec in metadata.items():
        tags = rec.get("tags") or []
        year_raw = rec.get("pubdate") or rec.get("year")
        year_int = int(str(year_raw)[:4]) if year_raw and str(year_raw)[:4].isdigit() else None
        for raw_tag in tags:
            if isinstance(raw_tag, str) and raw_tag.strip():
                key = raw_tag.lower()
                first_seen_label.setdefault(key, raw_tag)
                tag_articles[key].append(art_id)
                if year_int is not None:
                    tag_years[key][year_int] += 1

    index: Dict[str, Any] = {}
    for key, arts in tag_articles.items():
        if not arts:
            continue
        unique_arts = list(dict.fromkeys(arts))
        years = {str(y): c for y, c in sorted(tag_years[key].items(), key=lambda x: x[0], reverse=True)}
        label = consensus_casing(key, first_seen_label[key], cv_map)
        index[label] = {
            "freq": len(unique_arts),
            "articles": unique_arts,
            "years": years,
            "emily_hierarchies": [],
        }
    return index, cv_map


def merge_year_counters(counters: List[Dict[str, int]], dup: Optional[Dict[int, int]] = None) -> Dict[str, int]:
    total = Counter()
    for c in counters:
        for y, n in c.items():
            total[int(y)] += n
    if dup:
        for y, n in dup.items():
            total[y] -= n
            if total[y] <= 0:
                del total[y]
    return {str(y): n for y, n in sorted(total.items(), key=lambda x: x[0], reverse=True)}


def validate_no_cross_category_duplicates(index: Dict[str, Any]) -> None:
    tag_to_categories = defaultdict(set)
    for key, data in index.items():
        tag_name = key.split("|", 1)[1]
        for hier in data.get("emily_hierarchies", []):
            group_id = hier["group"]
            if group_id in GROUP_ID_TO_NAME:
                tag_to_categories[tag_name.lower()].add(GROUP_ID_TO_NAME[group_id])

    duplicates_found = [(tag, list(cats)) for tag, cats in tag_to_categories.items() if len(cats) > 1]
    if duplicates_found:
        logging.error("VALIDATION FAILED: Found tags in multiple categories:")
        for tag, cats in duplicates_found:
            logging.error("  - '%s' appears in: %s", tag, ", ".join(cats))
        raise ValueError(f"Found {len(duplicates_found)} tags in multiple categories!")
    logging.info("✓ Validation passed: No cross-category duplicates found")


def create_lean_version(full_index: Dict[str, Any]) -> Dict[str, Any]:
    lean_index = {}
    for key, data in full_index.items():
        lean_entry = {
            "emily_hierarchies": data.get("emily_hierarchies", []),
            "nearly_identical_tags": data.get("nearly_identical_tags", []),
        }
        if lean_entry["emily_hierarchies"] or lean_entry["nearly_identical_tags"]:
            lean_index[key] = lean_entry
    return lean_index


def is_miscellaneous(hierarchies: List[Dict[str, Any]]) -> bool:
    misc_gid = CATEGORY_SLUG_TO_GID.get("miscellaneous")
    return misc_gid is not None and any(hier.get("group") == misc_gid for hier in hierarchies)


def _extract_year(rec: Dict[str, Any]) -> Optional[int]:
    y = rec.get("pubdate") or rec.get("year")
    if not y:
        return None
    y4 = str(y)[:4]
    return int(y4) if y4.isdigit() else None


def compute_year_totals_for_article_ids(metadata: Dict[str, Any], article_ids: Optional[Set[str]] = None) -> Dict[str, int]:
    years = Counter()
    source = article_ids if article_ids is not None else metadata.keys()
    for aid in source:
        rec = metadata.get(aid)
        if rec:
            year = _extract_year(rec)
            if year:
                years[year] += 1
    return {str(y): years[y] for y in sorted(years.keys())}


def write_year_totals_json(path: Path, year_map: Dict[str, int]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump({"value": year_map}, fh, ensure_ascii=False, indent=2)
    logging.info("Wrote year totals → %s (%d years, total=%d)", path, len(year_map), sum(year_map.values()))


# ---------------------------------------------------------------------------
# Execution logic mirroring the historical main() implementation
# ---------------------------------------------------------------------------


def _execute() -> TopicTrendsExportResult:
    metadata = load_metadata(META_PATH)
    cv_cache = load_case_variants()

    global _CASE_COUNTS, _METADATA_RECORDS
    _CASE_COUNTS = build_case_counts_from_metadata(metadata)
    _METADATA_RECORDS = metadata

    tag_index, cv_cache = build_tag_index_from_metadata(metadata, cv_cache)
    if not tag_index:
        raise ValueError(
            "Metadata payload contains no tag assignments; ensure upstream steps enrich 'hash_id_to_metadata.pkl' with tags."
        )
    clusters, cluster_gid_map, _ = load_authoritative_clusters()

    excluded_tags: Set[str] = set()
    if APPLY_EXCLUDED_TAGS:
        excluded_tags = load_excluded_tags()

    lower_to_entry = {lbl.lower(): data for lbl, data in tag_index.items()}

    merged_entries: Dict[str, Any] = {}
    umbrella_to_sub: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    sub_owner: Dict[str, str] = {}

    processed_cluster_n = excluded_cluster_n = excluded_by_group_n = 0

    for canonical, variants in clusters.items():
        if APPLY_EXCLUDED_TAGS and should_exclude_cluster(canonical, variants, excluded_tags):
            excluded_cluster_n += 1
            logging.debug("Excluding cluster '%s' due to exclusion list.", canonical)
            continue

        auth_gid = cluster_gid_map.get(canonical.lower())
        if auth_gid in EXCLUDED_GROUP_IDS:
            excluded_by_group_n += 1
            continue

        cluster_labels = {canonical, *variants}

        chosen = canonical
        if not USE_CLUSTER_KEY_AS_LABEL:
            best_freq = lower_to_entry.get(canonical.lower(), {}).get("freq", 0)
            for t in cluster_labels:
                freq = lower_to_entry.get(t.lower(), {}).get("freq", 0)
                if freq > best_freq:
                    chosen, best_freq = t, freq

        art_ids: List[str] = []
        years_list: List[Dict[str, int]] = []
        chosen_indiv_entry = lower_to_entry.get(chosen.lower())

        for t in cluster_labels:
            entry = lower_to_entry.get(t.lower())
            if entry:
                art_ids.extend(entry["articles"])
                years_list.append(entry["years"])

        art_counter = Counter(art_ids)
        if not art_counter:
            continue

        unique_art = list(dict.fromkeys(art_ids))
        dup_by_year: Dict[int, int] = {
            yr: count - 1
            for aid, count in art_counter.items() if count > 1
            if (y_raw := (metadata.get(aid, {}).get("pubdate") or metadata.get(aid, {}).get("year")))
            and str(y_raw)[:4].isdigit() and (yr := int(str(y_raw)[:4]))
        }
        years_merged = merge_year_counters([ {str(y): c for y, c in entry.items()} for entry in years_list ], dup_by_year)

        chosen_lc = chosen.lower()
        chosen_cased = consensus_casing(chosen_lc, chosen, cv_cache)

        nearly_identical_raw = [t for t in cluster_labels if t.lower() != chosen.lower() and t.lower() in lower_to_entry]
        nearly_identical = deduplicate_variants_by_case(nearly_identical_raw, cv_cache)

        hier = []
        if auth_gid in ALLOWED_GROUP_IDS:
            hier.append({"group": auth_gid, "labels": [GROUP_ID_TO_NAME.get(auth_gid, auth_gid), chosen_cased]})

        merged_entries[chosen_cased] = {
            "freq": len(unique_art),
            "articles": unique_art,
            "years": years_merged,
            "emily_hierarchies": hier,
            "nearly_identical_tags": sorted(nearly_identical),
        }
        processed_cluster_n += 1

        # Collect sub-term data for trend analysis
        if chosen_indiv_entry:
            umbrella_to_sub[chosen_cased].append((chosen_cased, chosen_indiv_entry))
            sub_owner[chosen_cased.lower()] = chosen_cased

        for t in nearly_identical:
            entry = lower_to_entry.get(t.lower())
            if not entry:
                continue
            if (owner := sub_owner.get(t.lower())) and owner != chosen_cased:
                logging.warning("Sub-term '%s' already in umbrella '%s', skipping for '%s'", t, owner, chosen_cased)
                continue
            umbrella_to_sub[chosen_cased].append((t, entry))
            sub_owner[t.lower()] = chosen_cased

    # Filter miscellaneous category
    miscellaneous_entries = {lbl: d for lbl, d in merged_entries.items() if is_miscellaneous(d["emily_hierarchies"])}
    other_entries = {lbl: d for lbl, d in merged_entries.items() if not is_miscellaneous(d["emily_hierarchies"])}

    sorted_misc = sorted(miscellaneous_entries.items(), key=lambda x: x[1]["freq"], reverse=True)[:MISCELLANEOUS_LIMIT]
    excluded_misc_count = len(miscellaneous_entries) - len(sorted_misc)

    final_merged_entries = dict(other_entries)
    final_merged_entries.update(dict(sorted_misc))

    # Filter by frequency and assign final IDs
    filtered_entries = {lbl: data for lbl, data in final_merged_entries.items() if data.get("freq", 0) >= MIN_FREQUENCY}

    sorted_items = sorted(filtered_entries.items(), key=lambda kv: kv[1]["freq"], reverse=True)
    label_to_id: Dict[str, str] = {}
    new_index: Dict[str, Any] = {}
    for idx, (lbl, data) in enumerate(sorted_items, start=1):
        term_id = f"T{idx:06d}"
        label_to_id[lbl] = term_id
        new_index[f"{term_id}|{lbl}"] = data

    sub_label_to_id: Dict[str, str] = {
        sub_lbl: f"T{i + len(label_to_id) + 1:06d}"
        for i, sub_lbl in enumerate(
            {s_lbl for subs in umbrella_to_sub.values() for s_lbl, _ in subs} - set(label_to_id)
        )
    }

    for umbrella_lbl, subs in umbrella_to_sub.items():
        umbrella_id = label_to_id.get(umbrella_lbl)
        if not umbrella_id:
            continue
        umbrella_key = f"{umbrella_id}|{umbrella_lbl}"
        trends = {}
        for sub_lbl, sub_entry in subs:
            sub_id = label_to_id.get(sub_lbl) or sub_label_to_id.get(sub_lbl)
            if not sub_id:
                continue
            trends[f"{sub_id}|{sub_lbl}"] = {
                "freq": sub_entry["freq"],
                "articles": sub_entry["articles"],
                "years": sub_entry["years"],
            }
        if trends:
            new_index[umbrella_key]["individual_tag_trends"] = dict(
                sorted(trends.items(), key=lambda kv: kv[1]["freq"], reverse=True)
            )

    used_lower = {lbl.lower() for lbl in label_to_id}
    used_lower.update(sub.lower() for sub in sub_label_to_id)
    for data in new_index.values():
        used_lower.update(t.lower() for t in data.get("nearly_identical_tags", []))
    save_case_variants(cv_cache, used_lower)

    validate_no_cross_category_duplicates(new_index)

    year_totals_full = compute_year_totals_for_article_ids(metadata)
    write_year_totals_json(YEAR_TOTALS_FULL_OUTPUT_PATH, year_totals_full)
    used_article_ids: Set[str] = set().union(*(data.get("articles", []) for data in new_index.values()))
    year_totals_used = compute_year_totals_for_article_ids(metadata, used_article_ids)
    write_year_totals_json(YEAR_TOTALS_USED_OUTPUT_PATH, year_totals_used)

    with OUTPUT_PATH_FULL.open("w", encoding="utf-8") as fh:
        json.dump(new_index, fh, ensure_ascii=False, indent=2)
    lean_index = create_lean_version(new_index)
    with OUTPUT_PATH_LEAN.open("w", encoding="utf-8") as fh:
        json.dump(lean_index, fh, ensure_ascii=False, indent=2)

    # Export category-specific summaries
    category_entries = defaultdict(list)
    seen_tags_per_category = defaultdict(set)
    for key, data in new_index.items():
        tag_name = key.split("|", 1)[1]
        for hier in data.get("emily_hierarchies", []):
            group_id = hier["group"]
            if group_id in GROUP_ID_TO_NAME:
                category_name = next((slug for slug, gid in CATEGORY_SLUG_TO_GID.items() if gid == group_id), None)
                if category_name and tag_name not in seen_tags_per_category[category_name]:
                    seen_tags_per_category[category_name].add(tag_name)
                    all_variants = [tag_name] + data.get("nearly_identical_tags", [])
                    entry_summary = {
                        "tag": tag_name,
                        "frequency": data["freq"],
                        "variants": all_variants,
                        "variant_count": len(all_variants),
                        "individual_trends": len(data.get("individual_tag_trends", {})),
                    }
                    category_entries[category_name].append(entry_summary)

    CATEGORY_SUMMARY_DIR.mkdir(exist_ok=True)
    master_summary = {}
    for category, entries in sorted(category_entries.items()):
        entries.sort(key=lambda x: x["frequency"], reverse=True)
        for i, entry in enumerate(entries, 1):
            entry["rank"] = i

        json_path = CATEGORY_SUMMARY_DIR / f"{category}_summary.json"
        group_label = GROUP_ID_TO_NAME.get(CATEGORY_SLUG_TO_GID.get(category, ""), "")
        summary_data = {
            "category": category,
            "group_name": group_label,
            "total_entries": len(entries),
            "total_frequency": sum(e["frequency"] for e in entries),
            "entries": entries,
        }
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        master_summary[category] = {
            "group_name": group_label,
            "total_entries": len(entries),
            "total_frequency": sum(e["frequency"] for e in entries),
            "all_tags": [
                {"rank": e["rank"], "tag": e["tag"], "freq": e["frequency"], "variants": e["variants"]}
                for e in entries
            ],
        }

    master_path = CATEGORY_SUMMARY_DIR / "master_category_summary.json"
    with master_path.open("w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2, ensure_ascii=False)

    if PRINT_SUMMARY:
        category_counts = defaultdict(int)
        for category_slug, entries in category_entries.items():
            gid = CATEGORY_SLUG_TO_GID.get(category_slug)
            if gid:
                category_counts[gid] += len(entries)

        gstats = ", ".join(
            f"{GROUP_ID_TO_NAME[g]}: {category_counts[g]}" for g in sorted(ALLOWED_GROUP_IDS)
        )

        logging.info("Processed %d clusters (excluded via list=%d, excluded by group=%d) → %d entries", processed_cluster_n, excluded_cluster_n, excluded_by_group_n, len(new_index))
        if excluded_misc_count > 0:
            logging.info("Limited miscellaneous category to top %d, removing %d entries", MISCELLANEOUS_LIMIT, excluded_misc_count)
        logging.info("Outputs → %s, %s, %s, %s, summaries=%s", OUTPUT_PATH_FULL, OUTPUT_PATH_LEAN, YEAR_TOTALS_FULL_OUTPUT_PATH, YEAR_TOTALS_USED_OUTPUT_PATH, CATEGORY_SUMMARY_DIR)
        logging.info("Group distribution – %s", gstats)

    return TopicTrendsExportResult(
        clusters_processed=processed_cluster_n,
        entries_retained=len(new_index),
        excluded_clusters=excluded_cluster_n,
        excluded_by_group=excluded_by_group_n,
        full_output_path=OUTPUT_PATH_FULL,
        lean_output_path=OUTPUT_PATH_LEAN,
        year_totals_full_path=YEAR_TOTALS_FULL_OUTPUT_PATH,
        year_totals_used_path=YEAR_TOTALS_USED_OUTPUT_PATH,
        category_summary_dir=CATEGORY_SUMMARY_DIR,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run(config: TopicTrendsExportConfig) -> TopicTrendsExportResult:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    global AUTHORITATIVE_CLUSTERS_PATH, EXCLUDED_TAGS_PATH, APPLY_EXCLUDED_TAGS
    global META_PATH, CASE_VARIANTS_PATH, MISCELLANEOUS_LIMIT, USE_CLUSTER_KEY_AS_LABEL
    global RECREATE_CASE_VARIANTS_ON_MISS, COMPUTE_CASE_VARIANTS, PRINT_SUMMARY
    global CATEGORY_SLUG_TO_GID, STATIC_GROUP_LABELS, EXCLUDED_GROUP_IDS, ALLOWED_GROUP_IDS, GROUP_ID_TO_NAME
    global OUTPUT_PATH_FULL, OUTPUT_PATH_LEAN, YEAR_TOTALS_FULL_OUTPUT_PATH, YEAR_TOTALS_USED_OUTPUT_PATH
    global CATEGORY_SUMMARY_DIR, MIN_FREQUENCY

    AUTHORITATIVE_CLUSTERS_PATH = config.authoritative_clusters_path
    EXCLUDED_TAGS_PATH = config.excluded_tags_path
    APPLY_EXCLUDED_TAGS = config.apply_excluded_tags
    META_PATH = config.metadata_path
    CASE_VARIANTS_PATH = config.case_variants_path or config.output_dir / "tag_case_variants.json"
    MISCELLANEOUS_LIMIT = config.miscellaneous_limit
    USE_CLUSTER_KEY_AS_LABEL = config.use_cluster_key_as_label
    RECREATE_CASE_VARIANTS_ON_MISS = config.recreate_case_variants_on_miss
    COMPUTE_CASE_VARIANTS = config.compute_case_variants
    PRINT_SUMMARY = config.print_summary

    CATEGORY_SLUG_TO_GID = dict(config.slug_to_group)
    STATIC_GROUP_LABELS = dict(config.group_labels)
    EXCLUDED_GROUP_IDS = set(config.excluded_group_ids)
    ALLOWED_GROUP_IDS = {gid for gid in STATIC_GROUP_LABELS if gid not in EXCLUDED_GROUP_IDS}
    GROUP_ID_TO_NAME = dict(STATIC_GROUP_LABELS)

    OUTPUT_PATH_FULL = config.output_dir / config.full_output_name
    OUTPUT_PATH_LEAN = config.output_dir / config.lean_output_name
    YEAR_TOTALS_FULL_OUTPUT_PATH = config.output_dir / config.year_totals_full_name
    YEAR_TOTALS_USED_OUTPUT_PATH = config.output_dir / config.year_totals_used_name
    CATEGORY_SUMMARY_DIR = config.output_dir / config.category_summary_dir
    MIN_FREQUENCY = config.min_frequency

    return _execute()


__all__ = [
    "TopicTrendsExportConfig",
    "TopicTrendsExportResult",
    "run",
]


# ---------------------------------------------------------------------------
# CLI helpers for manual execution
# ---------------------------------------------------------------------------


def _resolve_path(base_dir: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    return candidate


def _config_from_yaml(step_config: Mapping[str, Any], base_dir: Path) -> TopicTrendsExportConfig:
    kwargs: Dict[str, Any] = {
        "authoritative_clusters_path": _resolve_path(base_dir, step_config["authoritative_clusters_path"]),
        "metadata_path": _resolve_path(base_dir, step_config["metadata_path"]),
        "output_dir": _resolve_path(base_dir, step_config.get("output_dir", base_dir / "artifacts" / "topic_trends")),
        "apply_excluded_tags": bool(step_config.get("apply_excluded_tags", False)),
        "use_cluster_key_as_label": bool(step_config.get("use_cluster_key_as_label", True)),
        "compute_case_variants": bool(step_config.get("compute_case_variants", False)),
        "recreate_case_variants_on_miss": bool(step_config.get("recreate_case_variants_on_miss", True)),
        "miscellaneous_limit": int(step_config.get("miscellaneous_limit", 300)),
        "min_frequency": int(step_config.get("min_frequency", 24)),
        "print_summary": bool(step_config.get("print_summary", True)),
    }

    for key in ("master_tag_clusters_path", "excluded_tags_path", "case_variants_path"):
        value = step_config.get(key)
        if value:
            kwargs[key] = _resolve_path(base_dir, value)

    if "excluded_group_ids" in step_config:
        kwargs["excluded_group_ids"] = tuple(step_config["excluded_group_ids"])
    if "slug_to_group" in step_config:
        kwargs["slug_to_group"] = dict(step_config["slug_to_group"])
    if "group_labels" in step_config:
        kwargs["group_labels"] = dict(step_config["group_labels"])

    for key in (
        "full_output_name",
        "lean_output_name",
        "year_totals_full_name",
        "year_totals_used_name",
        "category_summary_dir",
    ):
        if key in step_config:
            kwargs[key] = str(step_config[key])

    return TopicTrendsExportConfig(**kwargs)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Compile topic trends export artefacts.")
    default_cfg = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    parser.add_argument(
        "--config",
        type=Path,
        default=default_cfg,
        help="Path to pipeline YAML configuration (expects a 'topic_trends' step).",
    )
    parser.add_argument(
        "--step-key",
        default="topic_trends",
        help="Step key inside the configuration file to read from (default: topic_trends).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the result payload as JSON for scripting.",
    )
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        config_data = yaml.safe_load(handle) or {}

    steps_cfg = config_data.get("steps") or {}
    step_cfg = steps_cfg.get(args.step_key)
    if step_cfg is None:
        raise KeyError(f"Step '{args.step_key}' not found in configuration {config_path}")

    topic_config = _config_from_yaml(step_cfg, config_path.parent)
    result = run(topic_config)

    payload = {
        "summary": {
            "clusters_processed": result.clusters_processed,
            "entries_retained": result.entries_retained,
            "excluded_total": result.excluded_clusters + result.excluded_by_group,
        },
        "outputs": {
            "full": str(result.full_output_path),
            "lean": str(result.lean_output_path),
            "year_totals_full": str(result.year_totals_full_path),
            "year_totals_used": str(result.year_totals_used_path),
            "category_summaries": str(result.category_summary_dir),
        },
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        summary = payload["summary"]
        print(
            "Topic trends export complete: {entries} entries (clusters={clusters}, excluded={excluded}).".format(
                entries=summary["entries_retained"],
                clusters=summary["clusters_processed"],
                excluded=summary["excluded_total"],
            )
        )
        print("Outputs:")
        for key, value in payload["outputs"].items():
            print(f"  - {key}: {value}")


if __name__ == "__main__":
    _cli()
