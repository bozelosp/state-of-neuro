# state_of_neuro/scripts/canonical_clustering.py
"""Canonicalize curated terms with the recovered similarity and variant workflow.

Recovered from bozelosp/state-of-neuroscience at 61358677e2ce862dd465c99c61541782d8549a70.
Source: canonicalization_of_tags_provided_by_Emily/gpt_fusion_canonicalizer.py.
Lookup grouping and greedy similarity semantics are retained; configuration
replaces module globals, and recorded decisions replace silent API fallbacks.
"""

from __future__ import annotations

import csv
import json
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from rapidfuzz.fuzz import token_sort_ratio
from tqdm import tqdm

GPT_PROMPT = (
    "{context}\n"
    "Tags: {tags}\n"
    "Respond with just the chosen canonical term."
)

GPT_INSTRUCTION = (
    "You are a domain expert. Select the best canonical form for the following tag variants."
)

TAG_SCHEMA = {
    "type": "object",
    "properties": {"canonical": {"type": "string"}},
    "required": ["canonical"],
    "additionalProperties": False,
}



def levenshtein_ratio(a: str, b: str) -> float:
    """Compute normalized token sort ratio between ``a`` and ``b``."""
    return token_sort_ratio(a, b) / 100.0

def _connected_groups(
    terms: List[str], edges: Iterable[tuple[str, str]]
) -> List[List[str]]:
    """Return term groups linked by ``edges`` using bidirectional union."""

    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in tqdm(list(edges), desc="Union edges", unit="edge"):
        union(a, b)

    groups: Dict[str, List[str]] = defaultdict(list)
    for term in tqdm(terms, desc="Assigning groups", unit="term"):
        groups[find(term.lower())].append(term)
    return [g for g in groups.values() if len(g) > 1]

def _expand_acronym_variants(mapping: Dict[str, set[str]]) -> Dict[str, set[str]]:
    """Return mapping where each key includes all symmetric acronym variants."""

    expanded: Dict[str, set[str]] = {}
    for term, exps in mapping.items():
        group = set(exps)
        group.add(term)
        for t in group:
            expanded.setdefault(t, set()).update(group - {t})
    return expanded

def _expand_species_variants(mapping: Dict[str, str]) -> Dict[str, set[str]]:
    """Return mapping from any species term to all known synonyms."""

    canonical_to_terms: Dict[str, set[str]] = defaultdict(set)
    for term, canon in mapping.items():
        canonical_to_terms[canon].add(term)
        canonical_to_terms[canon].add(canon)

    expanded: Dict[str, set[str]] = {}
    for terms in canonical_to_terms.values():
        for t in terms:
            expanded.setdefault(t, set()).update(terms - {t})
    return expanded

def _species_groups(terms: List[str], mapping: Dict[str, str]) -> List[List[str]]:
    """Group terms referring to the same species using symmetric expansion."""

    variants = _expand_species_variants(mapping)
    valid = {t.lower() for t in terms}
    edges = []
    for term in tqdm(terms, desc="Checking species", unit="term"):
        for alt in variants.get(term.lower(), set()):
            if alt in valid:
                edges.append((term.lower(), alt))
    return _connected_groups(terms, edges)

def _acronym_groups(terms: List[str], mapping: Dict[str, set[str]]) -> List[List[str]]:
    """Group terms connected by acronym expansions using symmetric edges."""

    expanded = _expand_acronym_variants(mapping)
    valid = {t.lower() for t in terms}
    edges = []
    for term in tqdm(terms, desc="Checking acronyms", unit="term"):
        for alt in expanded.get(term.lower(), set()):
            if alt in valid:
                edges.append((term.lower(), alt))
    return _connected_groups(terms, edges)


@dataclass
class CanonicalConfig:
    categories_dir: Path
    embeddings_path: Path
    specificity_path: Path
    decisions_path: Path
    output_path: Path
    acronym_path: Path | None = None
    species_path: Path | None = None
    singular_plural_path: Path | None = None
    model: str | None = None
    temperature: float | None = 0.0
    cosine_threshold: float = 0.8
    levenshtein_threshold: float = 0.825


def load_categories(path: Path) -> dict[str, dict[str, int]]:
    """Read the historical reorganized category CSVs (tag/count columns)."""
    categories = {}
    for file in sorted(path.glob("*.csv")):
        with file.open(newline="", encoding="utf-8") as handle:
            categories[file.stem] = {
                (row.get("tag") or row["Tag"]).strip(): int(row.get("count") or 1)
                for row in csv.DictReader(handle)
            }
    if not categories:
        raise FileNotFoundError(f"No category CSVs in {path}")
    return categories


def cosine_similarity(a, b) -> float:
    """Float32 cosine calculation from historical embedding_utils.py."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


def similarity_clusters(terms, embeddings, cosine_threshold, levenshtein_threshold):
    """Recover the original greedy seed grouping and OR similarity rule."""
    assigned = set()
    clusters = []
    # Deliberately require real vectors; the historical random-vector fallback
    # cannot reproduce semantic clustering and is not carried into this port.
    vectors = {term: embeddings[term] for term in terms}
    for index, term in enumerate(terms):
        if term in assigned:
            continue
        bucket = [term]
        for other in terms[index + 1:]:
            if other in assigned:
                continue
            lev = levenshtein_ratio(term.lower(), other.lower())
            cos = cosine_similarity(vectors[term], vectors[other])
            if lev >= levenshtein_threshold or cos >= cosine_threshold:
                bucket.append(other)
                assigned.add(other)
        assigned.update(bucket)
        clusters.append(bucket)
    return clusters


def group_terms(terms, embeddings, acronym_map, species_map, config):
    """Historical order: acronym groups, species groups, then similarity."""
    clusters, used = [], set()
    if acronym_map:
        for bucket in _acronym_groups(terms, acronym_map):
            clusters.append(bucket)
            used.update(bucket)
    if species_map:
        remaining = [term for term in terms if term not in used]
        for bucket in _species_groups(remaining, species_map):
            clusters.append(bucket)
            used.update(bucket)
    remaining = [term for term in terms if term not in used]
    clusters.extend(similarity_clusters(remaining, embeddings,
                                       config.cosine_threshold, config.levenshtein_threshold))
    return clusters


def run(config: CanonicalConfig, client=None) -> dict:
    """Run the recovered clustering algorithm with recorded or live naming."""
    from .specificity_scoring import validate_flags
    from .term_normalization import SP

    categories = load_categories(config.categories_dir)
    if config.embeddings_path.suffix == ".pkl":
        with config.embeddings_path.open("rb") as handle:
            embeddings = pickle.load(handle)
    else:
        embeddings = json.loads(config.embeddings_path.read_text())
    flags = json.loads(config.specificity_path.read_text())
    decisions = json.loads(config.decisions_path.read_text()) if config.decisions_path.exists() else {}
    acronym_map = {}
    if config.acronym_path:
        for term, expansion in json.loads(config.acronym_path.read_text()).items():
            expansions = [expansion] if isinstance(expansion, str) else expansion
            acronym_map[term.lower()] = {value.lower() for value in expansions}
    species_map = json.loads(config.species_path.read_text()) if config.species_path else {}
    sp = SP()
    if config.singular_plural_path:
        if not config.singular_plural_path.is_file():
            raise FileNotFoundError(config.singular_plural_path)
        sp.load(str(config.singular_plural_path))
    output = {}
    for category, frequencies in categories.items():
        # The source's curated CSV membership is authoritative. Specificity is
        # retained for editorial review, not a newly invented exclusion rule.
        for tag in frequencies.keys() & flags.keys():
            validate_flags(flags[tag], tag)
        terms = sorted(frequencies, key=lambda term: (-frequencies[term], term))
        buckets = group_terms(terms, embeddings, acronym_map, species_map, config)
        clusters = []
        for bucket in buckets:
            context_parts = []
            for term in bucket:
                expansions = []
                if species_map.get(term.lower()):
                    expansions.append(f"species={species_map[term.lower()]}")
                if term.lower() in acronym_map:
                    expansions.append("acronym=" + ", ".join(sorted(acronym_map[term.lower()])))
                info = f"{term} (freq={frequencies[term]})"
                if expansions:
                    info += " [" + "; ".join(expansions) + "]"
                context_parts.append(info)
            prompt = GPT_PROMPT.format(tags=", ".join(bucket), context="\n".join(context_parts))
            # Keys include the full prompt so changes in context do not reuse a
            # naming decision for a different cluster or frequency context.
            key = prompt
            if len(bucket) == 1:
                canonical = bucket[0]
            elif key in decisions:
                canonical = decisions[key]["canonical"]
            else:
                if not config.model:
                    raise ValueError(f"Missing canonical decision for {bucket!r}; supply decisions_path or set model")
                if client is None:
                    from openai import OpenAI
                    client = OpenAI()
                response = client.responses.create(
                    model=config.model, instructions=GPT_INSTRUCTION, input=prompt,
                    **({"temperature": config.temperature} if config.temperature is not None else {}),
                    text={"format": {"type": "json_schema", "strict": True,
                                     "name": "canonical_form", "schema": TAG_SCHEMA}},
                )
                canonical = json.loads(response.output_text)["canonical"]
                if not isinstance(canonical, str) or not canonical.strip():
                    raise ValueError("Canonical classifier returned an empty/non-string label")
                decisions[key] = {"canonical": canonical.strip(), "model": config.model}
                config.decisions_path.parent.mkdir(parents=True, exist_ok=True)
                config.decisions_path.write_text(json.dumps(decisions, indent=2), encoding="utf-8")
            if not isinstance(canonical, str) or not canonical.strip():
                raise ValueError("Canonical decision must contain a nonempty string label")
            canonical = canonical.strip()
            members = list(bucket)
            # Expand only recorded forms, using the recovered authoritative
            # compiler's lookup; do not synthesize English inflection rules.
            for term in list(bucket):
                members.extend(form for form in sp.expand(term) if form not in members)
            clusters.append({"canonical": canonical, "variants": members,
                             "specificity": {tag: flags[tag] for tag in bucket if tag in flags}})
        output[category] = clusters
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"categories": len(output), "clusters": sum(map(len, output.values())),
            "output_path": str(config.output_path)}
