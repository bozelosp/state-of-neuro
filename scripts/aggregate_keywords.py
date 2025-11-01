# state_of_neuro/scripts/aggregate_keywords.py
"""Step 5 of the pipeline – aggregate keywords for specificity analysis.

This stage pulls together the per-abstract keyword batches generated in Step 4,
joins them with the hashed metadata, and emits:

* a ranked list of the top ``N`` keywords by total frequency (default 15k)
* per-year publication counts per keyword for downstream trend work

The outputs are stored under ``output_dir`` so the specificity scorer and later
curation passes can filter and collapse tags before category assignment.
"""

from __future__ import annotations

import json
import logging
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

@dataclass
class KeywordAggregationConfig:
    """Configuration for keyword aggregation."""

    keywords_dir: Path
    metadata_path: Path
    output_dir: Path
    top_n: int = 15_000


@dataclass
class KeywordAggregationResult:
    """Summary statistics for aggregation."""

    unique_keywords: int
    top_keywords: int
    top_keywords_path: Path
    year_counts_path: Path


def run(config: KeywordAggregationConfig) -> KeywordAggregationResult:
    if not config.keywords_dir.exists():
        raise FileNotFoundError(f"Keywords directory not found: {config.keywords_dir}")
    if not config.metadata_path.exists():
        raise FileNotFoundError(f"Metadata pickle not found: {config.metadata_path}")

    keyword_files = sorted(config.keywords_dir.glob("keywords_batch_*.json"))
    if not keyword_files:
        raise FileNotFoundError(f"No keyword batches found in {config.keywords_dir}")

    metadata = _load_metadata(config.metadata_path)
    total_counts: Counter[str] = Counter()
    year_counts: Dict[str, Counter[int]] = defaultdict(Counter)

    for keywords_path in keyword_files:
        payload = _load_keyword_batch(keywords_path)
        for hash_id, keywords in payload.items():
            if not isinstance(keywords, Sequence):
                continue
            record = metadata.get(hash_id)
            if not record:
                logging.debug("Hash %s missing in metadata, skipping", hash_id)
                continue
            year = _extract_year(record.get("pubdate") or record.get("year"))
            unique_keywords = {kw.strip() for kw in keywords if isinstance(kw, str) and kw.strip()}
            for keyword in unique_keywords:
                total_counts[keyword] += 1
                if year is not None:
                    year_counts[keyword][year] += 1

    unique_keywords_total = len(total_counts)
    top_n = config.top_n if config.top_n > 0 else unique_keywords_total
    top_keywords = total_counts.most_common(top_n)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    top_keywords_path = config.output_dir / "top_keywords.parquet"
    year_counts_path = config.output_dir / "keyword_year_counts.json"

    _write_top_keywords_parquet(top_keywords, top_keywords_path)
    _write_year_counts_json(top_keywords, year_counts, year_counts_path)

    return KeywordAggregationResult(
        unique_keywords=unique_keywords_total,
        top_keywords=len(top_keywords),
        top_keywords_path=top_keywords_path,
        year_counts_path=year_counts_path,
    )


def _load_metadata(path: Path) -> MutableMapping[str, Mapping[str, object]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, MutableMapping):
        raise ValueError("Metadata pickle must contain a mapping of hash_id to records")
    return payload


def _load_keyword_batch(path: Path) -> Mapping[str, Sequence[str]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Keyword batch must be a JSON object: {path}")
    return payload  # type: ignore[return-value]


def _extract_year(value: object) -> Optional[int]:
    if value is None:
        return None
    text = str(value)
    if len(text) < 4 or not text[:4].isdigit():
        return None
    return int(text[:4])


def _write_top_keywords_parquet(entries: Iterable[tuple[str, int]], path: Path) -> None:
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency check
        raise RuntimeError(
            "Keyword aggregation requires pandas to write Parquet outputs. "
            "Install pandas (pip install pandas) and retry."
        ) from exc

    df = pd.DataFrame(entries, columns=["keyword", "frequency"])  # type: ignore[arg-type]
    df.to_parquet(path, index=False)


def _write_year_counts_json(
    top_keywords: Iterable[tuple[str, int]],
    year_counts: Mapping[str, Counter[int]],
    path: Path,
) -> None:
    payload = {
        keyword: {str(year): count for year, count in sorted(year_counts.get(keyword, {}).items())}
        for keyword, _ in top_keywords
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


__all__ = ["KeywordAggregationConfig", "KeywordAggregationResult", "run"]
