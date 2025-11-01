# state_of_neuro/tests/unit/test_aggregate_keywords.py
"""Unit tests for keyword aggregation step."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, Tuple

import scripts.aggregate_keywords as agg


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_aggregate_keywords_counts_and_years(tmp_path: Path, monkeypatch) -> None:
    keywords_dir = tmp_path / "keywords"
    keywords_dir.mkdir()
    _write_json(
        keywords_dir / "keywords_batch_0001.json",
        {
            "hash1": ["Brain", " brain ", "Neuron"],
            "hash2": ["Brain", "Glia"],
        },
    )

    metadata = {
        "hash1": {"year": "2020", "title": "t1", "abstract": "a1"},
        "hash2": {"pubdate": "2021-05-01", "title": "t2", "abstract": "a2"},
    }
    metadata_path = tmp_path / "hash_id_to_metadata.pkl"
    with metadata_path.open("wb") as handle:
        pickle.dump(metadata, handle)

    output_dir = tmp_path / "aggregated"

    captured_top: Iterable[Tuple[str, int]] = ()
    captured_years: Dict[str, Dict[str, int]] = {}

    def fake_write_top(entries: Iterable[Tuple[str, int]], path: Path) -> None:
        nonlocal captured_top
        captured_top = list(entries)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("dummy", encoding="utf-8")

    def fake_write_years(top_entries: Iterable[Tuple[str, int]], year_counts: dict, path: Path) -> None:
        nonlocal captured_years
        captured_years = {
            keyword: {str(year): count for year, count in sorted(year_counts.get(keyword, {}).items())}
            for keyword, _ in top_entries
        }
        path.write_text("dummy", encoding="utf-8")

    monkeypatch.setattr(agg, "_write_top_keywords_parquet", fake_write_top)
    monkeypatch.setattr(agg, "_write_year_counts_json", fake_write_years)

    config = agg.KeywordAggregationConfig(
        keywords_dir=keywords_dir,
        metadata_path=metadata_path,
        output_dir=output_dir,
        top_n=4,
    )
    result = agg.run(config)

    assert result.unique_keywords == 4
    assert result.top_keywords == 4

    counts = dict(captured_top)
    assert counts == {"Brain": 2, "brain": 1, "Neuron": 1, "Glia": 1}

    assert captured_years == {
        "Brain": {"2020": 1, "2021": 1},
        "brain": {"2020": 1},
        "Neuron": {"2020": 1},
        "Glia": {"2021": 1},
    }

    assert result.top_keywords_path.exists()
    assert result.year_counts_path.exists()
