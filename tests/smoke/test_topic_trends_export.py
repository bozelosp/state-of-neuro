# state_of_neuro/tests/smoke/test_topic_trends_export.py
"""Exercise the curated-input export path, not published-dataset parity."""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_export_cli_from_curated_inputs(tmp_path: Path) -> None:
    """Existing curated inputs retain article denominators through export."""
    (tmp_path / "clusters.json").write_text(json.dumps({
        "G03": [{"label": "hippocampus", "members": ["hippocampal region"]}],
    }), encoding="utf-8")
    metadata = {
        "article-a": {"year": "2020", "tags": ["hippocampus", "hippocampal region"]},
        "article-b": {"pubdate": "2021-06-01", "tags": ["hippocampus"]},
        "article-c": {"year": "2021", "tags": ["unrelated term"]},
    }
    with (tmp_path / "metadata.pkl").open("wb") as handle:
        pickle.dump(metadata, handle)
    config = tmp_path / "export.yaml"
    config.write_text(yaml.safe_dump({"steps": {"topic_trends": {
        "authoritative_clusters_path": "clusters.json",
        "metadata_path": "metadata.pkl",
        "output_dir": "output",
        "min_frequency": 1,
        "print_summary": False,
    }}}), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/topic_trends_export.py"),
         "--config", str(config), "--json"],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    report = json.loads(result.stdout)
    assert report["summary"]["entries_retained"] == 1
    outputs = {key: json.loads(Path(report["outputs"][key]).read_text())
               for key in ("full", "lean", "year_totals_full", "year_totals_used")}
    entry = outputs["full"]["T000001|hippocampus"]
    assert entry["freq"] == 2
    assert set(entry["articles"]) == {"article-a", "article-b"}
    assert entry["years"] == {"2020": 1, "2021": 1}
    assert outputs["year_totals_full"] == {"value": {"2020": 1, "2021": 2}}
    assert outputs["year_totals_used"] == {"value": {"2020": 1, "2021": 1}}
    assert outputs["lean"]["T000001|hippocampus"]["nearly_identical_tags"] == [
        "hippocampal region",
    ]
    summary = json.loads((Path(report["outputs"]["category_summaries"])
                         / "brain_regions_summary.json").read_text())
    assert summary["total_entries"] == 1
    assert summary["entries"][0]["frequency"] == 2
