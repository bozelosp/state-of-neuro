# state_of_neuro/tests/smoke/test_recovered_pipeline.py
"""Run recovered scoring, clustering and category compilation through export."""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_keywords_to_canonical_categories_to_export(tmp_path: Path) -> None:
    """Merge actual synonyms and preserve editorial category/count semantics."""
    categories = tmp_path / "categories"
    categories.mkdir()
    (categories / "brain_regions.csv").write_text(
        "tag,count\nhippocampus,2\nhippocampal formation,1\ncortex,1\n")
    (categories / "methods_metrics_technology.csv").write_text(
        "tag,count\nfMRI,2\nfunctional MRI,1\n")
    data = {
        "a": {"year": "2020"}, "b": {"year": "2021"}, "c": {"year": "2021"},
    }
    with (tmp_path / "metadata.pkl").open("wb") as handle:
        pickle.dump(data, handle)
    keywords = tmp_path / "keywords"
    keywords.mkdir()
    (keywords / "keywords_batch_0001.json").write_text(json.dumps({
        "a": ["hippocampus", "hippocampal formation", "fMRI", "functional MRI"],
        "b": ["hippocampus", "fMRI"], "c": ["cortex"],
    }))
    flags = {tag: {"too_generic": False, "generic": False,
                   "specific_enough": True, "very_specific": False}
             for tag in ["hippocampus", "hippocampal formation", "cortex", "fMRI", "functional MRI"]}
    (tmp_path / "flags.json").write_text(json.dumps(flags))
    # These vectors exercise cosine clustering, not a supplied cluster result.
    (tmp_path / "vectors.json").write_text(json.dumps({
        "hippocampus": [1, 0], "hippocampal formation": [0.99, 0.01],
        "cortex": [0, 1], "fMRI": [1, 0], "functional MRI": [0, 1],
    }))
    (tmp_path / "acronyms.json").write_text('{"fMRI": ["functional MRI"]}')
    (tmp_path / "decisions.json").write_text(json.dumps({
        "hippocampus (freq=2)\nhippocampal formation (freq=1)\n"
        "Tags: hippocampus, hippocampal formation\nRespond with just the chosen canonical term.":
            {"canonical": "hippocampus"},
        "fMRI (freq=2) [acronym=functional mri]\nfunctional MRI (freq=1)\n"
        "Tags: fMRI, functional MRI\nRespond with just the chosen canonical term.":
            {"canonical": "fMRI"},
    }))
    config = tmp_path / "pipeline.yaml"
    config.write_text(yaml.safe_dump({"steps": {
        "attach_keywords": {"metadata_path": "metadata.pkl", "keywords_dir": "keywords",
                            "output_path": "tagged.pkl", "records_path": "records.json"},
        "specificity_scoring": {"records_path": "records.json", "output_path": "flags.json"},
        "term_embeddings": {"categories_dir": "categories", "output_path": "vectors.json"},
        "canonical_clustering": {"categories_dir": "categories", "embeddings_path": "vectors.json",
                                 "specificity_path": "flags.json", "decisions_path": "decisions.json",
                                 "output_path": "clusters.json", "acronym_path": "acronyms.json"},
        "category_pipeline": {"clusters_path": "clusters.json", "output_dir": "master"},
        "topic_trends": {"authoritative_clusters_path": "master/MASTER_TAG_CLUSTERS.json",
                         "metadata_path": "tagged.pkl", "output_dir": "export", "min_frequency": 1},
    }}))
    for step in ["attach_keywords", "specificity_scoring", "term_embeddings",
                 "canonical_clustering", "category_pipeline", "topic_trends"]:
        result = subprocess.run([sys.executable, str(ROOT / "scripts/run_step.py"), step,
                                 "--config", str(config), "--json"],
                                cwd=tmp_path, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["step"] == step
    master = json.loads((tmp_path / "master/MASTER_TAG_CLUSTERS.json").read_text())
    assert {entry["label"] for entry in master["G03"]} == {"hippocampus", "cortex"}
    assert master["G02"] == [{"label": "fMRI", "gid": "G02", "members": ["fMRI", "functional MRI"]}]
    output = json.loads((tmp_path / "export/topic_trends_full.json").read_text())
    by_label = {key.split("|", 1)[1]: value for key, value in output.items()}
    assert by_label["hippocampus"]["freq"] == 2
    assert by_label["hippocampus"]["years"] == {"2020": 1, "2021": 1}
    assert by_label["fMRI"]["freq"] == 2
    assert by_label["cortex"]["freq"] == 1
    assert json.loads((tmp_path / "export/year_totals_full.json").read_text()) == {
        "value": {"2020": 1, "2021": 2},
    }
    with (tmp_path / "metadata.pkl").open("rb") as handle:
        assert pickle.load(handle) == data  # The original article metadata stays intact.
