# state_of_neuro/tests/unit/test_recovered_curation.py
"""Check recovered curation decisions and real provider request handling."""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from scripts import canonical_clustering as canon
from scripts import category_assignment, specificity_scoring, term_embeddings
from scripts.run_category_pipeline import CategoryPipelineConfig, run as compile_categories


def write_json(path: Path, value) -> Path:
    path.write_text(json.dumps(value))
    return path


def decision(name="specific_enough"):
    return {key: key == name for key in specificity_scoring.SCHEMA["required"]}


def test_specificity_calls_original_prompt_and_reuses_decisions(tmp_path):
    records = write_json(tmp_path / "records.json", [{"label": "hippocampus", "frequency": 9}])
    config = specificity_scoring.SpecificityConfig(records, tmp_path / "scores.json", model="test-model")
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(output_text=json.dumps(decision()))

    result = specificity_scoring.run(config, SimpleNamespace(responses=SimpleNamespace(create=create)))
    assert result["new_decisions"] == 1
    assert calls[0]["instructions"] == specificity_scoring.build_system_prompt()
    assert calls[0]["input"] == 'Tag: "hippocampus"'
    assert calls[0]["temperature"] == 0.0
    assert json.loads(config.output_path.read_text())["hippocampus"]["specific_enough"] is True
    config.model = None
    assert specificity_scoring.run(config)["new_decisions"] == 0
    assert len(calls) == 1


def test_specificity_failure_does_not_fabricate_generic_score(tmp_path):
    records = write_json(tmp_path / "records.json", [{"label": "hippocampus", "frequency": 9}])
    output = tmp_path / "scores.json"

    async def fail(**kwargs):
        raise RuntimeError("provider failed")

    with pytest.raises(RuntimeError, match="provider failed"):
        specificity_scoring.run(specificity_scoring.SpecificityConfig(records, output, model="test-model"),
                               SimpleNamespace(responses=SimpleNamespace(create=fail)))
    assert not output.exists()


def test_recovered_grouping_and_missing_vectors():
    config = canon.CanonicalConfig(*([Path("unused")] * 5))
    terms = ["fMRI", "functional MRI", "mouse", "Mus musculus", "axon", "axons", "cortex"]
    vectors = {"axon": [1, 0], "axons": [0, 1], "cortex": [-1, 0]}
    groups = canon.group_terms(terms, vectors, {"fmri": {"functional mri"}},
                              {"mus musculus": "mouse"}, config)
    assert groups == [["fMRI", "functional MRI"], ["mouse", "Mus musculus"],
                      ["axon", "axons"], ["cortex"]]
    with pytest.raises(KeyError):
        canon.similarity_clusters(["axon", "missing"], vectors, 0.8, 0.825)


def test_category_assignment_uses_nearest_embedding_and_specificity_ranking(tmp_path):
    records = write_json(tmp_path / "records.json", [
        {"label": "hippocampus", "frequency": 10}, {"label": "fMRI", "frequency": 8}])
    vectors = write_json(tmp_path / "vectors.json", {"hippocampus": [1, 0], "fMRI": [0, 1]})
    cats = write_json(tmp_path / "categories.json", {"brain_regions": [1, 0], "methods_metrics_technology": [0, 1]})
    scores = write_json(tmp_path / "flags.json", {"hippocampus": decision(), "fMRI": decision("very_specific")})
    result = category_assignment.run(category_assignment.CategoryAssignmentConfig(
        records, vectors, cats, scores, tmp_path / "assigned", max_similarity=1.0))
    assert result["terms_assigned"] == 2
    with (tmp_path / "assigned/brain_regions.csv").open() as handle:
        row = list(csv.DictReader(handle))[0]
    assert row["tag"] == "hippocampus"
    assert float(row["composite_score"]) == 0.025 * 10 + 5 * 5 + 0.25
    with (tmp_path / "assigned/methods_metrics_technology.csv").open() as handle:
        assert list(csv.DictReader(handle))[0]["tag"] == "fMRI"


def test_category_compilation_preserves_review_and_applies_source_resolutions(tmp_path):
    clusters = write_json(tmp_path / "clusters.json", {
        "brain_regions": [{"canonical": "hippocampus", "variants": ["hippocampal formation"]},
                          {"canonical": "fMRI", "variants": ["functional MRI"]}],
        "methods_metrics_technology": [{"canonical": "fMRI", "variants": ["functional MRI"]}],
    })
    preserved = write_json(tmp_path / "preserved.json", {
        "brain_regions": [{"canonical": "Hippocampal formation", "variants": ["hippocampus"]}],
    })
    result = compile_categories(CategoryPipelineConfig(clusters, tmp_path / "master", preserved))
    master = json.loads(Path(result["output_path"]).read_text())
    assert [entry["label"] for entry in master["G03"]] == ["Hippocampal formation"]
    assert master["G02"][0]["label"] == "fMRI"
    assert "hippocampus" in master["G03"][0]["members"]


def test_term_embeddings_fetch_only_missing_terms_and_respect_response_indices(tmp_path):
    records = write_json(tmp_path / "records.json", [{"label": "a", "frequency": 2}, {"label": "b", "frequency": 1}])
    output = write_json(tmp_path / "vectors.json", {"a": [1, 0]})
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[0, 1])])

    config = term_embeddings.TermEmbeddingsConfig(output, records_path=records)
    result = term_embeddings.run(config, SimpleNamespace(embeddings=SimpleNamespace(create=create)))
    assert result["new_vectors"] == 1
    assert calls[0]["input"] == ["b"]
    assert calls[0]["encoding_format"] == "float"
    assert json.loads(output.read_text()) == {"a": [1, 0], "b": [0, 1]}


def test_canonical_naming_uses_provider_and_retains_recorded_plural_forms(tmp_path):
    categories = tmp_path / "categories"
    categories.mkdir()
    (categories / "brain_regions.csv").write_text("tag,count\naxon,2\naxons,1\n")
    vectors = write_json(tmp_path / "vectors.json", {"axon": [1, 0], "axons": [0, 1]})
    flags = write_json(tmp_path / "flags.json", {"axon": decision(), "axons": decision()})
    plurals = write_json(tmp_path / "plurals.json", [{"tag": "axon", "singular": "axon", "plural": "axons"}])
    config = canon.CanonicalConfig(categories, vectors, flags, tmp_path / "names.json",
                                   tmp_path / "clusters.json", singular_plural_path=plurals,
                                   model="test-model")
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(output_text='{"canonical": "axon"}')

    canon.run(config, SimpleNamespace(responses=SimpleNamespace(create=create)))
    result = json.loads(config.output_path.read_text())["brain_regions"]
    assert result[0]["canonical"] == "axon"
    assert set(result[0]["variants"]) == {"axon", "axons"}
    assert calls[0]["instructions"] == canon.GPT_INSTRUCTION
    config.model = None
    canon.run(config)
    assert len(calls) == 1
    assert json.loads(config.output_path.read_text())["brain_regions"] == result
