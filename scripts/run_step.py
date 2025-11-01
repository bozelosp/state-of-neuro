# state_of_neuro/scripts/run_step.py
"""Command-line front-end for running individual pipeline steps.

Reads a YAML configuration (see ``configs/default.yaml``) and wires the typed
config objects defined in ``scripts/`` to their corresponding ``run`` functions.
Typical invocation:

``python scripts/run_step.py --config configs/default.yaml abstract_ingestion``.

Use ``--dry-run`` to inspect resolved configuration without executing a step
and ``--json`` to receive machine-readable output for orchestration tooling.

Supported steps map directly to the ``steps`` stanza in the YAML config, and
any dataclass returned by a step is serialised to JSON (falling back to the
``__dict__`` representation) when ``--json`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Dict

import yaml

try:  # Support both package-style (`python -m`) and script-style invocation.
    from . import source_acquisition
    from .prepare_abstracts import (
        AbstractIngestionConfig,
        DEFAULT_HASH_FIELDS,
        run as run_abstract_ingestion,
    )
    from .filter_neuroscience_journals import (
        ScimagoFilterConfig,
        run as run_journal_filter,
    )
    from .aggregate_keywords import (
        KeywordAggregationConfig,
        KeywordAggregationResult,
        run as run_keyword_aggregation,
    )
    from .prepare_embeddings import EmbeddingPrepConfig, run as run_embedding_prep
    from .generate_keywords import KeywordGenerationConfig, run as run_keyword_generation
    from .generate_embeddings import EmbeddingGenerationConfig, run as run_embedding_generation
    from .run_category_pipeline import (
        CategoryPipelineConfig,
        run as run_category_pipeline_step,
    )
    from .export_for_frontend import FrontendExportConfig, run as run_frontend_export
except ImportError:  # pragma: no cover - executed only when run as a stand-alone script.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts import source_acquisition
    from scripts.prepare_abstracts import (
        AbstractIngestionConfig,
        DEFAULT_HASH_FIELDS,
        run as run_abstract_ingestion,
    )
    from scripts.filter_neuroscience_journals import (
        ScimagoFilterConfig,
        run as run_journal_filter,
    )
    from scripts.aggregate_keywords import (
        KeywordAggregationConfig,
        KeywordAggregationResult,
        run as run_keyword_aggregation,
    )
    from scripts.prepare_embeddings import EmbeddingPrepConfig, run as run_embedding_prep
    from scripts.generate_keywords import KeywordGenerationConfig, run as run_keyword_generation
    from scripts.generate_embeddings import EmbeddingGenerationConfig, run as run_embedding_generation
    from scripts.run_category_pipeline import (
        CategoryPipelineConfig,
        run as run_category_pipeline_step,
    )
    from scripts.export_for_frontend import FrontendExportConfig, run as run_frontend_export


def main(argv: list[str] | None = None) -> int:
    """Entry point for invoking individual pipeline steps."""
    default_config_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"

    parser = argparse.ArgumentParser(description="Run neuroscience pipeline steps.")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "step",
        choices=[
            "source_acquisition",
            "filter_neuroscience_journals",
            "abstract_ingestion",
            "keyword_generation",
            "keyword_aggregation",
            "embedding_generation",
            "embedding_preparation",
            "category_pipeline",
            "frontend_export",
            "trend_pipeline",
        ],
        help="Step to execute.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve configuration and exit without running the step.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit results (or dry-run config) as JSON for downstream tooling.",
    )
    args = parser.parse_args(argv)

    config_path = args.config.expanduser().resolve()
    config_data = _load_config(config_path)
    steps_cfg = config_data.get("steps", {})
    step_cfg = steps_cfg.get(args.step)
    if step_cfg is None:
        raise KeyError(f"Step '{args.step}' not found in configuration.")

    base_dir = config_path.parent

    if args.step == "source_acquisition":
        step_config = _parse_source_acquisition_config(step_cfg, base_dir)
        if args.dry_run:
            _emit_config("source_acquisition", step_config, args.json)
            return 0
        result = source_acquisition.run(step_config)
    elif args.step == "filter_neuroscience_journals":
        step_config = _parse_journal_filter_config(step_cfg, base_dir)
        if args.dry_run:
            _emit_config("filter_neuroscience_journals", step_config, args.json)
            return 0
        result = run_journal_filter(step_config)
    elif args.step == "abstract_ingestion":
        step_config = _parse_abstract_ingestion_config(step_cfg, base_dir)
        if args.dry_run:
            _emit_config("abstract_ingestion", step_config, args.json)
            return 0
        result = run_abstract_ingestion(step_config)
    elif args.step == "keyword_generation":
        step_config = _parse_keyword_generation_config(step_cfg, base_dir)
        if args.dry_run:
            _emit_config("keyword_generation", step_config, args.json)
            return 0
        result = run_keyword_generation(step_config)
    elif args.step == "embedding_generation":
        step_config = _parse_embedding_generation_config(step_cfg, base_dir)
        if args.dry_run:
            _emit_config("embedding_generation", step_config, args.json)
            return 0
        result = run_embedding_generation(step_config)
    elif args.step == "embedding_preparation":
        step_config = _parse_embedding_prep_config(step_cfg, base_dir)
        if args.dry_run:
            _emit_config("embedding_preparation", step_config, args.json)
            return 0
        result = run_embedding_prep(step_config)
    elif args.step == "category_pipeline":
        step_config = _parse_category_pipeline_config(step_cfg, base_dir)
        if args.dry_run:
            _emit_config("category_pipeline", step_config, args.json)
            return 0
        result = run_category_pipeline_step(step_config)
    elif args.step == "frontend_export":
        step_config = _parse_frontend_export_config(step_cfg, base_dir)
        if args.dry_run:
            _emit_config("frontend_export", step_config, args.json)
            return 0
        result = run_frontend_export(step_config)
    elif args.step == "trend_pipeline":
        raise NotImplementedError("Trend pipeline step is not implemented in this toolkit.")
    else:
        raise RuntimeError(f"Unsupported step: {args.step}")

    _emit_result(args.step, result, args.json)
    return 0


def _load_config(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_path(base_dir: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    return candidate


def _parse_source_acquisition_config(config: Mapping[str, Any], base_dir: Path) -> source_acquisition.SourceAcquisitionConfig:
    return source_acquisition.SourceAcquisitionConfig(
        listing_path=_resolve_path(base_dir, config["listing_path"]),
        manifest_directory=_resolve_path(base_dir, config["manifest_directory"]),
        ftp_base_url=config["ftp_base_url"],
        default_year=int(config["default_year"]),
        manifest_name_template=config.get("manifest_name_template", "pubmed_baseline_files_{year}.csv"),
        listing_url=config.get("listing_url"),
        download_dir=_resolve_path(base_dir, config["download_dir"]) if config.get("download_dir") else None,
        max_files=int(config["max_files"]) if config.get("max_files") is not None else None,
        verify_md5=bool(config.get("verify_md5", True)),
    )


def _parse_journal_filter_config(config: Mapping[str, Any], base_dir: Path) -> ScimagoFilterConfig:
    return ScimagoFilterConfig(
        input_csv=_resolve_path(base_dir, config["input_csv"]),
        output_csv=_resolve_path(base_dir, config["output_csv"]),
        area_name=config.get("area_name", "neuroscience"),
        areas_column=config.get("areas_column", "Areas"),
        metric_column=config.get("metric_column", "Cites / Doc. (2years)"),
        delimiter=config.get("delimiter", ";"),
    )


def _parse_abstract_ingestion_config(config: Mapping[str, Any], base_dir: Path) -> AbstractIngestionConfig:
    manifest_path = config.get("manifest_path")
    resolved_manifest = None
    if manifest_path:
        resolved_manifest = _resolve_path(base_dir, manifest_path)
    return AbstractIngestionConfig(
        input_dir=_resolve_path(base_dir, config["input_dir"]),
        output_dir=_resolve_path(base_dir, config["output_dir"]),
        metadata_output=_resolve_path(base_dir, config["metadata_output"]),
        batch_size=int(config["batch_size"]),
        min_abstract_length=int(config["min_abstract_length"]),
        manifest_path=resolved_manifest,
        hash_fields=tuple(config.get("hash_fields", DEFAULT_HASH_FIELDS)),
    )


def _parse_keyword_generation_config(config: Mapping[str, Any], base_dir: Path) -> KeywordGenerationConfig:
    prompt_path = config.get("prompt_path")
    return KeywordGenerationConfig(
        input_dir=_resolve_path(base_dir, config["input_dir"]),
        output_dir=_resolve_path(base_dir, config["output_dir"]),
        responses_path=_resolve_path(base_dir, config["responses_path"]),
        max_concurrency=int(config.get("max_concurrency", 50)),
        prompt_path=_resolve_path(base_dir, prompt_path) if prompt_path else None,
    )


def _parse_keyword_aggregation_config(config: Mapping[str, Any], base_dir: Path) -> KeywordAggregationConfig:
    return KeywordAggregationConfig(
        keywords_dir=_resolve_path(base_dir, config["keywords_dir"]),
        metadata_path=_resolve_path(base_dir, config["metadata_path"]),
        output_dir=_resolve_path(base_dir, config["output_dir"]),
        top_n=int(config.get("top_n", 15_000)),
    )


def _parse_embedding_generation_config(config: Mapping[str, Any], base_dir: Path) -> EmbeddingGenerationConfig:
    return EmbeddingGenerationConfig(
        input_dir=_resolve_path(base_dir, config["input_dir"]),
        output_dir=_resolve_path(base_dir, config["output_dir"]),
        responses_path=_resolve_path(base_dir, config["responses_path"]),
        max_concurrency=int(config.get("max_concurrency", 50)),
    )


def _parse_embedding_prep_config(config: Mapping[str, Any], base_dir: Path) -> EmbeddingPrepConfig:
    return EmbeddingPrepConfig(
        input_dir=_resolve_path(base_dir, config["input_dir"]),
        output_dir=_resolve_path(base_dir, config["output_dir"]),
        embed_dim=int(config["embed_dim"]),
        block_size=int(config["block_size"]),
    )


def _parse_category_pipeline_config(config: Mapping[str, Any], base_dir: Path) -> CategoryPipelineConfig:
    return CategoryPipelineConfig(
        categories_dir=_resolve_path(base_dir, config["categories_dir"]),
        output_dir=_resolve_path(base_dir, config["output_dir"]),
        overrides_dir=_resolve_path(base_dir, config["overrides_dir"]) if config.get("overrides_dir") else None,
        exclusion_list=_resolve_path(base_dir, config["exclusion_list"]) if config.get("exclusion_list") else None,
    )


def _parse_frontend_export_config(config: Mapping[str, Any], base_dir: Path) -> FrontendExportConfig:
    return FrontendExportConfig(
        output_dir=_resolve_path(base_dir, config["output_dir"]),
        schema_version=int(config["schema_version"]),
    )


def _emit_config(step: str, config: Any, as_json: bool) -> None:
    payload = {"step": step, "config": _to_serialisable(config)}
    if as_json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"[DRY-RUN] {step} configuration:\n{yaml.safe_dump(payload, sort_keys=False)}")


def _emit_result(step: str, result: Any, as_json: bool) -> None:
    summary = _result_summary(step, result)
    payload = {
        "step": step,
        "summary": summary,
        "result": _to_serialisable(result),
    }
    if as_json:
        print(json.dumps(payload, indent=2))
    else:
        print(summary)


def _result_summary(step: str, result: Any) -> str:
    if step == "source_acquisition":
        summary = f"Manifest written to {result.manifest_path} (sha256={result.checksum})"
        if getattr(result, "download_dir", None):
            summary += f"; downloaded {result.downloaded_files} file(s) into {result.download_dir}"
        return summary
    if step == "filter_neuroscience_journals":
        return (
            "Filtered {retained}/{total} journals into {output} (dropped={dropped}).".format(
                retained=result.retained_rows,
                total=result.input_rows,
                output=result.output_path,
                dropped=result.dropped_rows,
            )
        )
    if step == "abstract_ingestion":
        return (
            "Curated {retained}/{total} abstracts across {batches} batches. Manifest: {manifest} (changed={changed})".format(
                retained=result.retained_abstracts,
                total=result.total_abstracts,
                batches=result.batches_written,
                manifest=result.manifest_path,
                changed=result.manifest_changed,
            )
        )
    if step == "keyword_generation":
        return (
            "Generated {keywords} keywords across {total} abstract(s) in {batches} batch(es).".format(
                keywords=result.keywords_generated,
                total=result.total_abstracts,
                batches=result.batches_processed,
            )
        )
    if step == "keyword_aggregation":
        return (
            "Keyword aggregation retained {top} of {unique} unique keywords (top list: {top_path}, year counts: {year_path}).".format(
                top=result.top_keywords,
                unique=result.unique_keywords,
                top_path=result.top_keywords_path,
                year_path=result.year_counts_path,
            )
        )
    if step == "embedding_generation":
        return (
            "Generated embeddings for {written}/{total} abstract(s) across {batches} batch(es).".format(
                written=result.embeddings_written,
                total=result.total_abstracts,
                batches=result.batches_processed,
            )
        )
    if step == "embedding_preparation":
        return (
            "Prepared {total} embeddings across {batches} batches into {output}".format(
                total=result.total_vectors,
                batches=result.batches_written,
                output=result.output_dir,
            )
        )
    if step == "frontend_export":
        return "Frontend export step completed."
    return f"Step {step} completed."


def _to_serialisable(obj: Any) -> Any:
    if is_dataclass(obj):
        data = asdict(obj)
        return {key: _to_serialisable(value) for key, value in data.items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {key: _to_serialisable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_serialisable(value) for value in obj]
    return obj


if __name__ == "__main__":
    sys.exit(main())
