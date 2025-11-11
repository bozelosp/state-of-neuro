# State of Neuroscience 2025
*A collaboration between [The Transmitter](https://www.thetransmitter.org/) and World Wide Neuro*

**Team & roles**

- **Panos Bozelos** — Data engineering; developed this repo, from PubMed data sourcing through trend-analysis, handed off to Moritz Stefaner for visualisation.
- **Moritz Stefaner** — Visualisation & interaction design for the app/microsite.
- **The Transmitter (Simons Foundation)**
  - **Kristin Ozelli — Executive Editor**: Oversees operations, manages the editorial team, and steers production; led the project on the editorial side.
  - **Emily Singer — Chief Opinion Editor**: Commissions/edits scientist-written content and develops community resources for the package.
  - **Rebecca Horne — Art Director**: Leads multimedia strategy; commissions and directs illustration, photography, and video.
- **Tim Vogels** — Project advisor (oversight & feedback); professor of theoretical neuroscience at ISTA and co-founder of World Wide Neuro.

## Overview
Audit-ready rebuild of [The Transmitter](https://www.thetransmitter.org/) topic trends analysis project. Within the codebase we use "topics" and "keywords" interchangeably for the extracted tags, but when talking about the project externally we frame the work as topic trend analysis. The workflow ingests PubMed baselines, filters for neuroscience content, produces keywords and embeddings, and assembles canonical tag indices ready for editorial handoff.

## Step Status

### Step summary
| Step | Script(s) | Description | Outputs |
| --- | --- | --- | --- |
| Step 1: Source acquisition | `scripts/source_acquisition.py` | Fetch PubMed baseline listings, build a manifest, optionally download `.xml.gz` archives with MD5 checks. | `artifacts/manifests/pubmed_baseline_files_<date>.csv`, associated `.sha256`, `artifacts/pubmed_baseline/*.xml.gz` |
| Step 2: Neuroscience journal filter | `scripts/filter_neuroscience_journals.py` | Keep journals dedicated to neuroscience based on SCImago 2023 metadata. | `artifacts/journals/neuroscience_only.csv` |
| Step 3: Abstract ingestion | `scripts/prepare_abstracts.py` | Normalize abstracts, enforce length/hash rules, batch PubMed citation rows, update metadata cache. | `artifacts/abstracts/abstracts_batch_<####>.json`, `hash_id_manifest.csv`, `hash_id_to_metadata.pkl` |
| Step 4: Keyword extraction | `scripts/generate_keywords.py` | Replay cached keyword responses produced alongside the fixture prompts. | `artifacts/keywords/keywords_batch_<####>.json` |
| Step 5: Keyword aggregation & trend seeding | `scripts/aggregate_keywords.py` | Walk `hash_id_to_metadata.pkl` + keyword batches, deduplicate, and rank the top ~15,000 keywords with total counts (plus per-year breakdowns) for specificity analysis. | `artifacts/keywords/aggregated/top_keywords.parquet`, `keyword_year_counts.json` |
| Step 6: Embedding generation *(optional)* | `scripts/generate_embeddings.py` | Replay cached OpenAI `text-embedding-3-large` responses bundled for testing. | `data/embeddings_raw/embeddings_batch_<####>.json` |
| Step 7: Specificity scoring *(queued – 11/12/25)* | — | :hourglass_flowing_sand: Score the top keywords so we can drop generic tags before canonicalisation. | — |
| Step 8: Canonical clustering *(queued – 11/12/25)* | — | :hourglass_flowing_sand: Collapse nearly identical tags via semantic distance, Levenshtein, bidirectional species expansion, and singular↔plural normalization. | — |
| Step 9: Category assignment *(queued – 11/12/25)* | `scripts/run_category_pipeline.py` | :hourglass_flowing_sand: Build the production “master” cluster file (authoritative categories + variants) from the curated tag set. | — |
| Step 10: Topic-trend export | `scripts/topic_trends_export.py` | Consume the master clusters + `hash_id_to_metadata.pkl` to generate the Moritz-ready topic-trend bundles (full/lean) and year totals. | `artifacts/topic_trends/topic_trends_full.json`, `topic_trends_lean.json`, `year_totals_full.json`, `year_totals_used.json`, `category_summaries/` |
| CLI helper | `scripts/run_step.py` | Entry point for running individual steps via config. | Step-specific logs and manifests |

### Execution Order
1. **Step 1: Source acquisition** (`python scripts/run_step.py source_acquisition …`).
2. **Step 2: Neuroscience journal filter** (`python scripts/run_step.py filter_neuroscience_journals …`).
3. **Step 3: Abstract ingestion** (`python scripts/run_step.py abstract_ingestion …`).
4. **Step 4: Keyword extraction** (`python scripts/run_step.py keyword_generation …`). Capture keywords to disk, then run heavy analytics against the saved batches rather than the ingestion step.
5. **Step 5: Keyword aggregation** (`python scripts/run_step.py keyword_aggregation …`) to compute the top ~15,000 keywords and year counts used by the specificity scorer.
6. **Step 6: Embedding generation** (`python scripts/run_step.py embedding_generation …`, optional) when downstream analysis needs vectors.

Steps 7–9 remain as queued placeholders slated for Wednesday's implementation pass (12/11/2025). Once they land, run `python scripts/topic_trends_export.py --config configs/default.yaml` to produce the Moritz-ready bundles.

## Step Details
Implemented steps ship with deterministic fixtures and golden outputs:
- **Step 1: Source acquisition** (`scripts/source_acquisition.py`): Build manifests (and optional downloads) from cached or live PubMed listings; configure listing paths, year, `download_dir`, `max_files`, and `verify_md5`.
- **Step 2: Neuroscience journal filter** (`scripts/filter_neuroscience_journals.py`): Trim SCImago exports to a neuroscience-only whitelist; configure area, delimiter, and metric column.
- **Step 3: Abstract ingestion** (`scripts/prepare_abstracts.py`): Normalize abstracts, enforce length, and hash into batched JSON plus metadata manifest; configure minimum length, batch size, and hash fields.
- **Step 4: Keyword extraction** (`scripts/generate_keywords.py`): Replay cached keyword responses using the fixture prompt bundle (swap in production caches when needed); configure input/output folders, response JSON, concurrency, and the optional prompt bundle. Live replays hold the OpenAI temperature at 0.0 for deterministic outputs.
- **Step 5: Keyword aggregation & trend seeding** (`scripts/aggregate_keywords.py`): Combine `hash_id_to_metadata.pkl` with the generated keyword batches, deduplicate, and emit the top ~15,000 keywords plus per-year counts for specificity analysis (requires `pandas` to write the Parquet output).
- **Step 6: Embedding generation** (`scripts/generate_embeddings.py`, optional): Replay cached `text-embedding-3-large` vectors packaged for regression tests (replace with your own caches for full runs); configure paths and concurrency.

Queued placeholders slated for Monday's implementation (11/11/2025):
- **Step 7: Specificity scoring *(queued)***: Restore the scorer to grade keyword granularity before canonicalization; annotations already live under `data/specificity/` awaiting wiring.
- **Step 8: Canonical clustering *(queued)***: Rehydrate the canonical heuristic bundle—semantic distance checks against embedding centroids, Levenshtein distance for near-miss strings, bidirectional species-name expansion, and singular↔plural normalization—to collapse near-identical tags under one canonical concept and surface override manifests for review.
- **Step 9: Category pipeline *(queued)*** (`scripts/run_category_pipeline.py`): Rebuild automated category exports and golden fixtures so downstream loaders regain parity.

### Category Tagging Scope
The keyword aggregation output becomes the input to the (queued) specificity scorer. Once Steps 7–9 land, their curated clusters will feed into `scripts/topic_trends_export.py`, which will combine the master clusters with `hash_id_to_metadata.pkl` to produce the per-year topic counts, lean/full bundles, and category summaries that power Moritz Stefaner's visualization. The reinstated Step 8 heuristics (semantic distance, Levenshtein, bidirectional species expansion, singular↔plural swaps) ensure near-identical tags collapse under shared concepts, while the restored `scripts/run_category_pipeline.py` stub for Step 9 marks where automated category tagging and export regeneration will land on Monday (11/11/2025).

## Test Strategy
Pytest runs a fully offline suite backed by deterministic fixtures and golden outputs. Each implemented pipeline step replays miniature PubMed snapshots plus synthetic keyword and embedding response caches under `tests/fixtures`, so we can assert bit-for-bit reproducibility without touching external services or reissuing OpenAI workloads.

### Fixtures at a Glance
- **Step 1: Source acquisition:** `tests/fixtures/source_acquisition/` (FTP listing HTML + expected manifest rows).
- **Step 2: Journal filter:** `tests/fixtures/journals/` (SCImago sample CSV).
- **Step 3: Abstract ingestion:** `tests/fixtures/abstracts/abstracts_batch_0001.json` (mini abstract set).
- **Step 4: Keyword generation:** `tests/fixtures/keywords/` (toy response fixtures) with the optional prompt bundle in `configs/keyword_prompt.json`.
- **Step 5: Keyword aggregation:** regression fixtures land Monday; run against your own keywords for now.
- **Step 6: Embedding generation:** `tests/fixtures/embeddings/` (deterministic `text-embedding-3-large` response fixtures).

Golden outputs for the assertions above live in `tests/golden/`.

### Test Suite Layout
- `tests/unit/`: Deterministic functional coverage for pipeline steps 1–6. Assertions diff against the matching `tests/golden/` payloads.
- `tests/housekeeping/`: Code quality and repository convention checks (currently: path-header enforcement).
- `tests/smoke/`: Confirms the postponed category pipeline and Moritz dashboard export stubs continue to raise `NotImplementedError` until those milestones land.

### Running Tests
- Full suite: `pytest`
- Per focus: `pytest tests/unit`, `pytest tests/housekeeping`, or `pytest tests/smoke`

The bundled fixtures exist solely for regression coverage and keep the suite offline. Production-scale cached keywords or embeddings are **not** distributed—provision your own when reproducing full runs.

## Pipeline Operations
The State of Neuroscience project uses these commands to refresh the trend dataset delivered to [The Transmitter](https://www.thetransmitter.org/).

### Running the Pipeline
```bash
# First, verify everything works with the test fixtures
pytest tests

# Expected output when tests pass:
# ============================= test session starts ==============================
# platform darwin -- Python 3.9.15, pytest-7.2.2, pluggy-1.0.0
# rootdir: /Users/pbozelos/Dropbox/simons/state-of-neuro, configfile: pytest.ini
# 
# All tests passed! ✓
# 
# 13 tests ran successfully:
#
# Functionality tests (8 unit tests):
#   - Source acquisition (3 tests)
#   - Journal filtering
#   - Abstract preparation
#   - Keywords generation (2 tests)
#   - Embeddings preparation
#   - Embeddings generation
#   - Keyword aggregation
#
# Housekeeping tests (1 test):
#   - Path header enforcement
#
# Smoke tests (2 tests):
#   - Postponed pipeline stubs

# Then run the implemented steps on real data / config defaults
python scripts/run_step.py source_acquisition
python scripts/run_step.py filter_neuroscience_journals
python scripts/run_step.py abstract_ingestion
python scripts/run_step.py keyword_generation
python scripts/run_step.py keyword_aggregation
python scripts/run_step.py embedding_generation  # optional

# Steps 7–9 (specificity, canonical clustering, category assignment) land Monday
# Once they're in place:
python scripts/topic_trends_export.py --config configs/default.yaml  # consumes MASTER clusters + hash metadata
```

Paths in `configs/default.yaml` default to `artifacts/` for pipelines and `data/embeddings_raw` for raw embedding caches; adjust if you run outside the repo.
Swap in your own manifest, keyword-response, and embedding-response paths before replaying real PubMed workloads.

## Repository Layout
- `README.md`: Single consolidated reference for the pipeline initiative.
- `configs/`: YAML defaults plus the specificity manifest (`configs/specificity/manifest.json`).
- `scripts/`: Step implementations, deterministic fixtures, and CLI entry point.
- `tests/`: Fixtures, golden outputs, unit tests (functionality), housekeeping tests (code quality), and smoke tests.
- `artifacts/` (generated at runtime): step outputs when you run the pipeline.

## Coming Soon
- Step 7: Specificity scoring (`—`, queued) - bring the scorer back online so we can grade keyword granularity before canonicalization.
- Step 8: Canonical clustering (`—`, queued) - reinstate the canonical toolkit (semantic distance + Levenshtein + bidirectional species expansion + singular↔plural normalization) to collapse near-identical tags under one concept and surface override manifests.
- Step 9: Category pipeline (`scripts/run_category_pipeline.py`, queued) - rebuild category tagging automation and regenerate curated exports plus golden fixtures.
- Expanded historical notes on editorial decision-making so future collaborators understand why certain keyword judgments and canonical overrides were made.
