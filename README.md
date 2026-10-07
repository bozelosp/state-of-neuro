# State of Neuroscience 2025
*A collaboration between [The Transmitter](https://www.thetransmitter.org/) and World Wide Neuro*

## Reproduction status

**The 2025 interactive is published; this repository is an incomplete rebuild of
its analysis pipeline. It does not currently reproduce the published report
from PubMed end to end.** Steps 7–9 are unavailable here, and the production
input/output dataset is not included. The old November 2025 implementation
dates were plans, not completed releases.

- [Published interactive](https://stateofneuroscience.thetransmitter.org/)
- [Published scientific methods](https://www.thetransmitter.org/state-of-neuroscience-2025-map-methods/)

The methods describe approximately 346,000 papers from 1975–2024, keyword
extraction, curated categories, term standardization and the displayed trends.
Those methods describe the shipped analysis, not the capabilities of this
partial rebuild. The interactive's “code and dataset” link points here; as of
2026-10-07, there is no downloadable reproduction bundle in this repository or
its GitHub releases. Display data served by the interactive does not provide
the article-level inputs required by the exporter below.

## What you can reproduce

| Path | Available now | Limit |
| --- | --- | --- |
| Explore the published results | Use the interactive linked above. | Viewing released results is not rerunning the analysis. |
| Run the offline examples | Run the tests below against small synthetic fixtures. | These are software examples, not the published corpus or scientific validation. |
| Export from existing curated inputs | Run `scripts/topic_trends_export.py` with authoritative clusters and tagged article metadata. | This bypasses Steps 7–9 only if you already have those inputs. The published inputs and an output-parity reference are not distributed here. |
| Rebuild from PubMed | Some individual preprocessing steps are implemented. | The missing stages and data prevent an end-to-end reproduction. |

### Offline examples

Use Python 3.11 for the existing serialized fixture checks:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install pytest PyYAML beautifulsoup4 openai
python -m pytest
```

Tests use local fixtures and mocked clients; they do not call OpenAI or download
the corpus. `tests/smoke/test_topic_trends_export.py` exercises the real export
CLI with a tiny curated cluster and tagged metadata, including publication
counts and year totals. Successful tests do not establish published-data parity.

### Export from already curated inputs

This is a conditional path, not a way to recover the missing dataset. Supply:

1. An authoritative cluster JSON file: a list of objects with `label`, `gid`
   and `members`, or an object mapping group IDs to lists of those objects.
   The exporter uses the `G01`–`G07` group definitions in
   `scripts/topic_trends_export.py`.
2. A trusted `hash_id_to_metadata.pkl`: a dictionary keyed by article ID,
   with each article's `tags` and `pubdate` or `year`. The pickle must come
   from a trusted source because loading it executes Python deserialization.
   Raw Step 3 metadata is insufficient: Step 4 writes separate keyword
   batches and does not enrich that pickle with tags.
3. The original export settings and, if used, case-variant and exclusion
   inputs. Defaults filter out animal models (`G05`), apply a minimum frequency
   of 24 and limit miscellaneous entries to 300. Do not assume these defaults
   or an arbitrary historical cluster file match the released report.

Set `steps.topic_trends` in a copy of `configs/default.yaml` to those inputs
and a new output directory. Paths resolve relative to the configuration file.
Then run:

```bash
python scripts/topic_trends_export.py --config /path/to/reproduction.yaml --json
```

The existing exporter writes `topic_trends_full.json`, `topic_trends_lean.json`,
`year_totals_full.json`, `year_totals_used.json` and `category_summaries/`.
The full bundle includes article IDs and yearly counts; the lean bundle holds
hierarchies and variants. Neither is the interactive application itself.
A successful export from supplied inputs does not prove that its output equals
the published 2025 dataset.

## Implementation inventory

| Step | Entry point | Actual status |
| --- | --- | --- |
| 1. Source acquisition | `scripts/source_acquisition.py` | Builds manifests and optionally downloads PubMed XML archives. Default download limit is three files. |
| 2. Journal filter | `scripts/filter_neuroscience_journals.py` | Filters a supplied SCImago CSV; the production 2023 input is absent. |
| 3. Abstract ingestion | `scripts/prepare_abstracts.py` | Normalizes supplied citation **JSON**. This is not an XML-to-JSON corpus extraction pipeline. |
| 4. Keyword generation | `scripts/generate_keywords.py` | Configured runner replays supplied response caches. Default cache is synthetic; the standalone API helper is not a full-corpus workflow. |
| 5. Keyword aggregation | `scripts/aggregate_keywords.py` | Counts supplied keywords by year; Parquet output additionally requires `pandas` and a Parquet engine such as `pyarrow`. |
| 6. Embeddings | `scripts/generate_embeddings.py`, `scripts/prepare_embeddings.py` | Replays cached vectors and prepares batches. Default vectors are synthetic. |
| 7. Specificity scoring | No implementation | The specificity manifest is an example, not production annotations. |
| 8. Canonical clustering | No implementation | Canonical fixtures/golden files do not implement clustering. |
| 9. Category assignment | `scripts/run_category_pipeline.py` | Unavailable; raises an explicit error. |
| 10. Topic-trend export | `scripts/topic_trends_export.py` | Implemented for supplied curated clusters and tagged metadata, as described above. |
| Additional analytics / frontend export | `scripts/run_trend_pipeline.py`, `scripts/export_for_frontend.py` | Unavailable; these do not build the published interactive. |

`python scripts/run_step.py --help` lists individual step names. The CLI reports
unavailable stages explicitly, including when the default config lacks their
inputs. There is no supported “run all” command. The configuration's fixture
paths must not be mistaken for production caches.

## Recovery evidence and remaining gap

The public history was checked through
[`aeb029c`](https://github.com/bozelosp/state-of-neuro/commit/aeb029cf0d0a11471a9f8f5134b3acc1a6ba8317)
on 2026-10-07. The category, trend and frontend placeholders were already
present in the initial
[`6b1361f`](https://github.com/bozelosp/state-of-neuro/commit/6b1361f713bb92b6a061cf78f77dec6aeace9df1)
commit; the five subsequent commits do not restore them. The initial commit's
“v1.0.0” wording does not identify a downloadable GitHub release dataset.

Historical source inspection found earlier curation routines and a separate
fixture-only replay implementation, but no verified production input bundle
connecting them to the published outputs. Copying those routines or toy
clusters into this rebuild would not establish faithful reproduction. They
have therefore not been substituted for the missing steps.

A complete reproduction still needs the original corpus selection and journal
input, article metadata with keyword assignments, production caches and curated
cluster decisions, and the release's export settings and reference outputs.
Until those are recovered and matched, no end-to-end or bit-for-bit reproduction
claim is made. The published scientific methods and results remain unchanged.

**Team & roles**

- **Panos Bozelos** — Data engineering; developed the analysis and data handoff to Moritz Stefaner for visualisation.
- **Moritz Stefaner** — Visualisation & interaction design for the app/microsite.
- **The Transmitter (Simons Foundation)**
  - **Kristin Ozelli — Executive Editor**: Oversees operations, manages the editorial team, and steers production; led the project on the editorial side.
  - **Emily Singer — Chief Opinion Editor**: Commissions/edits scientist-written content and develops community resources for the package.
  - **Rebecca Horne — Art Director**: Leads multimedia strategy; commissions and directs illustration, photography, and video.
- **Tim Vogels** — Project advisor (oversight & feedback); professor of theoretical neuroscience at ISTA and co-founder of World Wide Neuro.
