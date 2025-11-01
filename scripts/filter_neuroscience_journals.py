# state_of_neuro/scripts/filter_neuroscience_journals.py
"""Filter SCImago Journal & Country Rank data to neuroscience-only journals.

Keeps only titles whose **sole** SCImago area assignment is ``Neuroscience`` so
the reproducible pipeline can deterministically rebuild the whitelist from the
public CSV export.

Example
-------
```python
from pathlib import Path
from scripts import filter_neuroscience_journals

config = filter_neuroscience_journals.ScimagoFilterConfig(
    input_csv=Path("data/journal_files/scimagojr_2023.csv"),
    output_csv=Path("artifacts/journals/neuro_scimagojr.csv"),
)
filter_neuroscience_journals.run(config)
```

The expected input is the SCImago CSV export (semicolon-delimited, UTF-8). Rows
with multiple areas (e.g. ``"Neuroscience; Medicine"``) are discarded to ensure
the downstream article ingestion only uses journals with purely neuroscience
coverage.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class ScimagoFilterConfig:
    """Options controlling the SCImago neuroscience filter."""

    input_csv: Path
    output_csv: Path
    area_name: str = "neuroscience"
    areas_column: str = "Areas"
    metric_column: str = "Cites / Doc. (2years)"
    delimiter: str = ";"


@dataclass
class ScimagoFilterResult:
    """Summary of the filtering operation."""

    input_rows: int
    retained_rows: int
    dropped_rows: int
    output_path: Path


def run(config: ScimagoFilterConfig) -> ScimagoFilterResult:
    """Filter a SCImago CSV down to neuroscience-exclusive journals."""

    if not config.input_csv.exists():
        raise FileNotFoundError(f"SCImago CSV not found: {config.input_csv}")

    with config.input_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=config.delimiter)
        if reader.fieldnames is None:
            raise ValueError("Input CSV must include a header row.")
        fieldnames = list(reader.fieldnames)
        rows, input_rows = _collect_neuroscience_rows(
            reader,
            area_name=config.area_name,
            areas_column=config.areas_column,
        )

    rows.sort(
        key=lambda row: _normalise_metric(row.get(config.metric_column, "")),
        reverse=True,
    )

    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with config.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=config.delimiter)
        writer.writeheader()
        writer.writerows(rows)

    retained_count = len(rows)
    return ScimagoFilterResult(
        input_rows=input_rows,
        retained_rows=retained_count,
        dropped_rows=input_rows - retained_count,
        output_path=config.output_csv,
    )


def _collect_neuroscience_rows(
    reader: Iterable[dict[str, str]],
    *,
    area_name: str,
    areas_column: str,
) -> tuple[List[dict[str, str]], int]:
    area_name_lower = area_name.strip().lower()
    retained: List[dict[str, str]] = []
    total = 0
    for row in reader:
        total += 1
        areas_raw = row.get(areas_column, "")
        areas = [
            segment.strip().lower()
            for segment in areas_raw.split(";")
            if segment.strip()
        ]
        if len(areas) == 1 and areas[0] == area_name_lower:
            retained.append(row)
    return retained, total


def _normalise_metric(value: str) -> float:
    value = value.strip()
    if not value:
        return float("-inf")
    try:
        return float(value.replace(",", "."))
    except ValueError:
        return float("-inf")


__all__ = [
    "ScimagoFilterConfig",
    "ScimagoFilterResult",
    "run",
]
