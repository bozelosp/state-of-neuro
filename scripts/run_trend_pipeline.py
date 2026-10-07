# state_of_neuro/scripts/run_trend_pipeline.py
"""Additional trend analytics are unavailable in this public repository.

This placeholder is not the published interactive or the implemented
topic_trends_export.py. See README.md for the available export path and limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrendPipelineConfig:
    """Configuration for running emerging trends analysis."""

    metrics_dir: Path
    faiss_dir: Path
    scenario: str


UNAVAILABLE_MESSAGE = (
    "Trend pipeline unavailable in this public repository. See README.md "
    "for reproduction status; topic_trends_export.py only exports supplied "
    "curated inputs."
)


def run(config: TrendPipelineConfig) -> None:
    """Report the missing implementation without generating substitute results."""
    raise NotImplementedError(UNAVAILABLE_MESSAGE)
