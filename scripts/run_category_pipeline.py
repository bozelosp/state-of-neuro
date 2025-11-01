# state_of_neuro/scripts/run_category_pipeline.py
"""Stub for the category tagging pipeline (Step 9 placeholder).

This module intentionally raises ``NotImplementedError`` so we remember to
restore the category tagging workflow on Monday. The future implementation will
consume canonical clusters, apply category tagging heuristics, and emit
category-level exports for downstream analytics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CategoryPipelineConfig:
    """Configuration stub for the category tagging pipeline."""

    categories_dir: Path
    output_dir: Path
    overrides_dir: Path | None = None
    exclusion_list: Path | None = None


def run(config: CategoryPipelineConfig) -> None:  # pragma: no cover - stub
    """Placeholder runner until the category tagging pipeline is rebuilt."""

    raise NotImplementedError(
        "Category pipeline reconstruction pending — restore tagging heuristics by Monday."
    )

