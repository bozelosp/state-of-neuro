# state_of_neuro/scripts/run_category_pipeline.py
"""Category assignment is unavailable in this public repository (Step 9).

Steps 7–9 and the production curated inputs are not distributed here. See
README.md for the conditional export path using existing curated inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CategoryPipelineConfig:
    """Parameters for the unavailable category pipeline."""

    categories_dir: Path
    output_dir: Path
    overrides_dir: Path | None = None
    exclusion_list: Path | None = None


UNAVAILABLE_MESSAGE = (
    "Category pipeline unavailable in this public repository. See README.md "
    "for reproduction status and the curated-input export path."
)


def run(config: CategoryPipelineConfig) -> None:
    """Report the missing implementation without generating substitute results."""
    raise NotImplementedError(UNAVAILABLE_MESSAGE)
