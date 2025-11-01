# state_of_neuro/tests/smoke/test_stubs.py
"""Smoke placeholders ensuring step stubs raise explicit errors."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from scripts import export_for_frontend, run_category_pipeline


@pytest.mark.parametrize(
    ("runner", "config"),
    [
        (
            run_category_pipeline.run,
            run_category_pipeline.CategoryPipelineConfig(
                categories_dir=Path("./categories"),
                output_dir=Path("./output"),
            ),
        ),
        (
            export_for_frontend.run,
            export_for_frontend.FrontendExportConfig(
                output_dir=Path("./web"),
                schema_version=1,
            ),
        ),
    ],
)
def test_step_stubs_raise_not_implemented(runner, config) -> None:
    """Ensure each step stub signals unimplemented status."""
    with pytest.raises(NotImplementedError):
        runner(config)
