# state_of_neuro/scripts/export_for_frontend.py
"""Frontend application export is unavailable in this public repository.

The published interactive is maintained separately. The implemented
topic_trends_export.py writes analysis bundles, not that application.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class FrontendExportConfig:
    """Configuration for assembling Moritz Stefaner’s dashboard artefacts."""

    output_dir: Path
    schema_version: int


UNAVAILABLE_MESSAGE = (
    "Frontend export unavailable in this public repository. See README.md "
    "for the published interactive and the separate topic-trend data "
    "exporter."
)


def run(config: FrontendExportConfig) -> None:
    """Report the missing implementation without generating substitute results."""
    raise NotImplementedError(UNAVAILABLE_MESSAGE)
