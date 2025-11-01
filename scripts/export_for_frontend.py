# state_of_neuro/scripts/export_for_frontend.py
"""Stub for the postponed Moritz visualisation export step (final pipeline handoff).

The implementation is intentionally deferred while the Next.js dashboard is
being rebuilt; keeping this placeholder allows ``scripts/run_step.py`` to keep
a consistent interface for the eventual Moritz export stage.

Pending work includes:
- Normalising trend metrics into the schema consumed by Moritz Stefaner’s app.
- Writing the topic-trend input bundle into ``output_dir`` for that dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class FrontendExportConfig:
    """Configuration for assembling Moritz Stefaner’s dashboard artefacts."""

    output_dir: Path
    schema_version: int


def run(config: FrontendExportConfig) -> None:
    """Placeholder entry point for the Moritz dashboard export handoff."""
    raise NotImplementedError("Dashboard export logic for Moritz is not yet implemented.")
