# state_of_neuro/tests/unit/test_filter_neuroscience_journals.py
"""Unit tests for SCImago neuroscience journal filtering."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts import filter_neuroscience_journals


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "journals"
GOLDEN = ROOT / "tests" / "golden" / "journals"


def test_filter_neuroscience_journals(tmp_path: Path) -> None:
    input_csv = FIXTURES / "scimagojr_sample.csv"
    output_csv = tmp_path / "neuro_scimagojr_sample.csv"

    config = filter_neuroscience_journals.ScimagoFilterConfig(
        input_csv=input_csv,
        output_csv=output_csv,
    )
    result = filter_neuroscience_journals.run(config)

    assert result.input_rows == 5
    assert result.retained_rows == 3
    assert result.dropped_rows == 2
    assert output_csv.exists()
    assert output_csv.read_text(encoding="utf-8") == GOLDEN.joinpath("neuro_scimagojr_sample.csv").read_text(encoding="utf-8")
