# state_of_neuro/tests/housekeeping/test_path_headers.py
"""Enforce file header convention across pipeline source files."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKED_ROOTS = [
    ROOT / "scripts",
    ROOT / "tests",
]
SKIP_PARTS = {"__pycache__", "fixtures", "golden"}


def _candidate_files() -> list[Path]:
    files: list[Path] = []
    for base in CHECKED_ROOTS:
        if not base.exists():
            continue
        for entry in base.rglob("*.py"):
            if any(part in SKIP_PARTS for part in entry.relative_to(ROOT).parts):
                continue
            files.append(entry)
    return files


def test_all_python_files_have_path_header() -> None:
    mismatches: list[str] = []
    for file_path in _candidate_files():
        expected = f"# state_of_neuro/{file_path.relative_to(ROOT).as_posix()}"
        first_line = file_path.read_text(encoding="utf-8").splitlines()
        actual = first_line[0].strip() if first_line else ""
        if actual != expected:
            mismatches.append(f"{file_path}: expected '{expected}', found '{actual}'")
    assert not mismatches, "Missing path headers:\n" + "\n".join(mismatches)
