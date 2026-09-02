"""Shared fixtures. Keeps the project root importable as ``src`` during test runs."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_analysis_frame  # noqa: E402


@pytest.fixture(scope="session")
def analysis_frame():
    """The prepared analytical cohort (regenerated from raw XPT files if missing)."""
    return load_analysis_frame()
