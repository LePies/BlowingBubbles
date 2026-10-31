"""Streamlit AppTest smoke test for the UI (skipped without the ui group)."""

from __future__ import annotations

from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = Path(__file__).parents[1] / "blowing_bubbles" / "ui" / "app.py"


def test_app_renders_three_tabs():
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    assert not at.exception
    tab_labels = [t.label for t in at.tabs]
    assert tab_labels == ["Single", "Batch", "Live Tune"]
