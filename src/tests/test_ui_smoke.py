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
    assert tab_labels == ["Single", "Batch", "Live Tune", "Analysis"]


def test_batch_tab_rerun_does_not_raise_widget_error():
    """Regression: writing batch_out session key after its widget exists."""
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    assert not at.exception
    at.session_state["batch_out"] = "/tmp/some_out"
    at.session_state["batch_report"] = None
    at.run()
    assert not at.exception


def test_browse_button_click_does_not_raise_widget_error():
    """Regression: Browse must not write the text input's own widget key."""
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    assert not at.exception
    at.session_state["batch_in_picked"] = "/tmp/picked"
    at.run()
    assert not at.exception
    browse = [b for b in at.button if "Browse" in (b.label or "")]
    assert len(browse) == 2
    browse[0].click().run()
    assert not at.exception
