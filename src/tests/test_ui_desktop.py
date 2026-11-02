"""Headless tests for the desktop entry-point helpers (no WebView2 needed)."""

from __future__ import annotations

import socket
from unittest.mock import patch

from blowing_bubbles.ui import desktop


def test_free_port_returns_usable_port():
    port = desktop.free_port()
    assert 1024 <= port <= 65535
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", port))  # must be bindable -> genuinely free


def test_app_path_points_at_streamlit_app():
    assert desktop.APP_PATH.name == "app.py"
    assert desktop.APP_PATH.exists()


def test_start_streamlit_inprocess_serves_healthz():
    port = desktop.free_port()
    thread = desktop.start_streamlit_inprocess(port)
    assert thread.daemon and thread.is_alive()
    assert desktop.wait_for_server(port, timeout=30.0) is True


def test_wait_for_server_times_out_on_dead_port():
    port = desktop.free_port()
    assert desktop.wait_for_server(port, timeout=0.5) is False


def test_pick_folder_returns_none_without_backend():
    from blowing_bubbles.ui import folder_picker

    with patch.dict("sys.modules", {"webview": None, "tkinter": None}):
        assert folder_picker.pick_folder() is None
