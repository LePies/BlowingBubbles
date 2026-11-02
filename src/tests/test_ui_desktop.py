"""Headless tests for the desktop entry-point helpers (no WebView2 needed)."""

from __future__ import annotations

import socket

from blowing_bubbles.ui import desktop


def test_free_port_returns_usable_port():
    port = desktop.free_port()
    assert 1024 <= port <= 65535
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", port))  # must be bindable -> genuinely free


def test_app_path_points_at_streamlit_app():
    assert desktop.APP_PATH.name == "app.py"
    assert desktop.APP_PATH.exists()


def test_start_streamlit_command_shape():
    import sys
    from unittest.mock import patch

    with patch("blowing_bubbles.ui.desktop.subprocess.Popen") as popen:
        desktop.start_streamlit(9999)
    cmd = popen.call_args[0][0]
    assert cmd[:3] == [sys.executable, "-m", "streamlit"]
    assert "9999" in cmd
    assert "--server.headless" in cmd


def test_wait_for_server_times_out_on_dead_port():
    port = desktop.free_port()
    assert desktop.wait_for_server(port, timeout=0.5) is False
