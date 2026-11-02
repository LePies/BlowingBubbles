r"""Desktop entry point: Streamlit UI inside a native pywebview window.

Runs the Streamlit server on a random free localhost port in a background
thread, then opens a pywebview window pointed at it (Edge WebView2 on
Windows). Packaged with PyInstaller via ``desktop.spec``::

    uv sync --group desktop
    uv run pyinstaller desktop.spec

The pywebview import is lazy so this module stays importable (and testable)
on machines without a display or WebView2 runtime.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

APP_PATH = Path(__file__).with_name("app.py")


def free_port() -> int:
    """Pick an unused TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def start_streamlit(port: int) -> subprocess.Popen:
    """Launch the Streamlit server for ``app.py`` on ``port`` (headless)."""
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.headless",
        "true",
        "--server.port",
        str(port),
        "--browser.gatherUsageStats",
        "false",
    ]
    return subprocess.Popen(cmd)


def wait_for_server(port: int, timeout: float = 30.0) -> bool:
    """Block until the Streamlit server answers on ``port``."""
    import urllib.request

    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/healthz"
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2).read()
            return True
        except OSError:
            time.sleep(0.25)
    return False


def main() -> None:
    import webview  # noqa: PLC0415 — lazy: needs WebView2 runtime at run time

    port = free_port()
    server = start_streamlit(port)
    try:
        if not wait_for_server(port):
            server.terminate()
            raise RuntimeError(f"Streamlit server did not start on port {port}")
        webview.create_window(
            "BlowingBubbles",
            f"http://127.0.0.1:{port}",
            width=1280,
            height=850,
        )
        webview.start()
    finally:
        server.terminate()


if __name__ == "__main__":
    main()
