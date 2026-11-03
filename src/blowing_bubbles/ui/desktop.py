r"""Run the Streamlit UI inside a native pywebview window (desktop entry point).

:func:`main` picks a free localhost port (:func:`free_port`), runs the
Streamlit server for ``app.py`` in-process on a daemon thread via
``streamlit.web.bootstrap.run`` (:func:`start_streamlit_inprocess`), polls its
``/healthz`` endpoint for up to 30 s (:func:`wait_for_server`), then opens a
1280x850 pywebview window titled "BlowingBubbles" pointed at it (Edge WebView2
on Windows) and blocks until the window is closed. Packaged with PyInstaller
via ``desktop.spec``::

    uv sync --group desktop
    uv run pyinstaller desktop.spec

The server is started in-process rather than as ``python -m streamlit``
because inside a PyInstaller bundle the subprocess form cannot resolve the
streamlit module and the server never starts. The pywebview import is lazy
so this module stays importable (and testable) on machines without a
display or WebView2 runtime.
"""

from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path


def _app_path() -> Path:
    """Locate app.py — next to this file in source, in the bundle when frozen.

    Onefile builds extract to ``sys._MEIPASS``; onedir builds use
    ``_internal/`` next to the launcher (exposed as ``sys._MEIPASS`` too in
    PyInstaller >= 6). Candidates are ``_MEIPASS/app.py``,
    ``_MEIPASS/_internal/app.py`` and the source path; the first existing one is
    returned, else the source path.
    """
    frozen_dir = getattr(sys, "_MEIPASS", None)
    candidates = []
    if frozen_dir:
        candidates.append(Path(frozen_dir) / "app.py")
        candidates.append(Path(frozen_dir) / "_internal" / "app.py")
    candidates.append(Path(__file__).with_name("app.py"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


APP_PATH = _app_path()

# Populated by the server thread if bootstrap.run raises; surfaced by main().
START_ERROR: list[str] = []


def free_port() -> int:
    """Pick an unused TCP port on localhost.

    Returns
    -------
    int
        Port number the OS assigned to a temporary socket bound to
        ``127.0.0.1``. The socket is closed before returning, so another
        process could in principle take the port before it is reused.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def start_streamlit_inprocess(port: int) -> threading.Thread:
    """Start the Streamlit server for ``app.py`` on a daemon thread.

    Parameters
    ----------
    port : int
        Localhost port the server binds to (address ``127.0.0.1``).

    Returns
    -------
    threading.Thread
        The started daemon thread, named ``"streamlit-server"``.

    Notes
    -----
    Streamlit's config is re-parsed via ``load_config_options`` with headless
    mode, development mode off, usage statistics off and the given port and
    address, and the same options are passed to ``bootstrap.run``.

    ``bootstrap.run`` installs signal handlers, which Python only allows in the
    main thread. The server always runs on the new daemon thread, so
    ``signal.signal`` is replaced by a no-op (process-wide) for the duration of
    the call and restored afterwards.

    Any exception raised by the server thread is not propagated; its traceback
    is appended to the module-level ``START_ERROR`` list, which :func:`main`
    reports if the server does not come up.
    """
    from streamlit.web import bootstrap

    # In a frozen app the config is typically already parsed before we run,
    # and bootstrap.run only applies flag_options when a config.toml changes.
    # Force a re-parse with dot-separated flag names so the server binds the
    # chosen port; this replaces the parsed option objects wholesale, which
    # also works in frozen builds where config files are never re-read.
    from streamlit.web.bootstrap import load_config_options

    load_config_options(
        {
            "global.developmentMode": False,
            "server.port": port,
            "server.address": "127.0.0.1",
            "server.headless": True,
            "browser.gatherUsageStats": False,
        }
    )
    flag_options = {
        "global.developmentMode": False,
        "server.port": port,
        "server.address": "127.0.0.1",
        "server.headless": True,
        "browser.gatherUsageStats": False,
    }

    def _run() -> None:
        """Run ``bootstrap.run``; record any exception in ``START_ERROR``."""
        import signal
        import traceback

        try:
            if threading.current_thread() is not threading.main_thread():
                # bootstrap.run installs signal handlers, which Python only
                # allows in the main thread; patch it out in that case.
                original = signal.signal
                signal.signal = lambda *a, **k: None  # type: ignore[assignment]
                try:
                    bootstrap.run(str(APP_PATH), False, [], flag_options)
                finally:
                    signal.signal = original  # type: ignore[assignment]
            else:
                bootstrap.run(str(APP_PATH), False, [], flag_options)
        except BaseException:
            # Daemon-thread exceptions are invisible in a windowed exe —
            # record them so main() can report the real cause.
            START_ERROR.append(traceback.format_exc())

    thread = threading.Thread(target=_run, daemon=True, name="streamlit-server")
    thread.start()
    return thread


def wait_for_server(port: int, timeout: float = 30.0) -> bool:
    """Block until the Streamlit server answers on ``port``.

    Parameters
    ----------
    port : int
        Localhost port of the server.
    timeout : float, default 30.0
        Maximum waiting time (s).

    Returns
    -------
    bool
        ``True`` once ``http://127.0.0.1:<port>/healthz`` responds, ``False``
        if it did not within ``timeout``.

    Notes
    -----
    Polls every 0.25 s after a failed request (each request has a 2 s timeout).
    """
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
    """Launch the desktop app: start the server, then open the pywebview window.

    Raises
    ------
    RuntimeError
        If the Streamlit server does not answer within 30 s; the message
        includes the server thread's traceback when one was recorded in
        ``START_ERROR``.

    Notes
    -----
    Imports pywebview lazily (requires the WebView2 runtime on Windows) and
    blocks in ``webview.start()`` until the window is closed. The server runs on
    a daemon thread, so it stops when the process exits.
    """
    import webview

    port = free_port()
    start_streamlit_inprocess(port)
    if not wait_for_server(port):
        detail = f"\nServer thread error:\n{START_ERROR[0]}" if START_ERROR else ""
        raise RuntimeError(f"Streamlit server did not start on port {port}.{detail}")
    webview.create_window(
        "BlowingBubbles",
        f"http://127.0.0.1:{port}",
        width=1280,
        height=850,
    )
    webview.start()


if __name__ == "__main__":
    main()
