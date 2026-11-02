"""Native folder picker for the desktop app (Streamlit-independent core).

``pick_folder`` tries, in order: pywebview's native dialog (when running
inside the packaged desktop app), then tkinter's dialog (available on most
Windows installs). Returns ``None`` when no dialog backend is available —
callers should keep a plain text field as fallback (browser mode).
"""

from __future__ import annotations


def _pick_with_pywebview() -> str | None:
    try:
        import webview  # noqa: PLC0415
    except ImportError:
        return None
    windows = webview.windows
    if not windows:
        return None
    try:
        result = windows[0].create_file_dialog(webview.FOLDER_DIALOG)
    except Exception:
        return None
    # Newer pywebview returns a list/tuple, older returns a bare string.
    if isinstance(result, (list, tuple)):
        return result[0] if result else None
    return result or None


def _pick_with_tkinter() -> str | None:
    try:
        import tkinter  # noqa: PLC0415
        from tkinter import filedialog  # noqa: PLC0415
    except ImportError:
        return None
    try:
        root = tkinter.Tk()
        root.withdraw()
        path = filedialog.askdirectory()
        root.destroy()
        return path or None
    except Exception:
        return None


def pick_folder() -> str | None:
    """Open a native 'select folder' dialog; ``None`` if unavailable."""
    return _pick_with_pywebview() or _pick_with_tkinter()
