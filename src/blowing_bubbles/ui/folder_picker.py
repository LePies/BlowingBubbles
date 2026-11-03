"""Open a native folder picker for the desktop app (Streamlit-independent core).

:func:`pick_folder` tries, in order: pywebview's native dialog (when running
inside the packaged desktop app), then tkinter's dialog (available on most
Windows installs). It returns ``None`` when no dialog backend is available or
the dialog is cancelled — callers should keep a plain text field as fallback
(browser mode).
"""

from __future__ import annotations

from pathlib import Path


def _pick_with_pywebview() -> str | None:
    """Return a folder from pywebview's dialog; ``None`` if no window or cancel."""
    try:
        import webview
    except ImportError:
        return None
    windows = webview.windows
    if not windows:
        return None
    try:
        result = webview.create_file_dialog(
            webview.FOLDER_DIALOG, directory=str(Path.home())
        )
    except Exception:
        try:
            result = windows[0].create_file_dialog(webview.FOLDER_DIALOG)
        except Exception:
            return None
    # Newer pywebview returns a list/tuple, older returns a bare string.
    if isinstance(result, (list, tuple)):
        return result[0] if result else None
    return result or None


def _pick_with_tkinter() -> str | None:
    """Return a folder from a hidden-root tkinter dialog; ``None`` on failure."""
    try:
        import tkinter
        from tkinter import filedialog
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
    """Open a native 'select folder' dialog and return the chosen path.

    Returns
    -------
    str or None
        Selected folder path, or ``None`` if the user cancelled or no dialog
        backend (pywebview window, tkinter) is available.

    Notes
    -----
    Blocks until the dialog is closed. The pywebview dialog is used only when
    a pywebview window exists (packaged desktop app); otherwise a hidden tkinter
    root window is created and destroyed for the dialog. Because the backends
    are chained with ``or``, cancelling the pywebview dialog falls through to
    the tkinter dialog.
    """
    return _pick_with_pywebview() or _pick_with_tkinter()
