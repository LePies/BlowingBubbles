# PyInstaller spec for the BlowingBubbles desktop app (pywebview window).
# Build:  uv sync --group desktop && uv run pyinstaller desktop.spec --clean
# Output: dist/BlowingBubbles.exe (Windows) / dist/BlowingBubbles (macOS/Linux)

from pathlib import Path

import streamlit
from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

STREAMLIT_DIR = Path(streamlit.__file__).resolve().parent

# Collect every streamlit submodule and every data file shipped with the
# package (static frontend, vendor JS, protobufs, templates). Streamlit
# lazy-imports many submodules at server start; missing ones kill the
# server thread inside a frozen exe.
streamlit_datas, streamlit_binaries, streamlit_hidden = collect_all("streamlit")

hiddenimports = streamlit_hidden + collect_submodules("streamlit") + [
    "pywebview",
    "blowing_bubbles.ui.app",
    "blowing_bubbles.ui.analysis",
    "blowing_bubbles.ui.analysis_plots",
    "blowing_bubbles.ui.batch",
    "blowing_bubbles.ui.live",
    "blowing_bubbles.ui.plots",
    "blowing_bubbles.ui.desktop",
]

# Several packages (streamlit, pywebview, altair, ...) call
# importlib.metadata.version()/entry_points at import time; PyInstaller
# does not bundle dist-info by default, so copy it explicitly.
METADATA_PACKAGES = [
    "streamlit",
    "pywebview",
    "altair",
    "pandas",
    "numpy",
    "scipy",
    "matplotlib",
    "pyarrow",
]

datas = list(streamlit_datas)
for pkg in METADATA_PACKAGES:
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass  # package not installed in this env; skip

a = Analysis(
    [str(Path("src/blowing_bubbles/ui/desktop.py"))],
    pathex=[str(Path("src"))],
    binaries=streamlit_binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "PyQt5", "PyQt6", "PySide6", "wx"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="BlowingBubbles",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # windowed app: no console window on Windows
    icon=None,
)
