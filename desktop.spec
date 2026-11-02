# PyInstaller spec for the BlowingBubbles desktop app (pywebview window).
# Build:  uv sync --group desktop && uv run pyinstaller desktop.spec
# Output: dist/BlowingBubbles.exe (Windows) / dist/BlowingBubbles (macOS/Linux)

from pathlib import Path

import streamlit

STREAMLIT_DIR = Path(streamlit.__file__).resolve().parent

hiddenimports = [
    "streamlit",
    "streamlit.web.bootstrap",
    "pywebview",
    "blowing_bubbles.ui.app",
    "blowing_bubbles.ui.analysis",
    "blowing_bubbles.ui.analysis_plots",
    "blowing_bubbles.ui.batch",
    "blowing_bubbles.ui.live",
    "blowing_bubbles.ui.plots",
]

datas = [
    (str(STREAMLIT_DIR / "static"), "streamlit/static"),
    (str(STREAMLIT_DIR / "vendor"), "streamlit/vendor"),
]

a = Analysis(
    [str(Path("src/blowing_bubbles/ui/desktop.py"))],
    pathex=[str(Path("src"))],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "PyQt5", "PyQt6", "PySide6"],
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
