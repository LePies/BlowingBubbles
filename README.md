# BlowingBubbles

Python tools for electrochemical impedance spectroscopy (EIS): load spectra, evaluate
the R–CPE equivalent circuit, and fit branch parameters.

## License

Released under the [MIT License](LICENSE): free use, modification, and redistribution.

## Citation (published scientific work)

If you use BlowingBubbles in **published** scientific work (papers, theses, preprints,
citable reports), please cite the entry in [`CITATION.bib`](CITATION.bib) or
[`CITATION.cff`](CITATION.cff). Private or internal use does not require a citation.
See [`ACADEMIC_CITATION.md`](ACADEMIC_CITATION.md).

## Interactive UI

An optional Streamlit UI (single-spectrum fitting, batch folder fitting, and a
live-tune mode) is included. Install the extra group and launch it:

```bash
uv sync --group ui
streamlit run src/blowing_bubbles/ui/app.py   # or: python -m blowing_bubbles.ui
```

### Windows desktop app

The same UI can be packaged as a standalone desktop app (native window via
Edge WebView2, no browser chrome):

```bash
uv sync --group desktop
uv run pyinstaller desktop.spec --clean
# produces dist/BlowingBubbles/ with the launcher BlowingBubbles.exe inside
# (onedir build: starts much faster than onefile; ship the whole folder)
```

The exe needs the WebView2 runtime on the target machine (preinstalled on
Windows 11; a bootstrapper is available for Windows 10).

## Documentation

Build locally with `Build-OmpyDocs` or see `docs/getting_started.rst`.
