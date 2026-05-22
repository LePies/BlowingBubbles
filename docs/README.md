# BlowingBubbles — Documentation

Sphinx + Furo documentation for this project.

## Build (uv)

From the **project root**:

```powershell
uv sync
uv run sphinx-build -b html docs docs/_build
```

Open `docs/_build/index.html` in a browser.

From inside `docs/` you can also run `make html` or `.\make.bat`.

Dependencies are managed in `pyproject.toml` under `[dependency-groups] docs`.
The file `requirements-docs.txt` mirrors those pins for convenience.
