# Examples (Sphinx-Gallery)

Python scripts in this folder are run by **Sphinx-Gallery** when you build the docs.
Each ``.py`` file becomes a gallery page under ``docs/auto_examples/``.

## Add an example

1. Create ``examples/my_example.py`` with a module docstring (title + description).
2. Build docs: ``Build-OmpyDocs`` or ``uv run sphinx-build -b html docs docs/_build``.

## Run locally without building docs

```powershell
uv sync
uv run python examples/plot_circuit_demo.py
```

For matplotlib plots, name scripts ``plot_*.py`` and add ``matplotlib`` to your project dependencies.
