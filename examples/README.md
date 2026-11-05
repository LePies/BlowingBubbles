# Examples (Sphinx-Gallery)

Python scripts in this folder are run by **Sphinx-Gallery** when you build the docs.
Each ``.py`` file becomes a gallery page under ``docs/auto_examples/``.

## Add an example

1. Create ``examples/my_example.py`` with a module docstring (title + description).
2. Build docs: ``uv run sphinx-build -W -b html docs/source docs/build/html``.

## Run locally without building docs

```powershell
uv sync
uv run python examples/plot_circuit_demo.py
```

For matplotlib plots, name scripts ``plot_*.py`` and add ``matplotlib`` to your project dependencies.

## Demo notebooks

The ``notebooks/`` folder holds interactive demo notebooks (numbered tutorials, like
the nest-soc project) that walk through the Python API: loading and inspecting spectra,
fitting the R-CPE circuit, and refining with ``quick_fit`` plus Lin-KK fit-quality checks.

Open them with Jupyter:

```powershell
uv sync
uv run jupyter notebook examples/notebooks
```

They are also rendered into the documentation under *Examples* (the docs build copies
them into ``docs/source/notebooks/``; re-execute the notebooks to refresh their outputs).
