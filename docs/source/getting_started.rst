.. _getting_started:

===============
Getting started
===============

Environment
-----------

This project uses `uv <https://docs.astral.sh/uv/>`_ for dependencies and virtual environments.
From the project root:

.. code-block:: bash

   uv sync

Linting and type checking
-------------------------

.. code-block:: bash

   uv run ruff check .
   uv run ruff format .
   uv run ty check

Examples (Sphinx-Gallery)
-------------------------

Scripts in :file:`examples/` are executed when you build the docs. The gallery
appears under :doc:`auto_examples/index`.

Demo notebooks
--------------

Interactive, numbered demo notebooks live in :file:`examples/notebooks/`. They walk
through the Python API step by step (loading and inspecting spectra, fitting the
R\u2013CPE circuit, refining with ``quick_fit`` and Lin-KK quality checks) and are
rendered under *Examples* in these docs. To run them locally:

.. code-block:: bash

   uv sync
   uv run jupyter notebook examples/notebooks

Interactive UI
--------------

An optional Streamlit UI provides three tabs: **Single** (load one spectrum,
run ``fit``, inspect the Nyquist overlay and parameters, refine with
``quick_fit``), **Batch** (fit every CSV in a folder into an output folder
with a report CSV and summary figure), and **Live Tune** (adjust
R / C / tau / exponent / L with sliders and watch the Nyquist update in
real time).

.. code-block:: bash

   uv sync --group ui
   streamlit run src/blowing_bubbles/ui/app.py
   # or: python -m blowing_bubbles.ui

Desktop app (Windows)
+++++++++++++++++++++

Package the UI as a standalone desktop application with a native window
(Edge WebView2 via pywebview, no browser UI):

.. code-block:: bash

   uv sync --group desktop
   uv run pyinstaller desktop.spec --clean
   # produces dist/BlowingBubbles/ (onedir; launch BlowingBubbles.exe inside)

The target machine needs the WebView2 runtime (preinstalled on Windows 11;
a bootstrapper is available for Windows 10).

.. code-block:: bash

   uv run python examples/plot_circuit_demo.py

Add new ``examples/your_example.py`` files with a module docstring (title underlined
with ``=``). For matplotlib figures, use filenames like ``plot_demo.py``.

Using the package
-----------------

After ``uv sync``, import from the installed package:

.. code-block:: python

   from blowing_bubbles.classes import EISData

   # eis = EISData.from_csv("path/to/spectrum.csv")
   # result = eis.fit(n_restarts=200, rc_groups=5)
   # params = result.parameters

Building this documentation
---------------------------

The documentation is built with Sphinx. From the project root:

.. code-block:: bash

   uv sync
   uv run ruff check .
   uv run ty check
   uv run sphinx-build -W -b html docs/source docs/build/html

Open :file:`docs/build/html/index.html` in a browser.

Alternatively, from :file:`docs/` run ``make html`` (or ``make.bat`` on Windows).
