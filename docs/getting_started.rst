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

.. code-block:: bash

   uv run python examples/plot_circuit_demo.py

Add new ``examples/your_example.py`` files with a module docstring (title underlined
with ``=``). For matplotlib figures, use filenames like ``plot_demo.py``.

Package icon
------------

To show a logo in the sidebar and on the home page, re-run init with an image path
(``.png``, ``.jpg``, or ``.svg``), or copy your icon to :file:`docs/_static/package_icon.png`
and set ``html_logo`` / ``html_favicon`` in :file:`docs/conf.py`.

.. code-block:: powershell

   Init-OmpyDocs -ProjectPath . -IconPath .\assets\logo.png -Force

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

.. code-block:: bash

   Build-OmpyDocs

Or manually:

.. code-block:: bash

   uv sync
   uv run ruff check .
   uv run ty check
   uv run sphinx-build -b html docs docs/_build

Open :file:`docs/_build/index.html` in a browser.

Alternatively, from :file:`docs/` run ``make html`` (or ``make.bat`` on Windows).
