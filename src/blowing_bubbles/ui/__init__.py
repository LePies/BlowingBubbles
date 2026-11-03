r"""Provide the optional Streamlit UI for BlowingBubbles.

Launch with::

    streamlit run src/blowing_bubbles/ui/app.py

or::

    python -m blowing_bubbles.ui

or, as a native desktop window, via :mod:`blowing_bubbles.ui.desktop`.

Requires the optional ``ui`` dependency group (``uv sync --group ui``).
The submodules :mod:`.plots`, :mod:`.batch`, :mod:`.live`, :mod:`.analysis`,
:mod:`.analysis_plots` and :mod:`.folder_picker` are Streamlit-independent
helpers and can be used without the UI installed.
"""

__all__ = [
    "analysis",
    "analysis_plots",
    "batch",
    "folder_picker",
    "live",
    "plots",
]
