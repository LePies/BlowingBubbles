r"""Optional Streamlit UI for BlowingBubbles.

Launch with::

    streamlit run src/blowing_bubbles/ui/app.py

or::

    python -m blowing_bubbles.ui

Requires the optional ``ui`` dependency group (``uv sync --group ui``).
The submodules :mod:`.plots`, :mod:`.batch` and :mod:`.live` are
Streamlit-independent helpers and can be used without the UI installed.
"""

__all__ = ["analysis", "analysis_plots", "batch", "live", "plots"]
