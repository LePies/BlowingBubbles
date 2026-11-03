"""Build the figures for the Analysis tab (Streamlit-independent).

:func:`resistance_overview_figure` plots the per-file resistance breakdown and
:func:`difference_figure` the difference spectra from
:mod:`blowing_bubbles.ui.analysis`. ``RESISTANCE_SERIES`` lists the default
``(label, color)`` series. Importing this module selects the non-interactive
``Agg`` matplotlib backend.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

RESISTANCE_SERIES = [
    ("Rs", "tab:blue"),
    ("R1_slowest", "tab:orange"),
    ("R_rest", "tab:red"),
    ("Rp", "tab:green"),
    ("Rtot", "black"),
]


def resistance_overview_figure(
    table: pd.DataFrame,
    series: list[tuple[str, str]] | None = None,
    extra_series: dict[str, str] | None = None,
) -> plt.Figure:
    """Plot the resistance breakdown vs file, one line per series.

    Parameters
    ----------
    table : pandas.DataFrame
        Table from :func:`~blowing_bubbles.ui.analysis.analysis_table` (with a
        ``"file"`` column). Rows with a false ``"success"`` value are dropped;
        the remaining rows are plotted in table order.
    series : list of tuple of str, optional
        ``(label, color)`` pairs; each label is also the table column plotted.
        ``None`` uses ``RESISTANCE_SERIES``. Labels without a matching column
        are skipped.
    extra_series : dict of str to str, optional
        Additional series mapping a label to a table column (used for
        user-selected ``R_i`` columns and the aggregated ``R_rest``, see
        :func:`~blowing_bubbles.ui.analysis.series_columns`). Colors cycle
        through a fixed palette.

    Returns
    -------
    matplotlib.figure.Figure
        Line plot of resistance (Ω) with the file names as x tick labels.
    """
    ok = table[table.get("success", True).astype(bool)] if len(table) else table
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(ok))
    all_series: list[tuple[str, str]] = list(series or RESISTANCE_SERIES)
    if extra_series:
        palette = ["tab:orange", "tab:red", "tab:purple", "tab:brown", "tab:pink"]
        for i, label in enumerate(extra_series):
            all_series.append((label, palette[i % len(palette)]))
    for label, color in all_series:
        if label not in ok.columns and not (extra_series and label in extra_series):
            continue
        col = extra_series.get(label, label) if extra_series else label
        if col not in ok.columns:
            continue
        ax.plot(
            x,
            ok[col].to_numpy(dtype=float),
            marker="o",
            markersize=4,
            linewidth=1.5,
            color=color,
            label=label,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(ok["file"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(r"Resistance ($\Omega$)")
    ax.grid(True, alpha=0.7)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def difference_figure(
    diff: dict[str, tuple[np.ndarray, np.ndarray]],
    reference: str,
    mode: str = "imag",
) -> plt.Figure:
    r"""Plot difference spectra :math:`\Delta Z` vs frequency.

    Mirrors the im_diff plots.

    Parameters
    ----------
    diff : dict of str to tuple of numpy.ndarray
        ``{name: (freq, delta_z)}`` from
        :func:`~blowing_bubbles.ui.analysis.difference_spectra`, with ``freq``
        in Hz and complex ``delta_z`` in Ω.
    reference : str
        Name of the reference file; its (zero) entry is not plotted.
    mode : {"imag", "mag"}, default "imag"
        ``"imag"`` plots :math:`-\mathrm{Im}(\Delta Z)`; any other value plots
        :math:`|\Delta Z|`.

    Returns
    -------
    matplotlib.figure.Figure
        Semilog-x plot with one line per compared file.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, (freq, dz) in diff.items():
        if name == reference:
            continue
        y = -dz.imag if mode == "imag" else np.abs(dz)
        ax.semilogx(freq, y, marker="o", markersize=3, label=name)
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_xlabel("f (Hz)")
    ax.set_ylabel(r"$-\mathrm{Im}(\Delta Z)$" if mode == "imag" else r"$|\Delta Z|$")
    ax.set_title(f"Difference vs reference: {reference}")
    ax.grid(True, alpha=0.7)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig
