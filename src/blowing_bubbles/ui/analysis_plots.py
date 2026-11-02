"""Figures for the Analysis tab (Streamlit-independent)."""

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
    """Resistance breakdown vs file index — one line per series.

    ``series`` is a list of ``(label, color)``; ``extra_series`` maps a
    label to a table column (used for user-selected R_i columns and the
    aggregated R_rest).
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
    """Difference spectra Delta Z vs frequency (mirrors the im_diff plots)."""
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
