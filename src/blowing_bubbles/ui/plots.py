"""Build matplotlib figures shared by the UI tabs (Streamlit-independent).

:func:`nyquist_figure`, :func:`residual_figure` and
:func:`batch_summary_figure` return figures for the Single/Live Tune and Batch
tabs. Model curves are evaluated on ``FREQ_GRID`` (400 log-spaced points from
1e-5 Hz to 1e5 Hz). Importing this module selects the non-interactive ``Agg``
matplotlib backend.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt

from blowing_bubbles.classes import CircuitParameters, EISData
from blowing_bubbles.classes.circuit_classes import FitResults

FREQ_GRID = np.logspace(-5, 5, 400)


def nyquist_figure(
    data: EISData | None,
    fits: list[tuple[str, FitResults | CircuitParameters]] | None = None,
    unit: str = r"$\Omega$",
) -> plt.Figure:
    r"""Plot the measured spectrum as a Nyquist plot overlaid with model curves.

    Parameters
    ----------
    data : EISData or None
        Measured spectrum, drawn as black markers. ``None`` plots models only.
    fits : list of tuple, optional
        ``(label, model)`` pairs where ``model`` is a :class:`FitResults` or
        :class:`CircuitParameters`; each is drawn as a curve evaluated on
        ``FREQ_GRID``. ``None`` draws no models.
    unit : str, default r"$\Omega$"
        Impedance unit appended to the axis labels (matplotlib mathtext).

    Returns
    -------
    matplotlib.figure.Figure
        Figure with equal axis scaling, :math:`Z'` vs :math:`-Z''`.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    if data is not None:
        ax.plot(
            data.Real,
            -data.Imag,
            "o",
            color="black",
            markersize=4,
            label="Measured",
        )
    for label, model in fits or []:
        z = (
            model.parameters.get_impedance(FREQ_GRID)
            if isinstance(model, FitResults)
            else model.get_impedance(FREQ_GRID)
        )
        ax.plot(z.real, -z.imag, label=label)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel(f"Z' {unit}")
    ax.set_ylabel(f"-Z'' {unit}")
    if (data is not None) or fits:
        ax.legend(loc="best")
    fig.tight_layout()
    return fig


def residual_figure(
    data: EISData,
    fit: FitResults,
    unit: str = r"$\Omega$",
) -> plt.Figure:
    r"""Plot the magnitude of the complex residuals versus frequency.

    Parameters
    ----------
    data : EISData
        Spectrum the fit was made on (supplies the frequencies, Hz).
    fit : FitResults
        Fit whose ``residuals_magnitude`` is plotted; must have the same length
        as ``data.Freq``.
    unit : str, default r"$\Omega$"
        Impedance unit appended to the y-axis label (matplotlib mathtext).

    Returns
    -------
    matplotlib.figure.Figure
        Semilog-x plot of :math:`|\hat{Z}_k - Z_k|`.
    """
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.semilogx(data.Freq, fit.residuals_magnitude, "o-", markersize=3)
    ax.grid(True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel("f (Hz)")
    ax.set_ylabel(f"|residual| {unit}")
    fig.tight_layout()
    return fig


def batch_summary_figure(
    rows: list[dict],
    data_by_name: dict[str, EISData],
    params_by_name: dict[str, CircuitParameters],
) -> plt.Figure:
    """Plot a grid of Nyquist overlays, one panel per successfully fitted file.

    Parameters
    ----------
    rows : list of dict
        Batch report rows; rows with a truthy ``"success"`` get a panel, titled
        by their ``"file"`` value.
    data_by_name : dict of str to EISData
        Measured spectra keyed by file name.
    params_by_name : dict of str to CircuitParameters
        Fitted parameters keyed by file name, drawn on ``FREQ_GRID``.

    Returns
    -------
    matplotlib.figure.Figure
        Grid with at most three columns; unused panels are hidden.

    See Also
    --------
    blowing_bubbles.ui.batch.run_batch : Saves this figure as
        ``batch_summary.png``.
    """
    ok = [r for r in rows if r.get("success")]
    n = max(len(ok), 1)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
    )
    for idx, row in enumerate(ok):
        ax = axes[idx // ncols][idx % ncols]
        name = row["file"]
        data = data_by_name.get(name)
        params = params_by_name.get(name)
        if data is not None:
            ax.plot(data.Real, -data.Imag, "o", color="black", markersize=2)
        if params is not None:
            z = params.get_impedance(FREQ_GRID)
            ax.plot(z.real, -z.imag)
        ax.set_title(name, fontsize=9)
        ax.set_aspect("equal")
        ax.grid(True)
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.tight_layout()
    return fig
