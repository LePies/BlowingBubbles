"""Batch folder fitting (Streamlit-independent).

Fits every EIS CSV in an input folder and writes per-file parameters,
per-frequency residuals, a batch report CSV, and a summary figure into an
output folder. Used by the Batch tab of the UI but usable headlessly.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from blowing_bubbles.classes import CircuitParameters, EISData
from blowing_bubbles.classes.circuit_classes import FitResults
from blowing_bubbles.ui.plots import batch_summary_figure


def _fit_one(
    path: Path,
    fit_kwargs: dict,
    quick_fit_kwargs: dict | None,
    area: float | None = None,
) -> tuple[FitResults, float]:
    data = EISData.from_csv(str(path), area=area)
    start = time.time()
    result = data.fit(**fit_kwargs)
    if quick_fit_kwargs is not None:
        result = data.quick_fit(
            initial_parameters=result.parameters, **quick_fit_kwargs
        )
    return result, time.time() - start


def run_batch(
    input_dir: str | Path,
    output_dir: str | Path,
    fit_kwargs: dict | None = None,
    quick_fit_kwargs: dict | None = None,
    pattern: str = "*.csv",
    area: float | None = None,
    progress=None,
) -> pd.DataFrame:
    """Fit every CSV in ``input_dir`` and write results to ``output_dir``.

    Parameters
    ----------
    input_dir, output_dir : path-like
        Source folder of spectra and destination for results.
    fit_kwargs : dict
        Keyword arguments for :meth:`EISData.fit`.
    quick_fit_kwargs : dict, optional
        If given, each coarse fit is refined with :meth:`EISData.quick_fit`
        using the coarse parameters as the initial guess.
    pattern : str
        Glob pattern for spectrum files (default ``"*.csv"``).
    area : float, optional
        Electrode area in cm^2. If given, every spectrum is normalized to
        area-specific impedance (Ohm cm^2) on load, before fitting.
    progress : callable, optional
        Called as ``progress(i, n, filename)`` after each file.

    Returns
    -------
    pandas.DataFrame
        The batch report (also written to ``output_dir/batch_report.csv``).

    Output layout under ``output_dir``::

        fits/<stem>_parameters.json   CircuitParameters.save()
        fits/<stem>_fit.csv           frequency, measured Z, fitted Z, residual
        batch_report.csv              one row per input file
        batch_summary.png             Nyquist grid of successful fits
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    fits_dir = output_dir / "fits"
    fits_dir.mkdir(parents=True, exist_ok=True)
    fit_kwargs = dict(fit_kwargs or {})

    files = sorted(input_dir.glob(pattern))
    rows: list[dict] = []
    data_by_name: dict[str, EISData] = {}
    params_by_name: dict[str, CircuitParameters] = {}
    for i, path in enumerate(files):
        if progress is not None:
            progress(i, len(files), path.name)
        row = {"file": path.name}
        try:
            result, elapsed = _fit_one(path, fit_kwargs, quick_fit_kwargs, area)
            data = EISData.from_csv(str(path), area=area)
            stem = path.stem
            result.parameters.save(fits_dir / f"{stem}_parameters.json")
            z_fit = result.parameters.get_impedance(data.Freq)
            pd.DataFrame(
                {
                    "Freq (Hz)": data.Freq,
                    "Zreal (Ohm)": data.Real,
                    "Zimag (Ohm)": data.Imag,
                    "Zfit_real": z_fit.real,
                    "Zfit_imag": z_fit.imag,
                    "residual_magnitude": result.residuals_magnitude,
                }
            ).to_csv(fits_dir / f"{stem}_fit.csv", index=False)
            row.update(
                success=bool(result.success),
                cost=float(result.cost),
                nfev=int(result.nfev),
                n_groups=int(result.n_groups),
                runtime_s=round(elapsed, 3),
                error="",
            )
            data_by_name[path.name] = data
            params_by_name[path.name] = result.parameters
        except Exception as exc:
            row.update(
                success=False,
                cost=np.nan,
                nfev=np.nan,
                n_groups=np.nan,
                runtime_s=np.nan,
                error=f"{type(exc).__name__}: {exc}",
            )
        rows.append(row)
    if progress is not None:
        progress(len(files), len(files), "")

    report = pd.DataFrame(rows)
    report.to_csv(output_dir / "batch_report.csv", index=False)
    if params_by_name:
        fig = batch_summary_figure(rows, data_by_name, params_by_name)
        fig.savefig(output_dir / "batch_summary.png", dpi=120)
    with (output_dir / "batch_report.json").open("w", encoding="utf-8") as f:
        json.dump({"files": len(files)}, f)
    return report


def load_report(output_dir: str | Path) -> pd.DataFrame:
    """Load a previously written ``batch_report.csv`` from ``output_dir``."""
    return pd.read_csv(Path(output_dir) / "batch_report.csv")


def load_fit(output_dir: str | Path, stem: str) -> CircuitParameters:
    """Load saved fit parameters for ``stem`` from an output folder."""
    return CircuitParameters.load(Path(output_dir) / "fits" / f"{stem}_parameters.json")


def make_synthetic_csv(path: str | Path, seed: int = 42) -> Path:
    """Write a synthetic R-CPE spectrum CSV (demo/testing helper)."""
    rng = np.random.default_rng(seed)
    true = CircuitParameters(
        resistance_source=10.0,
        inductance=1e-7,
        inductance_exponent=0.95,
        resistances=np.array([50.0, 20.0, 5.0]),
        capacitances=np.array([1e-4, 1e-3, 1e-2]),
        exponents=np.array([0.98, 0.95, 0.90]),
    )
    freqs = np.logspace(-2, 5, 60)
    z = true.get_impedance(freqs)
    frame = pd.DataFrame(
        {
            "Freq (Hz)": freqs,
            "Zreal (Ohm)": z.real + rng.normal(0, 0.05, freqs.size),
            "Zimag (Ohm)": z.imag + rng.normal(0, 0.05, freqs.size),
        }
    )
    path = Path(path)
    frame.to_csv(path, index=False)
    return path
