r"""Analyse fitted batch results (Streamlit-independent).

Derived from the im_diff / run overview scripts. :func:`analysis_table` builds
the per-file resistance breakdown (Rs, Rp, Rtot, per-branch R and tau) from a
batch output folder, :func:`series_columns` and :func:`max_branches` select the
branch columns to plot, and :func:`difference_spectra` computes model
difference spectra :math:`\Delta Z = Z - Z_\mathrm{ref}` against a reference
file. Used by the Analysis tab of the UI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from blowing_bubbles.classes import CircuitParameters, EISData
from blowing_bubbles.ui import batch as batch_mod


def resistance_summary(params: CircuitParameters) -> dict[str, float]:
    r"""Return Rs, Rp, Rtot and the slowest-branch split of a fitted circuit.

    Parameters
    ----------
    params : CircuitParameters
        Fitted circuit.

    Returns
    -------
    dict of str to float
        Resistances (Ω, or Ω cm² for area-normalized data) under the keys
        ``"Rs"`` (source resistance), ``"R1_slowest"`` (branch with the largest
        :math:`\tau`), ``"R_rest"`` (sum of the other branches), ``"Rp"`` (sum
        of all branch resistances) and ``"Rtot"`` (``Rs + Rp``).

    Notes
    -----
    Mirrors ``_resistance_summary`` from the run overview script. A circuit
    without branches gives ``R1_slowest = R_rest = Rp = 0``.
    """
    r_s = float(params.resistance_source)
    r_branch = np.asarray(params.resistances, dtype=float)
    taus = np.asarray(params.taus, dtype=float)
    if r_branch.size == 0:
        r_1 = r_rest = 0.0
    else:
        slowest = int(np.argmax(taus))
        r_1 = float(r_branch[slowest])
        r_rest = float(np.sum(np.delete(r_branch, slowest)))
    r_p = float(np.sum(r_branch))
    return {
        "Rs": r_s,
        "R1_slowest": r_1,
        "R_rest": r_rest,
        "Rp": r_p,
        "Rtot": r_s + r_p,
    }


def analysis_table(
    out_dir: str | Path,
    report: pd.DataFrame | None = None,
) -> pd.DataFrame:
    r"""Build the per-file analysis table for a batch output folder.

    Parameters
    ----------
    out_dir : str or pathlib.Path
        Batch output folder written by
        :func:`~blowing_bubbles.ui.batch.run_batch`.
    report : pandas.DataFrame, optional
        Batch report with at least a ``"file"`` column. ``None`` loads
        ``out_dir/batch_report.csv``.

    Returns
    -------
    pandas.DataFrame
        One row per report row with columns ``file``, ``success``, ``cost``
        and, when ``fits/<stem>_parameters.json`` exists, the
        :func:`resistance_summary` keys, ``n_branches``, ``tau_min`` and
        ``tau_max`` (s) and per-branch resistances ``R_1``, ``R_2``, ... in
        the stored (increasing-:math:`\tau`) branch order.

    See Also
    --------
    resistance_summary
    series_columns
    """
    out_dir = Path(out_dir)
    if report is None:
        report = batch_mod.load_report(out_dir)
    rows = []
    for _, row in report.iterrows():
        stem = Path(str(row["file"])).stem
        params_path = out_dir / "fits" / f"{stem}_parameters.json"
        entry = {
            "file": row["file"],
            "success": row.get("success", False),
            "cost": row.get("cost", np.nan),
        }
        if params_path.exists():
            params = CircuitParameters.load(params_path)
            entry.update(resistance_summary(params))
            entry["n_branches"] = len(params)
            entry["tau_min"] = float(np.min(params.taus)) if len(params) else np.nan
            entry["tau_max"] = float(np.max(params.taus)) if len(params) else np.nan
            # Per-branch resistances, tau-sorted (index 0 = slowest branch).
            for i, r in enumerate(np.asarray(params.resistances, dtype=float)):
                entry[f"R_{i + 1}"] = float(r)
        rows.append(entry)
    return pd.DataFrame(rows)


def max_branches(table: pd.DataFrame) -> int:
    """Return the highest branch index ``i`` of an ``R_i`` column (0 if none).

    Parameters
    ----------
    table : pandas.DataFrame
        Table from :func:`analysis_table`.

    Returns
    -------
    int
        Largest ``i`` among columns named ``R_<i>``.
    """
    n = 0
    for col in table.columns:
        if col.startswith("R_") and col[2:].isdigit():
            n = max(n, int(col[2:]))
    return n


def series_columns(table: pd.DataFrame, selected: list[int]) -> dict[str, str]:
    """Map plot series labels to table columns for the chosen branch indices.

    Selected branches get their own line (``R_1``, ``R_2``, ... in tau-sorted
    order); all unselected branches are summed into ``R_rest``.

    Parameters
    ----------
    table : pandas.DataFrame
        Table from :func:`analysis_table`; modified in place (see Notes).
    selected : list of int
        1-based branch indices to plot individually. Indices outside
        ``1..max_branches(table)`` are ignored.

    Returns
    -------
    dict of str to str
        Series label to column name, suitable as ``extra_series`` for
        :func:`~blowing_bubbles.ui.analysis_plots.resistance_overview_figure`.
        Contains ``"R_rest"`` (only if some branch is unselected) and one
        ``"R_i"`` entry per selected branch. ``Rs``, ``Rp`` and ``Rtot`` are not
        included; they are table columns already.

    Notes
    -----
    If any branch is unselected, the column ``"R_rest_selected"`` (row-wise sum
    of the unselected ``R_i`` columns, NaN treated as 0) is added to ``table``.
    """
    n = max_branches(table)
    selected = [i for i in selected if 1 <= i <= n]
    cols: dict[str, str] = {}
    rest_indices = [i for i in range(1, n + 1) if i not in selected]
    if rest_indices:
        rest_col = "R_rest_selected"
        table[rest_col] = table[[f"R_{i}" for i in rest_indices]].sum(axis=1)
        cols["R_rest"] = rest_col
    for i in selected:
        cols[f"R_{i}"] = f"R_{i}"
    return cols


def load_spectrum(
    out_dir: str | Path, in_dir: str | Path, name: str, area: float | None = None
) -> EISData:
    """Load the measured spectrum for ``name`` from the batch input folder.

    Parameters
    ----------
    out_dir : str or pathlib.Path
        Batch output folder (unused; kept for a uniform call signature).
    in_dir : str or pathlib.Path
        Batch input folder containing the CSV spectra.
    name : str
        CSV file name within ``in_dir``.
    area : float, optional
        Electrode area (cm²) to normalize the impedance by on load. ``None``
        leaves it unnormalized.

    Returns
    -------
    EISData
        The measured spectrum.
    """
    return EISData.from_csv(str(Path(in_dir) / name), area=area)


def difference_spectra(
    in_dir: str | Path,
    out_dir: str | Path,
    names: list[str],
    reference: str,
    area: float | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    r"""Compute model difference spectra :math:`\Delta Z = Z - Z_\mathrm{ref}`.

    Parameters
    ----------
    in_dir : str or pathlib.Path
        Batch input folder; the reference CSV is read only for its frequencies.
    out_dir : str or pathlib.Path
        Batch output folder holding ``fits/<stem>_parameters.json``.
    names : list of str
        File names to compare against the reference.
    reference : str
        File name of the reference spectrum.
    area : float, optional
        Electrode area (cm²) passed on when loading the reference CSV.

    Returns
    -------
    dict of str to tuple of numpy.ndarray
        ``{name: (freq, delta_z)}`` with ``freq`` (Hz) the reference's
        measured frequencies and ``delta_z`` the complex difference (Ω), both
        of shape ``(M,)``. The reference is included with ``delta_z`` all zeros.

    Notes
    -----
    Both :math:`Z` and :math:`Z_\mathrm{ref}` are the *fitted model*
    impedances (from the saved parameters) evaluated on the reference's
    frequency grid; no measured data are interpolated or subtracted.
    """
    ref_data = load_spectrum(out_dir, in_dir, reference, area)
    ref_params = batch_mod.load_fit(out_dir, Path(reference).stem)
    freq = ref_data.Freq
    z_ref = ref_params.get_impedance(freq)
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {
        reference: (freq.copy(), np.zeros_like(z_ref))
    }
    for name in names:
        if name == reference:
            continue
        params = batch_mod.load_fit(out_dir, Path(name).stem)
        z = params.get_impedance(freq)
        result[name] = (freq.copy(), z - z_ref)
    return result
