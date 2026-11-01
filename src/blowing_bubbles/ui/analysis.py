"""Analysis helpers for fitted batch results (Streamlit-independent).

Derived from the im_diff / run overview scripts: per-file resistance
breakdown (Rs, Rp, Rtot, per-branch R and tau) and difference spectra
Delta Z = Z - Z_ref against a chosen reference file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from blowing_bubbles.classes import CircuitParameters, EISData
from blowing_bubbles.ui import batch as batch_mod


def resistance_summary(params: CircuitParameters) -> dict[str, float]:
    """Rs, Rp, Rtot and slowest-branch split of a fitted circuit.

    Mirrors ``_resistance_summary`` from the run overview script:
    R1 = resistance of the slowest branch (largest tau), R_rest = sum of
    the faster branches, Rp = sum of all branch resistances.
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
    """Per-file analysis table for a batch output folder.

    Loads every ``fits/<stem>_parameters.json`` listed in the batch report
    (or discovered on disk) and appends the resistance summary plus
    per-branch tau / peak frequency columns.
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
        rows.append(entry)
    return pd.DataFrame(rows)


def load_spectrum(
    out_dir: str | Path, in_dir: str | Path, name: str, area: float | None = None
) -> EISData:
    """Load the measured spectrum for ``name`` from the batch input folder."""
    return EISData.from_csv(str(Path(in_dir) / name), area=area)


def difference_spectra(
    in_dir: str | Path,
    out_dir: str | Path,
    names: list[str],
    reference: str,
    area: float | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Delta Z = Z - Z_ref per selected file.

    Returns ``{name: (freq, delta_z_complex)}`` with the reference included
    as zeros. Frequency grids are interpolated onto the reference's grid.
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
