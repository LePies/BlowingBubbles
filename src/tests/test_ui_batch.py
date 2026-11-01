"""Headless tests for the UI batch and live-tune helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from blowing_bubbles.classes import EISData
from blowing_bubbles.ui import batch as batch_mod
from blowing_bubbles.ui import live as live_mod
from blowing_bubbles.ui import plots


@pytest.fixture()
def spectra_folder(tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    for i in range(2):
        batch_mod.make_synthetic_csv(in_dir / f"cell_{i:02d}.csv", seed=42 + i)
    return in_dir


def test_run_batch_writes_outputs(spectra_folder, tmp_path):
    out_dir = tmp_path / "out"
    progress_calls = []
    report = batch_mod.run_batch(
        spectra_folder,
        out_dir,
        fit_kwargs={"n_restarts": 5, "rc_groups": 2, "seed": 0},
        progress=lambda i, n, name: progress_calls.append((i, n, name)),
    )
    assert list(report["file"]) == ["cell_00.csv", "cell_01.csv"]
    assert report["success"].all()
    assert (out_dir / "batch_report.csv").exists()
    assert (out_dir / "batch_summary.png").exists()
    for stem in ("cell_00", "cell_01"):
        assert (out_dir / "fits" / f"{stem}_parameters.json").exists()
        fit_csv = pd.read_csv(out_dir / "fits" / f"{stem}_fit.csv")
        assert {"Freq (Hz)", "Zfit_real", "residual_magnitude"} <= set(fit_csv.columns)
    assert progress_calls[-1][0] == 2


def test_run_batch_continues_on_error(spectra_folder, tmp_path):
    (spectra_folder / "broken.csv").write_text("not,a,valid\nspectrum,1,2\n")
    out_dir = tmp_path / "out"
    report = batch_mod.run_batch(
        spectra_folder, out_dir, fit_kwargs={"n_restarts": 3, "rc_groups": 1, "seed": 0}
    )
    assert len(report) == 3
    broken = report[report["file"] == "broken.csv"].iloc[0]
    assert not broken["success"]
    assert broken["error"]


def test_run_batch_normalizes_by_area(spectra_folder, tmp_path):
    out_raw = tmp_path / "raw"
    out_norm = tmp_path / "norm"
    kwargs = {"n_restarts": 3, "rc_groups": 1, "seed": 0}
    batch_mod.run_batch(spectra_folder, out_raw, fit_kwargs=kwargs)
    batch_mod.run_batch(spectra_folder, out_norm, fit_kwargs=kwargs, area=2.0)
    raw = pd.read_csv(out_raw / "fits" / "cell_00_fit.csv")
    norm = pd.read_csv(out_norm / "fits" / "cell_00_fit.csv")
    assert norm["Zreal (Ohm)"].iloc[0] == pytest.approx(
        2.0 * raw["Zreal (Ohm)"].iloc[0]
    )
    assert norm["Zfit_real"].iloc[0] == pytest.approx(2.0 * raw["Zfit_real"].iloc[0])


def test_load_report_and_fit(spectra_folder, tmp_path):
    out_dir = tmp_path / "out"
    batch_mod.run_batch(
        spectra_folder, out_dir, fit_kwargs={"n_restarts": 3, "rc_groups": 1, "seed": 0}
    )
    report = batch_mod.load_report(out_dir)
    assert len(report) == 2
    params = batch_mod.load_fit(out_dir, "cell_00")
    assert len(params) >= 1


def test_live_tau_roundtrip():
    params = live_mod.default_parameters(2)
    live_mod.set_branch_tau(params, 0, 1e-3)
    assert live_mod.branch_tau(params, 0) == pytest.approx(1e-3)
    assert params.capacitances[0] == pytest.approx(1e-3 / params.resistances[0])
    assert live_mod.branch_peak_frequency(params, 0) > 0.0


@pytest.fixture()
def demo_data(tmp_path):
    csv_path = batch_mod.make_synthetic_csv(tmp_path / "demo.csv", seed=1)
    return EISData(pd.read_csv(csv_path))


def test_live_residual_rms_finite(demo_data):
    params = live_mod.default_parameters(3)
    rms = live_mod.live_residual_rms(demo_data, params)
    assert np.isfinite(rms) and rms >= 0.0


def test_analysis_table_and_difference(spectra_folder, tmp_path):
    from blowing_bubbles.ui import analysis as analysis_mod

    out_dir = tmp_path / "out"
    batch_mod.run_batch(
        spectra_folder, out_dir, fit_kwargs={"n_restarts": 3, "rc_groups": 1, "seed": 0}
    )
    table = analysis_mod.analysis_table(out_dir)
    assert len(table) == 2
    for col in ("Rs", "Rp", "Rtot", "R1_slowest", "R_rest"):
        assert col in table.columns
    assert np.allclose(table["Rtot"], table["Rs"] + table["Rp"])

    names = ["cell_00.csv", "cell_01.csv"]
    diff = analysis_mod.difference_spectra(spectra_folder, out_dir, names, names[0])
    assert set(diff) == set(names)
    freq, dz_ref = diff[names[0]]
    assert np.allclose(dz_ref, 0.0)
    assert len(freq) == len(diff[names[1]][0])


def test_nyquist_figure_runs(demo_data):
    import matplotlib.pyplot as plt

    params = live_mod.default_parameters(2)
    fig = plots.nyquist_figure(demo_data, [("live", params)])
    assert fig.axes
    plt.close(fig)
