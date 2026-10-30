"""Tests for EISData consistency, input checks, area handling and wrappers."""

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest

from blowing_bubbles.classes import data_classes
from blowing_bubbles.classes.circuit_classes import CircuitParameters
from blowing_bubbles.classes.data_classes import EISData
from blowing_bubbles.scripts import global_minimize

FREQ = np.logspace(-2, 5, 40)
TRUTH = CircuitParameters(
    10.0, 0.0, 1.0,
    np.array([20.0, 50.0]), np.array([1e-5, 1e-2]), np.array([0.9, 0.85]),
)


def _frame(freq=FREQ):
    z = TRUTH.get_impedance(freq)
    return pd.DataFrame({"freq": freq, "real": z.real, "imag": z.imag})


def _assert_consistent(eis):
    """Arrays, DataFrame and derived resistances must all agree."""
    r_key, i_key, f_key = eis.keys
    np.testing.assert_array_equal(eis.Real, eis.data[r_key].to_numpy())
    np.testing.assert_array_equal(eis.Imag, eis.data[i_key].to_numpy())
    np.testing.assert_array_equal(eis.Freq, eis.data[f_key].to_numpy())
    fresh = EISData(eis.data.reset_index(drop=True), columns=eis.keys)
    assert eis.Rs == pytest.approx(fresh.Rs)
    assert eis.Rtot == pytest.approx(fresh.Rtot)
    assert eis.Rp == pytest.approx(eis.Rtot - eis.Rs)


# --- area / normalization ----------------------------------------------------


def test_area_normalizes_impedance():
    frame = _frame()
    original = frame.copy()
    eis = EISData(frame, area=2.5)
    np.testing.assert_allclose(eis.Real, 2.5 * original["real"].to_numpy())
    np.testing.assert_allclose(eis.Imag, 2.5 * original["imag"].to_numpy())
    np.testing.assert_allclose(eis.Freq, original["freq"].to_numpy())
    assert eis.normalized is True
    assert eis.normalization_area == 2.5
    assert eis.unit == "Ohm cm^2"
    raw = EISData(original)
    assert eis.Rs == pytest.approx(2.5 * raw.Rs)
    assert eis.Rtot == pytest.approx(2.5 * raw.Rtot)
    _assert_consistent(eis)
    # The caller's DataFrame is not modified.
    pd.testing.assert_frame_equal(frame, original)


def test_area_keeps_non_default_unit():
    eis = EISData(_frame(), area=2.0, unit="mOhm m^2")
    assert eis.unit == "mOhm m^2"


def test_no_area_means_not_normalized():
    eis = EISData(_frame())
    assert eis.normalized is False
    assert eis.normalization_area is None
    assert eis.unit == "Ohm"


def test_normalize_sets_bool_flag():
    eis = EISData(_frame())
    eis.normalize(3.0, "Ohm cm^2")
    assert eis.normalized is True
    assert eis.unit == "Ohm cm^2"
    _assert_consistent(eis)


@pytest.mark.parametrize("area", [0.0, -1.0, np.nan, np.inf])
def test_invalid_area_raises(area):
    with pytest.raises(ValueError, match="area"):
        EISData(_frame(), area=area)


def test_from_csv_with_area(tmp_path):
    path = tmp_path / "spectrum.csv"
    _frame().to_csv(path, index=False)
    raw = EISData.from_csv(str(path))
    eis = EISData.from_csv(str(path), area=4.0)
    assert raw.normalized is False
    assert eis.normalized is True
    assert eis.unit == "Ohm cm^2"
    np.testing.assert_allclose(eis.Real, 4.0 * raw.Real)
    np.testing.assert_allclose(eis.Imag, 4.0 * raw.Imag)
    assert eis.Rp == pytest.approx(4.0 * raw.Rp)


def test_arithmetic_keeps_normalization_flag():
    eis = EISData(_frame(), area=2.0)
    shifted = eis + 1.0
    assert shifted.normalized is True
    assert shifted.normalization_area == 2.0


# --- slicing keeps the object consistent --------------------------------------


def test_limit_data_updates_everything():
    eis = EISData(_frame())
    eis.limit_data(5, 30)
    assert eis.Freq.size == 25
    np.testing.assert_allclose(eis.Freq, FREQ[5:30])
    _assert_consistent(eis)
    assert eis.Rtot == pytest.approx(TRUTH.get_impedance(FREQ[5:6]).real[0])


def test_cut_frequencies_above_updates_everything(capsys):
    eis = EISData(_frame())
    rs_before = eis.Rs
    eis.cut_frequencies_above(1e3)
    assert np.all(eis.Freq < 1e3)
    _assert_consistent(eis)
    # Highest remaining frequency now defines Rs.
    assert eis.Rs > rs_before
    # No print() noise.
    assert capsys.readouterr().out == ""


def test_cut_frequencies_below_updates_everything():
    eis = EISData(_frame())
    rtot_before = eis.Rtot
    eis.cut_frequencies_below(1.0)
    assert np.all(eis.Freq > 1.0)
    _assert_consistent(eis)
    assert eis.Rtot < rtot_before


def test_cut_then_normalize_does_not_warn():
    eis = EISData(_frame())
    eis.cut_frequencies_below(1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        eis.normalize(2.0, "Ohm cm^2")
    _assert_consistent(eis)


def test_cut_removing_everything_raises_and_keeps_data():
    eis = EISData(_frame())
    with pytest.raises(ValueError, match="remove all"):
        eis.cut_frequencies_above(FREQ.min() / 10)
    assert eis.Freq.size == FREQ.size
    _assert_consistent(eis)


# --- input sanity checks -------------------------------------------------------


@pytest.mark.parametrize("column", ["freq", "real", "imag"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_input_raises(column, bad):
    frame = _frame()
    frame.loc[3, column] = bad
    with pytest.raises(ValueError, match="NaN or infinite"):
        EISData(frame)


@pytest.mark.parametrize("bad", [0.0, -5.0])
def test_non_positive_frequency_raises(bad):
    frame = _frame()
    frame.loc[0, "freq"] = bad
    with pytest.raises(ValueError, match="strictly positive"):
        EISData(frame)


def test_duplicate_frequencies_warn_and_keep_order():
    frame = _frame()
    frame.loc[1, "freq"] = frame.loc[0, "freq"]
    with pytest.warns(UserWarning, match="duplicate"):
        eis = EISData(frame)
    np.testing.assert_array_equal(eis.Freq, frame["freq"].to_numpy())


def test_clean_input_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        EISData(_frame())


# --- dead code -----------------------------------------------------------------


def test_simple_rs_placeholder_removed():
    assert not hasattr(EISData, "simpleRs")


# --- wrapper defaults ----------------------------------------------------------


@pytest.mark.parametrize("func", [EISData.quick_fit, data_classes.quick_fit])
def test_quick_fit_defaults_match_impendance_fit(func):
    wrapper = inspect.signature(func).parameters
    core = inspect.signature(global_minimize.impendance_fit).parameters
    for name in ("z_margin_frac", "inductance_bounds", "weighting"):
        assert wrapper[name].default == core[name].default
    assert wrapper["z_margin_frac"].default == 0.0
    assert wrapper["inductance_bounds"].default is False
    assert wrapper["weighting"].default == "modulus"


def test_quick_fit_complex_fit_defaults_unchanged():
    assert inspect.signature(EISData.quick_fit).parameters[
        "complex_fit"
    ].default is True
    assert inspect.signature(data_classes.quick_fit).parameters[
        "complex_fit"
    ].default is False


@pytest.fixture
def captured_fit(monkeypatch):
    calls = []

    def fake_fit(**kwargs):
        calls.append(kwargs)
        return "sentinel"

    monkeypatch.setattr(global_minimize, "impendance_fit", fake_fit)
    return calls


def test_module_quick_fit_passes_through(captured_fit):
    eis = EISData(_frame())
    assert data_classes.quick_fit(eis) == "sentinel"
    kwargs = captured_fit[0]
    assert kwargs["z_margin_frac"] == 0.0
    assert kwargs["inductance_bounds"] is False
    assert kwargs["weighting"] == "modulus"
    data_classes.quick_fit(eis, weighting="unit")
    assert captured_fit[1]["weighting"] == "unit"


def test_method_quick_fit_passes_through(captured_fit):
    eis = EISData(_frame())
    assert eis.quick_fit(TRUTH, weighting="unit") == "sentinel"
    kwargs = captured_fit[0]
    assert kwargs["data"] is eis
    assert kwargs["n_groups"] == 2
    assert kwargs["complex_fit"] is True
    assert kwargs["z_margin_frac"] == 0.0
    assert kwargs["inductance_bounds"] is False
    assert kwargs["weighting"] == "unit"


def test_quick_fit_runs_with_unit_weighting():
    eis = EISData(_frame())
    res = eis.quick_fit(TRUTH, seed=0, weighting="unit")
    assert np.isfinite(res.cost)
