"""Regression tests for the v0.3.0 correctness fixes."""

import copy
import pickle

import numpy as np
import pandas as pd
import pytest

from blowing_bubbles.classes.circuit_classes import CircuitParameters
from blowing_bubbles.classes.data_classes import EISData
from blowing_bubbles.scripts.fit_circuit import fit_circuit
from blowing_bubbles.scripts.global_minimize import impendance_fit, initial_fit

FREQ = np.logspace(-2, 5, 60)
TRUTH = CircuitParameters(
    10.0, 0.0, 1.0,
    np.array([20.0, 50.0]), np.array([1e-5, 1e-2]), np.array([0.9, 0.85]),
)


def _frame(names, params=TRUTH, freq=FREQ):
    z = params.get_impedance(freq)
    f_key, r_key, i_key = names
    return pd.DataFrame({f_key: freq, r_key: z.real, i_key: z.imag})


@pytest.mark.parametrize(
    "names",
    [
        ("Freq", "Real", "Imaginary"),
        ("frequency", "Z_real", "Z_imag"),
        ("freq/Hz", "Re(Z)/Ohm", "Im(Z)/Ohm"),
        ("f", "r", "i"),
    ],
)
def test_find_keys_picks_correct_columns(names):
    eis = EISData(_frame(names))
    assert eis.keys == (names[1], names[2], names[0])


def test_find_keys_ambiguous_raises_and_explicit_columns_work():
    frame = _frame(("freq", "real", "imag"))
    frame["real_fit"] = frame["real"]
    with pytest.raises(ValueError, match="Ambiguous"):
        EISData(frame)
    eis = EISData(frame, columns=("real", "imag", "freq"))
    assert eis.keys == ("real", "imag", "freq")


@pytest.mark.parametrize("descending", [False, True])
def test_rs_rtot_independent_of_row_order(descending):
    frame = _frame(("freq", "real", "imag"))
    if descending:
        frame = frame.iloc[::-1].reset_index(drop=True)
    eis = EISData(frame)
    z = TRUTH.get_impedance(FREQ)
    assert eis.Rtot == pytest.approx(z.real[0])
    assert eis.Rs == pytest.approx(z.real[-1])


def test_rs_interpolates_inductive_crossing():
    p = TRUTH.copy()
    p.inductance = 1e-6
    eis = EISData(_frame(("freq", "real", "imag"), params=p))
    assert 10.0 <= eis.Rs < 11.0


def test_arithmetic_uses_complex_impedance():
    eis = EISData(_frame(("freq", "real", "imag")))
    z = eis.Z
    np.testing.assert_allclose((eis + eis).Z, 2 * z)
    np.testing.assert_allclose((eis - eis).Z, 0 * z, atol=1e-12)
    np.testing.assert_allclose((eis * eis).Z, z * z)
    np.testing.assert_allclose((eis / eis).Z, np.ones_like(z))
    np.testing.assert_allclose((eis * 2.0).Freq, FREQ)
    other = EISData(_frame(("freq", "real", "imag"), freq=FREQ * 2))
    with pytest.raises(ValueError):
        eis + other


def test_from_array_roundtrip():
    back = CircuitParameters.from_array(TRUTH.to_array(), len(TRUTH))
    np.testing.assert_allclose(back.to_array(), TRUTH.to_array())


def test_copy_and_pickle():
    for clone in (copy.deepcopy(TRUTH), pickle.loads(pickle.dumps(TRUTH))):
        np.testing.assert_allclose(clone.to_array(), TRUTH.to_array())
    assert set(TRUTH.as_dict()) >= {"resistance_source", "resistances", "taus"}


def test_complex_fit_false_converges_on_ideal_circuit():
    ideal = CircuitParameters(
        10.0, 0.0, 1.0, np.array([20.0, 50.0]), np.array([1e-5, 1e-2]), np.ones(2)
    )
    eis = EISData(_frame(("freq", "real", "imag"), params=ideal))
    start = ideal.copy()
    start.resistances = start.resistances * 1.3
    res = impendance_fit(
        eis, n_groups=2, complex_fit=False, initial_parameters=start,
        n_restarts=3, seed=0,
    )
    np.testing.assert_allclose(res.parameters.resistances, [20.0, 50.0], rtol=1e-4)
    np.testing.assert_allclose(res.parameters.exponents, [1.0, 1.0])


def test_impendance_fit_does_not_mutate_initial_parameters():
    start = CircuitParameters(
        10.0, 0.0, 1.0, np.array([50.0, 20.0]), np.array([1e-2, 1e-5]), np.ones(2)
    )
    before = start.resistances.copy()
    # Bypass the constructor's sort to get an unsorted instance.
    start.resistances, start.capacitances = before, np.array([1e-2, 1e-5])
    start.taus = start.resistances * start.capacitances
    eis = EISData(_frame(("freq", "real", "imag")))
    impendance_fit(eis, n_groups=2, complex_fit=True, initial_parameters=start)
    np.testing.assert_array_equal(start.resistances, before)


def test_zero_peaks_falls_back_to_one_branch():
    # Arc apex below the lowest measured frequency: -Im(Z) is monotonic.
    p = CircuitParameters(
        10.0, 0.0, 1.0, np.array([50.0]), np.array([10.0]), np.ones(1)
    )
    eis = EISData(_frame(("freq", "real", "imag"), params=p))
    guess = initial_fit(eis)
    assert len(guess) == 1
    assert np.all(np.isfinite(guess.capacitances))
    res = fit_circuit(eis, n_restarts=2, rc_groups=1, seed=0)
    assert len(res.parameters) == 1


def test_expansion_never_returns_worse_fit():
    eis = EISData(_frame(("freq", "real", "imag")))
    one = fit_circuit(eis, n_restarts=4, rc_groups=2, initial_peaks=2, seed=0)
    more = fit_circuit(eis, n_restarts=4, rc_groups=4, initial_peaks=2, seed=0)
    assert more.cost <= one.cost


def test_plot_full_fit_returns_axes():
    import matplotlib

    matplotlib.use("Agg")
    ax = TRUTH.plot_full_fit(unit="Ohm")
    assert ax is not None and len(ax.lines) > 0
