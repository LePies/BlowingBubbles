"""Regression tests: inductor-exponent bounds, inductance start value, tau window."""

import numpy as np
import pandas as pd
import pytest

from blowing_bubbles.classes.circuit_classes import CircuitParameters
from blowing_bubbles.classes.data_classes import EISData
from blowing_bubbles.scripts.global_minimize import (
    estimate_inductance_from_imag,
    impendance_fit,
    tau_bounds_from_frequencies,
)

FREQ = np.logspace(-2, 5, 70)


def _eis(params, freq=FREQ):
    z = params.get_impedance(freq)
    return EISData(pd.DataFrame({"freq": freq, "real": z.real, "imag": z.imag}))


def test_inductance_estimate_zero_without_inductive_points():
    # Two capacitive arcs, no inductance: Im Z < 0 everywhere and tends to 0.
    p = CircuitParameters(10.0, 0.0, 1.0, np.array([20.0, 50.0]),
                          np.array([1e-5, 1e-2]), np.array([0.9, 0.85]))
    z = p.get_impedance(FREQ)
    assert estimate_inductance_from_imag(FREQ, z.imag) == 0.0


def test_inductance_estimate_positive_from_inductive_tail():
    p = CircuitParameters(1.0, 1e-6, 1.0, np.array([2.0]), np.array([1e-3]), np.array([1.0]))
    freq = np.logspace(-1, 6, 80)
    z = p.get_impedance(freq)
    L = estimate_inductance_from_imag(freq, z.imag)
    assert 0.5e-6 < L <= 1.05e-6


def _low_nl_circuit():
    # Inductor with a low exponent, as found in collapsed EK3 fits.
    return CircuitParameters(0.3, 1e-3, 0.4, np.array([0.3, 0.4]),
                             np.array([1e-4 / 0.3, 1e-1 / 0.4]), np.array([0.9, 0.9]))


def test_default_inductor_exponent_bounds():
    data = _eis(_low_nl_circuit())
    res = impendance_fit(data, n_groups=2, complex_fit=True, n_restarts=2, seed=0,
                         initial_parameters=_low_nl_circuit())
    assert 0.7 - 1e-12 <= res.parameters.inductance_exponent <= 1.0


def test_inductor_exponent_bounds_can_be_widened():
    truth = _low_nl_circuit()
    res = impendance_fit(_eis(truth), n_groups=2, complex_fit=True, n_restarts=1, seed=0,
                         initial_parameters=truth, inductance_exponent_bounds=(0.2, 1.0))
    assert res.parameters.inductance_exponent == pytest.approx(0.4, rel=1e-4)


def test_exponent_bounds_still_apply_to_inductor():
    truth = _low_nl_circuit()
    res = impendance_fit(_eis(truth), n_groups=2, complex_fit=True, n_restarts=1, seed=0,
                         initial_parameters=truth, exponent_bounds=(0.3, 1.0))
    assert res.parameters.inductance_exponent >= 0.3 - 1e-12
    assert res.parameters.inductance_exponent < 0.7


def test_tau_bounds_from_frequencies_has_two_pi():
    lo, hi = tau_bounds_from_frequencies(np.array([0.01, 1.0, 1e5]))
    assert lo == pytest.approx(1 / (2 * np.pi * 1e5))
    assert hi == pytest.approx(1 / (2 * np.pi * 0.01))
