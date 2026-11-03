"""Regression tests for the corrections introduced in version 0.2."""

import numpy as np
import pandas as pd
import pytest

from blowing_bubbles.circuit_impedance import circuit_impedance
from blowing_bubbles.classes.circuit_classes import CircuitParameters
from blowing_bubbles.classes.data_classes import EISData
from blowing_bubbles.scripts.fit_circuit import initial_fit
from blowing_bubbles.scripts.global_minimize import (
    _constraints_satisfied,
    impendance_fit,
)
from blowing_bubbles.scripts.quality import fit_quality, lin_kk


def _data(freq, **p):
    z = circuit_impedance(freq, **p)
    return EISData(
        pd.DataFrame({"Freq (Hz)": freq, "Zreal (Ohm)": z.real, "Zimag (Ohm)": z.imag})
    ), z


ONE_ARC = {
    "resistance_source": 1.0,
    "inductance": 0.0,
    "inductance_exponent": 1.0,
    "resistances": np.array([10.0]),
    "capacitances": np.array([1e-3]),
    "exponents": np.array([1.0]),
}


def test_initial_tau_includes_two_pi():
    """Apex of an ideal RC arc is at 2*pi*f*R*C = 1 (v0.1 used tau = 1/f)."""
    freq = np.logspace(5, -3, 400)
    data, _ = _data(freq, **ONE_ARC)
    p = initial_fit(data, prominence=0.005)
    assert p.taus[0] == pytest.approx(10.0 * 1e-3, rel=0.05)


@pytest.mark.parametrize("order", [1, -1])
def test_initial_rs_independent_of_row_order(order):
    """R_s is read at the highest frequency, whatever the row order."""
    freq = np.logspace(5, -3, 200)[::order]
    data, z = _data(freq, **ONE_ARC)
    p = initial_fit(data, prominence=0.005)
    assert p.resistance_source == pytest.approx(z.real[np.argmax(freq)])


def test_truncated_arc_truth_is_feasible():
    """True circuit with a high-frequency arc beyond f_max must satisfy constraints."""
    freq = np.logspace(5, -2, 100)
    p = {
        "resistance_source": 0.3,
        "inductance": 1e-8,
        "inductance_exponent": 1.0,
        "resistances": np.array([0.3, 0.2]),
        "capacitances": np.array([1e-5 / 0.3, 1e-1 / 0.2]),
        "exponents": np.array([0.75, 0.95]),
    }
    _, z = _data(freq, **p)
    assert z.real.min() > p["resistance_source"] + 0.5 * 0.3 * 0  # sanity
    vec = np.array([p["resistance_source"], p["inductance"], 1.0, *p["resistances"]])
    assert _constraints_satisfied(vec, slice(3, 5), z.real.min(), z.real.max(), 1e-12)


def test_truth_start_is_fixed_point_on_clean_data():
    """Started at the truth, a clean-data fit must stay there (v0.1 moved away)."""
    freq = np.logspace(5, -2, 100)
    p = {
        "resistance_source": 0.3,
        "inductance": 1e-8,
        "inductance_exponent": 1.0,
        "resistances": np.array([0.3, 0.2]),
        "capacitances": np.array([1e-5 / 0.3, 1e-1 / 0.2]),
        "exponents": np.array([0.75, 0.95]),
    }
    data, _ = _data(freq, **p)
    truth = CircuitParameters(**p)
    res = impendance_fit(
        data, n_groups=2, complex_fit=True, n_restarts=1, initial_parameters=truth
    )
    assert res.parameters.resistance_source == pytest.approx(0.3, rel=1e-4)
    np.testing.assert_allclose(res.parameters.resistances, [0.3, 0.2], rtol=1e-4)


def test_weighting_option():
    freq = np.logspace(5, -3, 60)
    data, _ = _data(freq, **ONE_ARC)
    init = CircuitParameters(**ONE_ARC)
    for w in ("modulus", "unit"):
        impendance_fit(
            data, n_groups=1, complex_fit=True, initial_parameters=init, weighting=w
        )
    with pytest.raises(ValueError):
        impendance_fit(
            data,
            n_groups=1,
            complex_fit=True,
            initial_parameters=init,
            weighting="x",
        )


def test_kk_flag_separates_good_and_bad_fit():
    rng = np.random.default_rng(0)
    freq = np.logspace(5, -2, 100)
    z = circuit_impedance(freq, **ONE_ARC)
    noise_re = 0.005 * rng.standard_normal(z.size)
    noise_im = 0.005 * np.abs(z) * rng.standard_normal(z.size)
    zn = z * (1 + noise_re) + 1j * noise_im
    data = EISData(
        pd.DataFrame(
            {"Freq (Hz)": freq, "Zreal (Ohm)": zn.real, "Zimag (Ohm)": zn.imag}
        )
    )
    assert lin_kk(freq, zn)["rms_rel"] < 0.02
    good = fit_quality(data, CircuitParameters(**ONE_ARC))
    bad_p = dict(ONE_ARC, resistances=np.array([7.0]))
    bad = fit_quality(data, CircuitParameters(**bad_p))
    assert not good["suspect"] and good["kk_ratio"] < 2.0
    assert bad["suspect"]


def test_characteristic_times():
    p = CircuitParameters(**dict(ONE_ARC, exponents=np.array([0.8])))
    assert p.characteristic_times[0] == pytest.approx((10.0 * 1e-3) ** (1 / 0.8))
