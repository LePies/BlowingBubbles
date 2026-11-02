"""Tests for parameter bounds in the fitting functions.

Covers the bound parameters

- ``resistance_source_bounds``
- ``resistance_branch_bounds``
- ``exponent_bounds``

of :func:`impendance_fit`, :func:`fit_circuit` and :func:`quick_fit`.
"""

import numpy as np
import pandas as pd
import pytest

from blowing_bubbles.classes.circuit_classes import CircuitParameters, FitResults
from blowing_bubbles.classes.data_classes import EISData, fit_circuit, quick_fit
from blowing_bubbles.scripts.global_minimize import impendance_fit


def single_semicircle(
    n_freq: int = 50,
    R_s: float = 10.0,
    R_1: float = 50.0,
    C_1: float = 1e-4,
    seed: int = 42,
    noise_std: float = 0.5,
) -> EISData:
    """Synthetic spectrum of a single semicircle (one R-C branch) with noise."""
    rng = np.random.default_rng(seed)
    freq = np.logspace(-2, 5, n_freq)
    omega = 2 * np.pi * freq

    x = omega * R_1 * C_1
    Z_real = R_s + R_1 / (1 + x**2)
    Z_imag = -R_1 * x / (1 + x**2)

    Z_real += rng.normal(0, noise_std, n_freq)
    Z_imag += rng.normal(0, noise_std, n_freq)

    df = pd.DataFrame({"frequency": freq, "real": Z_real, "imag": Z_imag})
    return EISData(df)


def initial_params_1branch() -> CircuitParameters:
    """Initial parameters for a 1-branch fit."""
    return CircuitParameters(
        resistance_source=10.0,
        inductance=0.0,
        inductance_exponent=1.0,
        resistances=np.array([50.0]),
        capacitances=np.array([1e-4]),
        exponents=np.array([1.0]),
    )


@pytest.fixture(scope="module")
def eis() -> EISData:
    return single_semicircle()


@pytest.fixture
def initial_params() -> CircuitParameters:
    return initial_params_1branch()


def _assert_rs_within(result: FitResults, lo: float, hi: float) -> None:
    assert isinstance(result, FitResults)
    assert lo <= result.parameters.resistance_source <= hi


# --- impendance_fit -----------------------------------------------------------


def test_impedance_fit_default_no_bounds(eis, initial_params):
    """No bounds specified (backward compatibility)."""
    result = impendance_fit(
        eis,
        n_groups=1,
        complex_fit=True,
        initial_parameters=initial_params,
        n_restarts=1,
    )
    assert isinstance(result, FitResults)
    assert result.parameters is not None


def test_impedance_fit_default_rs_strictly_positive(eis, initial_params):
    """Default lower bound keeps R_s > 0, so the R_s relative error is finite."""
    result = impendance_fit(
        eis,
        n_groups=1,
        complex_fit=True,
        initial_parameters=initial_params,
        n_restarts=1,
    )
    rs = result.parameters.resistance_source
    assert rs > 0.0
    assert np.isfinite(abs(rs - 10.0) / rs)


def test_impedance_fit_zero_rs_bound_keeps_floor(eis, initial_params):
    """An explicit non-positive R_s_min must not re-enable R_s = 0."""
    result = impendance_fit(
        eis,
        n_groups=1,
        complex_fit=True,
        initial_parameters=initial_params,
        n_restarts=1,
        resistance_source_bounds=(0.0, np.inf),
    )
    assert result.parameters.resistance_source > 0.0


def test_impedance_fit_resistance_source_bounds(eis, initial_params):
    result = impendance_fit(
        eis,
        n_groups=1,
        complex_fit=True,
        initial_parameters=initial_params,
        n_restarts=1,
        resistance_source_bounds=(8.0, 12.0),
    )
    _assert_rs_within(result, 8.0, 12.0)


def test_impedance_fit_resistance_branch_bounds(eis, initial_params):
    result = impendance_fit(
        eis,
        n_groups=1,
        complex_fit=True,
        initial_parameters=initial_params,
        n_restarts=1,
        resistance_branch_bounds=(40.0, 60.0),
    )
    assert isinstance(result, FitResults)
    for R in result.parameters.resistances:
        assert 40.0 <= R <= 60.0


def test_impedance_fit_exponent_bounds(eis, initial_params):
    """``exponent_bounds`` with ``complex_fit=True``."""
    result = impendance_fit(
        eis,
        n_groups=1,
        complex_fit=True,
        initial_parameters=initial_params,
        n_restarts=1,
        exponent_bounds=(0.8, 1.0),
    )
    assert isinstance(result, FitResults)
    for exp in result.parameters.exponents:
        assert 0.8 <= exp <= 1.0


def test_impedance_fit_all_bounds(eis, initial_params):
    result = impendance_fit(
        eis,
        n_groups=1,
        complex_fit=True,
        initial_parameters=initial_params,
        n_restarts=1,
        resistance_source_bounds=(8.0, 12.0),
        resistance_branch_bounds=(40.0, 60.0),
        exponent_bounds=(0.8, 1.0),
    )
    assert isinstance(result, FitResults)


def test_exponent_bounds_ignored_when_complex_fit_false(eis, initial_params):
    """With ``complex_fit=False`` exponents are fixed to 1 and the bounds unused."""
    result = impendance_fit(
        eis,
        n_groups=1,
        complex_fit=False,
        initial_parameters=initial_params,
        n_restarts=3,
        exponent_bounds=(0.3, 0.7),  # would exclude 1.0 if it were applied
    )
    assert isinstance(result, FitResults)
    np.testing.assert_array_equal(result.parameters.exponents, 1.0)


# --- fit_circuit --------------------------------------------------------------


def test_fit_circuit_default_no_bounds(eis):
    """No bounds specified (backward compatibility)."""
    result = fit_circuit(eis, n_restarts=1, rc_groups=1, verbose=False)
    assert isinstance(result, FitResults)


def test_fit_circuit_all_bounds(eis):
    result = fit_circuit(
        eis,
        n_restarts=1,
        rc_groups=1,
        verbose=False,
        resistance_source_bounds=(8.0, 12.0),
        resistance_branch_bounds=(40.0, 60.0),
        exponent_bounds=(0.8, 1.0),
    )
    _assert_rs_within(result, 8.0, 12.0)


# --- quick_fit ----------------------------------------------------------------


def test_standalone_quick_fit_default(eis, initial_params):
    result = quick_fit(
        eis,
        n_groups=1,
        complex_fit=True,
        initial_parameters=initial_params,
        n_restarts=1,
    )
    assert isinstance(result, FitResults)


def test_standalone_quick_fit_all_bounds(eis, initial_params):
    result = quick_fit(
        eis,
        n_groups=1,
        complex_fit=True,
        initial_parameters=initial_params,
        n_restarts=1,
        resistance_source_bounds=(8.0, 12.0),
        resistance_branch_bounds=(40.0, 60.0),
        exponent_bounds=(0.8, 1.0),
    )
    _assert_rs_within(result, 8.0, 12.0)


def test_eisdata_quick_fit_method(eis, initial_params):
    """``EISData.quick_fit`` forwards the bounds."""
    result = eis.quick_fit(
        initial_params, complex_fit=True, resistance_source_bounds=(8.0, 12.0)
    )
    assert isinstance(result, FitResults)
    assert result.parameters.resistance_source >= 8.0


# --- different data configurations --------------------------------------------


@pytest.mark.parametrize("noise_std", [0.1, 0.5, 1.0])
def test_bounds_with_different_noise_levels(noise_std):
    eis = single_semicircle(noise_std=noise_std)
    result = impendance_fit(
        eis,
        n_groups=1,
        complex_fit=True,
        initial_parameters=initial_params_1branch(),
        n_restarts=1,
        resistance_source_bounds=(8.0, 12.0),
    )
    _assert_rs_within(result, 8.0, 12.0)


@pytest.mark.parametrize("seed", [0, 42, 100])
def test_bounds_with_different_seeds(seed):
    eis = single_semicircle(seed=seed)
    result = impendance_fit(
        eis,
        n_groups=1,
        complex_fit=True,
        initial_parameters=initial_params_1branch(),
        n_restarts=1,
        resistance_source_bounds=(8.0, 12.0),
        resistance_branch_bounds=(40.0, 60.0),
    )
    assert isinstance(result, FitResults)
