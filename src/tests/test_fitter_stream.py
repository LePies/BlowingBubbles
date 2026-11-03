"""Tests for fitter bounds, restart perturbations and parameter uncertainties."""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import OptimizeResult

from blowing_bubbles.classes.circuit_classes import CircuitParameters, FitResults
from blowing_bubbles.classes.data_classes import EISData
from blowing_bubbles.scripts import global_minimize as gm
from blowing_bubbles.scripts.global_minimize import (
    _perturb_init,
    _positive_floor,
    impedance_fit,
    impendance_fit,
)

FREQ = np.logspace(-2, 4, 40)
IDEAL = CircuitParameters(
    10.0, 0.0, 1.0, np.array([20.0, 50.0]), np.array([1e-5, 1e-2]), np.ones(2)
)


def _eis(z, freq=FREQ):
    return EISData(pd.DataFrame({"freq": freq, "real": z.real, "imag": z.imag}))


# --- (a) strictly positive R / tau lower bounds -------------------------------


def test_positive_floor_composes_with_user_bounds():
    assert _positive_floor(1e-9) == 1e-9
    assert _positive_floor(1e-9, 0.0, None) == 1e-9
    assert _positive_floor(1e-9, -1.0, None) == 1e-9
    # A positive user bound wins, whether above or below the floor.
    assert _positive_floor(1e-9, 5.0, 10.0) == 5.0
    assert _positive_floor(1e-9, 1e-12, None) == 1e-12
    # Never above a tiny user upper bound.
    assert 0 < _positive_floor(1e-9, 0.0, 1e-10) < 1e-10


def test_redundant_branch_keeps_positive_r_and_finite_c():
    # One arc fitted with three branches: surplus branches collapse towards 0.
    p = CircuitParameters(
        10.0, 0.0, 1.0, np.array([50.0]), np.array([1e-4]), np.ones(1)
    )
    eis = _eis(p.get_impedance(FREQ))
    for bounds in (None, (0.0, 1e3)):
        res = impedance_fit(
            eis,
            n_groups=3,
            complex_fit=False,
            n_restarts=4,
            seed=0,
            resistance_branch_bounds=bounds,
        )
        assert np.all(res.parameters.resistances > 0)
        assert np.all(np.isfinite(res.parameters.capacitances))
        assert np.all(res.parameters.capacitances > 0)
        assert np.all(res.parameters.taus > 0)


# --- (b), (c) restart perturbations -------------------------------------------


def _layout(complex_fit, n):
    head = 3 if complex_fit else 2
    n_params = head + (2 + complex_fit) * n
    lower = np.zeros(n_params)
    upper = np.full(n_params, np.inf)
    return head, n_params, lower, upper


def test_perturb_preserves_negative_inductance():
    n = 2
    _head, n_params, lower, upper = _layout(False, n)
    lower[1], upper[1] = -1e-5, 0.0
    base = np.ones(n_params)
    base[1] = -2e-6
    rng = np.random.default_rng(0)
    inits = [
        _perturb_init(
            base,
            rng,
            0.5,
            lower,
            upper,
            complex_fit=False,
            n_groups=n,
            local_exponents=False,
        )
        for _ in range(50)
    ]
    L = np.array([x[1] for x in inits])
    assert np.all(L < 0)
    assert np.all(L >= -1e-5)
    assert np.unique(L).size == L.size


def test_negative_inductance_fit_through_impedance_fit():
    # Strong negative slope of Im(Z): the slope estimate gives L bounds [l_lo, 0].
    p = IDEAL.copy()
    p.inductance = -1e-3
    eis = _eis(p.get_impedance(FREQ))
    start = IDEAL.copy()
    start.inductance = -5e-4
    res = impedance_fit(
        eis,
        n_groups=2,
        complex_fit=False,
        initial_parameters=start,
        n_restarts=4,
        seed=0,
        inductance_bounds=True,
    )
    assert res.parameters.inductance == pytest.approx(-1e-3, rel=1e-4)


@pytest.mark.parametrize("local", [False, True])
def test_perturb_exponents_respect_user_bounds(local):
    n = 3
    head, n_params, lower, upper = _layout(True, n)
    e_slice = slice(head + 2 * n, head + 3 * n)
    lower[2], upper[2] = 0.6, 0.7
    lower[e_slice], upper[e_slice] = 0.6, 0.7
    base = np.ones(n_params)
    base[2] = 0.65
    base[e_slice] = 0.65

    def draw(seed):
        rng = np.random.default_rng(seed)
        return np.array(
            [
                _perturb_init(
                    base,
                    rng,
                    0.5,
                    lower,
                    upper,
                    complex_fit=True,
                    n_groups=n,
                    local_exponents=local,
                )
                for _ in range(40)
            ]
        )

    inits = draw(3)
    exps = np.concatenate([inits[:, 2], inits[:, e_slice].ravel()])
    assert np.all((exps >= 0.6) & (exps <= 0.7))
    if not local:
        # Uniform over the user interval, not the default (0.2, 1.0).
        assert np.ptp(exps) > 0.05
    np.testing.assert_array_equal(inits, draw(3))


def test_restarts_with_exponent_bounds_stay_inside():
    p = CircuitParameters(
        10.0,
        0.0,
        1.0,
        np.array([20.0, 50.0]),
        np.array([1e-5, 1e-2]),
        np.array([0.9, 0.85]),
    )
    eis = _eis(p.get_impedance(FREQ))
    res = impedance_fit(
        eis,
        n_groups=2,
        complex_fit=True,
        initial_parameters=p,
        n_restarts=4,
        seed=0,
        exponent_bounds=(0.8, 0.95),
    )
    e = np.append(res.parameters.exponents, res.parameters.inductance_exponent)
    assert np.all((e >= 0.8) & (e <= 0.95))


# --- (d) alias -----------------------------------------------------------------


def test_impendance_fit_is_alias():
    assert impendance_fit is impedance_fit
    assert gm.impendance_fit is gm.impedance_fit


# --- (e) parameter uncertainties ------------------------------------------------


def test_standard_errors_match_monte_carlo():
    sigma = 0.01
    z0 = IDEAL.get_impedance(FREQ)
    rng = np.random.default_rng(0)
    est, se = [], []
    for k in range(30):
        noise = rng.normal(size=z0.size) + 1j * rng.normal(size=z0.size)
        eis = _eis(z0 + sigma * np.abs(z0) * noise)
        res = impedance_fit(
            eis,
            n_groups=2,
            complex_fit=False,
            initial_parameters=IDEAL,
            n_restarts=1,
            seed=k,
        )
        est.append(res.parameters.to_array())
        se.append(res.standard_errors)
    est, se = np.array(est), np.array(se)
    n = 2
    for i in [3, 4, 3 + n, 4 + n]:  # R_1, R_2, C_1, C_2
        ratio = se[:, i].mean() / est[:, i].std(ddof=1)
        assert 1 / 1.5 < ratio < 1.5, (i, ratio)


def test_covariance_shape_symmetry_and_fixed_parameters():
    z0 = IDEAL.get_impedance(FREQ)
    rng = np.random.default_rng(1)
    noise = rng.normal(size=z0.size) + 1j * rng.normal(size=z0.size)
    eis = _eis(z0 + 0.01 * np.abs(z0) * noise)
    res = impedance_fit(
        eis, n_groups=2, complex_fit=False, initial_parameters=IDEAL, seed=0
    )
    cov = res.covariance
    assert cov.shape == (9, 9)
    np.testing.assert_allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() >= -1e-12 * np.abs(cov).max()
    se = res.standard_errors
    np.testing.assert_allclose(se, np.sqrt(np.diag(cov)))
    # n_L and e_i are fixed for complex_fit=False.
    assert se[2] == 0 and np.all(se[-2:] == 0)
    assert np.all(se[3:7] > 0)
    corr = res.correlation
    np.testing.assert_allclose(np.diag(corr)[3:7], 1.0)
    assert np.all(np.abs(corr) <= 1 + 1e-12)
    unc = res.parameter_uncertainties()
    np.testing.assert_allclose(unc["resistances"], se[3:5])
    assert len(res.parameter_names) == 9


def test_complex_fit_covariance_is_symmetric_psd():
    p = CircuitParameters(
        10.0,
        1e-6,
        0.9,
        np.array([20.0, 50.0]),
        np.array([1e-5, 1e-2]),
        np.array([0.9, 0.85]),
    )
    z0 = p.get_impedance(FREQ)
    rng = np.random.default_rng(2)
    noise = rng.normal(size=z0.size) + 1j * rng.normal(size=z0.size)
    eis = _eis(z0 + 0.01 * np.abs(z0) * noise)
    res = impedance_fit(eis, n_groups=2, complex_fit=True, initial_parameters=p, seed=0)
    cov = res.covariance
    np.testing.assert_allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() >= -1e-9 * np.abs(cov).max()
    assert np.all(np.isfinite(res.standard_errors))


def test_fit_results_permutes_covariance_to_tau_order():
    # Branches given in decreasing tau: FitResults must sort them and the
    # covariance with the same permutation.
    n = 2
    x = np.array([1.0, 0.0, 1.0, 2.0, 3.0, 1.0, 1e-3, 0.9, 0.8])
    #             Rs   L    nL   R1   R2   C1   C2    e1   e2  (tau1 > tau2)
    res_ls = OptimizeResult(
        x=x,
        fun=np.zeros(4),
        cost=0.0,
        success=True,
        status=1,
        message="",
        nfev=1,
        njev=1,
        jac=None,
        grad=None,
        optimality=None,
        active_mask=None,
    )
    cov = np.diag(np.arange(1.0, 10.0))
    fr = FitResults(res_ls, n, True, covariance=cov)
    np.testing.assert_allclose(fr.parameters.resistances, [3.0, 2.0])
    np.testing.assert_allclose(fr.standard_errors**2, [1, 2, 3, 5, 4, 7, 6, 9, 8])
    # Backward compatible: no covariance -> attributes are None.
    assert FitResults(res_ls, n, True).covariance is None
