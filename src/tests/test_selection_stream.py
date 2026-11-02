"""Tests for model selection in iterative branch building (``fit_circuit``)."""

import numpy as np
import pandas as pd
import pytest

from blowing_bubbles.classes.circuit_classes import CircuitParameters
from blowing_bubbles.classes.data_classes import EISData
from blowing_bubbles.scripts.fit_circuit import fit_circuit, information_criterion

FREQ = np.logspace(-2, 5, 60)
TWO_ARC = CircuitParameters(
    10.0, 0.0, 1.0,
    np.array([20.0, 50.0]), np.array([1e-5, 1e-2]), np.array([0.9, 0.85]),
)
# Three arcs with tau = 1e-6, 3e-3 and 6 s (3+ decades apart) inside the
# measured window 1e-2 .. 1e6 Hz.
THREE_ARC = CircuitParameters(
    5.0, 0.0, 1.0,
    np.array([10.0, 30.0, 60.0]),
    np.array([1e-7, 1e-4, 1e-1]),
    np.array([0.95, 0.9, 0.9]),
)
FREQ_3 = np.logspace(-2, 6, 80)


def _data(params, freq, noise=0.0, seed=0):
    z = params.get_impedance(freq)
    if noise:
        rng = np.random.default_rng(seed)
        z = z * (1 + noise * (rng.normal(size=freq.size)
                              + 1j * rng.normal(size=freq.size)))
    return EISData(pd.DataFrame({"freq": freq, "real": z.real, "imag": z.imag}))


@pytest.fixture(scope="module")
def clean():
    return _data(TWO_ARC, FREQ)


def test_bic_rejects_duplicate_branch_on_clean_data(clean):
    res = fit_circuit(clean, 4, rc_groups=4, seed=0, selection="bic")
    assert res.n_groups == 2
    np.testing.assert_allclose(np.sort(res.parameters.taus), TWO_ARC.taus, rtol=1e-4)
    # The third branch was tried and rejected; expansion stopped there.
    assert [h["accepted"] for h in res.history] == [True, False]
    assert res.history[1]["n_groups"] == 3


@pytest.mark.parametrize("noise_seed", [0, 1])
def test_bic_finds_three_arcs_in_noisy_data(noise_seed):
    data = _data(THREE_ARC, FREQ_3, noise=0.005, seed=noise_seed)
    res = fit_circuit(
        data, 4, rc_groups=5, initial_peaks=1, seed=noise_seed, selection="bic"
    )
    assert res.n_groups == 3
    np.testing.assert_allclose(np.sort(res.parameters.taus), THREE_ARC.taus, rtol=0.2)
    bics = [h["bic"] for h in res.history]
    accepted = [h for h in res.history if h["accepted"]]
    assert accepted[-1]["n_groups"] == 3
    # BIC decreased for every accepted step and rose for the rejected 4th branch.
    assert all(b1 < b0 for b0, b1 in zip(bics[:-2], bics[1:-1], strict=True))
    assert bics[-1] > bics[-2]


def test_selection_none_keeps_old_behaviour():
    # Slightly noisy data: a redundant third branch always lowers the cost a
    # little. (On clean data the two-branch fit is exact to round-off, so a
    # third branch cannot lower the cost and this behaviour is not exercised.)
    noisy = _data(TWO_ARC, FREQ, noise=0.002, seed=0)
    res = fit_circuit(noisy, 4, rc_groups=3, seed=0)
    # Without a criterion any cost decrease is accepted, even a redundant branch.
    assert res.n_groups == 3
    assert all(h["accepted"] for h in res.history)
    assert all(set(h) == {"n_groups", "cost", "accepted"} for h in res.history)


def test_history_entries(clean):
    res = fit_circuit(clean, 4, rc_groups=4, seed=0, selection="aic")
    assert len(res.history) >= 1
    for h in res.history:
        assert set(h) == {"n_groups", "cost", "accepted", "aic"}
    n = [h["n_groups"] for h in res.history]
    assert n == list(range(n[0], n[0] + len(n)))
    assert res.history[-1]["n_groups"] <= 4
    final = [h for h in res.history if h["accepted"]][-1]
    assert final["n_groups"] == res.n_groups
    assert final["cost"] == pytest.approx(res.cost)
    assert final["aic"] == pytest.approx(
        information_criterion(res, clean, "modulus", kind="aic")
    )


def test_min_improvement_stops_expansion(clean):
    # selection=None alone adds a third branch here (round-off decrease, see
    # test_selection_none_keeps_old_behaviour); a relative floor stops it.
    res = fit_circuit(clean, 4, rc_groups=4, seed=0, min_improvement=0.999999)
    assert res.n_groups == 2


@pytest.mark.parametrize(
    "kwargs",
    [{"selection": "BIC"}, {"selection": "lasso"}, {"min_improvement": -0.1}],
)
def test_invalid_options_raise(clean, kwargs):
    with pytest.raises(ValueError):
        fit_circuit(clean, 2, rc_groups=3, seed=0, **kwargs)


def test_kk_stops_early_on_clean_data(clean):
    res = fit_circuit(clean, 4, rc_groups=4, seed=0, selection="kk")
    assert res.n_groups == 2
    # Within the Lin-KK floor already, so no further branch was even tried.
    assert len(res.history) == 1
    assert res.history[0]["kk_ratio"] < 3.0


def test_kk_keeps_adding_while_suspect():
    data = _data(THREE_ARC, FREQ_3, noise=0.005, seed=0)
    res = fit_circuit(data, 4, rc_groups=5, initial_peaks=1, seed=0, selection="kk")
    ratios = [h["kk_ratio"] for h in res.history]
    assert ratios[0] > 3.0  # one arc cannot describe three
    assert res.n_groups >= 3
    assert [h for h in res.history if h["accepted"]][-1]["kk_ratio"] <= 3.0
