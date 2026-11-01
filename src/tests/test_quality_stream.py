"""Tests for the Lin-KK refactor, its input validation and the scripts exports."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from blowing_bubbles.circuit_impedance import circuit_impedance
from blowing_bubbles.scripts.quality import _relative_rms, lin_kk


def _lin_kk_reference(freq, z, c_threshold=0.85, max_elements=None):
    """Verbatim copy of the v0.2 ``lin_kk`` (before the per-M precomputation)."""
    f = np.asarray(freq, dtype=float)
    z = np.asarray(z, dtype=complex)
    omega = 2.0 * np.pi * f
    w = 1.0 / np.maximum(np.abs(z), np.finfo(float).tiny)
    n = f.size
    decades = np.log10(omega.max() / omega.min())
    m_min = max(3, int(np.ceil(2.0 * decades)))
    m_max = max_elements if max_elements is not None else max(m_min, n // 2)
    m_min = min(m_min, m_max)
    best = None
    for m in range(m_min, m_max + 1):
        tau = np.logspace(np.log10(1.0 / omega.max()), np.log10(1.0 / omega.min()), m)
        basis = 1.0 / (1.0 + 1j * omega[:, None] * tau[None, :])
        A = np.hstack([np.ones((n, 1)), (1j * omega)[:, None], basis])
        Aw = A * w[:, None]
        Ar = np.vstack([Aw.real, Aw.imag])
        b = np.concatenate([(z * w).real, (z * w).imag])
        coef, *_ = np.linalg.lstsq(Ar, b, rcond=None)
        rk = coef[2:]
        pos = np.sum(np.abs(rk[rk >= 0]))
        neg = np.sum(np.abs(rk[rk < 0]))
        mu = 1.0 - neg / pos if pos > 0 else 0.0
        z_fit = A @ coef
        best = {
            "M": m,
            "mu": float(mu),
            "rms_rel": _relative_rms(z_fit, z),
            "z_fit": z_fit,
        }
        if mu <= c_threshold:
            break
    return best


def _spectrum(seed=0, n=80, noise=0.003):
    """Two-arc spectrum with inductance and a little multiplicative noise."""
    rng = np.random.default_rng(seed)
    freq = np.logspace(5, -2, n)
    z = circuit_impedance(
        freq,
        resistance_source=0.2,
        inductance=2e-7,
        inductance_exponent=1.0,
        resistances=np.array([0.3, 0.5]),
        capacitances=np.array([1e-4, 0.05]),
        exponents=np.array([0.9, 0.8]),
    )
    z = z * (1 + noise * rng.standard_normal(n)) + 1j * noise * np.abs(
        z
    ) * rng.standard_normal(n)
    return freq, z


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"c_threshold": 0.5},
        {"c_threshold": 0.99},
        {"max_elements": 5},
        {"c_threshold": -1.0, "max_elements": 30},  # never stops early
    ],
)
@pytest.mark.parametrize("seed", [0, 1])
def test_lin_kk_matches_reference(kwargs, seed):
    freq, z = _spectrum(seed)
    new = lin_kk(freq, z, **kwargs)
    ref = _lin_kk_reference(freq, z, **kwargs)
    assert set(new) == set(ref) == {"M", "mu", "rms_rel", "z_fit"}
    assert new["M"] == ref["M"]
    assert new["mu"] == pytest.approx(ref["mu"], rel=1e-12, abs=1e-12)
    assert new["rms_rel"] == pytest.approx(ref["rms_rel"], rel=1e-12, abs=1e-15)
    np.testing.assert_allclose(new["z_fit"], ref["z_fit"], rtol=1e-12, atol=0)


def test_lin_kk_types():
    freq, z = _spectrum()
    out = lin_kk(list(freq), list(z))
    assert isinstance(out["M"], int)
    assert isinstance(out["mu"], float)
    assert isinstance(out["rms_rel"], float)
    assert out["z_fit"].shape == freq.shape
    assert np.iscomplexobj(out["z_fit"])


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_lin_kk_rejects_bad_frequency(bad):
    freq, z = _spectrum()
    freq = freq.copy()
    freq[10] = bad
    with pytest.raises(ValueError, match="finite and > 0"):
        lin_kk(freq, z)


def test_lin_kk_rejects_empty_and_mismatched():
    freq, z = _spectrum()
    with pytest.raises(ValueError, match="empty"):
        lin_kk(np.array([]), np.array([], dtype=complex))
    with pytest.raises(ValueError, match="same shape"):
        lin_kk(freq, z[:-1])


def test_scripts_all_exports_importable():
    scripts = importlib.import_module("blowing_bubbles.scripts")
    assert sorted(scripts.__all__) == ["fit_circuit", "fit_quality", "lin_kk"]
    for name in scripts.__all__:
        assert callable(getattr(scripts, name))
    from blowing_bubbles.scripts import fit_circuit, fit_quality
    from blowing_bubbles.scripts import lin_kk as exported_lin_kk
    from blowing_bubbles.scripts.fit_circuit import fit_circuit as fc_direct
    from blowing_bubbles.scripts.quality import fit_quality as fq_direct

    assert fit_circuit is fc_direct
    assert fit_quality is fq_direct
    assert exported_lin_kk is lin_kk


@pytest.mark.parametrize(
    "code",
    [
        "import blowing_bubbles.classes; import blowing_bubbles.scripts",
        "import blowing_bubbles.scripts; import blowing_bubbles.classes",
        "from blowing_bubbles.classes import EISData; EISData.fit",
        "from blowing_bubbles.scripts import fit_circuit; "
        "from blowing_bubbles.classes import EISData; EISData.fit",
    ],
)
def test_no_circular_import_in_fresh_interpreter(code):
    # Make the child import this checkout's package, not an installed copy.
    src = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (src, env.get("PYTHONPATH")) if p)
    code += "; import blowing_bubbles; print(blowing_bubbles.__file__)"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert Path(proc.stdout.strip()).resolve().is_relative_to(Path(src))
