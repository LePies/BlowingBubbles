r"""Label-free quality checks for fitted spectra.

The central check compares the residual of the equivalent-circuit fit with the
residual of a linear Kramers-Kronig (Lin-KK) fit of the same spectrum
(Schönleber et al., Electrochim. Acta 131 (2014) 20-27). Lin-KK is a
linear least-squares fit of many RC elements with fixed, log-spaced time
constants, so it has no local minima; its residual estimates the noise and
non-stationarity floor of the measurement. An equivalent-circuit fit whose
residual is many times larger than that floor has not captured the spectrum
(typically a local minimum) and is flagged without any reference annotation.
"""

from __future__ import annotations

import numpy as np

from blowing_bubbles.circuit_impedance import circuit_impedance


def _relative_rms(z_model: np.ndarray, z_data: np.ndarray) -> float:
    w = 1.0 / np.maximum(np.abs(z_data), np.finfo(float).tiny)
    return float(np.sqrt(np.mean(np.abs((z_model - z_data) * w) ** 2)))


def lin_kk(
    freq: np.ndarray,
    z: np.ndarray,
    c_threshold: float = 0.85,
    max_elements: int | None = None,
) -> dict:
    r"""Linear Kramers-Kronig test with the :math:`\mu` criterion.

    Model: :math:`Z = R_0 + j\omega L + \sum_k R_k / (1 + j\omega\tau_k)` with
    :math:`\tau_k` log-spaced over :math:`[1/\omega_{max}, 1/\omega_{min}]`,
    solved by modulus-weighted linear least squares. Starting from two
    elements per decade, the number of elements :math:`M` is increased until :math:`\mu = 1 - \sum_{R_k<0}|R_k| /
    \sum_{R_k\ge0}|R_k|` drops below ``c_threshold``.

    Returns a dict with ``M``, ``mu``, ``rms_rel`` (relative RMS residual) and
    ``z_fit``.
    """
    f = np.asarray(freq, dtype=float)
    z = np.asarray(z, dtype=complex)
    omega = 2.0 * np.pi * f
    w = 1.0 / np.maximum(np.abs(z), np.finfo(float).tiny)
    n = f.size
    # At least two elements per decade: with coarser grids an RC element whose
    # tau falls between grid points is not resolved and the residual
    # overstates the noise floor (the mu criterion alone stops too early).
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
        best = {"M": m, "mu": float(mu), "rms_rel": _relative_rms(z_fit, z), "z_fit": z_fit}
        if mu <= c_threshold:
            break
    return best


def fit_quality(data, parameters, kk_ratio_threshold: float = 3.0) -> dict:
    r"""Label-free quality metrics for a fitted spectrum.

    Parameters
    ----------
    data : EISData
        The measured spectrum.
    parameters : CircuitParameters
        Fitted circuit (e.g. ``FitResults.parameters``).
    kk_ratio_threshold : float
        A fit is flagged ``suspect`` when its relative RMS residual exceeds
        ``kk_ratio_threshold`` times the Lin-KK residual.

    Returns
    -------
    dict
        ``rms_rel_fit``, ``rms_rel_kk``, ``kk_ratio``, ``kk_M``, ``kk_mu``,
        ``r_total_fit`` (:math:`R_s + \sum R_i`), ``r_total_lowf``
        (:math:`Z'` at the lowest measured frequency), ``r_total_ratio`` and
        the boolean ``suspect``.
    """
    f = np.asarray(data.Freq, dtype=float)
    z = np.asarray(data.Real, dtype=float) + 1j * np.asarray(data.Imag, dtype=float)
    p = parameters
    z_fit = circuit_impedance(
        f,
        resistance_source=p.resistance_source,
        inductance=p.inductance,
        inductance_exponent=p.inductance_exponent,
        resistances=p.resistances,
        capacitances=p.capacitances,
        exponents=p.exponents,
    )
    kk = lin_kk(f, z)
    rms_fit = _relative_rms(z_fit, z)
    ratio = rms_fit / max(kk["rms_rel"], np.finfo(float).tiny)
    r_total_fit = float(p.resistance_source + np.sum(p.resistances))
    r_total_lowf = float(np.real(z[np.argmin(f)]))
    return {
        "rms_rel_fit": rms_fit,
        "rms_rel_kk": kk["rms_rel"],
        "kk_ratio": float(ratio),
        "kk_M": kk["M"],
        "kk_mu": kk["mu"],
        "r_total_fit": r_total_fit,
        "r_total_lowf": r_total_lowf,
        "r_total_ratio": r_total_fit / r_total_lowf if r_total_lowf != 0 else np.nan,
        "suspect": bool(ratio > kk_ratio_threshold),
    }
