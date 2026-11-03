r"""Label-free quality checks for fitted spectra.

The central check compares the residual of the equivalent-circuit fit with the
residual of a linear Kramers-Kronig (Lin-KK) fit of the same spectrum
(Schönleber et al., Electrochim. Acta 131 (2014) 20-27). Lin-KK is a
linear least-squares fit of many RC elements with fixed, log-spaced time
constants, so it has no local minima; its residual estimates the noise and
non-stationarity floor of the measurement. An equivalent-circuit fit whose
residual is many times larger than that floor has not captured the spectrum
(typically a local minimum) and is flagged without any reference annotation.

Public entry points: :func:`fit_quality` (metrics and ``suspect`` flag for a
fitted spectrum) and :func:`lin_kk` (the Lin-KK test itself).
"""

from __future__ import annotations

import numpy as np

from blowing_bubbles.circuit_impedance import circuit_impedance


def _relative_rms(z_model: np.ndarray, z_data: np.ndarray) -> float:
    """Return the RMS of ``|z_model - z_data| / |z_data|`` over all points."""
    w =1.0 / np.maximum(np.abs(z_data), np.finfo(float).tiny)
    return float(np.sqrt(np.mean(np.abs((z_model - z_data) * w) ** 2)))


def lin_kk(
    freq: np.ndarray,
    z: np.ndarray,
    c_threshold: float = 0.85,
    max_elements: int | None = None,
) -> dict:
    r"""Run a linear Kramers-Kronig (Lin-KK) test with the :math:`\mu` criterion.

    Fits a Kramers-Kronig-compliant RC ladder by linear least squares; its
    residual estimates the noise floor of the measurement (see
    :func:`fit_quality`).

    Parameters
    ----------
    freq : numpy.ndarray, shape (n_freq,)
        Frequencies in Hz; all must be finite and positive.
    z : numpy.ndarray, shape (n_freq,)
        Complex impedance in Ω.
    c_threshold : float, default 0.85
        Stop adding elements once :math:`\mu \le` ``c_threshold``.
    max_elements : int, optional
        Maximum number of RC elements. ``None`` means
        ``max(m_min, n_freq // 2)``, with ``m_min`` the starting count.

    Returns
    -------
    dict
        ``M`` (int, number of RC elements used), ``mu`` (float),
        ``rms_rel`` (float, relative RMS residual, see Notes) and ``z_fit``
        (complex numpy.ndarray, shape (n_freq,), fitted impedance in Ω).

    Raises
    ------
    ValueError
        If ``freq`` is empty, contains non-finite or non-positive values, or
        does not have the same shape as ``z``.

    See Also
    --------
    fit_quality

    Notes
    -----
    Model (Schönleber et al., Electrochim. Acta 131 (2014) 20-27):

    .. math::

        Z = R_0 + j\omega L + \sum_{k=1}^{M} \frac{R_k}{1 + j\omega\tau_k},

    with :math:`\tau_k` log-spaced over :math:`[1/\omega_{\max},
    1/\omega_{\min}]`. :math:`R_0`, :math:`L` and :math:`R_k` are real and
    solved by modulus-weighted linear least squares (weights
    :math:`1/|Z|`). Starting from
    :math:`M = \max(3, \lceil 2 \cdot \text{decades} \rceil)` (at least two
    elements per decade, capped at ``max_elements``), :math:`M` is increased
    by one until

    .. math::

        \mu = 1 - \frac{\sum_{R_k<0}|R_k|}{\sum_{R_k\ge0}|R_k|}

    drops to ``c_threshold`` or below (:math:`\mu = 0` if all
    :math:`R_k \le 0`). If it never does, the largest :math:`M` is used.
    The relative RMS residual is
    :math:`\sqrt{\mathrm{mean}(|Z_{\mathrm{fit}} - Z|^2 / |Z|^2)}`.
    """
    f = np.asarray(freq, dtype=float)
    z = np.asarray(z, dtype=complex)
    if f.size == 0:
        raise ValueError("lin_kk: freq is empty.")
    if f.shape != z.shape:
        raise ValueError(
            f"lin_kk: freq and z must have the same shape, got {f.shape} and {z.shape}."
        )
    if not np.all(np.isfinite(f)) or np.any(f <= 0):
        raise ValueError(
            "lin_kk: all frequencies must be finite and > 0 (the tau grid is "
            "log-spaced up to 1/omega_min)."
        )
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

    # Everything that does not depend on M is built once: the R_0 and L
    # columns of the design matrix (unweighted and weighted, stacked as
    # [real; imag]), the right-hand side and the log10 tau range. Per M only
    # the RC basis columns are rebuilt.
    log_tau_lo = np.log10(1.0 / omega.max())
    log_tau_hi = np.log10(1.0 / omega.min())
    a_fixed = np.column_stack([np.ones(n), 1j * omega])
    aw_fixed = a_fixed * w[:, None]
    zw = z * w
    b = np.concatenate([zw.real, zw.imag])
    omega_col = omega[:, None]
    w_col = w[:, None]

    m = mu = coef = basis = None
    for m in range(m_min, m_max + 1):
        tau = np.logspace(log_tau_lo, log_tau_hi, m)
        basis = 1.0 / (1.0 + 1j * omega_col * tau[None, :])
        basis_w = basis * w_col
        Ar = np.empty((2 * n, m + 2))
        Ar[:n, :2] = aw_fixed.real
        Ar[n:, :2] = aw_fixed.imag
        Ar[:n, 2:] = basis_w.real
        Ar[n:, 2:] = basis_w.imag
        coef, *_ = np.linalg.lstsq(Ar, b, rcond=None)
        rk = coef[2:]
        pos = np.sum(np.abs(rk[rk >= 0]))
        neg = np.sum(np.abs(rk[rk < 0]))
        mu = 1.0 - neg / pos if pos > 0 else 0.0
        if mu <= c_threshold:
            break
    if coef is None:
        return None
    # The fitted spectrum and residual are only needed for the selected M.
    z_fit = np.hstack([a_fixed, basis]) @ coef
    return {
        "M": m,
        "mu": float(mu),
        "rms_rel": _relative_rms(z_fit, z),
        "z_fit": z_fit,
    }


def fit_quality(data, parameters, kk_ratio_threshold: float = 3.0) -> dict:
    r"""Compute label-free quality metrics for a fitted spectrum.

    Compares the relative RMS residual of the equivalent-circuit fit with the
    Lin-KK noise floor of the same spectrum and with its low-frequency
    resistance.

    Parameters
    ----------
    data : EISData
        The measured spectrum (``Freq`` in Hz, ``Real`` and ``Imag`` in Ω).
    parameters : CircuitParameters
        Fitted circuit (e.g. ``FitResults.parameters``).
    kk_ratio_threshold : float, default 3.0
        A fit is flagged ``suspect`` when its relative RMS residual exceeds
        ``kk_ratio_threshold`` times the Lin-KK residual.

    Returns
    -------
    dict
        Metrics with the keys

        - ``rms_rel_fit`` : relative RMS residual of the circuit fit;
        - ``rms_rel_kk`` : relative RMS residual of the Lin-KK fit;
        - ``kk_ratio`` : ``rms_rel_fit / rms_rel_kk``;
        - ``kk_M``, ``kk_mu`` : number of Lin-KK elements and final
          :math:`\mu` (see :func:`lin_kk`);
        - ``r_total_fit`` : :math:`R_s + \sum R_i` in Ω;
        - ``r_total_lowf`` : :math:`Z'` at the lowest measured frequency, Ω;
        - ``r_total_ratio`` : ``r_total_fit / r_total_lowf`` (NaN if
          ``r_total_lowf == 0``);
        - ``suspect`` : bool, ``kk_ratio > kk_ratio_threshold``.

    See Also
    --------
    lin_kk
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
