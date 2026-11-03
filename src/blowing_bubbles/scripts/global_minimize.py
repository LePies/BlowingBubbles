r"""Bounded nonlinear least-squares fitting of the equivalent circuit to EIS data.

Minimizes the (weighted) residuals between the measured impedance and the model
impedance :math:`Z_{\mathrm{pred}}(\omega_k; \mathbf{p})` at the measured
frequencies :math:`\omega_k`, using SciPy's trust-region-reflective solver with
multiple restarts. The main entry point is :func:`impedance_fit`; helpers build
start values (:func:`initial_fit`), time-constant bounds
(:func:`tau_bounds_from_frequencies`), inductance estimates and bounds
(:func:`estimate_inductance_from_imag`, :func:`inductance_bounds_from_estimate`)
and a reusable process pool for parallel restarts (:func:`make_process_pool`).
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import Executor, ProcessPoolExecutor
from itertools import repeat
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import (
    Bounds,
    LinearConstraint,
    OptimizeResult,
    least_squares,
    minimize,
)
from scipy.signal import find_peaks

from blowing_bubbles.classes.circuit_classes import CircuitParameters, FitResults

if TYPE_CHECKING:
    from blowing_bubbles.classes.data_classes import EISData


def initial_fit(
    eis: EISData,
    prominence: float = 0.01,
    tau_bounds: tuple[float, float] | None = None,
    max_peaks: int | None = None,
) -> CircuitParameters:
    r"""Build a heuristic initial guess from the Nyquist peak structure of ``-Im(Z)``.

    Each peak of :math:`-\mathrm{Im}(Z)` becomes one ideal RC branch. Useful as
    ``initial_parameters`` for :func:`impedance_fit`.

    Parameters
    ----------
    eis : EISData
        Measured spectrum (``Freq`` in Hz, ``Real`` and ``Imag`` in Ω).
    prominence : float, default 0.01
        Minimum peak prominence, relative to the peak-to-peak range of
        :math:`-\mathrm{Im}(Z)`.
    tau_bounds : tuple of float, optional
        ``(tau_min, tau_max)`` in s; the estimated time constants are clipped
        to this range. ``None`` applies no clipping.
    max_peaks : int, optional
        Keep only the ``max_peaks`` most prominent peaks. ``None`` keeps all.

    Returns
    -------
    CircuitParameters
        One branch per retained peak, with :math:`n_L = 1` and all branch
        exponents equal to 1.

    See Also
    --------
    impedance_fit

    Notes
    -----
    Peaks are found with :func:`scipy.signal.find_peaks`. If none is found
    (e.g. a truncated arc), a single branch is seeded at the point of largest
    :math:`-\mathrm{Im}(Z)`. Between neighbouring peaks the minimum of
    :math:`-\mathrm{Im}(Z)` marks a valley; the data edges bound the first and
    last arc.

    - :math:`R_i` is the arc width :math:`|Z'_{\mathrm{right}} - Z'_{\mathrm{left}}|`
      between its bracketing valleys (or edges), floored at
      :math:`10^{-6}` of the :math:`Z'` span so :math:`C_i` stays finite.
    - :math:`\tau_i = 1/(2\pi f_{\mathrm{peak}})` (apex of an ideal RC arc),
      optionally clipped to ``tau_bounds``; :math:`C_i = \tau_i / R_i`.
    - :math:`R_s` is :math:`Z'` at the highest measured frequency.
    - :math:`L` comes from :func:`estimate_inductance_from_imag`.
    """
    # Find all peaks with the given prominence
    prom = prominence * np.ptp(-eis.Imag)
    peaks, properties = find_peaks(-eis.Imag, prominence=prom)
    if len(peaks) == 0:
        # No interior maximum (e.g. a truncated arc): seed one branch at the
        # point of largest -Im(Z).
        peaks = np.array([int(np.argmax(-eis.Imag))])
    # If max_peaks is set, keep the most prominent peaks (descending prominence)
    if max_peaks is not None and len(peaks) > max_peaks:
        prominences = properties["prominences"]
        # Get indices of the most prominent peaks
        top_peak_indices = np.argsort(prominences)[-max_peaks:][::-1]
        # Sort the selected peaks in increasing index order for consistency
        top_peak_indices = np.sort(top_peak_indices)
        peaks = peaks[top_peak_indices]
    # Now find the valley bottom between peaks
    valleys = []
    if len(peaks) > 1:
        for i in range(len(peaks) - 1):
            # Search between each pair of peaks for the minimum (valley)
            start = peaks[i]
            end = peaks[i + 1]
            if end > start + 1:  # There is at least one point between the peaks
                valley_idx = np.argmin(-eis.Imag[start : end + 1]) + start
                valleys.append(valley_idx)

    # Mountain width: valley to valley, or edge to valley for the first/last arc
    mountain_widths = []
    mountain_center_freqs = []

    # Indexes bracketing each peak region
    all_bounds = [0, *valleys, len(eis.Real) - 1]

    for i in range(len(peaks)):
        # Peak i lies between all_bounds[i] and all_bounds[i + 1]
        left = all_bounds[i]
        right = all_bounds[i + 1]
        # The width of this "mountain" is freq[right] - freq[left]
        width = abs(eis.Real[right] - eis.Real[left])
        mountain_widths.append(width)
        peak_freq = float(eis.Freq[peaks[i]])
        if peak_freq <= 0:
            peak_freq = np.finfo(float).tiny
        mountain_center_freqs.append(peak_freq)

    # Floor zero widths so C = tau / R stays finite.
    r_floor = max(1e-6 * float(np.ptp(eis.Real)), np.finfo(float).tiny)
    resistances = np.maximum(np.array(mountain_widths), r_floor)
    # Apex of an ideal RC arc: omega_peak * R * C = 1 with omega = 2*pi*f.
    taus = 1 / (2 * np.pi * np.asarray(mountain_center_freqs, dtype=float))
    if tau_bounds is not None:
        taus = np.clip(taus, tau_bounds[0], tau_bounds[1])
    capacitances = taus / resistances
    # High-frequency intercept, independent of the row order of the data.
    resistances_source = float(eis.Real[int(np.argmax(eis.Freq))])
    inductance = estimate_inductance_from_imag(eis.Freq, eis.Imag)
    inductance_exponent = 1

    return CircuitParameters(
        resistance_source=resistances_source,
        inductance=inductance,
        inductance_exponent=inductance_exponent,
        resistances=resistances,
        capacitances=capacitances,
        exponents=np.ones_like(resistances),
    )


def tau_bounds_from_frequencies(freq: np.ndarray) -> tuple[float, float]:
    r"""Return the time-constant range whose apex lies inside the measured window.

    Intended as ``tau_bounds`` for :func:`impedance_fit` and
    :func:`initial_fit`.

    Parameters
    ----------
    freq : numpy.ndarray, shape (n_freq,)
        Measured frequencies in Hz. Non-positive entries are ignored.

    Returns
    -------
    tau_min : float
        :math:`1/(2\pi f_{\max})` in s.
    tau_max : float
        :math:`1/(2\pi f_{\min})` in s.

    Raises
    ------
    ValueError
        If ``freq`` contains no positive frequency.

    Notes
    -----
    The range contains every :math:`\tau` whose apex frequency
    :math:`f = 1/(2\pi\tau)` lies between :math:`f_{\min}` and
    :math:`f_{\max}`. (Before 0.3 the factor :math:`2\pi` was missing, which
    excluded arcs peaking above :math:`f_{\max}/2\pi`.) The bounds act on the
    stored products :math:`R_iQ_i`; for :math:`n_i = 1` these equal the
    characteristic times.
    """
    f = np.asarray(freq, dtype=float)
    f_pos = f[f > 0]
    if f_pos.size == 0:
        raise ValueError("Need at least one positive frequency for tau bounds.")
    two_pi = 2.0 * np.pi
    return (
        1.0 / (two_pi * float(np.max(f_pos))),
        1.0 / (two_pi * float(np.min(f_pos))),
    )


_INDUCTANCE_EST_TOL = 1e-15
_CPE_EXPONENT_MIN = 0.2
_CPE_EXPONENT_MAX = 1.0
# Default lower bounds on R_i and tau_1, relative to the data scales (|Z'| span
# and 1/omega_max). Strictly positive so C_i = tau_i / R_i stays finite and > 0.
_POSITIVE_FLOOR_REL = 1e-12
# Smallest magnitude of a perturbed restart start value.
_PERTURB_MIN_ABS = 1e-9
# TRF termination tolerances (SciPy default 1e-8). Tighter values let the
# solver reach parameters on their bounds (e.g. L = 0 with n_L >= 0.7) instead
# of stopping at a tiny interior value.
_TRF_TOL = 1e-12
# Default inductor-exponent bounds when neither ``exponent_bounds`` nor
# ``inductance_exponent_bounds`` is given (expert EK3 fits: n_L = 0.86-0.99).
_DEFAULT_INDUCTANCE_EXPONENT_BOUNDS = (0.7, 1.0)


def estimate_inductance_from_imag(
    freq: np.ndarray,
    imag: np.ndarray,
) -> float:
    r"""Estimate :math:`L` from a weighted slope of :math:`\mathrm{Im}(Z)`.

    Parameters
    ----------
    freq : numpy.ndarray, shape (n_freq,)
        Frequencies in Hz.
    imag : numpy.ndarray, shape (n_freq,)
        Imaginary part of the impedance in Ω.

    Returns
    -------
    float
        Estimated inductance in H (may be negative). ``0.0`` when the spectrum
        shows no inductive behaviour.

    See Also
    --------
    inductance_bounds_from_estimate

    Notes
    -----
    For :math:`Z = R_s + j\omega L + \cdots` with :math:`n_L = 1`,
    :math:`\mathrm{Im}(Z) \approx \omega L` at high frequency. The function
    regresses :math:`\mathrm{Im}(Z)` against :math:`\omega` through the origin,

    .. math::

        L = \frac{\sum_k w_k \omega_k \mathrm{Im}(Z_k)}{\sum_k w_k \omega_k^2},
        \qquad w_k = f_k / \max_k f_k,

    so higher frequencies weigh more. The points used are

    - the inductive points (:math:`f > 0`, :math:`\mathrm{Im}(Z) > 0`) if there
      are at least two of them;
    - otherwise, for a negative inductance, the top 20 % (at least 3) of the
      positive frequencies, provided all of them have :math:`\mathrm{Im}(Z) < 0`
      and :math:`\mathrm{Im}(Z)` decreases with frequency there.

    Returns 0 for a purely capacitive high-frequency tail (where
    :math:`\mathrm{Im}(Z) < 0` tends to 0) or with fewer than 3 positive
    frequencies in the fallback case.
    """
    f = np.asarray(freq, dtype=float)
    im = np.asarray(imag, dtype=float)
    # Only inductive points (Im Z > 0) carry information on L. Regressing over
    # the capacitive tail, where Im Z rises towards zero, returned a spurious
    # positive L for spectra without any inductive part.
    pos = (f > 0) & (im > 0)
    if np.count_nonzero(pos) >= 2:
        mask = pos
    else:
        # Negative L shows as Im Z < 0 that grows in magnitude with omega at
        # the top frequencies; a capacitive tail is negative but tends to 0.
        fp = f > 0
        if np.count_nonzero(fp) < 3:
            return 0.0
        order = np.argsort(f[fp])
        k = max(3, int(np.ceil(0.2 * order.size)))
        top = order[-k:]
        f_top, im_top = f[fp][top], im[fp][top]
        if not (np.all(im_top < 0) and np.polyfit(f_top, im_top, 1)[0] < 0):
            return 0.0
        mask = np.zeros_like(f, dtype=bool)
        mask[np.flatnonzero(fp)[top]] = True
    f = f[mask]
    im = im[mask]
    omega = 2.0 * np.pi * f
    w = f / float(np.max(f))
    # Weighted least squares through the origin: Im Z ~ omega * L.
    den = float(np.sum(w * omega**2))
    if den <= 0:
        return 0.0
    return float(np.sum(w * omega * im) / den)


def inductance_bounds_from_estimate(
    L_est: float,
    *,
    margin: float = 2.0,
) -> tuple[float | None, float | None]:
    r"""Return signed box bounds on :math:`L` from a slope estimate.

    Used by :func:`impedance_fit` when ``inductance_bounds=True``.

    Parameters
    ----------
    L_est : float
        Inductance estimate in H, e.g. from
        :func:`estimate_inductance_from_imag`.
    margin : float, default 2.0
        Factor applied to ``L_est`` to obtain the outer bound.

    Returns
    -------
    lower : float or None
        Lower bound on :math:`L` in H.
    upper : float or None
        Upper bound on :math:`L` in H.

    Notes
    -----
    - :math:`L_{\mathrm{est}} < 0`: ``margin * L_est <= L <= 0``.
    - :math:`L_{\mathrm{est}} > 0`: ``0 <= L <= margin * L_est``.

    ``(None, None)`` (no custom bounds) is returned when
    :math:`|L_{\mathrm{est}}| < 10^{-15}`, when ``margin * L_est`` is not
    finite, or when the resulting interval is empty (e.g. ``margin <= 0``).
    """
    L_est = float(L_est)
    if abs(L_est) < _INDUCTANCE_EST_TOL:
        return None, None
    bound = margin * L_est
    if not np.isfinite(bound):
        return None, None
    if L_est < 0:
        lower, upper = bound, 0.0
    else:
        lower, upper = 0.0, bound
    if lower >= upper:
        return None, None
    return lower, upper


def _normalize_tau_bounds(
    tau_bounds: tuple[float, float] | None,
) -> tuple[float, float] | None:
    """Return ``tau_bounds`` as an ordered float pair (swapped if reversed)."""
    if tau_bounds is None:
        return None
    tau_lo, tau_hi = float(tau_bounds[0]), float(tau_bounds[1])
    if tau_lo > tau_hi:
        tau_lo, tau_hi = tau_hi, tau_lo
    return tau_lo, tau_hi


def _positive_floor(
    floor: float,
    user_lo: float | None = None,
    user_hi: float | None = None,
) -> float:
    """Return a strictly positive lower bound that never tightens user bounds.

    Returns ``max(floor', user_lo)`` where ``floor'`` is *floor* reduced, if
    needed, to stay at or below a positive ``user_lo`` and at or below half a
    positive ``user_hi``. A user lower bound of ``0`` (or less) is replaced by
    the floor.
    """
    lo = float(floor)
    if user_lo is not None and user_lo > 0:
        lo = min(lo, float(user_lo))
    if user_hi is not None and user_hi > 0:
        lo = min(lo, 0.5 * float(user_hi))
    if user_lo is not None:
        lo = max(lo, float(user_lo))
    return lo


def _perturb_init(
    base: np.ndarray,
    rng: np.random.Generator,
    sigma: float,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    complex_fit: bool,
    n_groups: int,
    local_exponents: bool,
) -> np.ndarray:
    """Draw a random restart start point around *base*, clipped to the bounds.

    All entries get a log-normal multiplicative perturbation, which preserves
    their sign; magnitudes are floored at ``_PERTURB_MIN_ABS`` (zeros become
    positive), so a negative inductance allowed by the bounds survives. With
    ``complex_fit`` the exponents (``n_L`` and ``e_i``) are instead drawn
    around *base* (``local_exponents=True``) or uniformly within their bounds.
    The number and order of random draws depend only on the arguments, so a
    seeded generator gives a reproducible sequence of starts.
    """
    init = base * np.exp(rng.normal(scale=sigma, size=base.size))
    init = np.where(
        init < 0,
        np.minimum(init, -_PERTURB_MIN_ABS),
        np.maximum(init, _PERTURB_MIN_ABS),
    )
    if complex_fit:
        head = 3
        e_slice = slice(head + 2 * n_groups, head + 3 * n_groups)
        if local_exponents:
            # Stay near the base exponents (refinement around a seed).
            init[2] = base[2] + rng.normal(scale=0.5 * sigma)
            init[e_slice] = base[e_slice] + rng.normal(scale=0.5 * sigma, size=n_groups)
        else:
            # Global exploration of the exponents within their bounds.
            init[2] = rng.uniform(lower[2], upper[2])
            init[e_slice] = rng.uniform(lower[e_slice], upper[e_slice])
    return np.clip(init, lower, upper)


def _project_dtau(
    dtau: np.ndarray,
    tau_lo: float,
    tau_hi: float,
) -> np.ndarray:
    """Map Δτ so that ``cumsum(Δτ)`` lies in ``[tau_lo, tau_hi]`` with Δτ >= 0."""
    dtau = np.maximum(np.asarray(dtau, dtype=float), 0.0)
    tau = np.cumsum(dtau)
    tau = np.clip(tau, tau_lo, tau_hi)
    tau = np.maximum.accumulate(tau)
    out = np.empty_like(tau)
    out[0] = tau[0]
    if tau.size > 1:
        out[1:] = np.diff(tau)
    return out


def _build_center_linear_constraint(
    n_params: int,
    head: int,
    n_groups: int,
    z_min: float,
    z_max: float,
) -> LinearConstraint:
    """Build the hard limits R_s <= z_min and R_s + sum(R) - R_N/2 <= z_max.

    The lower limit follows from Re Z(omega) >= R_s for every branch and for the
    CPE inductor; it stays valid when the high-frequency arc is truncated by the
    measurement window (version 0.1 used R_s + R_1/2 >= z_min, which excludes
    the true circuit in that case).
    """
    A = np.zeros((2, n_params))
    A[0, 0] = 1.0
    A[1, 0] = 1.0
    A[1, head : head + n_groups] = 1.0
    A[1, head + n_groups - 1] -= 0.5
    return LinearConstraint(A, -np.inf * np.ones(2), np.array([z_min, z_max]))


def _center_positions(p: np.ndarray, r_slice: slice) -> tuple[float, float]:
    """Return ``(R_s, R_s + sum(R) - R_N/2)``, the constrained Nyquist positions."""
    Rs = p[0]
    R = p[r_slice]
    c_lo = Rs
    c_hi = Rs + float(np.sum(R)) - 0.5 * R[-1]
    return c_lo, c_hi


def _constraints_satisfied(
    p: np.ndarray,
    r_slice: slice,
    z_min: float,
    z_max: float,
    tol: float,
) -> bool:
    """Return whether both hard center constraints hold within ``tol``."""
    c_lo, c_hi = _center_positions(p, r_slice)
    return c_lo <= z_min + tol and c_hi <= z_max + tol


def _project_feasible(
    p: np.ndarray,
    z_min: float,
    z_max: float,
    head: int,
    r_slice: slice,
    lower: np.ndarray,
    upper: np.ndarray,
    n_groups: int,
    dtau_slice: slice | None = None,
    tau_bounds: tuple[float, float] | None = None,
) -> np.ndarray:
    """Clip to box bounds and satisfy the Nyquist center and τ constraints.

    Lowers ``R_s`` to ``z_min`` and scales the branch resistances down
    proportionally (to no less than 1 % per pass, four passes) so the upper
    center constraint holds, then projects the Δτ block with
    ``_project_dtau``. Returns a new array; *p* is not modified.
    """
    p = np.clip(p, lower, upper).copy()
    if n_groups == 0:
        return p

    for _ in range(4):
        Rs = p[0]

        if Rs > z_min:
            p[0] = max(lower[0], z_min)

        c_hi = p[0] + float(np.sum(p[r_slice])) - 0.5 * p[r_slice][-1]
        if c_hi > z_max:
            excess = c_hi - z_max
            r_sum = float(np.sum(p[r_slice]))
            if r_sum > 0:
                scale = max(0.01, 1.0 - excess / r_sum)
                p[r_slice] = np.maximum(p[r_slice] * scale, lower[r_slice])

        if p[0] > z_min:
            p[0] = max(lower[0], z_min)

        p = np.clip(p, lower, upper)

    if tau_bounds is not None and dtau_slice is not None:
        tau_lo, tau_hi = tau_bounds
        p[dtau_slice] = _project_dtau(p[dtau_slice], tau_lo, tau_hi)
        p = np.clip(p, lower, upper)

    return p


class _FitProblem:
    """Weighted residuals, Jacobian and constraint handling for one fit.

    Module-level and picklable so restarts can run in worker processes. The
    forward model is evaluated once per parameter vector and cached, so the
    Jacobian at an accepted step reuses the intermediates of the residual call.

    The optimizer vector is ``[R_s, L, (n_L,) R_1..R_n, Δτ_1..Δτ_n,
    (e_1..e_n)]``; the bracketed entries exist only with ``complex_fit``.
    """

    def __init__(
        self,
        freq: np.ndarray,
        z: np.ndarray,
        weights: np.ndarray,
        *,
        complex_fit: bool,
        n_groups: int,
        z_min: float,
        z_max: float,
        margin: float,
        margin_weight: float,
        lower: np.ndarray,
        upper: np.ndarray,
        tau_bounds: tuple[float, float] | None,
        max_nfev: int,
        polish: bool,
        polish_maxiter: int,
        constraint_tol: float,
    ) -> None:
        # log(j*omega) on the principal branch: (j*omega)**e == exp(e * log_s).
        omega = 2.0 * np.pi * np.asarray(freq, dtype=float)
        self.log_s = np.log(omega) + 0.5j * np.pi
        self.z = z
        self.w = weights
        self.m = z.size
        self.complex_fit = complex_fit
        self.n_groups = n_groups
        self.head = 3 if complex_fit else 2
        self.r_slice = slice(self.head, self.head + n_groups)
        self.dtau_slice = slice(self.head + n_groups, self.head + 2 * n_groups)
        self.e_slice = slice(self.head + 2 * n_groups, self.head + 3 * n_groups)
        self.n_params = self.head + (2 + complex_fit) * n_groups
        self.z_min = z_min
        self.z_max = z_max
        self.margin = margin
        self.margin_weight = margin_weight
        self.lower = lower
        self.upper = upper
        self.tau_bounds = tau_bounds
        self.max_nfev = max_nfev
        self.polish = polish
        self.polish_maxiter = polish_maxiter
        self.constraint_tol = constraint_tol
        self._ones = np.ones(n_groups)
        self._tau_chain = np.tri(n_groups)
        self._key: bytes | None = None
        self._state: tuple | None = None

    def __getstate__(self) -> dict:
        """Return the pickle state without the forward-model cache."""
        state = self.__dict__.copy()
        state["_key"] = None
        state["_state"] = None
        return state

    def _forward(self, p: np.ndarray) -> tuple:
        """Return cached ``(R, tau, s_L, S, D, Z)`` for the parameter vector *p*."""
        key = p.tobytes()
        if key != self._key:
            R = p[self.r_slice].copy()
            tau = np.cumsum(p[self.dtau_slice])
            e = p[self.e_slice] if self.complex_fit else self._ones
            n_l = p[2] if self.complex_fit else 1.0
            s_l = np.exp(n_l * self.log_s)
            S = np.exp(self.log_s[:, None] * e)
            D = 1.0 + tau * S
            Z = p[0] + p[1] * s_l + np.sum(R / D, axis=1)
            self._key = key
            self._state = (R, tau, s_l, S, D, Z)
        assert self._state is not None
        return self._state

    def model(self, p: np.ndarray) -> np.ndarray:
        """Return the model impedance (Ω, complex, shape (n_freq,)) at *p*."""
        return self._forward(np.asarray(p, dtype=float))[-1]

    def _margin_terms(self, p: np.ndarray) -> tuple[float, float]:
        """Return the signed lower/upper margin violations (positive = violated)."""
        c_lo, c_hi = _center_positions(p, self.r_slice)
        return self.margin + (c_lo - self.z_min), self.margin - (self.z_max - c_hi)

    def residuals(self, p: np.ndarray) -> np.ndarray:
        """Return weighted ``[Re; Im]`` residuals plus the two margin penalties.

        Shape ``(2 * n_freq + 2,)``.
        """
        p = np.asarray(p, dtype=float)
        diff = self._forward(p)[-1] - self.z
        m = self.m
        out = np.empty(2 * m + 2)
        out[:m] = self.w * diff.real
        out[m : 2 * m] = self.w * diff.imag
        pen_lo, pen_hi = self._margin_terms(p)
        out[2 * m] = self.margin_weight * max(0.0, pen_lo)
        out[2 * m + 1] = self.margin_weight * max(0.0, pen_hi)
        return out

    def jacobian(self, p: np.ndarray) -> np.ndarray:
        """Return the analytic Jacobian of :meth:`residuals`.

        Shape ``(2 * n_freq + 2, n_params)``.
        """
        p = np.asarray(p, dtype=float)
        R, tau, s_l, S, D, _ = self._forward(p)
        # Z_i = R_i / D_i with D_i = 1 + tau_i S_i, so at fixed tau:
        #   dZ/dR_i = 1/D_i,  dZ/dtau_i = -R_i S_i / D_i^2,
        #   dZ/de_i = dZ/dtau_i * tau_i * log(j omega).
        dz_dtau = -R * S / (D * D)
        J = np.empty((self.m, self.n_params), dtype=complex)
        J[:, 0] = 1.0
        J[:, 1] = s_l
        if self.complex_fit:
            J[:, 2] = p[1] * s_l * self.log_s
            J[:, self.e_slice] = dz_dtau * tau * self.log_s[:, None]
        J[:, self.r_slice] = 1.0 / D
        J[:, self.dtau_slice] = dz_dtau @ self._tau_chain

        m = self.m
        w = self.w[:, None]
        out = np.zeros((2 * m + 2, self.n_params))
        out[:m] = w * J.real
        out[m : 2 * m] = w * J.imag
        pen_lo, pen_hi = self._margin_terms(p)
        if pen_lo > 0:
            out[2 * m, 0] = self.margin_weight
        if pen_hi > 0:
            out[2 * m + 1, 0] = self.margin_weight
            out[2 * m + 1, self.r_slice] = self.margin_weight
            out[2 * m + 1, self.head + self.n_groups - 1] = 0.5 * self.margin_weight
        return out

    def cost(self, p: np.ndarray) -> float:
        """Return ``0.5 * sum(residuals(p) ** 2)``, as in ``least_squares``."""
        r = self.residuals(p)
        return 0.5 * float(np.dot(r, r))

    def project(self, p: np.ndarray) -> np.ndarray:
        """Project *p* onto the bounds and constraints (``_project_feasible``)."""
        return _project_feasible(
            p,
            self.z_min,
            self.z_max,
            self.head,
            self.r_slice,
            self.lower,
            self.upper,
            self.n_groups,
            dtau_slice=self.dtau_slice,
            tau_bounds=self.tau_bounds,
        )

    def satisfied(self, p: np.ndarray) -> bool:
        """Return whether the hard center constraints hold at *p*."""
        return _constraints_satisfied(
            p, self.r_slice, self.z_min, self.z_max, self.constraint_tol
        )

    def polish_if_infeasible(
        self, p: np.ndarray, cost: float
    ) -> tuple[np.ndarray, float]:
        """Project *p*; if still infeasible, polish it with ``trust-constr``.

        The polished point is kept if its cost does not exceed *cost* or it
        satisfies the constraints; otherwise the projected *p* and the input
        *cost* are returned. A failed polish emits a ``UserWarning``.
        """
        p = self.project(p)
        if self.satisfied(p):
            return p, self.cost(p)
        if not self.polish:
            return p, cost
        try:
            res_opt = minimize(
                self.cost,
                p,
                method="trust-constr",
                jac=lambda x: self.jacobian(x).T @ self.residuals(x),
                bounds=Bounds(self.lower, self.upper),
                constraints=[
                    _build_center_linear_constraint(
                        self.n_params, self.head, self.n_groups, self.z_min, self.z_max
                    )
                ],
                options={"maxiter": self.polish_maxiter, "verbose": 0},
            )
            p_out = self.project(res_opt.x)
            cost_out = self.cost(p_out)
            if cost_out <= cost or self.satisfied(p_out):
                return p_out, cost_out
        except (ValueError, np.linalg.LinAlgError) as exc:
            warnings.warn(f"Constraint polish failed: {exc}", stacklevel=3)
        return p, cost


def _solve_restart(problem: _FitProblem, init: np.ndarray) -> dict[str, Any]:
    """Run one TRF solve from *init*, then project (and polish) onto constraints.

    Returns a dict with ``x``, ``cost``, ``polished``, ``nfev``, ``njev`` and
    ``status``, or ``{"error": message}`` if the solver raised.
    """
    try:
        res_ls = least_squares(
            problem.residuals,
            init,
            jac=problem.jacobian,
            method="trf",
            bounds=(problem.lower, problem.upper),
            max_nfev=problem.max_nfev,
            x_scale="jac",
            ftol=_TRF_TOL,
            xtol=_TRF_TOL,
            gtol=_TRF_TOL,
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    p = problem.project(res_ls.x.copy())
    cost = problem.cost(p)
    polished = not problem.satisfied(p)
    if polished:
        p, cost = problem.polish_if_infeasible(p, cost)
    return {
        "x": p,
        "cost": cost,
        "polished": polished,
        "nfev": int(res_ls.nfev),
        "njev": int(res_ls.njev or 0),
        "status": int(res_ls.status),
    }


def _parameter_covariance(problem: _FitProblem, x: np.ndarray) -> np.ndarray:
    r"""Return the linearized covariance of the reported parameters at *x*.

    Uses the weighted impedance Jacobian :math:`J` (margin-penalty rows
    excluded) and the residual variance
    :math:`s^2 = \sum r_k^2 / \max(2M - p, 1)` of the weighted residuals:
    :math:`\Sigma_x = s^2 (J^T J)^+`. The pseudo-inverse (computed from a
    column-scaled SVD) handles rank deficiency. :math:`\Sigma_x` is mapped to
    the :meth:`CircuitParameters.to_array` layout
    ``[R_s, L, n_L, R_1..R_n, C_1..C_n, e_1..e_n]`` with the Jacobian
    :math:`G` of :math:`(R, \Delta\tau) \mapsto C_i = \tau_i / R_i`,
    :math:`\tau_i = \sum_{j \le i} \Delta\tau_j`: :math:`G \Sigma_x G^T`.
    Parameters fixed by the model (``n_L`` and ``e_i`` when
    ``complex_fit=False``) get zero variance. Branches are in optimizer
    order (``x`` is *not* yet sorted by tau).
    """
    m = problem.m
    n = problem.n_groups
    n_opt = problem.n_params
    J = problem.jacobian(x)[: 2 * m]
    r = problem.residuals(x)[: 2 * m]
    dof = max(2 * m - n_opt, 1)
    s2 = float(np.dot(r, r)) / dof

    # Column scaling keeps the SVD cutoff meaningful when parameters differ by
    # many orders of magnitude (Ohm vs. seconds).
    scale = np.linalg.norm(J, axis=0)
    scale[scale == 0] = 1.0
    _, sv, vt = np.linalg.svd(J / scale, full_matrices=False)
    cutoff = np.finfo(float).eps * max(J.shape) * (sv[0] if sv.size else 0.0)
    inv_sv2 = np.zeros_like(sv)
    keep = sv > cutoff
    inv_sv2[keep] = 1.0 / sv[keep] ** 2
    cov_opt = (vt.T * inv_sv2) @ vt
    cov_opt = s2 * cov_opt / np.outer(scale, scale)

    # G maps optimizer coordinates to [Rs, L, nL, R, C, e].
    R = x[problem.r_slice]
    tau = np.cumsum(x[problem.dtau_slice])
    G = np.zeros((3 + 3 * n, n_opt))
    G[0, 0] = 1.0
    G[1, 1] = 1.0
    if problem.complex_fit:
        G[2, 2] = 1.0
        G[3 + 2 * n :, problem.e_slice] = np.eye(n)
    G[3 : 3 + n, problem.r_slice] = np.eye(n)
    # dC_i/dR_i = -tau_i / R_i^2 ; dC_i/d(dtau_j) = 1 / R_i for j <= i.
    G[3 + n : 3 + 2 * n, problem.r_slice] = np.diag(-tau / R**2)
    G[3 + n : 3 + 2 * n, problem.dtau_slice] = np.tri(n) / R[:, None]
    cov = G @ cov_opt @ G.T
    return 0.5 * (cov + cov.T)


_BLAS_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _noop(_: int) -> None:
    """Do nothing; submitted to force worker processes to start."""
    return None


def make_process_pool(n_jobs: int | None = -1) -> ProcessPoolExecutor:
    """Create a process pool for restarts whose workers use single-threaded BLAS.

    Restart solves work on tiny matrices, so multithreaded BLAS in every worker
    only oversubscribes the CPU (and with many workers can exhaust memory).
    Pass the pool as ``executor=`` to :func:`impedance_fit` to reuse it across
    fits; the caller is responsible for shutting it down.

    Parameters
    ----------
    n_jobs : int or None, default -1
        Number of worker processes. ``None`` or ``0`` means 1; a negative value
        means ``os.cpu_count() + 1 + n_jobs`` (at least 1), so ``-1`` uses all
        cores.

    Returns
    -------
    concurrent.futures.ProcessPoolExecutor
        Pool with all workers already started.

    Notes
    -----
    ``OMP_NUM_THREADS``, ``OPENBLAS_NUM_THREADS`` and ``MKL_NUM_THREADS`` are
    set to ``"1"`` in ``os.environ`` while the workers start, then restored to
    their previous values (or removed).
    """
    jobs = _resolve_n_jobs(n_jobs)
    saved = {k: os.environ.get(k) for k in _BLAS_THREAD_VARS}
    os.environ.update(dict.fromkeys(_BLAS_THREAD_VARS, "1"))
    try:
        pool = ProcessPoolExecutor(max_workers=jobs)
        # Workers spawn lazily; start them all while the variables are set.
        list(pool.map(_noop, range(jobs)))
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return pool


def _resolve_n_jobs(n_jobs: int | None) -> int:
    """Map ``n_jobs`` to a worker count (None/0 -> 1, -k -> cpu_count + 1 - k)."""
    if n_jobs is None or n_jobs == 0:
        return 1
    if n_jobs < 0:
        return max(1, (os.cpu_count() or 1) + 1 + n_jobs)
    return n_jobs


def _run_restarts(
    problem: _FitProblem,
    inits: list[np.ndarray],
    n_jobs: int | None,
    executor: Executor | None,
    early_stop: int | None,
    early_stop_rtol: float = 1e-6,
) -> list[dict[str, Any]]:
    """Run restarts in order (or in parallel batches) with optional early stop.

    With ``early_stop=k`` the loop ends once ``k`` restarts reached the best
    cost within ``early_stop_rtol``. Parallel runs check after each batch
    (``n_jobs`` restarts, or ``os.cpu_count()`` with a supplied *executor* and
    ``n_jobs == 1``). A pool created here is shut down before returning.
    """
    results: list[dict[str, Any]] = []

    def converged() -> bool:
        """Return whether ``early_stop`` restarts share the best cost."""
        if early_stop is None:
            return False
        costs = [r["cost"] for r in results if "cost" in r]
        if len(costs) < early_stop:
            return False
        best = min(costs)
        # Absolute floor: on noise-free data all costs are ~1e-30 round-off.
        tol = max(early_stop_rtol * best, 1e-20)
        n_best = sum(c <= best + tol for c in costs)
        return n_best >= early_stop

    jobs = _resolve_n_jobs(n_jobs)
    if executor is None and jobs == 1:
        for init in inits:
            results.append(_solve_restart(problem, init))
            if converged():
                break
        return results

    own_pool = executor is None
    pool = make_process_pool(jobs) if own_pool else executor
    batch = jobs if jobs > 1 else (os.cpu_count() or 1)
    try:
        for start in range(0, len(inits), batch):
            chunk = inits[start : start + batch]
            results.extend(pool.map(_solve_restart, repeat(problem), chunk))
            if converged():
                break
    finally:
        if own_pool:
            pool.shutdown()
    return results


def impedance_fit(
    data: EISData,
    n_groups: int = 2,
    complex_fit: bool = False,
    initial_parameters: CircuitParameters | None = None,
    n_restarts: int = 1,
    seed: int | None = None,
    sigma: float = 0.5,
    tau_bounds: tuple[float, float] | None = None,
    penalty_margin_weight: float = 1.0,
    penalty_weight: float | None = None,
    z_margin_frac: float = 0.0,
    z_margin_abs: float = 0.0,
    max_nfev: int | None = None,
    polish: bool = True,
    polish_maxiter: int = 50,
    constraint_tol: float = 1e-9,
    inductance_bounds: bool = False,
    inductance_margin: float = 2.0,
    resistance_source_bounds: tuple[float, float] | None = None,
    resistance_branch_bounds: tuple[float, float] | None = None,
    exponent_bounds: tuple[float, float] | None = None,
    inductance_exponent_bounds: tuple[float, float] | None = None,
    weighting: str = "modulus",
    n_jobs: int | None = 1,
    executor: Executor | None = None,
    early_stop: int | None = None,
) -> FitResults:
    r"""Fit the equivalent circuit to ``data`` by bounded TRF least squares.

    The circuit is :math:`R_s` in series with an inductor (CPE-type,
    :math:`L (j\omega)^{n_L}`) and ``n_groups`` parallel R–CPE branches. Each
    restart runs SciPy's trust-region-reflective solver with an analytic
    Jacobian; the lowest-cost solution is returned together with linearized
    parameter uncertainties.

    Parameters
    ----------
    data : EISData
        Experimental EIS spectrum (``Freq`` in Hz, ``Real`` and ``Imag`` in Ω).
    n_groups : int, default 2
        Number of parallel R–CPE branches.
    complex_fit : bool, default False
        If ``True``, fit the CPE exponents :math:`e_i` and :math:`n_L`
        (bounded to ``0.2 <= e <= 1``; :math:`n_L` to ``(0.7, 1.0)`` by
        default, see ``inductance_exponent_bounds``). If ``False``, fix
        :math:`e_i = 1` and :math:`n_L = 1` (ideal capacitors and inductor).
        Residuals are complex (stacked real/imaginary) in both cases.
    initial_parameters : CircuitParameters, optional
        Starting point; its branch count must match ``n_groups`` (checked with
        ``assert``). It is copied and sorted by :math:`\tau`, not modified.
        ``None`` starts every parameter at ``1e-3`` except :math:`L`, which
        starts at :func:`estimate_inductance_from_imag`; the start is then
        projected onto the bounds and constraints.
    n_restarts : int, default 1
        Total number of starts (at least 1): the unperturbed start point plus
        ``n_restarts - 1`` log-normal perturbations of it. The lowest cost is
        kept. Perturbations keep the sign of every parameter (e.g. a negative
        :math:`L`); with ``complex_fit`` about half of the perturbed starts
        draw the exponents uniformly within their bounds, the others near the
        start values.
    seed : int, optional
        Seed for :func:`numpy.random.default_rng` used for the restarts.
        ``None`` gives non-reproducible restarts.
    sigma : float, default 0.5
        Log-normal scale of the restart perturbations.
    tau_bounds : tuple of float, optional
        ``(tau_min, tau_max)`` in s for each :math:`\tau_i` (swapped if given
        in reverse order). ``None`` (default) leaves the time constants
        unbounded (apart from the tiny positive floor on :math:`\tau_1`). Use
        ``tau_bounds=tau_bounds_from_frequencies(data.Freq)`` to restrict them
        to the measured window.
    penalty_margin_weight : float, default 1.0
        Weight on soft Nyquist margin violations (see Notes). Ignored when
        ``penalty_weight`` is given.
    penalty_weight : float, optional
        Alias of ``penalty_margin_weight``; takes precedence when not ``None``.
    z_margin_frac : float, default 0.0
        Soft margin as a fraction of the :math:`Z'` span (see Notes).
    z_margin_abs : float, default 0.0
        Soft margin in Ω; the larger of the two margins is used.
    max_nfev : int, optional
        Maximum function evaluations per restart. ``None`` means
        ``500 * n_params`` (number of optimizer parameters).
    polish : bool, default True
        If the projected TRF solution still violates the hard center
        constraints, polish it with SciPy's ``trust-constr`` method.
    polish_maxiter : int, default 50
        Iteration cap for polishing.
    constraint_tol : float, default 1e-9
        Tolerance (Ω) for declaring the hard constraints satisfied.
    inductance_bounds : bool, default False
        If ``False``, :math:`L \ge 0` only. If ``True``, set signed box bounds
        on :math:`L` from a frequency-weighted slope of :math:`\mathrm{Im}(Z)`
        vs :math:`\omega` (see :func:`estimate_inductance_from_imag` and
        :func:`inductance_bounds_from_estimate`); this allows a negative
        :math:`L`. If the estimate is negligible, the default applies.
    inductance_margin : float, default 2.0
        Margin factor for the :math:`L` bounds from the slope estimate.
    resistance_source_bounds : tuple of float, optional
        ``(R_s_min, R_s_max)`` box bounds in Ω for the source resistance
        :math:`R_s`. The lower bound is never below the strictly positive
        floor :math:`10^{-12}\,(\max Z' - \min Z')` (a smaller ``R_s_min``
        keeps the floor), so relative errors in :math:`R_s` stay finite.
        ``None`` uses the floor and an upper bound of :math:`\infty`.
    resistance_branch_bounds : tuple of float, optional
        ``(R_min, R_max)`` uniform box bounds in Ω for all branch resistances.
        If ``None``, defaults to :math:`(\epsilon, \infty)` with
        :math:`\epsilon = 10^{-12}\,(\max Z' - \min Z')`. A positive
        ``R_min`` replaces :math:`\epsilon`; ``R_min <= 0`` keeps it.
    exponent_bounds : tuple of float, optional
        ``(e_min, e_max)`` uniform box bounds for all CPE exponents (branch and,
        unless ``inductance_exponent_bounds`` is given, inductor). Only used
        when ``complex_fit=True``. Overrides the default ``(0.2, 1.0)``.
    inductance_exponent_bounds : tuple of float, optional
        Box bounds for the inductor exponent :math:`n_L`, intersected with
        ``(0.2, 1.0)``. If ``None`` (default), ``exponent_bounds`` applies when
        given, otherwise ``(0.7, 1.0)``. Pass ``(0.2, 1.0)`` for the pre-0.3
        behaviour. Only used when ``complex_fit=True``.
    weighting : {"modulus", "unit"}, default "modulus"
        ``"modulus"``: residuals divided by :math:`|Z_k|`, i.e. the relative
        error is minimised and every decade of frequency contributes.
        ``"unit"``: unweighted residuals (behaviour of version 0.1).
    n_jobs : int, default 1
        Worker processes for the restarts (``1`` or ``None``: sequential;
        ``-1``: all cores). Results do not depend on ``n_jobs`` unless
        ``early_stop`` is set. On Windows/macOS the calling script needs an
        ``if __name__ == "__main__":`` guard.
    executor : concurrent.futures.Executor, optional
        Reuse an existing pool (e.g. from :func:`make_process_pool`) instead of
        creating one per call. It is not shut down.
    early_stop : int, optional
        Stop once this many restarts have reached the best cost (relative
        tolerance ``1e-6``). ``None`` (default) runs all restarts.

    Returns
    -------
    FitResults
        Optimized parameters (branch capacitances in F, or CPE coefficients
        when ``complex_fit``), the unweighted residual vector ``fun`` (stacked
        real and imaginary parts, Ω), the weighted ``cost``, summed ``nfev`` and
        ``njev`` over all restarts, and the parameter uncertainties
        ``covariance``, ``standard_errors`` and ``correlation``.

    Raises
    ------
    ValueError
        If ``n_groups < 1``, ``weighting`` is unknown, or the sorted time
        constants of ``initial_parameters`` are not non-decreasing from 0
        (i.e. some :math:`\tau_i < 0`).
    RuntimeError
        If every restart failed with a solver error.

    Warns
    -----
    UserWarning
        If some (but not all) restarts failed, or a constraint polish failed.

    See Also
    --------
    initial_fit : Heuristic start values.
    tau_bounds_from_frequencies : Time-constant bounds from the data window.
    blowing_bubbles.classes.data_classes.EISData.fit
    blowing_bubbles.classes.data_classes.fit_circuit

    Notes
    -----
    **Objective.** With measured :math:`Z_k` and model
    :math:`\hat{Z}_k = Z_{\mathrm{pred}}(\omega_k; \mathbf{p})`, the residuals
    are the stacked real and imaginary parts of
    :math:`w_k (\hat{Z}_k - Z_k)`, with :math:`w_k = 1/|Z_k|` (``"modulus"``)
    or :math:`w_k = 1` (``"unit"``).

    **Constraints.** Two hard constraints keep the Nyquist arcs inside the
    measured :math:`Z'` range:

    .. math::

        R_s \le \min Z', \qquad R_s + \sum_i R_i - R_n/2 \le \max Z'.

    They are enforced by projecting every start point and every TRF solution
    (and, if needed, by polishing). Two soft penalty residuals, each equal to
    ``w_pen * max(0, violation)``, push the solver to satisfy them with the
    margin ``max(z_margin_abs, z_margin_frac * (max Z' - min Z'))``. The
    penalty weight is ``penalty_weight`` (or ``penalty_margin_weight``) times
    the median of :math:`w_k |Z_k|`.

    **Reparameterization.** Capacitances enter as non-negative increments
    :math:`\Delta\tau_i` with :math:`\tau_i = \sum_{j \le i} \Delta\tau_j` and
    :math:`C_i = \tau_i / R_i`, which enforces ordered time constants. The
    branch resistances :math:`R_i` and :math:`\tau_1` have tiny strictly
    positive default lower bounds (:math:`10^{-12}` of the :math:`Z'` span and
    of :math:`1/\omega_{\max}`), so every :math:`C_i` is finite and positive.
    With ``tau_bounds``, the TRF box bounds only enforce
    :math:`\tau_1 \ge \tau_{\min}` and :math:`\Delta\tau_i \le \tau_{\max}`;
    the full range :math:`\tau_i \in [\tau_{\min}, \tau_{\max}]` is enforced by
    the projection.

    **Uncertainties.** ``covariance``, ``standard_errors`` and ``correlation``
    refer to the reported parameters (:meth:`CircuitParameters.to_array`
    order). They come from a local linearization at the best solution,
    :math:`s^2 (J^T J)^{+}`, with :math:`J` the Jacobian of the *weighted*
    impedance residuals and :math:`s^2` their sum of squares divided by
    :math:`2M - p` (:math:`M` frequencies, :math:`p` free parameters). This
    assumes independent residuals whose scale matches ``weighting`` (e.g.
    relative errors for ``"modulus"``) and a well-converged fit; it ignores
    active box bounds, the Nyquist center constraints and the soft margin
    penalty, so parameters sitting on a bound get unreliable errors.
    Parameters fixed by the model (:math:`n_L`, :math:`e_i` with
    ``complex_fit=False``) have zero variance.

    **Inductor exponent.** A low :math:`n_L` gives the inductor term a large
    real part, which together with a branch whose apex lies above
    :math:`f_{\max}` can absorb the series resistance; hence the default
    bounds ``(0.7, 1.0)``.

    ``impendance_fit`` is a backward-compatible alias of this function.
    """
    if n_groups < 1:
        raise ValueError(f"n_groups must be >= 1, got {n_groups}.")

    rng = np.random.default_rng(seed)

    if initial_parameters is not None:
        initial_parameters = initial_parameters.copy()  # do not mutate caller's
        initial_parameters.sort_by_tau()

    X = data.Freq
    Y = data.Real + 1j * data.Imag

    if weighting == "modulus":
        W = 1.0 / np.maximum(np.abs(Y), np.finfo(float).tiny)
    elif weighting == "unit":
        W = np.ones(Y.shape, dtype=float)
    else:
        raise ValueError(f"Unknown weighting {weighting!r}; use 'modulus' or 'unit'.")

    z_min = float(np.min(data.Real))
    z_max = float(np.max(data.Real))
    z_span = max(z_max - z_min, np.finfo(float).eps)
    margin = max(z_margin_abs, z_margin_frac * z_span)

    pw = penalty_weight if penalty_weight is not None else penalty_margin_weight
    margin_weight = pw * float(np.median(np.abs(Y) * W))

    head = 3 if complex_fit else 2
    r_slice = slice(head, head + n_groups)
    dtau_slice = slice(head + n_groups, head + 2 * n_groups)
    e_slice = slice(head + 2 * n_groups, head + 3 * n_groups)
    n_params = head + (2 + complex_fit) * n_groups

    if max_nfev is None:
        max_nfev = 500 * n_params

    resolved_tau_bounds = _normalize_tau_bounds(tau_bounds)

    L_est = estimate_inductance_from_imag(data.Freq, data.Imag)
    l_lo, l_hi = (
        inductance_bounds_from_estimate(L_est, margin=inductance_margin)
        if inductance_bounds
        else (None, None)
    )

    if initial_parameters is None:
        parameters_init = np.ones(n_params) * 1e-3
        parameters_init[1] = L_est
    else:
        R = np.asarray(initial_parameters.resistances, dtype=float)
        tau = np.asarray(initial_parameters.taus, dtype=float)
        assert R.shape == (n_groups,), (
            "initial_parameters.resistances has shape "
            f"{R.shape}, expected ({n_groups},)"
        )
        assert tau.shape == (n_groups,), (
            f"initial_parameters.taus has shape {tau.shape}, expected ({n_groups},)"
        )

        dtau = np.diff(tau, prepend=0.0)
        if np.any(dtau < 0):
            raise ValueError(
                f"initial_parameters.taus must be non-decreasing (got {tau.tolist()})."
            )

        parameters_init = np.empty(n_params)
        parameters_init[0] = initial_parameters.resistance_source
        parameters_init[1] = initial_parameters.inductance
        parameters_init[r_slice] = R
        parameters_init[dtau_slice] = dtau
        if complex_fit:
            e = np.asarray(initial_parameters.exponents, dtype=float)
            assert e.shape == (n_groups,), (
                "initial_parameters.exponents has shape "
                f"{e.shape}, expected ({n_groups},)"
            )
            # Clipped to the (possibly user-set) exponent bounds below.
            parameters_init[2] = initial_parameters.inductance_exponent
            parameters_init[e_slice] = e

        if resolved_tau_bounds is not None:
            tau_lo, tau_hi = resolved_tau_bounds
            parameters_init[dtau_slice] = _project_dtau(
                parameters_init[dtau_slice], tau_lo, tau_hi
            )

    lower = np.zeros(n_params)
    upper = np.full(n_params, np.inf)
    if complex_fit:
        lower[2] = _CPE_EXPONENT_MIN
        upper[2] = _CPE_EXPONENT_MAX
        lower[e_slice] = _CPE_EXPONENT_MIN
        upper[e_slice] = _CPE_EXPONENT_MAX

    # Strictly positive R_i and tau_1 (hence every tau_i) keep C_i = tau_i / R_i
    # finite and positive, also after _project_feasible clips to the bounds.
    r_floor = _POSITIVE_FLOOR_REL * z_span
    lower[0] = r_floor
    f_max = float(np.max(X)) if np.size(X) else 0.0
    f_ref = f_max if f_max > 0 else 1.0
    tau_floor = _POSITIVE_FLOOR_REL / (2.0 * np.pi * f_ref)
    r_user = resistance_branch_bounds or (None, None)
    lower[r_slice] = _positive_floor(r_floor, *r_user)
    tau_user = resolved_tau_bounds or (None, None)
    lower[dtau_slice.start] = _positive_floor(tau_floor, *tau_user)

    if resolved_tau_bounds is not None and n_groups > 0:
        tau_lo, tau_hi = resolved_tau_bounds
        lower[dtau_slice.start] = max(lower[dtau_slice.start], tau_lo)
        upper[dtau_slice] = np.minimum(upper[dtau_slice], tau_hi)

    if l_lo is not None and l_hi is not None:
        lower[1] = l_lo
        upper[1] = l_hi
        parameters_init[1] = np.clip(parameters_init[1], l_lo, l_hi)

    # User-specified parameter bounds (override defaults if provided)
    if resistance_source_bounds is not None:
        rs_lo, rs_hi = resistance_source_bounds
        lower[0] = max(lower[0], rs_lo)
        upper[0] = min(upper[0], rs_hi)
        parameters_init[0] = np.clip(parameters_init[0], rs_lo, rs_hi)

    if resistance_branch_bounds is not None:
        r_lo, r_hi = resistance_branch_bounds
        lower[r_slice] = np.maximum(lower[r_slice], r_lo)
        upper[r_slice] = np.minimum(upper[r_slice], r_hi)
        parameters_init[r_slice] = np.clip(parameters_init[r_slice], r_lo, r_hi)

    if exponent_bounds is not None and complex_fit:
        exp_lo, exp_hi = exponent_bounds
        lower[2] = exp_lo
        upper[2] = exp_hi
        parameters_init[2] = np.clip(parameters_init[2], exp_lo, exp_hi)
        lower[e_slice] = exp_lo
        upper[e_slice] = exp_hi
        parameters_init[e_slice] = np.clip(parameters_init[e_slice], exp_lo, exp_hi)

    # Inductor exponent: kept near an inductor by default. With n_L free down to
    # 0.2 the term L (j omega)^n_L has a large real part and, together with a
    # branch whose apex lies above f_max, can absorb the series resistance
    # (R_s -> 0 in 24 % of the EK3 fits and most noisy synthetic fits).
    nl_bounds = inductance_exponent_bounds
    if nl_bounds is None and exponent_bounds is None:
        nl_bounds = _DEFAULT_INDUCTANCE_EXPONENT_BOUNDS
    if nl_bounds is not None and complex_fit:
        nl_lo, nl_hi = nl_bounds
        lower[2] = max(nl_lo, _CPE_EXPONENT_MIN)
        upper[2] = min(nl_hi, _CPE_EXPONENT_MAX)
        parameters_init[2] = np.clip(parameters_init[2], lower[2], upper[2])

    problem = _FitProblem(
        X,
        Y,
        W,
        complex_fit=complex_fit,
        n_groups=n_groups,
        z_min=z_min,
        z_max=z_max,
        margin=margin,
        margin_weight=margin_weight,
        lower=lower,
        upper=upper,
        tau_bounds=resolved_tau_bounds,
        max_nfev=max_nfev,
        polish=polish,
        polish_maxiter=polish_maxiter,
        constraint_tol=constraint_tol,
    )

    base_init = problem.project(parameters_init.copy())
    # Half of the restarts keep the exponents local, half explore globally.
    # Drawn up front so the result does not depend on n_jobs.
    inits = [base_init] + [
        problem.project(
            _perturb_init(
                base_init,
                rng,
                sigma,
                lower,
                upper,
                complex_fit=complex_fit,
                n_groups=n_groups,
                local_exponents=(i % 2 == 1),
            )
        )
        for i in range(1, max(1, n_restarts))
    ]
    results = _run_restarts(problem, inits, n_jobs, executor, early_stop)

    failures = [r["error"] for r in results if "error" in r]
    solved = [r for r in results if "error" not in r]
    if not solved:
        raise RuntimeError(
            f"All {len(failures)} restarts failed to produce a fit. "
            f"First error: {failures[0] if failures else 'unknown'}"
        )
    if failures:
        warnings.warn(
            f"{len(failures)} of {len(results)} restarts failed "
            f"(first: {failures[0]}).",
            stacklevel=2,
        )
    best = solved[0]
    for r in solved[1:]:
        if r["cost"] < best["cost"]:
            best = r

    best_x = best["x"].copy()
    diff = problem.model(best_x) - Y
    polish_count = sum(r["polished"] for r in solved)
    res_ls = OptimizeResult(
        x=best_x,
        fun=np.concatenate([diff.real, diff.imag]),
        cost=best["cost"],
        success=True,
        status=best["status"],
        message=(
            f"hybrid TRF ({len(results)} of {max(1, n_restarts)} restarts, "
            f"{polish_count} polished)"
        ),
        nfev=sum(r["nfev"] for r in solved),
        njev=sum(r["njev"] for r in solved),
        jac=None,
        grad=None,
        optimality=None,
        active_mask=None,
    )

    # Before x is rewritten to capacitances: covariance needs the dtau layout.
    covariance = _parameter_covariance(problem, best_x)

    tau = np.cumsum(res_ls.x[dtau_slice])
    R = res_ls.x[r_slice]
    res_ls.x[dtau_slice] = tau / R

    return FitResults(res_ls, n_groups, complex_fit, covariance=covariance)


impendance_fit = impedance_fit
"""Backward-compatible (misspelled) alias of :func:`impedance_fit`."""
