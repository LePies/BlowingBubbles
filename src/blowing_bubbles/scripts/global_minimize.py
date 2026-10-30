r"""Bounded nonlinear least-squares fitting of the equivalent circuit to EIS data.

Minimizes residuals between measured and model impedance
:math:`Z_{\mathrm{pred}}(\omega_k; \mathbf{p})` at frequencies :math:`\omega_k`.
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
    r"""Heuristic initial guess from Nyquist peak structure in ``-Im(Z)``.

    Peaks in the negative imaginary part define branch resistances (arc widths in
    :math:`Z'`) and time constants :math:`\tau \approx 1/f_{\mathrm{peak}}`.
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
    all_bounds = [0] + valleys + [len(eis.Real) - 1]

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
    """Characteristic τ scale from measured frequencies: (1/f_max, 1/f_min)."""
    f = np.asarray(freq, dtype=float)
    f_pos = f[f > 0]
    if f_pos.size == 0:
        raise ValueError("Need at least one positive frequency for tau bounds.")
    return (1.0 / float(np.max(f_pos)), 1.0 / float(np.min(f_pos)))


_INDUCTANCE_EST_TOL = 1e-15
_CPE_EXPONENT_MIN = 0.2
_CPE_EXPONENT_MAX = 1.0
# Default lower bounds on R_i and tau_1, relative to the data scales (|Z'| span
# and 1/omega_max). Strictly positive so C_i = tau_i / R_i stays finite and > 0.
_POSITIVE_FLOOR_REL = 1e-12
# Smallest magnitude of a perturbed restart start value.
_PERTURB_MIN_ABS = 1e-9


def estimate_inductance_from_imag(
    freq: np.ndarray,
    imag: np.ndarray,
) -> float:
    r"""Estimate :math:`L` from a weighted slope of :math:`\mathrm{Im}(Z)`.

    Regresses :math:`\mathrm{Im}(Z)` against :math:`\omega`. For
    :math:`Z = R_s + j\omega L + \cdots` with :math:`n_L = 1`,
    :math:`\mathrm{Im}(Z) \approx \omega L`, so :math:`L \approx d\mathrm{Im}/d\omega`.
    Weights favor higher frequencies where the inductive tail dominates.
    """
    f = np.asarray(freq, dtype=float)
    im = np.asarray(imag, dtype=float)
    mask = f > 0
    if np.count_nonzero(mask) < 2:
        return 0.0
    f = f[mask]
    im = im[mask]
    omega = 2.0 * np.pi * f
    w = f / float(np.max(f))
    w_mean_o = np.average(omega, weights=w)
    w_mean_i = np.average(im, weights=w)
    num = np.average((omega - w_mean_o) * (im - w_mean_i), weights=w)
    den = np.average((omega - w_mean_o) ** 2, weights=w)
    if den <= 0:
        return 0.0
    return float(num / den)


def inductance_bounds_from_estimate(
    L_est: float,
    *,
    margin: float = 2.0,
) -> tuple[float | None, float | None]:
    r"""Signed box bounds on :math:`L` from a slope estimate.

    If :math:`L_{\mathrm{est}} < 0`: ``margin * L_est < L < 0``.
    If :math:`L_{\mathrm{est}} > 0`: ``0 < L < margin * L_est``.
    If :math:`|L_{\mathrm{est}}|` is negligible, returns ``(None, None)``
    (no custom bounds).
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
    """Strictly positive lower bound that never tightens user-supplied bounds.

    Returns ``max(floor', user_lo)`` where ``floor'`` is *floor* reduced, if
    needed, to stay at or below a positive ``user_lo`` and below a positive
    ``user_hi``. A user lower bound of ``0`` (or less) is replaced by the floor.
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
    """Random restart start point around *base*, clipped to ``[lower, upper]``.

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
            init[e_slice] = base[e_slice] + rng.normal(
                scale=0.5 * sigma, size=n_groups
            )
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
    """Map Δτ so cumsum(Δτ) lies in [tau_lo, tau_hi] with Δτ >= 0."""
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
    """Hard limits: R_s <= z_min and R_s + sum(R) - R_N/2 <= z_max.

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
    """Clip to box bounds and satisfy Nyquist center / τ constraints."""
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
        state = self.__dict__.copy()
        state["_key"] = None
        state["_state"] = None
        return state

    def _forward(self, p: np.ndarray) -> tuple:
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
        return self._state

    def model(self, p: np.ndarray) -> np.ndarray:
        return self._forward(np.asarray(p, dtype=float))[-1]

    def _margin_terms(self, p: np.ndarray) -> tuple[float, float]:
        c_lo, c_hi = _center_positions(p, self.r_slice)
        return self.margin + (c_lo - self.z_min), self.margin - (self.z_max - c_hi)

    def residuals(self, p: np.ndarray) -> np.ndarray:
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
        r = self.residuals(p)
        return 0.5 * float(np.dot(r, r))

    def project(self, p: np.ndarray) -> np.ndarray:
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
        return _constraints_satisfied(
            p, self.r_slice, self.z_min, self.z_max, self.constraint_tol
        )

    def polish_if_infeasible(
        self, p: np.ndarray, cost: float
    ) -> tuple[np.ndarray, float]:
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
    """One TRF solve from *init*, projected (and polished) onto the constraints."""
    try:
        res_ls = least_squares(
            problem.residuals,
            init,
            jac=problem.jacobian,
            method="trf",
            bounds=(problem.lower, problem.upper),
            max_nfev=problem.max_nfev,
            x_scale="jac",
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
    r"""Linearized covariance of the reported parameters at the solution *x*.

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
    return None


def make_process_pool(n_jobs: int | None = -1) -> ProcessPoolExecutor:
    """Process pool for restarts whose workers use single-threaded BLAS.

    Restart solves work on tiny matrices, so multithreaded BLAS in every worker
    only oversubscribes the CPU (and with many workers can exhaust memory).
    Pass the pool as ``executor=`` to reuse it across fits.
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
    cost within ``early_stop_rtol``. Parallel runs check after each batch.
    """
    results: list[dict[str, Any]] = []

    def converged() -> bool:
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
    weighting: str = "modulus",
    n_jobs: int | None = 1,
    executor: Executor | None = None,
    early_stop: int | None = None,
) -> FitResults:
    r"""Bounded TRF least-squares fit of the equivalent circuit to ``data``.

    **Objective.** With measured :math:`Z_k` and model
    :math:`\hat{Z}_k = Z_{\mathrm{pred}}(\omega_k; \mathbf{p})`, residuals are the
    stacked real and imaginary parts of the difference. Soft penalties keep
    Nyquist arc centers inside the
    measured :math:`Z'` range.

    **Reparameterization.** Capacitances enter as non-negative increments
    :math:`\Delta\tau_i` with :math:`\tau_i = \sum_{j \le i} \Delta\tau_j` and
    :math:`C_i = \tau_i / R_i`, which enforces ordered time constants. The
    branch resistances :math:`R_i` and :math:`\tau_1` have tiny strictly
    positive default lower bounds (:math:`10^{-12}` of the :math:`Z'` span and
    of :math:`1/\omega_{\max}`), so every :math:`C_i` is finite and positive.

    **Uncertainties.** The returned :class:`FitResults` carries
    ``covariance``, ``standard_errors`` and ``correlation`` of the reported
    parameters (:meth:`CircuitParameters.to_array` order). They come from a
    local linearization at the best solution,
    :math:`s^2 (J^T J)^{+}`, with :math:`J` the Jacobian of the *weighted*
    impedance residuals and :math:`s^2` their sum of squares divided by
    :math:`2M - p` (:math:`M` frequencies, :math:`p` free parameters). This
    assumes independent residuals whose scale matches ``weighting`` (e.g.
    relative errors for ``"modulus"``) and a well-converged fit; it ignores
    active box bounds, the Nyquist center constraints and the soft margin
    penalty, so parameters sitting on a bound get unreliable errors.
    Parameters fixed by the model (:math:`n_L`, :math:`e_i` with
    ``complex_fit=False``) have zero variance.

    Parameters
    ----------
    data : EISData
        Experimental EIS spectrum.
    n_groups : int
        Number of parallel R–CPE branches.
    complex_fit : bool
        If ``True``, fit the CPE exponents (bounded to ``0.2 <= e <= 1``); if
        ``False``, fix :math:`e_i = 1` and :math:`n_L = 1` (ideal capacitors and
        inductor). Residuals are complex (stacked real/imaginary) in both cases.
    initial_parameters : CircuitParameters, optional
        Starting point; branch count must match ``n_groups``.
    n_restarts : int
        Number of log-normal perturbation restarts (best cost is kept).
        Perturbations keep the sign of every parameter (e.g. a negative
        :math:`L`); with ``complex_fit`` half of the restarts draw the
        exponents uniformly within their bounds.
    seed : int, optional
        RNG seed for restarts.
    sigma : float
        Log-normal scale for restart perturbations.
    tau_bounds : tuple of float, optional
        ``(tau_min, tau_max)`` for each :math:`\tau_i`. ``None`` (default)
        leaves the time constants unbounded (apart from the tiny positive
        floor on :math:`\tau_1`). Use :func:`tau_bounds_from_frequencies`
        to restrict them to the measured window,
        ``tau_bounds=tau_bounds_from_frequencies(data.Freq)``.
    penalty_margin_weight, penalty_weight : float
        Weight on soft Nyquist margin violations.
    z_margin_frac, z_margin_abs : float
        Optional soft margin on the constraints :math:`R_s \le \min Z'` and
        :math:`R_s + \sum R_i - R_n/2 \le \max Z'` (default 0).
    max_nfev : int, optional
        Maximum function evaluations per restart (default ``500 * n_params``).
    polish : bool
        Run a trust-region polish step if hard constraints are violated.
    polish_maxiter : int
        Iteration cap for polishing.
    constraint_tol : float
        Tolerance for declaring constraints satisfied.
    inductance_bounds : bool
        Default ``False`` (only :math:`L \ge 0`). If ``True``, set signed box
        bounds on :math:`L` from a frequency-weighted slope of
        :math:`\mathrm{Im}(Z)` vs :math:`\omega` (see
        :func:`estimate_inductance_from_imag`).
    inductance_margin : float
        Margin factor for ``L`` bounds from the slope estimate (default ``2.0``).
    resistance_source_bounds : tuple of float, optional
        ``(R_s_min, R_s_max)`` box bounds for the source resistance :math:`R_s`.
        If ``None``, defaults to ``(0, \infty)``.
    resistance_branch_bounds : tuple of float, optional
        ``(R_min, R_max)`` uniform box bounds for all branch resistances.
        If ``None``, defaults to ``(\epsilon, \infty)`` with
        :math:`\epsilon = 10^{-12}\,(\max Z' - \min Z')`. A positive
        ``R_min`` replaces :math:`\epsilon`; ``R_min <= 0`` keeps it.
    exponent_bounds : tuple of float, optional
        ``(e_min, e_max)`` uniform box bounds for all CPE exponents (branch and
        inductance exponent). Only used when ``complex_fit=True``. Overrides
        the default ``(0.2, 1.0)``. If ``None``, the default bounds are used.
    weighting : {"modulus", "unit"}
        ``"modulus"`` (default): residuals divided by :math:`|Z_k|`, i.e. the
        relative error is minimised and every decade of frequency contributes.
        ``"unit"``: unweighted residuals (behaviour of version 0.1).
    n_jobs : int, optional
        Worker processes for the restarts (``1``: sequential, default; ``-1``:
        all cores). Results do not depend on ``n_jobs`` unless ``early_stop`` is
        set. On Windows/macOS the calling script needs an
        ``if __name__ == "__main__":`` guard.
    executor : concurrent.futures.Executor, optional
        Reuse an existing process pool instead of creating one per call.
    early_stop : int, optional
        Stop once this many restarts have reached the best cost (relative
        tolerance ``1e-6``). ``None`` (default) runs all restarts.

    Returns
    -------
    FitResults
        Optimized parameters, residual vectors and parameter uncertainties.

    Notes
    -----
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
            "initial_parameters.taus has shape "
            f"{tau.shape}, expected ({n_groups},)"
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
        parameters_init[r_slice] = np.clip(
            parameters_init[r_slice], r_lo, r_hi
        )

    if exponent_bounds is not None and complex_fit:
        exp_lo, exp_hi = exponent_bounds
        lower[2] = exp_lo
        upper[2] = exp_hi
        parameters_init[2] = np.clip(parameters_init[2], exp_lo, exp_hi)
        lower[e_slice] = exp_lo
        upper[e_slice] = exp_hi
        parameters_init[e_slice] = np.clip(
            parameters_init[e_slice], exp_lo, exp_hi
        )

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
