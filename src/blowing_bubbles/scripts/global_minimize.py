r"""Bounded nonlinear least-squares fitting of the equivalent circuit to EIS data.

Minimizes residuals between measured and model impedance
:math:`Z_{\mathrm{pred}}(\omega_k; \mathbf{p})` at frequencies :math:`\omega_k`.
"""

import numpy as np
from scipy.optimize import (
    Bounds,
    LinearConstraint,
    OptimizeResult,
    least_squares,
    minimize,
)
from scipy.signal import find_peaks

from blowing_bubbles.circuit_impedance import (
    circuit_impedance,
    circuit_impedance_parameter_gradient,
)
from blowing_bubbles.classes.circuit_classes import CircuitParameters, FitResults
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

    resistances = np.array(mountain_widths)
    taus = 1 / np.asarray(mountain_center_freqs, dtype=float)
    if tau_bounds is not None:
        taus = np.clip(taus, tau_bounds[0], tau_bounds[1])
    capacitances = taus / resistances
    resistances_source = eis.Real[0]
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
    """Hard limits: R_s + R_1/2 >= z_min and R_s + sum(R) - R_N/2 <= z_max."""
    A = np.zeros((2, n_params))
    A[0, 0] = -1.0
    A[0, head] = -0.5
    A[1, 0] = 1.0
    A[1, head : head + n_groups] = 1.0
    A[1, head + n_groups - 1] -= 0.5
    return LinearConstraint(A, -np.inf * np.ones(2), np.array([-z_min, z_max]))


def _center_positions(p: np.ndarray, r_slice: slice) -> tuple[float, float]:
    Rs = p[0]
    R = p[r_slice]
    c_lo = Rs + 0.5 * R[0]
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
    return c_lo >= z_min - tol and c_hi <= z_max + tol


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
        R = p[r_slice].copy()
        Rs = p[0]

        if Rs + 0.5 * R[0] < z_min:
            p[0] = max(lower[0], z_min - 0.5 * R[0])

        c_hi = p[0] + float(np.sum(p[r_slice])) - 0.5 * p[r_slice][-1]
        if c_hi > z_max:
            excess = c_hi - z_max
            r_sum = float(np.sum(p[r_slice]))
            if r_sum > 0:
                scale = max(0.01, 1.0 - excess / r_sum)
                p[r_slice] = np.maximum(p[r_slice] * scale, lower[r_slice])

        if p[0] + 0.5 * p[r_slice][0] < z_min:
            p[0] = max(lower[0], z_min - 0.5 * p[r_slice][0])

        p = np.clip(p, lower, upper)

    if tau_bounds is not None and dtau_slice is not None:
        tau_lo, tau_hi = tau_bounds
        p[dtau_slice] = _project_dtau(p[dtau_slice], tau_lo, tau_hi)
        p = np.clip(p, lower, upper)

    return p


def impendance_fit(
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
    z_margin_frac: float = 0.05,
    z_margin_abs: float = 0.0,
    max_nfev: int | None = None,
    polish: bool = True,
    polish_maxiter: int = 50,
    constraint_tol: float = 1e-9,
    inductance_bounds: bool = True,
    inductance_margin: float = 2.0,
) -> FitResults:
    r"""Bounded TRF least-squares fit of the equivalent circuit to ``data``.

    **Objective.** With measured :math:`Z_k` and model
    :math:`\hat{Z}_k = Z_{\mathrm{pred}}(\omega_k; \mathbf{p})`, residuals are either
    stacked real/imaginary parts (``complex_fit=True``) or magnitudes
    (``complex_fit=False``). Soft penalties keep Nyquist arc centers inside the
    measured :math:`Z'` range.

    **Reparameterization.** Capacitances enter as non-negative increments
    :math:`\Delta\tau_i` with :math:`\tau_i = \sum_{j \le i} \Delta\tau_j` and
    :math:`C_i = \tau_i / R_i`, which enforces ordered time constants.

    Parameters
    ----------
    data : EISData
        Experimental EIS spectrum.
    n_groups : int
        Number of parallel R–CPE branches.
    complex_fit : bool
        If ``True``, fit CPE exponents and use complex residuals; if ``False``, fix
        :math:`e_i = 1` and minimize magnitude error. When ``True``, branch exponents
        and the inductance exponent :math:`n` are bounded to ``0.2 < n < 1``.
    initial_parameters : CircuitParameters, optional
        Starting point; branch count must match ``n_groups``.
    n_restarts : int
        Number of log-normal perturbation restarts (best cost is kept).
    seed : int, optional
        RNG seed for restarts.
    sigma : float
        Log-normal scale for restart perturbations.
    tau_bounds : tuple of float, optional
        ``(tau_min, tau_max)`` for each :math:`\tau_i`; default from data frequencies.
    penalty_margin_weight, penalty_weight : float
        Weight on soft Nyquist margin violations.
    z_margin_frac, z_margin_abs : float
        Margin on :math:`Z'` when enforcing arc-center constraints.
    max_nfev : int, optional
        Maximum function evaluations per restart (default ``500 * n_params``).
    polish : bool
        Run a trust-region polish step if hard constraints are violated.
    polish_maxiter : int
        Iteration cap for polishing.
    constraint_tol : float
        Tolerance for declaring constraints satisfied.
    inductance_bounds : bool
        If ``True``, set signed box bounds on :math:`L` from a frequency-weighted
        slope of :math:`\mathrm{Im}(Z)` vs :math:`\omega` (see
        :func:`estimate_inductance_from_imag`).
    inductance_margin : float
        Margin factor for ``L`` bounds from the slope estimate (default ``1.1``).

    Returns
    -------
    FitResults
        Optimized parameters and residual vectors.
    """
    rng = np.random.default_rng(seed)

    if initial_parameters is not None:
        initial_parameters.sort_by_tau()

    X = data.Freq
    Y = data.Real + 1j * data.Imag

    z_min = float(np.min(data.Real))
    z_max = float(np.max(data.Real))
    z_span = max(z_max - z_min, np.finfo(float).eps)
    margin = max(z_margin_abs, z_margin_frac * z_span)

    pw = penalty_weight if penalty_weight is not None else penalty_margin_weight
    margin_weight = pw * float(np.median(np.abs(Y)))

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
            parameters_init[2] = np.clip(
                initial_parameters.inductance_exponent,
                _CPE_EXPONENT_MIN,
                _CPE_EXPONENT_MAX,
            )
            parameters_init[e_slice] = np.clip(e, _CPE_EXPONENT_MIN, _CPE_EXPONENT_MAX)

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

    if resolved_tau_bounds is not None and n_groups > 0:
        tau_lo, tau_hi = resolved_tau_bounds
        lower[dtau_slice.start] = max(lower[dtau_slice.start], tau_lo)
        upper[dtau_slice] = np.minimum(upper[dtau_slice], tau_hi)

    if l_lo is not None and l_hi is not None:
        lower[1] = l_lo
        upper[1] = l_hi
        parameters_init[1] = np.clip(parameters_init[1], l_lo, l_hi)

    scipy_bounds = Bounds(lower, upper)
    linear_constraint = _build_center_linear_constraint(
        n_params, head, n_groups, z_min, z_max
    )

    def _branch_params(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        resistances = p[r_slice]
        tau = np.cumsum(p[dtau_slice])
        capacitances = tau / resistances
        exponents = p[e_slice] if complex_fit else np.ones(n_groups)
        return resistances, capacitances, exponents

    def _model_func(freq, p: np.ndarray):
        resistance_source = p[0]
        inductance = p[1]
        inductance_exponent = p[2] if complex_fit else 1.0
        resistances, capacitances, exponents = _branch_params(p)
        return circuit_impedance(
            freq,
            resistance_source=resistance_source,
            inductance=inductance,
            inductance_exponent=inductance_exponent,
            resistances=resistances,
            capacitances=capacitances,
            exponents=exponents,
        )

    def _impedance_residuals(p: np.ndarray) -> np.ndarray:
        Y_pred = _model_func(X, p)
        return np.concatenate([Y_pred.real - Y.real, Y_pred.imag - Y.imag])

    def _margin_residuals(p: np.ndarray) -> np.ndarray:
        c_lo, c_hi = _center_positions(p, r_slice)
        pen_margin_lo = max(0.0, margin - (c_lo - z_min))
        pen_margin_hi = max(0.0, margin - (z_max - c_hi))
        return np.array(
            [margin_weight * pen_margin_lo, margin_weight * pen_margin_hi],
            dtype=float,
        )

    _tau_chain = np.tri(n_groups)

    def _impedance_jacobian(p: np.ndarray) -> np.ndarray:
        inductance_exponent = p[2] if complex_fit else 1.0
        resistances, capacitances, exponents = _branch_params(p)
        tau = np.cumsum(p[dtau_slice])
        R = resistances

        g = circuit_impedance_parameter_gradient(
            X,
            resistance_source=p[0],
            inductance=p[1],
            inductance_exponent=inductance_exponent,
            resistances=resistances,
            capacitances=capacitances,
            exponents=exponents,
        )

        dZ_dtau = g["capacitances"] / R
        dZ_dR = g["resistances"] - g["capacitances"] * (tau / R**2)

        M = X.size
        J = np.empty((M, n_params), dtype=complex)
        J[:, 0] = g["resistance_source"]
        J[:, 1] = g["inductance"]
        if complex_fit:
            J[:, 2] = g["inductance_exponent"]
        J[:, r_slice] = dZ_dR
        J[:, dtau_slice] = dZ_dtau @ _tau_chain
        if complex_fit:
            J[:, e_slice] = g["exponents"]

        if complex_fit:
            return np.vstack([J.real, J.imag])

        diff = _model_func(X, p) - Y
        mag = np.abs(diff)
        safe = np.where(mag > 0, mag, 1.0)
        return (np.conj(diff)[:, None] * J).real / safe[:, None]

    def _margin_jacobian_rows(p: np.ndarray) -> np.ndarray:
        rows = np.zeros((2, n_params))
        c_lo, c_hi = _center_positions(p, r_slice)
        pen_margin_lo = margin - (c_lo - z_min)
        pen_margin_hi = margin - (z_max - c_hi)

        if pen_margin_lo > 0:
            rows[0, 0] = -margin_weight
            rows[0, head] = -0.5 * margin_weight

        if pen_margin_hi > 0:
            rows[1, 0] = margin_weight
            for i in range(n_groups - 1):
                rows[1, head + i] = margin_weight
            rows[1, head + n_groups - 1] = 0.5 * margin_weight

        return rows

    _eval_cache: dict = {"p": None, "r": None, "J": None}

    def _evaluate(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = np.concatenate([_impedance_residuals(p), _margin_residuals(p)])
        J = np.vstack([_impedance_jacobian(p), _margin_jacobian_rows(p)])
        return r, J

    def _total_cost(p: np.ndarray) -> float:
        r, _ = _evaluate(p)
        return 0.5 * float(np.dot(r, r))

    def _residuals(p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        r, J = _evaluate(p)
        _eval_cache["p"] = p.copy()
        _eval_cache["r"] = r
        _eval_cache["J"] = J
        return r

    def _jacobian(p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        cached_p = _eval_cache["p"]
        if cached_p is not None and cached_p.shape == p.shape and np.all(cached_p == p):
            return _eval_cache["J"]
        r, J = _evaluate(p)
        _eval_cache["p"] = p.copy()
        _eval_cache["r"] = r
        _eval_cache["J"] = J
        return J

    def _polish_if_infeasible(p: np.ndarray, cost: float) -> tuple[np.ndarray, float]:
        p = _project_feasible(
            p,
            z_min,
            z_max,
            head,
            r_slice,
            lower,
            upper,
            n_groups,
            dtau_slice=dtau_slice,
            tau_bounds=resolved_tau_bounds,
        )
        if _constraints_satisfied(p, r_slice, z_min, z_max, constraint_tol):
            return p, _total_cost(p)

        if not polish:
            return p, cost

        def objective(x: np.ndarray) -> float:
            return _total_cost(x)

        def grad(x: np.ndarray) -> np.ndarray:
            r, J = _evaluate(x)
            return J.T @ r

        try:
            res_opt = minimize(
                objective,
                p,
                method="trust-constr",
                jac=grad,
                bounds=scipy_bounds,
                constraints=[linear_constraint],
                options={"maxiter": polish_maxiter, "verbose": 0},
            )
            p_out = _project_feasible(
                res_opt.x,
                z_min,
                z_max,
                head,
                r_slice,
                lower,
                upper,
                n_groups,
                dtau_slice=dtau_slice,
                tau_bounds=resolved_tau_bounds,
            )
            cost_out = _total_cost(p_out)
            if cost_out <= cost or _constraints_satisfied(
                p_out, r_slice, z_min, z_max, constraint_tol
            ):
                return p_out, cost_out
        except Exception:
            pass

        return p, cost

    def _perturb(base: np.ndarray) -> np.ndarray:
        init = base * np.exp(rng.normal(scale=sigma, size=n_params))
        init = np.maximum(init, 1e-9)
        if complex_fit:
            init[2] = rng.uniform(_CPE_EXPONENT_MIN, _CPE_EXPONENT_MAX)
            init[e_slice] = rng.uniform(
                _CPE_EXPONENT_MIN, _CPE_EXPONENT_MAX, size=n_groups
            )
        return _project_feasible(
            np.clip(init, lower, upper),
            z_min,
            z_max,
            head,
            r_slice,
            lower,
            upper,
            n_groups,
            dtau_slice=dtau_slice,
            tau_bounds=resolved_tau_bounds,
        )

    best_cost: float | None = None
    best_x: np.ndarray | None = None
    polish_count = 0

    base_init = _project_feasible(
        parameters_init.copy(),
        z_min,
        z_max,
        head,
        r_slice,
        lower,
        upper,
        n_groups,
        dtau_slice=dtau_slice,
        tau_bounds=resolved_tau_bounds,
    )

    for i in range(max(1, n_restarts)):
        init = base_init if i == 0 else _perturb(base_init)
        try:
            res_ls = least_squares(
                _residuals,
                init,
                jac=_jacobian,
                method="trf",
                bounds=(lower, upper),
                max_nfev=max_nfev,
            )
        except Exception:
            continue

        p = _project_feasible(
            res_ls.x.copy(),
            z_min,
            z_max,
            head,
            r_slice,
            lower,
            upper,
            n_groups,
            dtau_slice=dtau_slice,
            tau_bounds=resolved_tau_bounds,
        )
        cost = _total_cost(p)

        if not _constraints_satisfied(p, r_slice, z_min, z_max, constraint_tol):
            polish_count += 1
            p, cost = _polish_if_infeasible(p, cost)

        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_x = p.copy()

    if best_x is None:
        raise RuntimeError("All restarts failed to produce a fit.")

    res_ls = OptimizeResult(
        x=best_x,
        fun=_impedance_residuals(best_x),
        cost=best_cost,
        success=True,
        message=f"hybrid TRF ({n_restarts} restarts, {polish_count} polished)",
        nfev=0,
        njev=0,
    )
    for attr, default in (
        ("jac", None),
        ("grad", None),
        ("optimality", None),
        ("active_mask", None),
        ("status", 0),
    ):
        if not hasattr(res_ls, attr):
            setattr(res_ls, attr, default)

    tau = np.cumsum(res_ls.x[dtau_slice])
    R = res_ls.x[r_slice]
    res_ls.x[dtau_slice] = tau / R

    res = FitResults(res_ls, n_groups, complex_fit)

    return res
