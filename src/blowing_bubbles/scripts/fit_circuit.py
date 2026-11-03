r"""Fit the R–CPE ladder by sequential branch expansion (IFBB).

The branch count grows from the initial peak-based guess up to ``rc_groups``.
After each fit, a new R–CPE branch is seeded where the residual magnitude is
largest on the frequency segment with the greatest integrated error.

Public entry points are :func:`fit_circuit` (the full fit) and
:func:`information_criterion` (BIC/AIC used for branch selection), plus the
residual-segment helpers :func:`segment_magnitude_integrals` and
:func:`argmax_residual_in_largest_area_segment`.
"""

import warnings

import numpy as np

from blowing_bubbles.classes.circuit_classes import FitResults
from blowing_bubbles.classes.data_classes import EISData
from blowing_bubbles.scripts.global_minimize import (  # noqa: F401 (re-export)
    _resolve_n_jobs,
    impendance_fit,
    initial_fit,
    make_process_pool,
)
from blowing_bubbles.scripts.quality import fit_quality


def _segment_boundaries_at_local_minima(y: np.ndarray) -> list[int]:
    """Return segment boundaries at interior local minima of ``y``.

    The result starts with 0 and ends with ``len(y)``, so consecutive entries
    delimit half-open index ranges, one per peak.
    """
    n = len(y)
    if n < 2:
        return [0, n]
    mins = [i for i in range(1, n - 1) if y[i] <= y[i - 1] and y[i] <= y[i + 1]]
    return [0] + mins + [n]


def segment_magnitude_integrals(
    magnitude: np.ndarray,
    log_freq: np.ndarray,
) -> tuple[list[tuple[int, int]], list[float]]:
    r"""Integrate :math:`|r|` over :math:`\log_{10} f` between local minima.

    Parameters
    ----------
    magnitude : numpy.ndarray
        Residual magnitude :math:`|r_k|`, shape ``(M,)``.
    log_freq : numpy.ndarray
        :math:`\log_{10} f_k` (f in Hz), shape ``(M,)``, in the same order as
        ``magnitude``.

    Returns
    -------
    segment_ranges : list of tuple of int
        Inclusive ``(start, end)`` index range of each segment.
    integrated_areas : list of float
        Trapezoidal integral of ``magnitude`` over ``log_freq`` on each segment,
        aligned with ``segment_ranges``.

    See Also
    --------
    argmax_residual_in_largest_area_segment

    Notes
    -----
    Segments lie between successive interior local minima of :math:`|r|`
    (a minimum belongs to the segment that starts at it). Segments with fewer
    than two points are dropped. The sign of each area follows the ordering of
    ``log_freq`` (negative for descending frequencies).
    """
    segment_indices = _segment_boundaries_at_local_minima(magnitude)
    integrated_areas: list[float] = []
    segment_ranges: list[tuple[int, int]] = []
    for i in range(len(segment_indices) - 1):
        start = segment_indices[i]
        end = segment_indices[i + 1]
        x_segment = log_freq[start:end]
        y_segment = magnitude[start:end]
        if len(x_segment) > 1:
            integrated_areas.append(float(np.trapezoid(y_segment, x_segment)))
            segment_ranges.append((start, end - 1))
    return segment_ranges, integrated_areas


def argmax_residual_in_largest_area_segment(
    residuals_magnitude: np.ndarray,
    segment_ranges: list[tuple[int, int]],
    integrated_areas: list[float],
) -> tuple[int, int | None]:
    """Return the index of the largest residual in the largest-area segment.

    Parameters
    ----------
    residuals_magnitude : numpy.ndarray
        Residual magnitude, shape ``(M,)``.
    segment_ranges : list of tuple of int
        Inclusive ``(start, end)`` index ranges, as returned by
        :func:`segment_magnitude_integrals`.
    integrated_areas : list of float
        Integrated residual per segment, aligned with ``segment_ranges``.

    Returns
    -------
    index : int
        Index into ``residuals_magnitude`` of the maximum within the segment with
        the largest integrated area (global maximum if there are no segments).
    segment : int or None
        Index of the selected segment, or ``None`` if ``integrated_areas`` is
        empty.

    See Also
    --------
    segment_magnitude_integrals
    """
    if not integrated_areas:
        return int(np.argmax(residuals_magnitude)), None
    seg_i = int(np.argmax(integrated_areas))
    start, end_idx = segment_ranges[seg_i]
    local_max = int(np.argmax(residuals_magnitude[start : end_idx + 1]))
    return start + local_max, seg_i


_SELECTION_MODES = (None, "bic", "aic", "kk")
# History key under which each selection mode records its criterion value.
_CRITERION_KEYS = {"bic": "bic", "aic": "aic", "kk": "kk_ratio"}

# Floor on the residual variance used by the information criteria, relative to
# the mean squared weighted impedance. A relative RMS misfit of 1e-8 is far
# below any measurement noise; below it, cost differences are optimizer
# round-off and must not count as evidence for an extra branch.
_RELATIVE_VARIANCE_FLOOR = 1e-16


def _validate_selection(selection: str | None, min_improvement: float) -> None:
    """Raise ``ValueError`` for an unknown ``selection`` or bad ``min_improvement``.

    ``min_improvement`` must lie in ``[0, 1)``.
    """
    if selection not in _SELECTION_MODES:
        raise ValueError(
            f"Unknown selection {selection!r}; use None, 'bic', 'aic' or 'kk'."
        )
    if not (0.0 <= min_improvement < 1.0):
        raise ValueError(
            f"min_improvement must be in [0, 1), got {min_improvement!r}."
        )


def _residual_weights(data: EISData, weighting: str) -> np.ndarray:
    """Return per-frequency residual weights matching ``impendance_fit``.

    :math:`1/|Z_k|` for ``"modulus"`` (floored away from zero), otherwise ones.
    """
    z_abs = np.abs(data.Real + 1j * data.Imag)
    if weighting == "modulus":
        return 1.0 / np.maximum(z_abs, np.finfo(float).tiny)
    return np.ones_like(z_abs)


def _weighted_rss(res: FitResults, data: EISData, weighting: str) -> float:
    r"""Return the weighted residual sum of squares over the :math:`2M` rows.

    Recomputed from the unweighted ``res.residuals_magnitude``
    (:math:`|r_k|^2 = r_{k,\mathrm{re}}^2 + r_{k,\mathrm{im}}^2`) so the margin
    penalty rows included in ``res.cost`` do not enter the criterion.
    """
    w = _residual_weights(data, weighting)
    return float(np.sum((res.residuals_magnitude * w) ** 2))


def information_criterion(
    res: FitResults, data: EISData, weighting: str = "modulus", kind: str = "bic"
) -> float:
    r"""Return the Gaussian least-squares BIC or AIC of a ``complex_fit=True`` fit.

    Parameters
    ----------
    res : FitResults
        Fit result (``n_groups`` branches).
    data : EISData
        Spectrum the fit was made on.
    weighting : {"modulus", "unit"}, default "modulus"
        Weighting used by the fit. Any value other than ``"modulus"`` is
        treated as ``"unit"``.
    kind : {"bic", "aic"}, default "bic"
        Criterion.

    Returns
    -------
    float
        Criterion value (lower is better).

    Raises
    ------
    ValueError
        If ``kind`` is not ``"bic"`` or ``"aic"``.

    See Also
    --------
    fit_circuit : Uses this criterion when ``selection`` is ``"bic"``/``"aic"``.

    Notes
    -----
    .. math::

        \mathrm{BIC} = N \ln(\mathrm{RSS}/N) + k \ln N, \qquad
        \mathrm{AIC} = N \ln(\mathrm{RSS}/N) + 2k

    with :math:`N = 2M` (real and imaginary rows of :math:`M` frequencies),
    :math:`\mathrm{RSS}` the weighted residual sum of squares (weights
    :math:`1/|Z_k|` for ``"modulus"``, 1 for ``"unit"``; margin penalty terms
    excluded) and :math:`k = 3 + 3n` free parameters (:math:`R_s, L, n_L` and
    :math:`R_i, C_i, e_i` per branch). :math:`\mathrm{RSS}/N` is floored at
    ``1e-16`` times the mean squared weighted impedance, so fits of noise-free
    data that differ only by round-off compare by the penalty term alone.
    """
    w = _residual_weights(data, weighting)
    z = data.Real + 1j * data.Imag
    n_obs = 2 * len(z)
    scale = float(np.sum(np.abs(z * w) ** 2)) / n_obs
    floor = max(_RELATIVE_VARIANCE_FLOOR * scale, np.finfo(float).tiny)
    variance = max(_weighted_rss(res, data, weighting) / n_obs, floor)
    k = 3 + 3 * int(res.n_groups)
    if kind == "bic":
        penalty = k * np.log(n_obs)
    elif kind == "aic":
        penalty = 2.0 * k
    else:
        raise ValueError(f"Unknown criterion {kind!r}; use 'bic' or 'aic'.")
    return float(n_obs * np.log(variance) + penalty)


def fit_circuit(
    data: EISData,
    n_restarts: int,
    rc_groups: int = 5,
    initial_peaks: int = 2,
    verbose: bool = False,
    seed: int | None = None,
    tau_bounds: tuple[float, float] | None = None,
    resistance_source_bounds: tuple[float, float] | None = None,
    resistance_branch_bounds: tuple[float, float] | None = None,
    exponent_bounds: tuple[float, float] | None = None,
    inductance_exponent_bounds: tuple[float, float] | None = None,
    weighting: str = "modulus",
    n_jobs: int | None = 1,
    early_stop: int | None = None,
    selection: str | None = None,
    min_improvement: float = 0.0,
) -> FitResults:
    r"""Fit the R–CPE ladder with iterative fast branch building (IFBB).

    Parameters
    ----------
    data : EISData
        Measured impedance (Ω) vs. frequency (Hz).
    n_restarts : int
        Random restarts for the initial
        :func:`~blowing_bubbles.scripts.global_minimize.impendance_fit` call;
        each branch-expansion attempt uses ``max(1, n_restarts // 2)``.
    rc_groups : int, default 5
        Maximum number of parallel R–CPE branches (values above 5 emit a warning).
    initial_peaks : int, default 2
        Maximum number of Nyquist peaks (and hence branches) in the initial
        guess, passed as ``max_peaks`` to
        :func:`~blowing_bubbles.scripts.global_minimize.initial_fit`.
    verbose : bool, default False
        Print branch counts and cost changes during expansion.
    seed : int, optional
        RNG seed for parameter perturbations between restarts. ``None`` uses
        fresh entropy.
    tau_bounds : tuple of float, optional
        Bounds on :math:`\tau_i = R_i C_i` (s) forwarded to the initial guess
        and the fitter. ``None`` uses the fitter's data-derived defaults.
    resistance_source_bounds : tuple of float, optional
        ``(R_s_min, R_s_max)`` box bounds (Ω) for the source resistance. The
        default (``None``) lower bound keeps :math:`R_s` strictly positive, so
        relative errors in :math:`R_s` stay finite.
    resistance_branch_bounds : tuple of float, optional
        ``(R_min, R_max)`` uniform box bounds (Ω) for all branch resistances.
        ``None`` uses the fitter's defaults.
    exponent_bounds : tuple of float, optional
        ``(e_min, e_max)`` uniform box bounds for all CPE exponents (and the
        inductor exponent). ``None`` uses the fitter's defaults.
    inductance_exponent_bounds : tuple of float, optional
        Bounds on the inductor exponent :math:`n_L`. ``None`` means
        ``(0.7, 1.0)`` unless ``exponent_bounds`` is given.
    weighting : {"modulus", "unit"}, default "modulus"
        Residual weighting passed to the fitter. ``"modulus"`` divides each
        residual by :math:`|Z_k|`, so small high-frequency arcs carry weight;
        ``"unit"`` reproduces the unweighted objective of version 0.1.
    n_jobs : int or None, default 1
        Worker processes for the restarts (``1``, ``0`` or ``None``:
        sequential; ``-1``: all cores). A process pool is created once and
        shared by all sub-fits. Scripts using ``n_jobs != 1`` on Windows/macOS need an
        ``if __name__ == "__main__":`` guard.
    early_stop : int, optional
        Stop the restarts of each sub-fit once this many reached the best cost.
        ``None`` runs all restarts.
    selection : {None, "bic", "aic", "kk"}, optional
        Criterion for accepting another branch.

        - ``None`` (default): unchanged behaviour; a branch is accepted whenever
          it lowers the cost, up to ``rc_groups`` branches.
        - ``"bic"``: accept only if the Bayesian information criterion
          :math:`N \ln(\mathrm{RSS}/N) + k \ln N` decreases, with
          :math:`N = 2M` weighted residual rows (real and imaginary) and
          :math:`k = 3 + 3n` free parameters. See
          :func:`information_criterion`. Recommended for automatic use; it
          penalises extra branches more strongly than AIC.
        - ``"aic"``: as ``"bic"`` with the Akaike penalty :math:`2k`.
        - ``"kk"``: stop before adding a branch once the current fit is no
          longer ``suspect`` according to
          :func:`~blowing_bubbles.scripts.quality.fit_quality`, i.e. its
          relative RMS residual is within ``kk_ratio_threshold`` (default 3)
          times the Lin-KK noise floor of the spectrum.
    min_improvement : float, default 0.0
        Minimum relative cost decrease :math:`(c_{old} - c_{new}) / c_{old}`
        required to accept a branch, in every ``selection`` mode. Must lie in
        ``[0, 1)``; ``0.0`` accepts any decrease.

    Returns
    -------
    FitResults
        Best fit after branch expansion; ``initialize_rc_groups`` records
        ``rc_groups``. ``history`` is a list with one dict per branch count
        tried, with keys ``n_groups``, ``cost``, ``accepted`` and, if
        ``selection`` is set, the criterion value under ``"bic"``, ``"aic"`` or
        ``"kk_ratio"``. A rejected candidate is recorded with
        ``accepted=False``; in ``"kk"`` mode the fit that stopped expansion is
        the last entry.

    Raises
    ------
    ValueError
        If ``selection`` is not one of the options above or ``min_improvement``
        is outside ``[0, 1)``.
    RuntimeError
        If the initial fit or a branch-expansion fit (all attempts) does not
        converge.

    Warns
    -----
    UserWarning
        If ``rc_groups`` is greater than 5.

    See Also
    --------
    information_criterion
    blowing_bubbles.scripts.global_minimize.impendance_fit
    blowing_bubbles.scripts.quality.fit_quality

    Notes
    -----
    **Algorithm.**

    #. Seed parameters with
       :func:`~blowing_bubbles.scripts.global_minimize.initial_fit`
       (Nyquist peak widths and characteristic frequencies, at most
       ``initial_peaks`` branches).
    #. Run :func:`~blowing_bubbles.scripts.global_minimize.impendance_fit` with
       ``complex_fit=True`` for the initial branch count.
    #. While ``n_groups < rc_groups``, insert a new branch at the frequency where
       the residual magnitude (divided by :math:`|Z_k|` for ``"modulus"``
       weighting) is largest on the segment with greatest
       :math:`\int |\hat{Z} - Z|\, d\log_{10} f`. The new branch is ideal
       (:math:`e = 1`) with :math:`\tau = 1/(2\pi f)` at that frequency and
       :math:`R` equal to 0.6 times the :math:`Z'` span of the segment. Re-fit
       up to three times with different seeds, stopping as soon as the cost does
       not exceed the previous fit's. If the new branch is not accepted (see
       ``selection`` and ``min_improvement``), expansion stops and the previous
       fit is returned.

    **Model selection.** Adding a branch can never raise the least-squares
    cost, so with ``selection=None`` the fit keeps adding branches up to
    ``rc_groups`` even when the extra branch only duplicates an existing one
    (same :math:`\tau`) and lowers the cost by round-off. The ``selection``
    criteria weigh the cost decrease against the extra parameters and stop
    expansion once another branch is not justified by the data.

    With ``verbose=True`` progress and cost changes are printed to stdout.
    """
    _validate_selection(selection, min_improvement)
    jobs = _resolve_n_jobs(n_jobs)
    pool = make_process_pool(jobs) if jobs > 1 else None
    try:
        return _fit_circuit_impl(
            data,
            n_restarts,
            rc_groups=rc_groups,
            initial_peaks=initial_peaks,
            verbose=verbose,
            seed=seed,
            tau_bounds=tau_bounds,
            resistance_source_bounds=resistance_source_bounds,
            resistance_branch_bounds=resistance_branch_bounds,
            exponent_bounds=exponent_bounds,
            weighting=weighting,
            fit_kwargs={
                "executor": pool,
                "early_stop": early_stop,
                "inductance_exponent_bounds": inductance_exponent_bounds,
            },
            selection=selection,
            min_improvement=min_improvement,
        )
    finally:
        if pool is not None:
            pool.shutdown()


def _fit_circuit_impl(
    data: EISData,
    n_restarts: int,
    *,
    rc_groups: int,
    initial_peaks: int,
    verbose: bool,
    seed: int | None,
    tau_bounds: tuple[float, float] | None,
    resistance_source_bounds: tuple[float, float] | None,
    resistance_branch_bounds: tuple[float, float] | None,
    exponent_bounds: tuple[float, float] | None,
    weighting: str,
    fit_kwargs: dict,
    selection: str | None = None,
    min_improvement: float = 0.0,
) -> FitResults:
    """Run the IFBB expansion for :func:`fit_circuit`.

    ``fit_kwargs`` (``executor``, ``early_stop``, ``inductance_exponent_bounds``)
    is forwarded to every ``impendance_fit`` call.
    """
    _validate_selection(selection, min_improvement)
    if rc_groups > 5:
        warnings.warn("rc_groups is greater than 5.", stacklevel=2)

    initial_parameters = initial_fit(
        data,
        prominence=0.005,
        tau_bounds=tau_bounds,
        max_peaks=initial_peaks,
    )

    n_groups = len(initial_parameters.resistances)
    if verbose:
        print(f"Initial fit with {n_groups} groups")
        print(initial_parameters)

    res = impendance_fit(
        data,
        n_groups=n_groups,
        complex_fit=True,
        n_restarts=n_restarts,
        seed=seed,
        sigma=0.5,
        weighting=weighting,
        initial_parameters=initial_parameters,
        tau_bounds=tau_bounds,
        resistance_source_bounds=resistance_source_bounds,
        resistance_branch_bounds=resistance_branch_bounds,
        exponent_bounds=exponent_bounds,
        **fit_kwargs,
    )
    if res is None:
        raise RuntimeError("Initial impedance fit did not converge.")

    def _criterion(fit: FitResults) -> tuple[float | None, bool]:
        """Return the criterion value and whether a ``"kk"`` fit is adequate."""
        if selection in ("bic", "aic"):
            return information_criterion(fit, data, weighting, kind=selection), False
        if selection == "kk":
            quality = fit_quality(data, fit.parameters)
            return quality["kk_ratio"], not quality["suspect"]
        return None, False

    def _entry(fit: FitResults, value: float | None, accepted: bool) -> dict:
        """Return the ``history`` record for one branch count."""
        entry = {
            "n_groups": int(fit.n_groups),
            "cost": float(fit.cost),
            "accepted": accepted,
        }
        if selection is not None:
            entry[_CRITERION_KEYS[selection]] = value
        return entry

    crit, adequate = _criterion(res)
    history = [_entry(res, crit, True)]

    for branch_count in range(n_groups, rc_groups):
        if adequate:
            if verbose:
                print(
                    f"   Fit with {branch_count} branches is within the Lin-KK "
                    f"noise floor (kk_ratio={crit:.3g}); not adding branches."
                )
            break
        if verbose:
            print(f"Fitting with {branch_count + 1} groups")
        magnitude = res.residuals_magnitude
        if weighting == "modulus":
            # Locate the gap on the same (relative) scale the fit minimises.
            magnitude = magnitude / np.maximum(
                np.abs(data.Real + 1j * data.Imag), np.finfo(float).tiny
            )
        log_freq = np.log10(data.Freq)
        segment_ranges, integrated_areas = segment_magnitude_integrals(
            magnitude, log_freq
        )
        arg_max, selected_seg = argmax_residual_in_largest_area_segment(
            magnitude, segment_ranges, integrated_areas
        )
        freq_add = data.Freq[arg_max]

        if selected_seg is None:
            segment_start, segment_end = 0, len(res.residuals) - 1
        else:
            segment_start, segment_end = segment_ranges[selected_seg]

        # Ideal (e = 1) branch with its apex at freq_add: omega * R * C = 1.
        tau_new = 1 / (2 * np.pi * freq_add)
        # Width of the selected residual segment along Z' (measured data),
        # scaled by 0.6 so the new arc does not overshoot the gap it targets.
        resistance_new = np.abs(data.Real[segment_end] - data.Real[segment_start]) * 0.6
        if not np.isfinite(resistance_new) or resistance_new <= 0:
            resistance_new = 0.05 * float(np.ptp(data.Real))

        new_params = res.parameters.copy()
        new_params.add_branch(resistance_new, tau_new / resistance_new, 1.0)

        max_attempts = 3
        attempt = 0
        res_new = None
        while attempt < max_attempts:
            # Distinct seed per attempt; identical seeds would repeat the same fit.
            attempt_seed = (
                None if seed is None else seed + 1000 * (branch_count + 1) + attempt
            )
            res_try = impendance_fit(
                data,
                n_groups=branch_count + 1,
                complex_fit=True,
                n_restarts=max(1, n_restarts // 2),
                seed=attempt_seed,
                weighting=weighting,
                initial_parameters=new_params,
                sigma=0.1,
                tau_bounds=tau_bounds,
                resistance_source_bounds=resistance_source_bounds,
                resistance_branch_bounds=resistance_branch_bounds,
                exponent_bounds=exponent_bounds,
                **fit_kwargs,
            )
            if res_try is not None and (res_new is None or res_try.cost < res_new.cost):
                res_new = res_try
            if res_new is not None and res_new.cost <= res.cost:
                break
            attempt += 1

        if res_new is None:
            raise RuntimeError("Branch expansion fit did not converge.")
        crit_new, adequate_new = _criterion(res_new)
        # With min_improvement = 0 this is the plain "cost decreased" test.
        cost_ok = res_new.cost < res.cost * (1.0 - min_improvement)
        crit_ok = selection not in ("bic", "aic") or crit_new < crit
        accepted = bool(cost_ok and crit_ok)
        history.append(_entry(res_new, crit_new, accepted))
        if not accepted:
            if verbose:
                reason = (
                    "did not lower the cost enough"
                    if not cost_ok
                    else f"did not lower the {selection.upper()}"
                )
                print(
                    f"   Adding branch {branch_count + 1} {reason}; "
                    f"keeping {branch_count} branches."
                )
            break

        if verbose:
            print(f"\033[90m   New parameters: \n{new_params}\033[0m")
            print(f"\033[90m   New cost: {res_new.cost}\033[0m")
            print(f"\033[90m   Old cost: {res.cost}\033[0m")
            delta = res_new.cost - res.cost
            print(f"\033[90m   Cost improvement: {delta}\033[0m")
            pct = delta / res.cost * 100
            print(f"\033[90m   Cost improvement percentage: {pct}%\033[0m")

        res = res_new
        crit, adequate = crit_new, adequate_new

    res.add_initial_rc_groups(rc_groups)
    res.history = history

    return res
