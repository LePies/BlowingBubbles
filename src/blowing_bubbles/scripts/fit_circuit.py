r"""Sequential model expansion (grow branch count until ``n_groups_max``).

After each fit, a new R–CPE branch is seeded where the residual magnitude is
largest on the frequency segment with the greatest integrated error.

See :func:`fit_circuit` for the public entry point.
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


def _segment_boundaries_at_local_minima(y: np.ndarray) -> list[int]:
    """Split index ranges at interior local minima (one segment per peak)."""
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

    Segments lie between successive minima of :math:`|r|`.
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
    """Return index of max residual magnitude in the largest integrated segment."""
    if not integrated_areas:
        return int(np.argmax(residuals_magnitude)), None
    seg_i = int(np.argmax(integrated_areas))
    start, end_idx = segment_ranges[seg_i]
    local_max = int(np.argmax(residuals_magnitude[start : end_idx + 1]))
    return start + local_max, seg_i


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
    weighting: str = "modulus",
    n_jobs: int | None = 1,
    early_stop: int | None = None,
) -> FitResults:
    r"""Fit the R–CPE ladder with iterative fast branch building (IFBB).

    **Algorithm.**

    #. Seed parameters with :func:`~blowing_bubbles.scripts.fit_circuit.initial_fit`
       (Nyquist peak widths and characteristic frequencies).
    #. Run :func:`~blowing_bubbles.scripts.global_minimize.impendance_fit` with
       ``complex_fit=True`` for the initial branch count.
    #. While ``n_groups < rc_groups``, insert a new branch at the frequency where
       the residual magnitude is largest on the segment with greatest
       :math:`\int |\hat{Z} - Z|\, d\log_{10} f` and re-fit (up to three attempts).
       If no attempt lowers the cost, expansion stops and the previous fit is
       returned.

    Parameters
    ----------
    data : EISData
        Measured impedance vs. frequency.
    n_restarts : int
        Random restarts passed to each
        :func:`~blowing_bubbles.scripts.global_minimize.impendance_fit` call.
    rc_groups : int
        Maximum number of parallel R–CPE branches (values above 5 emit a warning).
    verbose : bool
        Print branch counts and cost changes during expansion.
    seed : int, optional
        RNG seed for parameter perturbations between restarts.
    tau_bounds : tuple of float, optional
        Bounds on :math:`\tau_i = R_i C_i` forwarded to the fitter.
    resistance_source_bounds : tuple of float, optional
        ``(R_s_min, R_s_max)`` box bounds for the source resistance.
    resistance_branch_bounds : tuple of float, optional
        ``(R_min, R_max)`` uniform box bounds for all branch resistances.
    exponent_bounds : tuple of float, optional
        ``(e_min, e_max)`` uniform box bounds for all CPE exponents.
    weighting : {"modulus", "unit"}
        Residual weighting passed to the fitter. ``"modulus"`` (default) divides
        each residual by :math:`|Z_k|`, so small high-frequency arcs carry weight;
        ``"unit"`` reproduces the unweighted objective of version 0.1.
    n_jobs : int, optional
        Worker processes for the restarts (``1``: sequential, default; ``-1``:
        all cores). Scripts using ``n_jobs != 1`` on Windows/macOS need an
        ``if __name__ == "__main__":`` guard.
    early_stop : int, optional
        Stop the restarts of each sub-fit once this many reached the best cost.
        ``None`` (default) runs all restarts.

    Returns
    -------
    FitResults
        Best fit after branch expansion; ``initialize_rc_groups`` records ``rc_groups``.
    """
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
            fit_kwargs={"executor": pool, "early_stop": early_stop},
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
) -> FitResults:
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

    for branch_count in range(n_groups, rc_groups):
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
        if res_new.cost >= res.cost:
            if verbose:
                print(
                    f"   Adding branch {branch_count + 1} did not lower the cost; "
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

    res.add_initial_rc_groups(rc_groups)

    return res
