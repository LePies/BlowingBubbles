r"""Sequential model expansion (grow branch count until ``n_groups_max``).

After each fit, a new R–CPE branch is seeded where the residual magnitude is
largest on the frequency segment with the greatest integrated error.

See :func:`fit_circuit` for the public entry point.
"""

import numpy as np
from scipy.signal import find_peaks

from blowing_bubbles.classes.circuit_classes import CircuitParameters, FitResults
from blowing_bubbles.classes.data_classes import EISData
from blowing_bubbles.scripts.global_minimize import (
    estimate_inductance_from_imag,
    impendance_fit,
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


def initial_fit(
    eis: EISData,
    prominence: float = 0.01,
    tau_bounds: tuple[float, float] | None = None,
    max_peaks: int | None = None,
):
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
        # Find which bounds bracket the peak
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


def fit_circuit(
    data: EISData,
    n_restarts: int,
    rc_groups: int = 5,
    initial_peaks: int = 2,
    verbose: bool = False,
    seed: int | None = None,
    tau_bounds: tuple[float, float] | None = None,
) -> FitResults:
    r"""Fit the R–CPE ladder with iterative fast branch building (IFBB).

    **Algorithm.**

    #. Seed parameters with :func:`~blowing_bubbles.scripts.fit_circuit.initial_fit`
       (Nyquist peak widths and characteristic frequencies).
    #. Run :func:`~blowing_bubbles.scripts.global_minimize.impendance_fit` with
       ``complex_fit=True`` for the initial branch count.
    #. While ``n_groups < rc_groups``, insert a new branch at the frequency where
       the residual magnitude is largest on the segment with greatest
       :math:`\int |\hat{Z} - Z|\, d\log_{10} f`, re-fit, and keep the result if
       cost does not increase (up to three attempts per insertion).

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
        Bounds on :math:`\tau_i = R_i C_i` forwarded to the quick fitter.

    Returns
    -------
    FitResults
        Best fit after branch expansion; ``initialize_rc_groups`` records ``rc_groups``.
    """
    if rc_groups > 5:
        print("Warning: rc_groups is greater than 5.")

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
        initial_parameters=initial_parameters,
        tau_bounds=tau_bounds,
    )
    if res is None:
        raise RuntimeError("Initial impedance fit did not converge.")

    for branch_count in range(n_groups, rc_groups):
        if verbose:
            print(f"Fitting with {branch_count + 1} groups")
        magnitude = res.residuals_magnitude
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
        if segment_end == len(res.residuals) - 1:
            residuals_end = res.residuals[segment_end]
        else:
            residuals_end = res.residuals[segment_end + 1]
        residuals_start = res.residuals[segment_start]

        # Find the indices of the two taus surrounding 1/freq_add
        taus = res.parameters.taus
        target_tau = 1 / freq_add

        # Find all taus less than or equal to target_tau
        idx_below = np.where(taus <= target_tau)[0]
        # Find all taus greater than target_tau
        idx_above = np.where(taus > target_tau)[0]

        # Last tau below (max index less or equal), first tau above (min index greater)
        idx_lo = idx_below[-1] if len(idx_below) > 0 else None
        idx_hi = idx_above[0] if len(idx_above) > 0 else None

        new_params = res.parameters.copy()

        if idx_lo is None and idx_hi is None:
            raise ValueError(
                "Cannot insert new group: both idx_lo and idx_hi are None."
            )

        tau_new = 1 / freq_add

        if idx_hi is None:
            insert_pos = len(new_params.resistances)
        elif idx_lo is None:
            insert_pos = 0
        else:
            insert_pos = idx_hi

        resistance_new = np.abs(residuals_start - residuals_end) * 0.6
        capacitance_new = tau_new / resistance_new
        exponent_new = 1.0

        new_params.inductance *= 0.5
        new_params.inductance_exponent = 0.9
        new_params._set_branches(
            np.insert(new_params.resistances, insert_pos, resistance_new),
            np.insert(new_params.capacitances, insert_pos, capacitance_new),
            np.insert(new_params.exponents, insert_pos, exponent_new),
        )
        new_params.sort_by_tau()

        expected = branch_count + 1
        assert len(new_params) == expected, (
            f"Branch insert failed: expected {expected} groups, "
            f"got {len(new_params)}."
        )

        max_attempts = 3
        attempt = 0
        res_new = None
        while attempt < max_attempts:
            res_new = impendance_fit(
                data,
                n_groups=branch_count + 1,
                complex_fit=True,
                n_restarts=np.maximum(1, n_restarts // 2),
                seed=seed,
                initial_parameters=new_params,
                sigma=0.1,
                tau_bounds=tau_bounds,
            )
            if res_new is not None and res_new.cost <= res.cost:
                break
            attempt += 1

        if res_new is None:
            raise RuntimeError("Branch expansion fit did not converge.")

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
