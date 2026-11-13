# Changelog

## Unreleased

## 0.2.0 (2026-11-13)

Corrections found in a review of the method against synthetic ground truth.

### Fixed
- **Initial and inserted time constants missed a factor 2*pi.** `initial_fit`
  and the branch insertion used `tau = 1/f`; the apex of an RC arc is at
  `2*pi*f*R*C = 1`, so `tau = 1/(2*pi*f)` is now used.
- **Initial R_s depended on the row order of the data.** `initial_fit` took
  `Real[0]`, which is the *low*-frequency value for ascending data. It now
  uses the real part at the highest frequency.
- **Lower arc-centre constraint excluded the true circuit.** The hard limit
  `R_s + R_1/2 >= min Z'` (plus a 5 % soft margin) is violated whenever the
  high-frequency arc is truncated by the measurement window (56 % of the
  Case 2 synthetic circuits). It is replaced by the physical bound
  `R_s <= min Z'` (Re Z >= R_s for every branch and the CPE inductor). The
  soft margin now defaults to 0 (`z_margin_frac=0.0`).
- **Inductance bounds from the slope estimate are off by default**
  (`inductance_bounds=False`; L >= 0 still applies). The estimate excluded the
  true L in 14 % of the synthetic circuits.
- **New-branch resistance used residuals instead of data.** `R'` was
  `0.6*|r_start - r_end|` (difference of complex residuals at local minima,
  i.e. close to zero); it is now `0.6*|Z'_end - Z'_start|` as described.
- **The three insertion attempts reused the same seed** and therefore repeated
  the same fit; each attempt now gets its own seed and the best attempt is kept.
- **Restart exponents ignored the base point.** Every restart drew all CPE
  exponents uniformly; now half of the restarts perturb the base exponents
  locally (scale sigma/2) and half explore uniformly.

- **Series resistance collapsed to R_s = 0** in 20 of 83 EK3 fits and in most
  noisy synthetic fits (33/50 at 1 %, 42/50 at 5 % noise). The inductor
  exponent ran down to n_L ~ 0.4, where `L (jω)^n_L` has a large real part,
  and together with a branch peaking above f_max it absorbed R_s (R_tot stayed
  correct, so the R_tot-based screen did not catch it). The inductor exponent
  now defaults to `inductance_exponent_bounds=(0.7, 1.0)` (expert EK3 values
  0.86–0.99). `exponent_bounds`, when given, still applies to n_L as before;
  pass `inductance_exponent_bounds=(0.2, 1.0)` for the old range.
- **Spurious inductance start value.** `estimate_inductance_from_imag`
  regressed over the capacitive tail and returned L > 0 for spectra without
  any inductive part (6e-6 H for a pure two-arc test circuit). It now uses
  only inductive points (Im Z > 0), or, for a negative inductance, top
  frequencies where Im Z < 0 grows in magnitude; otherwise it returns 0.
- TRF termination tolerances tightened to 1e-12 (SciPy default 1e-8) so
  parameters can reach their bounds (e.g. L = 0) instead of stopping at a
  tiny interior value.
- `test_selection_none_keeps_old_behaviour` now uses slightly noisy data: on
  clean data the two-branch fit is exact to round-off, so a redundant branch
  can no longer lower the cost.
- **`complex_fit=False` never produced a fit** (residual/Jacobian shape
  mismatch hidden by a blanket `except`). It now means ideal elements
  (`e_i = n_L = 1`) with complex residuals.
- **Column detection** matched single letters (`"Imaginary"` was taken as the
  real part). Matching is now token-based, ambiguous names raise, and
  `EISData(..., columns=(real, imag, freq))` sets them explicitly.
- **`Rtot`/`Rs` depended on row order**; `Rs` is interpolated at the
  inductive zero crossing.
- **`EISData` arithmetic** squared the frequency column (`*`, `/`) or added
  extra columns (`+`, `-`); it now operates on the complex impedance.
- `CircuitParameters.from_array` (missing `cls`); `__dict__` method broke
  `copy`/`pickle` and is renamed `as_dict()`.
- `fit_circuit` kept a branch even when it raised the cost; it now keeps the
  previous fit and stops expanding.
- Spectra without an interior `-Im(Z)` peak crashed; `initial_fit` falls back
  to one branch and floors zero widths.
- `impendance_fit` no longer mutates `initial_parameters`; failed restarts are
  reported instead of silently skipped. `plot_full_fit` returns its axes.

### Added
- `weighting` option (`"modulus"`, default, or `"unit"`) on `impendance_fit`,
  `fit_circuit` and `EISData.fit`. Use `"unit"` for constant-variance noise and
  `"modulus"` for noise proportional to |Z|.
- TRF now uses Jacobian-based variable scaling (`x_scale="jac"`).
- `blowing_bubbles.scripts.quality`: `lin_kk` (linear Kramers-Kronig test) and
  `fit_quality` (fit residual relative to the Lin-KK noise floor, `suspect`
  flag for ratio > 3, total-resistance check), for label-free screening.
- `CircuitParameters.characteristic_times` ((R*Q)**(1/n)) and
  `peak_frequencies`. The stored `taus` remain the products R*Q.
- Regression tests in `src/tests/test_fixes_v02.py`.

### Performance
- Fit core rewritten around a cached evaluator: the forward model is computed
  once per parameter vector, the Jacobian reuses its intermediates, `log(jω)`
  is precomputed and argument validation is skipped in the hot loop
  (~1.6–3x less CPU per function evaluation).
- New `n_jobs` (process-parallel restarts, results independent of `n_jobs`)
  and `early_stop` options on `impendance_fit`, `fit_circuit` and
  `EISData.fit`; `make_process_pool()` creates a reusable pool with
  single-threaded BLAS workers.
- matplotlib is imported lazily.

### Changed
- `circuit_impedance` accepts lists and NumPy scalars and raises
  `TypeError`/`ValueError` instead of `AssertionError`.

### Behaviour changes
- Default weighting changed from unweighted to modulus weighting; pass
  `weighting="unit"` to reproduce the 0.1 objective.
