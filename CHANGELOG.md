# Changelog

## 0.2.0 (2026-10-30)

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

### Behaviour changes
- Default weighting changed from unweighted to modulus weighting; pass
  `weighting="unit"` to reproduce the 0.1 objective.
