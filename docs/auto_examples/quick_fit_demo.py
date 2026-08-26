"""
Quick Fitting Demo
=================

Demonstrate fine-tuning of EIS circuit parameters using
:meth:`~blowing_bubbles.classes.data_classes.EISData.quick_fit` and the
standalone :func:`~blowing_bubbles.classes.data_classes.quick_fit` function.

This example shows a typical workflow:

1. Generate or load EIS data
2. Obtain initial parameters from a coarse fit or prior knowledge
3. Use ``quick_fit`` to refine parameters with tighter constraints
4. Compare the results

The ``quick_fit`` method uses global optimization
(via ``scipy.optimize.dual_annealing``) and is designed for fast,
targeted optimization starting from good initial guesses.
It's particularly useful for:

- Fine-tuning parameters within specific bounds
- Quick re-fitting with different constraints
- Batch processing of multiple spectra with similar characteristics
- Interactive exploration of parameter space
"""

from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from blowing_bubbles.classes import CircuitParameters, EISData
from blowing_bubbles.classes.data_classes import quick_fit

# %%
# Generate synthetic EIS data
# We'll create data with known parameters so we can evaluate the fit quality

np.random.seed(42)

# True circuit parameters (what we're trying to recover)
true_params = CircuitParameters(
    resistance_source=10.0,
    inductance=1e-7,
    inductance_exponent=0.95,
    resistances=np.array([50.0, 20.0, 5.0]),
    capacitances=np.array([1e-4, 1e-3, 1e-2]),
    exponents=np.array([0.98, 0.95, 0.90]),
)

# Generate frequency points
n_points = 100
freqs = np.logspace(-2, 5, n_points)

# Calculate true impedance and add noise
z_true = true_params.get_impedance(freqs)
noise_level = 0.1
re_impedance = z_true.real + np.random.normal(0, noise_level, n_points)
im_impedance = z_true.imag + np.random.normal(0, noise_level, n_points)

data = pd.DataFrame(
    {
        "Freq (Hz)": freqs,
        "Zreal (Ohm)": re_impedance,
        "Zimag (Ohm)": im_impedance,
    }
)

eis_data = EISData(data)

# %%
# Step 1: Coarse initial fit
# Use the regular fit method to get a rough estimate of parameters

print("=" * 60)
print("Step 1: Running coarse initial fit (iterative branch building)")
print("=" * 60)

start_time = time.time()
initial_fit = eis_data.fit(
    n_restarts=50,
    rc_groups=3,
    verbose=False,
    seed=42,
)
coarse_time = time.time() - start_time

print(f"Coarse fit completed in {coarse_time:.2f} s")
print(f"Success: {initial_fit.success}")
print(f"Cost: {initial_fit.cost:.6e}")
print("\nInitial parameters:")
print(initial_fit.parameters)

# %%
# Step 2: Fine-tune with quick_fit (complex_fit=True)
# Use the parameters from the coarse fit as initial guess
# Enable complex_fit to also optimize the inductance exponent

print("\n" + "=" * 60)
print("Step 2: Fine-tuning with quick_fit (complex_fit=True)")
print("=" * 60)

start_time = time.time()
fine_tuned_fit = eis_data.quick_fit(
    initial_parameters=initial_fit.parameters,
    complex_fit=True,
    seed=42,
    sigma=0.25,
    resistance_source_bounds=(8.0, 12.0),
    resistance_branch_bounds=(1.0, 100.0),
    exponent_bounds=(0.85, 1.0),
    polish=True,
    polish_maxiter=100,
)
fine_tune_time = time.time() - start_time

print(f"Fine-tuning completed in {fine_tune_time:.4f} s")
print(f"Success: {fine_tuned_fit.success}")
print(f"Cost: {fine_tuned_fit.cost:.6e}")
print("\nFine-tuned parameters:")
print(fine_tuned_fit.parameters)

# %%
# Step 3: Fine-tune from a rough initial guess
# Show how quick_fit can recover good parameters from a reasonable starting point

print("\n" + "=" * 60)
print("Step 3: Fine-tuning from a rough initial guess")
print("=" * 60)

rough_initial_params = CircuitParameters(
    resistance_source=10.5,
    inductance=1e-8,
    inductance_exponent=0.9,
    resistances=np.array([45.0, 25.0, 8.0]),
    capacitances=np.array([8e-5, 8e-4, 8e-3]),
    exponents=np.array([0.95, 0.95, 0.95]),
)

print("\nRough initial parameters:")
print(rough_initial_params)

start_time = time.time()
refined_fit = eis_data.quick_fit(
    initial_parameters=rough_initial_params,
    complex_fit=True,
    seed=42,
    resistance_source_bounds=(8.0, 12.0),
    resistance_branch_bounds=(1.0, 100.0),
    exponent_bounds=(0.85, 1.0),
    polish=True,
)
refined_time = time.time() - start_time

print(f"\nRefined fit completed in {refined_time:.4f} s")
print(f"Success: {refined_fit.success}")
print(f"Cost: {refined_fit.cost:.6e}")
print("\nRefined parameters:")
print(refined_fit.parameters)

# %%
# Step 4: Compare all results

print("\n" + "=" * 60)
print("Step 4: Comparison of all fits")
print("=" * 60)

print("\nTrue parameters:")
print(true_params)


def calc_param_error(params1, params2):
    """Calculate normalized parameter errors (as percentages)."""
    rs_err = (
        abs(params1.resistance_source - params2.resistance_source)
        / params2.resistance_source
    )
    r_err = np.sum(
        np.abs(params1.resistances - params2.resistances)
    ) / np.sum(params2.resistances)
    c_err = np.sum(
        np.abs(params1.capacitances - params2.capacitances)
    ) / np.sum(params2.capacitances)
    exp_err = np.sum(
        np.abs(params1.exponents - params2.exponents)
    ) / len(params2.exponents)
    return rs_err * 100, r_err * 100, c_err * 100, exp_err * 100


coarse_rs_err, coarse_r_err, coarse_c_err, coarse_exp_err = calc_param_error(
    initial_fit.parameters, true_params
)
fine_rs_err, fine_r_err, fine_c_err, fine_exp_err = calc_param_error(
    fine_tuned_fit.parameters, true_params
)
refined_rs_err, refined_r_err, refined_c_err, refined_exp_err = calc_param_error(
    refined_fit.parameters, true_params
)

print("\nCoarse fit errors (percent):")
print(f"  Rs error: {coarse_rs_err:.2f}%")
print(f"  Resistance error: {coarse_r_err:.2f}%")
print(f"  Capacitance error: {coarse_c_err:.2f}%")
print(f"  Exponent error: {coarse_exp_err:.2f}%")

print("\nFine-tuned fit errors (percent):")
print(f"  Rs error: {fine_rs_err:.2f}%")
print(f"  Resistance error: {fine_r_err:.2f}%")
print(f"  Capacitance error: {fine_c_err:.2f}%")
print(f"  Exponent error: {fine_exp_err:.2f}%")

print("\nRefined fit errors (percent):")
print(f"  Rs error: {refined_rs_err:.2f}%")
print(f"  Resistance error: {refined_r_err:.2f}%")
print(f"  Capacitance error: {refined_c_err:.2f}%")
print(f"  Exponent error: {refined_exp_err:.2f}%")

print("\nSpeed comparison:")
print(f"  Coarse fit: {coarse_time:.2f} s")
print(f"  Fine-tune: {fine_tune_time:.4f} s")
print(f"  Refined from rough: {refined_time:.4f} s")
if fine_tune_time > 0:
    print(f"  Coarse vs Fine-tune speedup: {coarse_time / fine_tune_time:.1f}x")

# %%
# Step 5: Visualize all fits

# Create a denser frequency grid for smooth plotting
freqs_plot = np.logspace(-2, 5, 500)
unit = r"$\Omega$"

fig, ax = plt.subplots(figsize=(10, 8))

# Plot measured data
eis_data.plot_nyquist(axes=ax, label="Measured", color="black", markersize=4)

# Plot coarse fit
initial_fit.parameters.plot_nyquist(freqs_plot, axis=ax, unit=unit)

# Plot fine-tuned fit
Z_fine = fine_tuned_fit.parameters.get_impedance(freqs_plot)
ax.plot(Z_fine.real, -Z_fine.imag, label="Fine-tuned (complex)", linestyle="-")

# Plot refined fit
Z_refined = refined_fit.parameters.get_impedance(freqs_plot)
ax.plot(Z_refined.real, -Z_refined.imag, label="Refined from rough", linestyle="--")

# Plot true model (for reference)
Z_true = true_params.get_impedance(freqs_plot)
ax.plot(Z_true.real, -Z_true.imag, label="True parameters", linestyle=":")

ax.legend(loc="best")
ax.set_title("Quick Fit Demo: Measured vs. Fits vs. True Parameters")
plt.tight_layout()
plt.savefig("quick_fit_demo_comparison.png", dpi=150)
print("\nPlot saved as 'quick_fit_demo_comparison.png'")

# %%
# Step 6: Demonstrate standalone quick_fit function

print("\n" + "=" * 60)
print("Step 6: Using standalone quick_fit function")
print("=" * 60)

standalone_start = time.time()
standalone_fit = quick_fit(
    data=eis_data,
    n_groups=3,
    complex_fit=True,
    initial_parameters=initial_fit.parameters,
    n_restarts=1,
    seed=42,
    resistance_source_bounds=(8.0, 12.0),
    exponent_bounds=(0.85, 1.0),
)
standalone_time = time.time() - standalone_start

print(f"Standalone quick_fit completed in {standalone_time:.4f} s")
print(f"Success: {standalone_fit.success}")
print(f"Cost: {standalone_fit.cost:.6e}")

# %%
# Step 7: Show robustness with poor initial parameters

print("\n" + "=" * 60)
print("Step 7: Convergence with poor initial parameters")
print("=" * 60)

# Create a poor initial guess (far from true values)
poor_initial_params = CircuitParameters(
    resistance_source=5.0,  # Half of true value
    inductance=1e-5,  # 100x too large
    inductance_exponent=0.8,
    resistances=np.array([100.0, 50.0, 30.0]),
    capacitances=np.array([1e-6, 1e-4, 1e-3]),
    exponents=np.array([1.0, 1.0, 1.0]),
)

print("\nTrying quick_fit with poor initial parameters...")
start_time = time.time()
poor_fit = eis_data.quick_fit(
    initial_parameters=poor_initial_params,
    complex_fit=True,
    seed=42,
    max_nfev=1000,
)
poor_time = time.time() - start_time

print(f"Fit with poor initial guess completed in {poor_time:.4f} s")
print(f"Success: {poor_fit.success}")
print(f"Cost: {poor_fit.cost:.6e}")

poor_rs_err, poor_r_err, poor_c_err, poor_exp_err = calc_param_error(
    poor_fit.parameters, true_params
)
print("\nErrors from poor initial guess (percent):")
print(f"  Rs error: {poor_rs_err:.2f}%")
print(f"  Resistance error: {poor_r_err:.2f}%")
print(f"  Capacitance error: {poor_c_err:.2f}%")
print(f"  Exponent error: {poor_exp_err:.2f}%")

print("\n" + "=" * 60)
print("Demo completed!")
print("=" * 60)
print("\nKey takeaways:")
print("- quick_fit is much faster than full iterative fitting")
print("- It works best when starting from reasonable initial parameters")
print("- You can apply bounds to keep parameters in physically realistic ranges")
print("- The standalone quick_fit function is available for script-based workflows")
print("- Even with poor initial guesses, quick_fit can find good solutions")
