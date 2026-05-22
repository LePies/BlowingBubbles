"""
Circuit fitting demo
====================

Synthesize noisy EIS data from a parametric R–CPE ladder, fit it with
:meth:`~blowing_bubbles.classes.data_classes.EISData.fit` (iterative fast
branch building), and overlay the model on a Nyquist plot.

The model is :math:`[R_s L (R_i C_i)_n]`: series resistance and
non-ideal inductance, plus ``n`` parallel R–CPE branches.
"""

from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from blowing_bubbles.classes import CircuitParameters, EISData

# %%
# Simulate a spectrum

freqs = np.logspace(-5, 5, 1000)
np.random.shuffle(freqs)
freqs_sample = freqs[:100]

true_params = CircuitParameters(
    resistance_source=2.109195e-01,
    inductance=7.713453e-07,
    inductance_exponent=9.500716e-01,
    resistances=np.array([0.0897, 0.0051, 0.2093, 0.0002]),
    capacitances=np.array([0.0128, 1.8721, 0.6132, 1.224]),
    exponents=np.array([1.0, 1.0, 1.0, 1.0]),
)

np.random.seed(42)
n_sample = len(freqs_sample)
z_sample = true_params.get_impedance(freqs_sample)
noise = 1e-3
re_impedance_sample = z_sample.real + np.random.normal(0, noise, n_sample)
im_impedance_sample = z_sample.imag + np.random.normal(0, noise, n_sample)

data_sample = pd.DataFrame(
    {
        "Freq (Hz)": freqs_sample,
        "Zreal (Ohm)": re_impedance_sample,
        "Zimag (Ohm)": im_impedance_sample,
    }
)

# %%
# Fit the circuit (IFBB: start from peaks, grow branches up to ``rc_groups``)

eis_data = EISData(data_sample)

print("Fitting started (2 peaks -> up to 3 R–CPE branches)...")
start_time = time.time()
fit_result = eis_data.fit(
    n_restarts=100,
    rc_groups=3,
    verbose=True,
    seed=42,
)
elapsed = time.time() - start_time
print(f"Fitting completed in {elapsed:.2f} s.")
print(f"Success: {fit_result.success}")
print(f"Final cost: {fit_result.cost:.6e}")
print(fit_result.parameters)

# %%
# Nyquist plot: measured points and full-model overlay

freqs_plot = np.logspace(-5, 5, 100)
unit = r"$\Omega$"

fig, ax = plt.subplots()
eis_data.plot_nyquist(axes=ax)
fit_result.parameters.plot_full_fit(freqs_plot, ax, unit=unit)
ax.legend()
ax.set_title("Synthetic EIS: measurement vs. fit")
plt.tight_layout()
plt.show()
