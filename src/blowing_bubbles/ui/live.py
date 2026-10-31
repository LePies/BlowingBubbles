"""Live-tune helpers (Streamlit-independent).

Build and edit a :class:`CircuitParameters` interactively: tau sliders write
``C = tau / R`` so the characteristic time stays consistent with R.
"""

from __future__ import annotations

import numpy as np

from blowing_bubbles.classes import CircuitParameters


def set_branch_tau(params: CircuitParameters, index: int, tau: float) -> None:
    """Set branch ``index`` characteristic product ``R*C = tau`` in place."""
    params.capacitances[index] = tau / params.resistances[index]


def branch_tau(params: CircuitParameters, index: int) -> float:
    """Characteristic product ``R*C`` of branch ``index``."""
    return float(params.resistances[index] * params.capacitances[index])


def branch_peak_frequency(params: CircuitParameters, index: int) -> float:
    """Apex frequency ``f = 1 / (2 pi tau_c)`` of branch ``index`` (Hz)."""
    tau_c = params.resistances[index] * params.capacitances[index]
    tau_c = float(np.power(tau_c, 1.0 / params.exponents[index]))
    return float(1.0 / (2.0 * np.pi * tau_c))


def live_residual_rms(data, params: CircuitParameters) -> float:
    """Relative RMS residual of ``params`` against an :class:`EISData`."""
    z_model = params.get_impedance(data.Freq)
    z_data = data.Real + 1j * data.Imag
    denom = np.sum(np.abs(z_data) ** 2)
    if denom == 0.0:
        return float("nan")
    return float(np.sqrt(np.sum(np.abs(z_model - z_data) ** 2) / denom))


def default_parameters(n_branches: int = 3) -> CircuitParameters:
    """Reasonable starting circuit for the Live Tune tab."""
    return CircuitParameters(
        resistance_source=10.0,
        inductance=1e-7,
        inductance_exponent=0.95,
        resistances=np.full(n_branches, 20.0),
        capacitances=np.logspace(-4, -2, n_branches),
        exponents=np.full(n_branches, 0.95),
    )
