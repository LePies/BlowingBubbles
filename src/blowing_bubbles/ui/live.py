"""Edit circuit parameters for the Live Tune tab (Streamlit-independent).

Build and edit a :class:`CircuitParameters` interactively: tau sliders write
``C = tau / R`` (via :func:`set_branch_tau`) so the characteristic time stays
consistent with R. Also provides :func:`branch_tau`,
:func:`branch_peak_frequency`, :func:`live_residual_rms` and
:func:`default_parameters`.
"""

from __future__ import annotations

import numpy as np

from blowing_bubbles.classes import CircuitParameters


def set_branch_tau(params: CircuitParameters, index: int, tau: float) -> None:
    """Set the product ``R*C`` of branch ``index`` to ``tau`` in place.

    Parameters
    ----------
    params : CircuitParameters
        Circuit to modify.
    index : int
        Branch index in the current branch order.
    tau : float
        Target :math:`R_i C_i` (s).

    Notes
    -----
    Only the capacitance is changed (``C = tau / R``); the branches are not
    re-sorted, so the tau order may no longer hold afterwards.
    """
    params.capacitances[index] = tau / params.resistances[index]


def branch_tau(params: CircuitParameters, index: int) -> float:
    """Return the characteristic product ``R*C`` of branch ``index``.

    Parameters
    ----------
    params : CircuitParameters
        Circuit.
    index : int
        Branch index.

    Returns
    -------
    float
        :math:`R_i C_i` (s).
    """
    return float(params.resistances[index] * params.capacitances[index])


def branch_peak_frequency(params: CircuitParameters, index: int) -> float:
    r"""Return the apex frequency of branch ``index``.

    Parameters
    ----------
    params : CircuitParameters
        Circuit.
    index : int
        Branch index.

    Returns
    -------
    float
        Apex frequency (Hz).

    Notes
    -----
    For an R–CPE branch with exponent :math:`e_i`,

    .. math::

        \tau_c = (R_i C_i)^{1/e_i}, \qquad f = \frac{1}{2\pi\tau_c}.
    """
    tau_c = params.resistances[index] * params.capacitances[index]
    tau_c = float(np.power(tau_c, 1.0 / params.exponents[index]))
    return float(1.0 / (2.0 * np.pi * tau_c))


def live_residual_rms(data, params: CircuitParameters) -> float:
    r"""Return the relative RMS residual of ``params`` against measured data.

    Parameters
    ----------
    data : EISData
        Measured spectrum.
    params : CircuitParameters
        Model evaluated at ``data.Freq``.

    Returns
    -------
    float
        :math:`\sqrt{\sum_k |\hat{Z}_k - Z_k|^2 / \sum_k |Z_k|^2}`, or NaN if
        the data are all zero.
    """
    z_model = params.get_impedance(data.Freq)
    z_data = data.Real + 1j * data.Imag
    denom = np.sum(np.abs(z_data) ** 2)
    if denom == 0.0:
        return float("nan")
    return float(np.sqrt(np.sum(np.abs(z_model - z_data) ** 2) / denom))


def default_parameters(n_branches: int = 3) -> CircuitParameters:
    """Return a reasonable starting circuit for the Live Tune tab.

    Parameters
    ----------
    n_branches : int, default 3
        Number of R–CPE branches.

    Returns
    -------
    CircuitParameters
        :math:`R_s = 10` Ω, :math:`L = 10^{-7}` H, :math:`n_L = 0.95`, and
        branches with :math:`R_i = 20` Ω, :math:`C_i` log-spaced from
        :math:`10^{-4}` to :math:`10^{-2}` F and :math:`e_i = 0.95`.
    """
    return CircuitParameters(
        resistance_source=10.0,
        inductance=1e-7,
        inductance_exponent=0.95,
        resistances=np.full(n_branches, 20.0),
        capacitances=np.logspace(-4, -2, n_branches),
        exponents=np.full(n_branches, 0.95),
    )
