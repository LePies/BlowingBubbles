r"""Impedance evaluation and analytic Jacobians for BlowingBubbles.

The equivalent-circuit model combines ohmic resistance, a constant-phase inductor,
and :math:`n` parallel R–CPE branches (Debye / Cole–Cole style elements):

.. math::

    Z(\omega) = R_s + (j\omega)^{n_L} L
                + \sum_{i=1}^{n} \frac{R_i}{1 + R_i C_i (j\omega)^{e_i}}

with angular frequency :math:`\omega = 2\pi f`. Branch time constants are
:math:`\tau_i = R_i C_i`. Setting :math:`e_i = 1` recovers ideal capacitors;
:math:`e_i < 1` gives constant-phase elements (CPEs).
"""

import numpy as np

_EMPTY_BRANCH_ARRAY = np.zeros(0)


def circuit_assert_arguments(
    freq: float | np.ndarray,
    resistance_source: float = 0,
    inductance: float = 0,
    inductance_exponent: float = 1,
    resistances: np.ndarray = _EMPTY_BRANCH_ARRAY,
    capacitances: np.ndarray = _EMPTY_BRANCH_ARRAY,
    exponents: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float, float, np.ndarray, np.ndarray, np.ndarray, bool]:
    r"""Validate and normalize arguments for :func:`circuit_impedance`.

    Coerces scalar ``freq`` to a length-1 array and defaults ``exponents`` to ones.
    Returns the normalized inputs plus ``isfloat`` (``True`` when a scalar frequency
    was passed so the caller can return a scalar impedance).

    Raises
    ------
    TypeError
        If an argument is not numeric.
    ValueError
        If branch array shapes are inconsistent.
    """
    if isinstance(freq, (list, tuple)):
        freq = np.asarray(freq)
    isfloat = np.ndim(freq) == 0
    try:
        freq = np.atleast_1d(np.asarray(freq, dtype=float))
        resistance_source = float(resistance_source)
        inductance = float(inductance)
        inductance_exponent = float(inductance_exponent)
        resistances = np.asarray(resistances, dtype=float)
        capacitances = np.asarray(capacitances, dtype=float)
        exponents = (
            np.ones_like(capacitances)
            if exponents is None
            else np.asarray(exponents, dtype=float)
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Circuit arguments must be numeric: {exc}") from exc

    if not resistances.shape == capacitances.shape == exponents.shape:
        raise ValueError(
            "resistances, capacitances and exponents must have the same shape. "
            f"Current shapes: {resistances.shape}, {capacitances.shape}, "
            f"{exponents.shape}"
        )

    return (
        freq,
        resistance_source,
        inductance,
        inductance_exponent,
        resistances,
        capacitances,
        exponents,
        isfloat,
    )


def circuit_impedance(
    freq: float | np.ndarray,
    resistance_source: float = 0,
    inductance: float = 0,
    inductance_exponent: float = 1,
    resistances: np.ndarray = _EMPTY_BRANCH_ARRAY,
    capacitances: np.ndarray = _EMPTY_BRANCH_ARRAY,
    exponents: np.ndarray | None = None,
) -> np.complex128 | np.ndarray:
    r"""
    Calculate the impedance of a circuit.

    .. math::
        Z = R_s + (j \omega)^n L + \sum_{i=1}^n \frac{R_i}{1 + R_i C_i (j \omega)^e_i}

    where:
    - :math:`R_s` is the source resistance
    - :math:`L` is the inductance
    - :math:`n` is the inductance exponent
    - :math:`R_i` is the resistance of the i-th parallel branch
    - :math:`C_i` is the capacitance of the i-th parallel branch
    - :math:`e_i` is the exponent of the capacitance of the i-th parallel branch

    Parameters
    ----------
    freq : float | np.ndarray
        Frequency in Hz.
    resistance_source : float
        Resistance of the source in Ohm.
    inductance : float
        Inductance in H.
    inductance_exponent : float
        Exponent of the inductance.
    resistances : np.ndarray
        Resistances in Ohm.
    capacitances : np.ndarray
        Capacitances in F.
    exponents : np.ndarray | None
        Exponents of the capacitances.
    """

    (
        freq,
        resistance_source,
        inductance,
        inductance_exponent,
        resistances,
        capacitances,
        exponents,
        isfloat,
    ) = circuit_assert_arguments(
        freq,
        resistance_source,
        inductance,
        inductance_exponent,
        resistances,
        capacitances,
        exponents,
    )

    s = 1j * (2 * np.pi * freq)

    parallel_impedances = resistances / (
        1 + resistances * capacitances * s[:, None] ** exponents
    )

    series_impedances = (
        resistance_source
        + s**inductance_exponent * inductance
        + np.sum(parallel_impedances, axis=1)
    )

    if isfloat:
        series_impedances = series_impedances[0]

    return series_impedances


def circuit_impedance_parameter_gradient(
    freq: float | np.ndarray,
    resistance_source: float = 0,
    inductance: float = 0,
    inductance_exponent: float = 1,
    resistances: np.ndarray = _EMPTY_BRANCH_ARRAY,
    capacitances: np.ndarray = _EMPTY_BRANCH_ARRAY,
    exponents: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    r"""Analytic partial derivatives of :func:`circuit_impedance` w.r.t. each parameter.

    Returns a dict with keys ``resistance_source``, ``inductance``,
    ``inductance_exponent``, ``resistances``, ``capacitances``, and ``exponents``.
    Branch arrays have shape ``(n_freq, n_branches)``; global terms are
    length ``n_freq``.

    For branch :math:`i`, with :math:`Y_i = 1 + R_i C_i (j\omega)^{e_i}`:

    .. math::

        \frac{\partial Z}{\partial R_i} =
            \frac{1}{Y_i} - \frac{R_i C_i (j\omega)^{e_i}}{Y_i^2}, \quad
        \frac{\partial Z}{\partial C_i} = -\frac{R_i^2 (j\omega)^{e_i}}{Y_i^2}

    Parameters match :func:`circuit_impedance`.
    """
    (
        freq,
        resistance_source,
        inductance,
        inductance_exponent,
        resistances,
        capacitances,
        exponents,
        isfloat,
    ) = circuit_assert_arguments(
        freq,
        resistance_source,
        inductance,
        inductance_exponent,
        resistances,
        capacitances,
        exponents,
    )

    s = 1j * (2 * np.pi * freq)
    log_s = np.log(s)
    s_l = s**inductance_exponent
    S = s[:, None] ** exponents
    D2 = (1 + resistances * capacitances * S) ** 2
    d_capacitances = -(resistances**2) * S / D2

    return {
        "resistance_source": np.ones_like(freq),
        "inductance": s_l,
        "inductance_exponent": s_l * log_s * inductance,
        "resistances": 1 / D2,
        "capacitances": d_capacitances,
        "exponents": d_capacitances * capacitances * log_s[:, None],
    }
