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
    AssertionError
        If types or branch array shapes are inconsistent.
    """
    assert isinstance(freq, (float, int, np.ndarray)), (
        f"freq must be a float, int or np.ndarray. Current type: {type(freq)}"
    )
    assert isinstance(resistance_source, (float, int)), (
        "resistance_source must be a float, int. "
        f"Current type: {type(resistance_source)}"
    )
    assert isinstance(inductance, (float, int)), (
        f"inductance must be a float, int. Current type: {type(inductance)}"
    )
    assert isinstance(inductance_exponent, (float, int)), (
        "inductance_exponent must be a float, int. "
        f"Current type: {type(inductance_exponent)}"
    )
    assert isinstance(resistances, np.ndarray), (
        f"resistances must be a np.ndarray. Current type: {type(resistances)}"
    )
    assert isinstance(capacitances, np.ndarray), (
        f"capacitances must be a np.ndarray. Current type: {type(capacitances)}"
    )
    assert isinstance(exponents, (np.ndarray, type(None))), (
        f"exponents must be a np.ndarray or NoneType. Current type: {type(exponents)}"
    )

    if exponents is None:
        exponents = np.ones_like(capacitances)

    assert resistances.shape == capacitances.shape == exponents.shape, (
        f"resistances, capacitances and exponents must have the same shape. "
        f"Current shapes: {resistances.shape}, {capacitances.shape}, "
        f"{exponents.shape}"
    )

    isfloat = False

    if isinstance(freq, (float, int)):
        freq = np.array([freq])
        isfloat = True
    else:
        freq = np.array(freq)

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

    omega = 2 * np.pi * freq

    parralel_impedances = resistances / (
        1 + resistances * capacitances * (1j * omega[:, None]) ** exponents
    )

    series_impedances = (
        resistance_source
        + (1j * omega) ** inductance_exponent * inductance
        + np.sum(parralel_impedances, axis=1)
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

    gradient = {
        "resistance_source": np.zeros_like(freq),
        "inductance": np.zeros_like(freq),
        "inductance_exponent": np.zeros_like(freq),
        "resistances": np.zeros((freq.shape[0], resistances.shape[0])),
        "capacitances": np.zeros((freq.shape[0], capacitances.shape[0])),
        "exponents": np.zeros((freq.shape[0], exponents.shape[0])),
    }

    omega = 2 * np.pi * freq

    gradient["resistance_source"] = np.ones_like(freq)
    gradient["inductance"] = (1j * omega) ** inductance_exponent
    gradient["inductance_exponent"] = (
        (1j * omega) ** inductance_exponent * np.log(1j * omega) * inductance
    )

    gradient["resistances"] = (
        1 / (1 + resistances * capacitances * (1j * omega[:, None]) ** exponents) ** 2
    )
    gradient["capacitances"] = (
        -(resistances**2)
        * (1j * omega[:, None]) ** exponents
        / (1 + resistances * capacitances * (1j * omega[:, None]) ** exponents) ** 2
    )
    gradient["exponents"] = (
        -(resistances**2)
        * capacitances
        * (1j * omega[:, None]) ** exponents
        * np.log(1j * omega[:, None])
        / (1 + resistances * capacitances * (1j * omega[:, None]) ** exponents) ** 2
    )

    return gradient
