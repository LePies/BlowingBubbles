r"""Impedance evaluation and analytic Jacobians for BlowingBubbles.

The equivalent-circuit model combines ohmic resistance, a constant-phase inductor,
and :math:`n` parallel R–CPE branches (Debye / Cole–Cole style elements):

.. math::

    Z(\omega) = R_s + (j\omega)^{n_L} L
                + \sum_{i=1}^{n} \frac{R_i}{1 + R_i C_i (j\omega)^{e_i}}

with angular frequency :math:`\omega = 2\pi f`. Branch time constants are
:math:`\tau_i = R_i C_i`. Setting :math:`e_i = 1` recovers ideal capacitors;
:math:`e_i < 1` gives constant-phase elements (CPEs).

Public entry points are :func:`circuit_impedance` (forward model) and
:func:`circuit_impedance_parameter_gradient` (analytic partial derivatives);
:func:`circuit_assert_arguments` validates their shared arguments.
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
    r"""Validate and normalize the arguments of :func:`circuit_impedance`.

    Converts all inputs to floats / float arrays, coerces a scalar *freq* to a
    length-1 array and defaults *exponents* to ones.

    Parameters
    ----------
    freq : float or numpy.ndarray
        Frequency or frequencies (Hz); lists and tuples are accepted.
    resistance_source, inductance, inductance_exponent : float
        Global parameters :math:`R_s`, :math:`L`, :math:`n_L`.
    resistances, capacitances : numpy.ndarray, shape (n_branches,)
        Branch parameters :math:`R_i`, :math:`C_i`.
    exponents : numpy.ndarray, shape (n_branches,), optional
        Branch exponents :math:`e_i`. ``None`` (default) means all ones.

    Returns
    -------
    freq : numpy.ndarray, shape (n_freq,)
        Frequencies as a 1-D float array.
    resistance_source, inductance, inductance_exponent : float
        Global parameters as floats.
    resistances, capacitances, exponents : numpy.ndarray
        Branch parameters as float arrays.
    isfloat : bool
        ``True`` when a scalar frequency was passed, so the caller can return
        a scalar impedance.

    Raises
    ------
    TypeError
        If an argument cannot be converted to a float or float array.
    ValueError
        If *resistances*, *capacitances* and *exponents* differ in shape.
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
    r"""Calculate the impedance of the R–CPE ladder circuit.

    Parameters
    ----------
    freq : float or numpy.ndarray
        Frequency or frequencies :math:`f` (Hz); lists and tuples are accepted.
    resistance_source : float, default 0
        Series (source) resistance :math:`R_s` (Ω).
    inductance : float, default 0
        Inductance :math:`L` (H for :math:`n_L = 1`).
    inductance_exponent : float, default 1
        Inductor exponent :math:`n_L`.
    resistances : numpy.ndarray, shape (n_branches,), default empty
        Branch resistances :math:`R_i` (Ω).
    capacitances : numpy.ndarray, shape (n_branches,), default empty
        Branch capacitances :math:`C_i` (F for :math:`e_i = 1`).
    exponents : numpy.ndarray, shape (n_branches,), optional
        Branch CPE exponents :math:`e_i`. ``None`` (default) means all ones
        (ideal capacitors).

    Returns
    -------
    Z : numpy.complex128 or numpy.ndarray
        Complex impedance; a scalar if *freq* is a scalar, otherwise an array
        of shape ``(n_freq,)``.

    See Also
    --------
    circuit_assert_arguments : Argument validation (raises ``TypeError`` for
        non-numeric input and ``ValueError`` for mismatched branch shapes).
    circuit_impedance_parameter_gradient : Analytic partial derivatives.

    Notes
    -----
    .. math::

        Z(\omega) = R_s + (j\omega)^{n_L} L
                    + \sum_{i=1}^{n} \frac{R_i}{1 + R_i C_i (j\omega)^{e_i}}

    with :math:`\omega = 2\pi f` and :math:`n` the number of branches. With
    the default arguments the circuit is empty and :math:`Z = 0`. Capacitive
    branches give :math:`\mathrm{Im}(Z) < 0`, the inductor
    :math:`\mathrm{Im}(Z) > 0`.
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
    r"""Return the analytic partial derivatives of :func:`circuit_impedance`.

    Parameters
    ----------
    freq : float or numpy.ndarray
        Frequency or frequencies :math:`f` (Hz).
    resistance_source, inductance, inductance_exponent : float
        Global parameters :math:`R_s` (Ω), :math:`L`, :math:`n_L`; defaults
        0, 0, 1.
    resistances, capacitances : numpy.ndarray, shape (n_branches,)
        Branch parameters :math:`R_i` (Ω), :math:`C_i`; default empty.
    exponents : numpy.ndarray, shape (n_branches,), optional
        Branch exponents :math:`e_i`. ``None`` (default) means all ones.

    Returns
    -------
    gradient : dict of str to numpy.ndarray
        Partial derivatives :math:`\partial Z / \partial p`, keyed by
        parameter name: ``resistance_source``, ``inductance`` and
        ``inductance_exponent`` with shape ``(n_freq,)``; ``resistances``,
        ``capacitances`` and ``exponents`` with shape
        ``(n_freq, n_branches)``. Arrays are returned even for a scalar
        *freq* (``n_freq = 1``).

    See Also
    --------
    circuit_impedance : The differentiated model.
    circuit_assert_arguments : Argument validation.

    Notes
    -----
    With :math:`s = j\omega` and :math:`Y_i = 1 + R_i C_i s^{e_i}`:

    .. math::

        \frac{\partial Z}{\partial R_s} = 1, \quad
        \frac{\partial Z}{\partial L} = s^{n_L}, \quad
        \frac{\partial Z}{\partial n_L} = s^{n_L} L \ln s,

        \frac{\partial Z}{\partial R_i} =
            \frac{1}{Y_i} - \frac{R_i C_i s^{e_i}}{Y_i^2} = \frac{1}{Y_i^2},
        \quad
        \frac{\partial Z}{\partial C_i} = -\frac{R_i^2 s^{e_i}}{Y_i^2},
        \quad
        \frac{\partial Z}{\partial e_i} =
            -\frac{R_i^2 C_i s^{e_i} \ln s}{Y_i^2}.

    The derivative with respect to :math:`R_s` is returned as a real array.
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
