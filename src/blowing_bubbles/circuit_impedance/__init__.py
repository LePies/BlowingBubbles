r"""Forward model and gradients for the BlowingBubbles R–CPE equivalent circuit.

Re-exports :func:`~blowing_bubbles.circuit_impedance.impedance.circuit_impedance`
(complex impedance of the circuit) and
:func:`~blowing_bubbles.circuit_impedance.impedance.circuit_impedance_parameter_gradient`
(analytic partial derivatives) from
:mod:`blowing_bubbles.circuit_impedance.impedance`.
"""

from .impedance import circuit_impedance, circuit_impedance_parameter_gradient

__all__ = [
    "circuit_impedance",
    "circuit_impedance_parameter_gradient",
]
