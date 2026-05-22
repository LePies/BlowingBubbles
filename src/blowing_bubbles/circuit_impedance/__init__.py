r"""Forward model and gradients for the BlowingBubbles R–CPE equivalent circuit.

Provides :func:`circuit_impedance` and :func:`circuit_impedance_parameter_gradient`.
"""
from .impedance import circuit_impedance, circuit_impedance_parameter_gradient

__all__ = [
    "circuit_impedance",
    "circuit_impedance_parameter_gradient",
]