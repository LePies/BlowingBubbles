r"""Fitting routines for the BlowingBubbles equivalent circuit.

Public API
----------
fit_circuit
    Iterative fast branch building (IFBB) for an
    :class:`~blowing_bubbles.classes.data_classes.EISData` spectrum.
    See :mod:`blowing_bubbles.scripts.fit_circuit`.
fit_quality, lin_kk
    Label-free fit-quality metrics (Lin-KK residual floor, total-resistance
    check). See :mod:`blowing_bubbles.scripts.quality`.

Notes
-----
Importing the function ``fit_circuit`` here rebinds the package attribute
``blowing_bubbles.scripts.fit_circuit`` to the function. The submodule is
still importable with ``from blowing_bubbles.scripts.fit_circuit import ...``.
"""

from blowing_bubbles.scripts.fit_circuit import fit_circuit
from blowing_bubbles.scripts.quality import fit_quality, lin_kk

__all__ = ["fit_circuit", "fit_quality", "lin_kk"]
