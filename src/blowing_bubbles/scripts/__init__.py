r"""Fitting routines for the BlowingBubbles equivalent circuit.

Re-exports ``fit_circuit`` (iterative fast branch building for an
:class:`~blowing_bubbles.classes.data_classes.EISData` spectrum, from
:mod:`blowing_bubbles.scripts.fit_circuit`) and ``fit_quality`` / ``lin_kk``
(label-free fit-quality metrics: Lin-KK residual floor and total-resistance
check, from :mod:`blowing_bubbles.scripts.quality`). The least-squares core
lives in :mod:`blowing_bubbles.scripts.global_minimize`.

Importing the function ``fit_circuit`` here rebinds the package attribute
``blowing_bubbles.scripts.fit_circuit`` to the function. The submodule is
still importable with ``from blowing_bubbles.scripts.fit_circuit import ...``.
"""

from blowing_bubbles.scripts.fit_circuit import fit_circuit
from blowing_bubbles.scripts.quality import fit_quality, lin_kk

__all__ = ["fit_circuit", "fit_quality", "lin_kk"]
