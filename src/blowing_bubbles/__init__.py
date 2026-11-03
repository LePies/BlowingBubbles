r"""BlowingBubbles — equivalent-circuit fitting of electrochemical impedance spectra.

The main entry point is :class:`~blowing_bubbles.classes.EISData` (load, process
and fit a spectrum); fitted parameters are held in
:class:`~blowing_bubbles.classes.CircuitParameters` and
:class:`~blowing_bubbles.classes.FitResults`. Subpackages:

- :mod:`blowing_bubbles.classes` — data, parameter and fit-result containers;
- :mod:`blowing_bubbles.circuit_impedance` — impedance forward model and
  analytic gradients;
- :mod:`blowing_bubbles.scripts` — fitting routines and fit-quality metrics.

Licensed under MIT (see ``LICENSE``). Cite ``CITATION.bib`` in published
scientific work; see ``ACADEMIC_CITATION.md``.
"""

__version__ = "0.2.0"

__all__ = ["classes", "circuit_impedance", "scripts", "__version__"]
