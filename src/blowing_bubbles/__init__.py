r"""BlowingBubbles — equivalent-circuit EIS modeling and documentation.

Subpackages
-----------
classes
    Parameter containers and fit results (:mod:`blowing_bubbles.classes`).
circuit_impedance
    Impedance forward model and analytic gradients
    (:mod:`blowing_bubbles.circuit_impedance`).
scripts
    Fitting routines (:mod:`blowing_bubbles.scripts`).

License
-------
MIT (see ``LICENSE``). Cite ``CITATION.bib`` in published scientific work;
see ``ACADEMIC_CITATION.md``.
"""

__version__ = "0.2.0"

__all__ = ["classes", "circuit_impedance", "scripts", "__version__"]
