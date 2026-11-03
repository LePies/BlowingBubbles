.. _units:

=====================
Units and conventions
=====================

All quantities are SI unless noted otherwise.

Quantities and units
--------------------

===========================  ======  ==========================================
Quantity                     Unit    Notes
===========================  ======  ==========================================
Frequency :math:`f`          Hz      angular frequency :math:`\omega = 2\pi f`
Impedance :math:`Z`          Ohm     area-specific (Ohm cm^2) after normalization
Resistance :math:`R`         Ohm     :math:`R_s` series, :math:`R_i` per branch
Capacitance :math:`C`        F       CPE when :math:`e_i < 1`
Inductance :math:`L`         H
Time constant :math:`\tau`   s       :math:`\tau_i = R_i C_i`
===========================  ======  ==========================================

Conventions
-----------

- **Sign convention.** The measured sign is kept: capacitive behaviour has
  :math:`\mathrm{Im}(Z) < 0`. Nyquist plots show :math:`-\mathrm{Im}(Z)` on the
  vertical axis so that capacitive arcs open upward.
- **Normalization.** Passing an electrode *area* to
  :class:`~blowing_bubbles.classes.EISData` multiplies the impedance by the area,
  giving area-specific impedance in ``Ohm cm^2`` (the default unit ``"Ohm"``
  becomes ``"Ohm cm^2"``; a custom *unit* string is kept as is).
- **Model.** The fitted equivalent circuit is

  .. math::

     Z(\omega) = R_s + (j\omega)^{n_L} L
                 + \sum_{i=1}^{n} \frac{R_i}{1 + R_i C_i (j\omega)^{e_i}}

  with :math:`e_i = 1` for ideal capacitors and :math:`e_i < 1` for
  constant-phase elements (CPEs). See
  :mod:`blowing_bubbles.circuit_impedance` for details.
- **Fitting.** Least-squares residuals are computed on the complex impedance.
  The default ``weighting="modulus"`` divides each residual by :math:`|Z|`
  (relative error); ``weighting="unit"`` uses uniform weights.
