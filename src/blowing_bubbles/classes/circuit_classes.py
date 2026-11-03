r"""Parameter containers and fit result wrappers for equivalent-circuit EIS.

:class:`CircuitParameters` holds the parameters of the R–CPE ladder (with JSON
save/load and Nyquist plotting), :class:`FitResults` wraps a SciPy least-squares
result together with parameter uncertainties, and :func:`align_axes_height`
lines up subplot boxes. See :mod:`blowing_bubbles.circuit_impedance` for the
impedance model.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, overload

import numpy as np
from scipy.optimize import OptimizeResult

if TYPE_CHECKING:
    from matplotlib import pyplot as plt
    from matplotlib.axes import Axes

from blowing_bubbles.circuit_impedance import circuit_impedance

_CIRCUIT_PARAMS_FORMAT_VERSION = 1
_CIRCUIT_PARAMS_FILE_TYPE = "blowing_bubbles.CircuitParameters"
_FIT_LABEL = r"$[R_s L(R_i C_i)]$ fit"
_TEXT_BBOX = {
    "facecolor": "white",
    "alpha": 0.8,
    "edgecolor": "none",
    "boxstyle": "round,pad=0.2",
}


def align_axes_height(
    axes: list[Axes] | tuple[Axes, ...],
    reference: Axes | None = None,
) -> None:
    """Match the box height and bottom edge of subplots to a reference axes.

    Call after :meth:`matplotlib.figure.Figure.tight_layout`, which would
    otherwise override the positions again.

    Parameters
    ----------
    axes : list or tuple of matplotlib.axes.Axes
        Axes to align; their positions are modified in place (x position and
        width are kept).
    reference : matplotlib.axes.Axes, optional
        Axes whose bottom edge and height are copied. ``None`` (default) uses
        the last element of *axes*.
    """
    if reference is None:
        reference = axes[-1]
    ref = reference.get_position()
    for ax in axes:
        pos = ax.get_position()
        ax.set_position((pos.x0, ref.y0, pos.width, ref.height))


def _plot_full_fit(
    freqs: np.ndarray,
    parameters: CircuitParameters,
    axis: plt.Axes | None = None,
    unit: str | None = None,
):
    """Draw the total model curve and each branch arc; see ``plot_full_fit``.

    The total curve is evaluated at *freqs*; each single-branch arc is drawn on
    a fixed grid ``logspace(-10, 10, 500)`` Hz, shifted right by :math:`R_s`
    plus the resistances of the preceding (smaller-tau) branches. Only the
    inductance term is included in the first arc. Returns the axes drawn on.
    """
    if unit is None:
        warnings.warn(
            "No unit provided, using default unit: Omega cm^2",
            stacklevel=2,
        )
        unit = r"$\Omega$ cm$^{2}$"
    if axis is None:
        from matplotlib import pyplot as plt

        _, axis = plt.subplots()
    Z_pred = parameters.get_impedance(freqs)

    axis.plot(
        Z_pred.real,
        -Z_pred.imag,
        label=_FIT_LABEL,
        color="black",
        ls="--",
    )

    n_groups = len(parameters)

    for i in range(n_groups):
        rs_i = parameters.resistance_source + np.sum(parameters.resistances[:i])
        ri_i = parameters.resistances[i]
        branch_freq = np.logspace(-10, 10, 500)
        branch_r = np.array([ri_i])
        branch_c = np.array([parameters.capacitances[i]])
        branch_e = np.array([parameters.exponents[i]])

        if i == 0:
            # With inductance
            _Zi_pred_L = np.atleast_1d(
                circuit_impedance(
                    branch_freq,
                    resistance_source=rs_i,
                    resistances=branch_r,
                    capacitances=branch_c,
                    exponents=branch_e,
                    inductance=parameters.inductance,
                    inductance_exponent=parameters.inductance_exponent,
                )
            )
            mask_L = _Zi_pred_L.imag < 0
            axis.plot(
                _Zi_pred_L.real[mask_L],
                -_Zi_pred_L.imag[mask_L],
                label=rf"$(R_{n_groups} C_{n_groups})$ fit with one group (with $L$)",
            )
            # Calculate and print R1 at the midpoint of this arc, at y=0
            xline = _Zi_pred_L.real[mask_L]
            if len(xline) > 1:
                xmid = rs_i + ri_i / 2
                axis.text(
                    xmid,
                    -np.max(np.abs(Z_pred.imag)) * 0.2,
                    rf"$R_{n_groups}$",
                    fontsize=12,
                    va="bottom",
                    ha="center",
                    color=axis.lines[-1].get_color(),
                    fontweight="bold",
                    bbox=_TEXT_BBOX,
                )
        else:
            _Zi_pred = np.atleast_1d(
                circuit_impedance(
                    branch_freq,
                    resistance_source=rs_i,
                    resistances=branch_r,
                    capacitances=branch_c,
                    exponents=branch_e,
                    inductance=0,
                    inductance_exponent=1,
                )
            )
            mask = _Zi_pred.imag < 0
            axis.plot(
                _Zi_pred.real[mask],
                -_Zi_pred.imag[mask],
                label=rf"$(R_{n_groups - i} C_{n_groups - i})$ fit with one group",
            )
            # Calculate and print Ri at the midpoint of this arc, at y=0
            xline = _Zi_pred.real[mask]
            if len(xline) > 1:
                xmid = rs_i + ri_i / 2
                axis.text(
                    xmid,
                    -np.max(np.abs(Z_pred.imag)) * 0.2,
                    rf"$R_{n_groups - i}$",
                    fontsize=12,
                    va="bottom",
                    ha="center",
                    color=axis.lines[-1].get_color(),
                    fontweight="bold",
                    bbox=_TEXT_BBOX,
                )

    # datalim: equal aspect adjusts limits, not axes box height
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(True)
    axis.spines["right"].set_visible(False)
    axis.spines["top"].set_visible(False)
    axis.set_xlabel("Z' " + unit)
    axis.set_ylabel("-Z'' " + unit)
    return axis


def _tau_sort_index(resistances: np.ndarray, capacitances: np.ndarray) -> np.ndarray:
    r"""Branch order used by :meth:`CircuitParameters.sort_by_tau` (by :math:`R C`)."""
    return np.argsort(np.asarray(capacitances) * np.asarray(resistances))


class CircuitParameters:
    r"""Parameters of the R–CPE ladder used in BlowingBubbles EIS fitting.

    Parameters
    ----------
    resistance_source : float
        Series resistance :math:`R_s` (Ω).
    inductance : float
        Inductance :math:`L` (H for :math:`n_L = 1`) of the constant-phase
        inductor term.
    inductance_exponent : float
        Inductor exponent :math:`n_L`.
    resistances : numpy.ndarray, shape (n_branches,)
        Branch resistances :math:`R_i` (Ω).
    capacitances : numpy.ndarray, shape (n_branches,)
        Branch (pseudo-)capacitances :math:`C_i` (F for :math:`e_i = 1`).
    exponents : numpy.ndarray, shape (n_branches,)
        Branch CPE exponents :math:`e_i` (1 for an ideal capacitor).

    Attributes
    ----------
    resistance_source, inductance, inductance_exponent : float
        Global parameters :math:`R_s`, :math:`L`, :math:`n_L`.
    resistances, capacitances, exponents : numpy.ndarray, shape (n_branches,)
        Branch parameters, sorted by increasing :math:`\tau_i`.
    taus : numpy.ndarray, shape (n_branches,)
        Products :math:`\tau_i = R_i C_i` (units :math:`\mathrm{s}^{e_i}`),
        kept in sync with the branch arrays.

    See Also
    --------
    FitResults : Fit result holding a fitted ``CircuitParameters``.
    blowing_bubbles.circuit_impedance.circuit_impedance : Impedance model.

    Notes
    -----
    The impedance model is (see
    :func:`~blowing_bubbles.circuit_impedance.circuit_impedance`):

    .. math::

        Z(\omega) = R_s + (j\omega)^{n_L} L
                    + \sum_{i=1}^{n} \frac{R_i}{1 + R_i C_i (j\omega)^{e_i}}

    with :math:`\omega = 2\pi f`. The constructor copies the input arrays and
    sorts the branches by increasing :math:`\tau_i = R_i C_i`; the mutating
    methods (:meth:`add_branch`, :meth:`drop_branch`, :meth:`pop_branch`)
    preserve this order, while :meth:`sort_by_capacitance` replaces it. For
    area-normalized spectra the resistances are in Ω·cm² and the capacitances
    in F/cm².
    """

    def __init__(
        self,
        resistance_source: float,
        inductance: float,
        inductance_exponent: float,
        resistances: np.ndarray,
        capacitances: np.ndarray,
        exponents: np.ndarray,
    ):
        r"""Build parameters and sort branches by :math:`\tau_i = R_i C_i`."""
        self.resistance_source = float(resistance_source)
        self.inductance = float(inductance)
        self.inductance_exponent = float(inductance_exponent)
        self.resistances = np.array(resistances)
        self.capacitances = np.array(capacitances)
        self.exponents = np.array(exponents)
        self.taus = self.capacitances * self.resistances
        self.sort_by_tau()

    @property
    def characteristic_times(self) -> np.ndarray:
        r"""Characteristic time constants :math:`\tau_{c,i} = (R_i C_i)^{1/e_i}` (s).

        Array of shape ``(n_branches,)`` in the stored branch order.
        ``taus`` stores the product :math:`R_i C_i` (units
        :math:`\mathrm{s}^{e_i}`), which is what the optimiser orders and
        parameterises. The apex of branch :math:`i` lies at
        :math:`\omega_i = 1/\tau_{c,i}`.
        """
        return np.power(self.resistances * self.capacitances, 1.0 / self.exponents)

    @property
    def peak_frequencies(self) -> np.ndarray:
        r"""Apex frequencies :math:`f_i = 1 / (2\pi\,\tau_{c,i})` (Hz).

        Array of shape ``(n_branches,)``; see :attr:`characteristic_times`.
        """
        return 1.0 / (2.0 * np.pi * self.characteristic_times)

    def sort_by_tau(self):
        r"""Sort the branches by increasing :math:`\tau_i = R_i C_i`, in place.

        ``resistances``, ``capacitances``, ``exponents`` and ``taus`` are
        reordered together.
        """
        sort_idx = _tau_sort_index(self.resistances, self.capacitances)
        self.resistances = self.resistances[sort_idx]
        self.capacitances = self.capacitances[sort_idx]
        self.exponents = self.exponents[sort_idx]
        self.taus = self.capacitances * self.resistances

    def sort_by_capacitance(self):
        """Sort the branches by increasing capacitance, in place.

        ``resistances``, ``capacitances``, ``exponents`` and ``taus`` are
        reordered together.

        Notes
        -----
        This replaces the tau order established by the constructor. Methods
        that refer to tau order (e.g. :meth:`pop_branch`, which removes the
        last branch) then act on the capacitance order instead.
        """
        sort_idx = np.argsort(self.capacitances)
        self.resistances = self.resistances[sort_idx]
        self.capacitances = self.capacitances[sort_idx]
        self.exponents = self.exponents[sort_idx]
        self.taus = self.capacitances * self.resistances

    def _set_branches(
        self,
        resistances: np.ndarray,
        capacitances: np.ndarray,
        exponents: np.ndarray,
    ) -> None:
        """Replace the branch arrays and recompute ``taus`` (no re-sorting)."""
        self.resistances = np.asarray(resistances, dtype=float)
        self.capacitances = np.asarray(capacitances, dtype=float)
        self.exponents = np.asarray(exponents, dtype=float)
        self.taus = self.capacitances * self.resistances

    def add_branch(
        self,
        resistance: float,
        capacitance: float,
        exponent: float = 1.0,
    ) -> None:
        """Add an R–CPE branch and re-sort the branches by tau, in place.

        Parameters
        ----------
        resistance : float
            Branch resistance :math:`R` (Ω).
        capacitance : float
            Branch capacitance :math:`C` (F for an ideal capacitor).
        exponent : float, default 1.0
            CPE exponent :math:`e` (1 gives an ideal RC branch).
        """
        self._set_branches(
            np.append(self.resistances, resistance),
            np.append(self.capacitances, capacitance),
            np.append(self.exponents, exponent),
        )
        self.sort_by_tau()

    def drop_branch(self, index: int) -> None:
        """Remove the branch at *index*, in place.

        Parameters
        ----------
        index : int
            Position in the stored (tau-sorted) branch order; negative indices
            count from the end.

        Raises
        ------
        IndexError
            If *index* is out of range for the number of branches.
        """
        n = len(self)
        if index < 0:
            index += n
        if not 0 <= index < n:
            raise IndexError(f"branch index {index} out of range for {n} branches")
        keep = np.delete(np.arange(n), index)
        self._set_branches(
            self.resistances[keep],
            self.capacitances[keep],
            self.exponents[keep],
        )

    def pop_branch(self) -> tuple[float, float, float]:
        """Remove and return the last branch (largest tau), in place.

        Returns
        -------
        resistance : float
            Resistance of the removed branch (Ω).
        capacitance : float
            Capacitance of the removed branch.
        exponent : float
            CPE exponent of the removed branch.

        Raises
        ------
        IndexError
            If there are no branches.
        """
        if len(self) == 0:
            raise IndexError("no branches to pop")
        r = float(self.resistances[-1])
        c = float(self.capacitances[-1])
        e = float(self.exponents[-1])
        self.drop_branch(-1)
        return r, c, e

    def as_dict(self) -> dict:
        """Return the parameters as a flat dict of floats and arrays.

        Returns
        -------
        params : dict
            Keys ``resistance_source``, ``inductance``, ``inductance_exponent``
            (float) and ``resistances``, ``capacitances``, ``exponents``,
            ``taus`` (numpy.ndarray, shape ``(n_branches,)``). The arrays are
            the instance's own arrays, not copies.

        See Also
        --------
        to_dict : JSON-friendly nested layout used by :meth:`save`.
        """
        return {
            "resistance_source": self.resistance_source,
            "inductance": self.inductance,
            "inductance_exponent": self.inductance_exponent,
            "resistances": self.resistances,
            "capacitances": self.capacitances,
            "exponents": self.exponents,
            "taus": self.taus,
        }

    def __len__(self):
        """Return the number of parallel branches."""
        return len(self.taus)

    def __str__(self):
        """Return a multi-line, fixed-width summary of all parameters."""
        # Format floats to 8 chars (sign, int, dot, decimals).

        def _format_arr(arr):
            if isinstance(arr, np.ndarray):
                # For 1D array of floats
                return "[" + " ".join([f"{x:10.4e}" for x in arr]) + "]"

            else:
                return str(arr)

        return (
            f"  Resistance Source:      {self.resistance_source:12.4e}\n"
            f"  Inductance:             {self.inductance:12.4e}\n"
            f"  Inductance Exponent:    {self.inductance_exponent:12.4e}\n"
            f"  Resistances:            {_format_arr(self.resistances)}\n"
            f"  Capacitances:           {_format_arr(self.capacitances)}\n"
            f"  Exponents:              {_format_arr(self.exponents)}\n"
            f"  Taus:                   {_format_arr(self.taus)}\n"
        )

    def to_array(self):
        """Return the parameters as a flat vector.

        Returns
        -------
        array : numpy.ndarray, shape (3 + 3 * n_branches,)
            ``[R_s, L, n_L, R_1..R_n, C_1..C_n, e_1..e_n]`` with the branches
            in the stored (tau-sorted) order.

        See Also
        --------
        from_array : Inverse operation.
        """
        return np.array(
            [
                self.resistance_source,
                self.inductance,
                self.inductance_exponent,
                *self.resistances,
                *self.capacitances,
                *self.exponents,
            ]
        )

    def copy(self):
        """Return an independent copy (branch arrays are copied).

        Returns
        -------
        params : CircuitParameters
            New instance with the same parameter values.
        """
        return CircuitParameters(
            resistance_source=self.resistance_source,
            inductance=self.inductance,
            inductance_exponent=self.inductance_exponent,
            resistances=self.resistances.copy(),
            capacitances=self.capacitances.copy(),
            exponents=self.exponents.copy(),
        )

    @classmethod
    def from_array(cls, array: np.ndarray, n_groups: int) -> CircuitParameters:
        """Build parameters from a flat vector (inverse of :meth:`to_array`).

        Parameters
        ----------
        array : numpy.ndarray, shape (3 + 3 * n_groups,)
            ``[R_s, L, n_L, R_1..R_n, C_1..C_n, e_1..e_n]``. All entries after
            the capacitances are taken as exponents.
        n_groups : int
            Number of parallel branches :math:`n`.

        Returns
        -------
        params : CircuitParameters
            New instance; branches are re-sorted by tau.
        """
        return cls(
            resistance_source=array[0],
            inductance=array[1],
            inductance_exponent=array[2],
            resistances=array[3 : 3 + n_groups],
            capacitances=array[3 + n_groups : 3 + 2 * n_groups],
            exponents=array[3 + 2 * n_groups :],
        )

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict (branches in the stored tau order).

        Returns
        -------
        data : dict
            Nested dict of plain Python floats, see Notes.

        See Also
        --------
        from_dict : Inverse operation.
        save : Write this dict to a JSON file.

        Notes
        -----
        Layout (``tau`` is informational and ignored by :meth:`from_dict`)::

            {
              "format_version": 1,
              "type": "blowing_bubbles.CircuitParameters",
              "global": {"resistance_source": ..., "inductance": ...,
                         "inductance_exponent": ...},
              "branches": [
                {"resistance": ..., "capacitance": ..., "exponent": ...,
                 "tau": ...},
                ...
              ]
            }
        """
        return {
            "format_version": _CIRCUIT_PARAMS_FORMAT_VERSION,
            "type": _CIRCUIT_PARAMS_FILE_TYPE,
            "global": {
                "resistance_source": self.resistance_source,
                "inductance": self.inductance,
                "inductance_exponent": self.inductance_exponent,
            },
            "branches": [
                {
                    "resistance": float(r),
                    "capacitance": float(c),
                    "exponent": float(e),
                    "tau": float(t),
                }
                for r, c, e, t in zip(
                    self.resistances,
                    self.capacitances,
                    self.exponents,
                    self.taus,
                    strict=False,
                )
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> CircuitParameters:
        """Build parameters from a dict produced by :meth:`to_dict`.

        Parameters
        ----------
        data : dict
            Either the nested layout of :meth:`to_dict` (recognized by its
            ``"type"`` and ``"global"`` keys) or the legacy flat layout with
            keys ``resistance_source``, ``inductance``, ``inductance_exponent``,
            ``resistances``, ``capacitances`` and ``exponents`` (as returned by
            :meth:`as_dict`).

        Returns
        -------
        params : CircuitParameters
            New instance; branches are re-sorted by tau.
        """
        if data.get("type") == _CIRCUIT_PARAMS_FILE_TYPE and "global" in data:
            g = data["global"]
            branches = data.get("branches", [])
            return cls(
                resistance_source=g["resistance_source"],
                inductance=g["inductance"],
                inductance_exponent=g["inductance_exponent"],
                resistances=np.array([b["resistance"] for b in branches]),
                capacitances=np.array([b["capacitance"] for b in branches]),
                exponents=np.array([b["exponent"] for b in branches]),
            )
        # Legacy / flat layout (arrays at top level)
        return cls(
            resistance_source=data["resistance_source"],
            inductance=data["inductance"],
            inductance_exponent=data["inductance_exponent"],
            resistances=data["resistances"],
            capacitances=data["capacitances"],
            exponents=data["exponents"],
        )

    def save(self, path: str | Path) -> None:
        """Write the parameters to a human-readable JSON file.

        Parameters
        ----------
        path : str or pathlib.Path
            Output file. Missing parent directories are created and an
            existing file is overwritten.

        See Also
        --------
        load : Read the file back.
        to_dict : Layout of the written JSON object.

        Notes
        -----
        The file is UTF-8 encoded, indented by two spaces and ends with a
        newline. Branch rows include R, C, exponent and tau (tau is
        informational only).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")

    @classmethod
    def load(cls, path: str | Path) -> CircuitParameters:
        """Load parameters from a JSON file written by :meth:`save`.

        Parameters
        ----------
        path : str or pathlib.Path
            JSON file in the :meth:`to_dict` layout (the legacy flat layout
            is also accepted, see :meth:`from_dict`).

        Returns
        -------
        params : CircuitParameters
            Loaded parameters, branches sorted by tau.
        """
        with Path(path).open(encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @overload
    def get_impedance(self, freq: float) -> np.complex128: ...

    @overload
    def get_impedance(self, freq: np.ndarray) -> np.ndarray: ...

    def get_impedance(self, freq: float | np.ndarray) -> np.complex128 | np.ndarray:
        r"""Evaluate the model impedance :math:`Z(\omega)` at *freq*.

        Parameters
        ----------
        freq : float or numpy.ndarray
            Frequency or frequencies :math:`f` (Hz), with
            :math:`\omega = 2\pi f`.

        Returns
        -------
        Z : numpy.complex128 or numpy.ndarray
            Complex impedance (same unit as the resistances); a scalar for a
            scalar *freq*, otherwise an array with one entry per frequency.
            Capacitive behaviour gives :math:`\mathrm{Im}(Z) < 0`.

        See Also
        --------
        blowing_bubbles.circuit_impedance.circuit_impedance : Evaluated model.
        """
        return circuit_impedance(
            freq,
            self.resistance_source,
            self.inductance,
            self.inductance_exponent,
            self.resistances,
            self.capacitances,
            self.exponents,
        )

    def plot_nyquist(
        self,
        freq: np.ndarray | None = None,
        axis: plt.Axes | None = None,
        unit: str | None = None,
    ):
        r"""Plot the full-model Nyquist curve (:math:`Z'` vs. :math:`-Z''`).

        Parameters
        ----------
        freq : numpy.ndarray, optional
            Frequencies (Hz) at which the model is evaluated. ``None``
            (default) uses ``numpy.logspace(-5, 5, 1000)``.
        axis : matplotlib.axes.Axes, optional
            Axes to draw on. ``None`` (default) creates a new figure and axes.
        unit : str, optional
            Unit appended to the axis labels. ``None`` (default) uses
            :math:`\Omega\,\mathrm{cm}^2` and emits a warning.

        Returns
        -------
        axis : matplotlib.axes.Axes
            The axes drawn on (the given one or the newly created one).

        Warns
        -----
        UserWarning
            If *unit* is ``None``.

        See Also
        --------
        plot_full_fit : Same curve plus the individual branch arcs.

        Notes
        -----
        The curve is drawn as a dashed black line, the aspect ratio is set to
        equal, a grid is enabled and the top and right spines are hidden.
        """
        if unit is None:
            warnings.warn(
                "No unit provided, using default unit: Omega cm^2",
                stacklevel=2,
            )
            unit = r"$\Omega$ cm$^{2}$"
        if axis is None:
            from matplotlib import pyplot as plt

            _, axis = plt.subplots()
        if freq is None:
            freq = np.logspace(-5, 5, 1000)
        Z_pred = self.get_impedance(freq)
        axis.plot(
            Z_pred.real,
            -Z_pred.imag,
            label=_FIT_LABEL,
            color="black",
            ls="--",
        )
        axis.set_aspect("equal")
        axis.grid(True)
        axis.spines["right"].set_visible(False)
        axis.spines["top"].set_visible(False)
        axis.set_xlabel("Z' " + unit)
        axis.set_ylabel("-Z'' " + unit)
        return axis

    def plot_full_fit(
        self,
        freq: np.ndarray | None = None,
        axis: plt.Axes | None = None,
        unit: str | None = None,
    ):
        r"""Plot the total model Nyquist curve plus the decomposed branch arcs.

        Overlays the combined model (dashed black) and single-branch
        contributions, with a resistance label below each semicircle.

        Parameters
        ----------
        freq : numpy.ndarray, optional
            Frequencies (Hz) for the total model curve. ``None`` (default) uses
            ``numpy.logspace(-5, 5, 1000)``.
        axis : matplotlib.axes.Axes, optional
            Axes to draw on. ``None`` (default) creates a new figure and axes.
        unit : str, optional
            Unit appended to the axis labels. ``None`` (default) uses
            :math:`\Omega\,\mathrm{cm}^2` and emits a warning.

        Returns
        -------
        axis : matplotlib.axes.Axes
            The axes drawn on (the given one or the newly created one).

        Warns
        -----
        UserWarning
            If *unit* is ``None``.

        See Also
        --------
        plot_nyquist : Total model curve only.

        Notes
        -----
        Each branch arc is evaluated on the fixed grid
        ``numpy.logspace(-10, 10, 500)`` Hz (independent of *freq*) and shifted
        along :math:`Z'` by :math:`R_s` plus the resistances of the preceding
        branches; only its capacitive part (:math:`\mathrm{Im}(Z) < 0`) is
        drawn. The inductance is included only in the first (smallest-tau) arc.
        Labels number the branches in reverse tau order: the smallest-tau
        branch is labelled :math:`R_n`, the largest-tau branch :math:`R_1`.
        The aspect ratio is set to equal (adjusting the data limits).
        """
        if freq is None:
            freq = np.logspace(-5, 5, 1000)
        return _plot_full_fit(freq, self, axis, unit)


class FitResults(OptimizeResult):
    r"""Result of an equivalent-circuit fit, built from a SciPy least-squares result.

    Parameters
    ----------
    res_ls : scipy.optimize.OptimizeResult
        Least-squares result. ``res_ls.x`` holds capacitances (the
        optimizer's :math:`\Delta\tau` already converted back) in the layout
        given in Notes; ``res_ls.fun`` holds the stacked real and imaginary
        impedance residuals ``[Re(r_1..r_M), Im(r_1..r_M)]``.
    n_groups : int
        Number of parallel branches :math:`n` in the fit.
    complex_fit : bool
        Layout of ``res_ls.x``: ``True`` if the CPE exponents were fitted. If
        ``False``, :math:`n_L` and all :math:`e_i` are set to 1.
    tau_bounds : tuple of float, optional
        Time-constant bounds used in the fit, stored for reference.
    initialize_rc_groups : int, optional
        Requested maximum number of branches, stored for reference.
    covariance : numpy.ndarray, shape (3 + 3n, 3 + 3n), optional
        Parameter covariance in :meth:`CircuitParameters.to_array` layout with
        branches in the same (optimizer) order as ``res_ls.x``. It is permuted
        to the tau-sorted branch order of :attr:`parameters`. ``None``
        (default) leaves all uncertainty attributes ``None``.

    Attributes
    ----------
    parameters : CircuitParameters
        Fitted parameters rebuilt from the solution vector (sorted by tau).
    residuals : numpy.ndarray of complex, shape (M,)
        Complex residuals :math:`\hat{Z}_k - Z_k` per frequency, rebuilt from
        ``res_ls.fun``.
    residuals_magnitude : numpy.ndarray, shape (M,)
        :math:`|\hat{Z}_k - Z_k|`.
    n_groups : int
        Number of parallel branches in the fit.
    covariance : numpy.ndarray or None
        ``(3 + 3n, 3 + 3n)`` covariance matrix of the reported parameters in
        :meth:`CircuitParameters.to_array` order
        ``[R_s, L, n_L, R_1..R_n, C_1..C_n, e_1..e_n]`` (branches in the
        tau-sorted order of ``parameters``); ``None`` if not computed.
    standard_errors : numpy.ndarray or None
        ``sqrt(diag(covariance))`` (negative variances clipped to 0), same
        order. See :meth:`parameter_uncertainties` for a dict keyed like
        :meth:`CircuitParameters.as_dict` and :attr:`parameter_names` for
        labels.
    correlation : numpy.ndarray or None
        Correlation matrix, same order; rows/columns of parameters with zero
        variance (fixed parameters) are 0.
    tau_bounds : tuple of float or None
        As passed in.
    initialize_rc_groups : int or None
        As passed in, or set by :meth:`add_initial_rc_groups`.
    success, status, message, jac, cost, grad, optimality, active_mask, nfev, njev
        Copied from ``res_ls`` (see :func:`scipy.optimize.least_squares`).

    See Also
    --------
    CircuitParameters : Container for the fitted parameters.
    blowing_bubbles.scripts.global_minimize.impedance_fit : Produces this
        result.

    Notes
    -----
    Layout of ``res_ls.x`` depending on ``complex_fit``:

    - ``complex_fit=False``: ``[R_s, L, R_1..R_n, C_1..C_n]``
    - ``complex_fit=True``: ``[R_s, L, n_L, R_1..R_n, C_1..C_n, e_1..e_n]``

    Branches in ``res_ls.x`` (and in a ``covariance`` passed in) are in
    optimizer order; ``parameters`` and the stored uncertainties are sorted by
    tau.

    The uncertainties are a local linearization of the weighted least-squares
    problem at the solution; see
    :func:`~blowing_bubbles.scripts.global_minimize.impedance_fit`.
    """

    def __init__(
        self,
        res_ls: OptimizeResult,
        n_groups: int,
        complex_fit: bool,
        tau_bounds: tuple[float, float] | None = None,
        initialize_rc_groups: int | None = None,
        covariance: np.ndarray | None = None,
    ):
        """Populate the result from a least-squares solution."""
        # Parameter layout (must match impedance_fit least-squares vector):
        #   complex_fit=False:  [Rs, L,        R_1..R_n, C_1..C_n]
        #   complex_fit=True:   [Rs, L, n_L,   R_1..R_n, C_1..C_n, e_1..e_n]
        head = 3 if complex_fit else 2
        resistances = res_ls.x[head : head + n_groups]
        capacitances = res_ls.x[head + n_groups : head + 2 * n_groups]
        self.parameters = CircuitParameters(
            resistance_source=res_ls.x[0],
            inductance=res_ls.x[1],
            inductance_exponent=res_ls.x[2] if complex_fit else 1.0,
            resistances=resistances,
            capacitances=capacitances,
            exponents=(
                res_ls.x[head + 2 * n_groups : head + 3 * n_groups]
                if complex_fit
                else np.ones(n_groups)
            ),
        )
        self.covariance = None
        self.standard_errors = None
        self.correlation = None
        if covariance is not None:
            # Apply the same branch permutation as CircuitParameters.sort_by_tau.
            idx = _tau_sort_index(np.array(resistances), np.array(capacitances))
            perm = np.concatenate(
                [[0, 1, 2], 3 + idx, 3 + n_groups + idx, 3 + 2 * n_groups + idx]
            )
            cov = np.asarray(covariance, dtype=float)[np.ix_(perm, perm)]
            std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
            denom = np.outer(std, std)
            corr = np.zeros_like(cov)
            np.divide(cov, denom, out=corr, where=denom > 0)
            self.covariance = cov
            self.standard_errors = std
            self.correlation = corr
        res_real = res_ls.fun[: res_ls.fun.size // 2]
        res_imag = res_ls.fun[res_ls.fun.size // 2 :]
        self.residuals = res_real + 1j * res_imag
        self.residuals_magnitude = np.sqrt(res_real**2 + res_imag**2)
        self.success = res_ls.success
        self.status = res_ls.status
        self.message = res_ls.message
        self.jac = res_ls.jac
        self.cost = res_ls.cost
        self.grad = res_ls.grad
        self.optimality = res_ls.optimality
        self.active_mask = res_ls.active_mask
        self.nfev = res_ls.nfev
        self.njev = res_ls.njev
        self.n_groups = n_groups
        self.tau_bounds = tau_bounds
        self.initialize_rc_groups = initialize_rc_groups

    @property
    def parameter_names(self) -> list[str]:
        """Labels for the :attr:`covariance` / :attr:`standard_errors` entries.

        ``["R_s", "L", "n_L", "R_1", ..., "C_1", ..., "e_1", ...]``, numbered
        in the tau-sorted branch order of :attr:`parameters` (``R_1`` has the
        smallest tau).
        """
        n = self.n_groups
        return (
            ["R_s", "L", "n_L"]
            + [f"R_{i + 1}" for i in range(n)]
            + [f"C_{i + 1}" for i in range(n)]
            + [f"e_{i + 1}" for i in range(n)]
        )

    def parameter_uncertainties(self) -> dict | None:
        """Return the standard errors keyed like :meth:`CircuitParameters.as_dict`.

        Returns
        -------
        uncertainties : dict or None
            Keys ``resistance_source``, ``inductance``, ``inductance_exponent``
            (float) and ``resistances``, ``capacitances``, ``exponents``
            (numpy.ndarray, shape ``(n_groups,)``, copies) in the tau-sorted
            order of :attr:`parameters`. ``None`` if no covariance was
            computed.
        """
        if self.standard_errors is None:
            return None
        n = self.n_groups
        se = self.standard_errors
        return {
            "resistance_source": float(se[0]),
            "inductance": float(se[1]),
            "inductance_exponent": float(se[2]),
            "resistances": se[3 : 3 + n].copy(),
            "capacitances": se[3 + n : 3 + 2 * n].copy(),
            "exponents": se[3 + 2 * n : 3 + 3 * n].copy(),
        }

    def add_initial_rc_groups(self, rc_groups: int) -> None:
        """Record the requested maximum number of RC groups.

        Only sets :attr:`initialize_rc_groups` (in place); the fitted
        parameters are not changed.

        Parameters
        ----------
        rc_groups : int
            Value stored in :attr:`initialize_rc_groups`.
        """
        self.initialize_rc_groups = rc_groups
