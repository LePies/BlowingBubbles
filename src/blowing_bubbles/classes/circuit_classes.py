r"""Parameter containers and fit result wrappers for equivalent-circuit EIS.

See :mod:`blowing_bubbles.circuit_impedance` for the impedance model.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import overload

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from scipy.optimize import OptimizeResult

from blowing_bubbles.circuit_impedance import circuit_impedance

_CIRCUIT_PARAMS_FORMAT_VERSION = 1
_CIRCUIT_PARAMS_FILE_TYPE = "blowing_bubbles.CircuitParameters"
_FIT_LABEL = r"$[R_s L(R_i C_i)]$ fit"
_TEXT_BBOX = dict(
    facecolor="white",
    alpha=0.8,
    edgecolor="none",
    boxstyle="round,pad=0.2",
)


def align_axes_height(
    axes: list[Axes] | tuple[Axes, ...],
    reference: Axes | None = None,
) -> None:
    """Match subplot box height and bottom edge (use after tight_layout)."""
    if reference is None:
        reference = axes[-1]
    ref = reference.get_position()
    for ax in axes:
        pos = ax.get_position()
        ax.set_position((pos.x0, ref.y0, pos.width, ref.height))


def _plot_full_fit(
    freqs: np.ndarray,
    parameters: CircuitParameters,
    axis : plt.Axes | None = None,
    unit: str | None = None,
):
    if unit is None:
        warnings.warn(
            "No unit provided, using default unit: Omega cm^2",
            stacklevel=2,
        )
        unit = r"$\Omega$ cm$^{2}$"
    if axis is None:
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
                _Zi_pred_L.real[mask_L], -_Zi_pred_L.imag[mask_L],
                label=fr"$(R_{n_groups} C_{n_groups})$ fit with one group (with $L$)"
            )
            # Calculate and print R1 at the midpoint of this arc, at y=0
            xline = _Zi_pred_L.real[mask_L]
            if len(xline) > 1:
                xmid = rs_i + ri_i / 2
                axis.text(
                    xmid,
                    -np.max(np.abs(Z_pred.imag))*0.2,
                    fr"$R_{n_groups}$",
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
                _Zi_pred.real[mask], -_Zi_pred.imag[mask],
                label=fr"$(R_{n_groups-i} C_{n_groups-i})$ fit with one group"
            )
            # Calculate and print Ri at the midpoint of this arc, at y=0
            xline = _Zi_pred.real[mask]
            if len(xline) > 1:
                xmid = rs_i + ri_i / 2
                axis.text(
                    xmid,
                    -np.max(np.abs(Z_pred.imag))*0.2,
                    fr"$R_{n_groups-i}$",
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
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.set_xlabel("Z' "+ unit)
    axis.set_ylabel("-Z'' "+ unit)
    return axis


class CircuitParameters:
    r"""Container for the parametric R–CPE ladder used in BlowingBubbles EIS fitting.

    The impedance model is (see
    :func:`~blowing_bubbles.circuit_impedance.circuit_impedance`):

    .. math::

        Z(\omega) = R_s + (j\omega)^{n_L} L
                    + \sum_{i=1}^{n} \frac{R_i}{1 + R_i C_i (j\omega)^{e_i}}

    Branches are stored in increasing :math:`\tau_i = R_i C_i` order after construction.

    Parameters
    ----------
    resistance_source : float
        Series resistance :math:`R_s` (Ω).
    inductance : float
        Inductance :math:`L` (H) in the constant-phase inductor term.
    inductance_exponent : float
        Inductor CPE exponent :math:`n_L`.
    resistances, capacitances, exponents : np.ndarray
        Parallel branch parameters (:math:`R_i`, :math:`C_i`, :math:`e_i`), same length.
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

        ``taus`` stores the product :math:`R_i C_i` (units :math:`\mathrm{s}^{e_i}`),
        which is what the optimiser orders and parameterises. The apex of
        branch :math:`i` lies at :math:`\omega_i = 1/\tau_{c,i}`.
        """
        return np.power(self.resistances * self.capacitances, 1.0 / self.exponents)

    @property
    def peak_frequencies(self) -> np.ndarray:
        r"""Apex frequencies :math:`f_i = 1 / (2\pi\,\tau_{c,i})` (Hz)."""
        return 1.0 / (2.0 * np.pi * self.characteristic_times)

    def sort_by_tau(self):
        """
        Sort resistances, capacitances, and exponents by increasing tau.
        """
        tau = self.capacitances * self.resistances
        sort_idx = np.argsort(tau)
        self.resistances = self.resistances[sort_idx]
        self.capacitances = self.capacitances[sort_idx]
        self.exponents = self.exponents[sort_idx]
        self.taus = self.capacitances * self.resistances

    
    def sort_by_capacitance(self):
        """
        Sort resistances, capacitances, and exponents by increasing capacitance.
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
        """Append an RC branch and re-sort by tau."""
        self._set_branches(
            np.append(self.resistances, resistance),
            np.append(self.capacitances, capacitance),
            np.append(self.exponents, exponent),
        )
        self.sort_by_tau()

    def drop_branch(self, index: int) -> None:
        """Remove the branch at *index* (tau-sorted order; negative indices allowed)."""
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
        """Remove and return the last branch (largest tau): (R, C, exponent)."""
        if len(self) == 0:
            raise IndexError("no branches to pop")
        r = float(self.resistances[-1])
        c = float(self.capacitances[-1])
        e = float(self.exponents[-1])
        self.drop_branch(-1)
        return r, c, e

    def as_dict(self) -> dict:
        """Return the parameters as a flat dict of floats and arrays."""
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
        return len(self.taus)

    def __str__(self):
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
        return np.array([
            self.resistance_source,
            self.inductance,
            self.inductance_exponent,
            *self.resistances,
            *self.capacitances,
            *self.exponents,
        ])
    
    def copy(self):
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
        """Inverse of :meth:`to_array`."""
        return cls(
            resistance_source=array[0],
            inductance=array[1],
            inductance_exponent=array[2],
            resistances=array[3:3+n_groups],
            capacitances=array[3+n_groups:3+2*n_groups],
            exponents=array[3+2*n_groups:],
        )
    
    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict (branches listed in tau order)."""
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
                    self.taus, strict=False,
                )
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> CircuitParameters:
        """Build from a dict produced by :meth:`to_dict` or :meth:`load`."""
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
        """
        Write parameters to a human-readable JSON file.

        Branch rows include R, C, exponent, and tau (tau is informational only).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")

    @classmethod
    def load(cls, path: str | Path) -> CircuitParameters:
        """Load parameters from a file written by :meth:`save`."""
        with Path(path).open(encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @overload
    def get_impedance(self, freq: float) -> np.complex128: ...

    @overload
    def get_impedance(self, freq: np.ndarray) -> np.ndarray: ...

    def get_impedance(self, freq: float | np.ndarray) -> np.complex128 | np.ndarray:
        r"""Evaluate :math:`Z(\omega)` at ``freq`` (Hz).

        Uses :func:`circuit_impedance`.
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
        freq : np.ndarray, optional
            Frequencies in Hz (default ``logspace(-5, 5, 1000)``).
        axis : matplotlib.axes.Axes, optional
            Axes to draw on; creates a new figure if ``None``.
        unit : str, optional
            Axis unit label suffix (default :math:`\Omega\,\mathrm{cm}^2`).
        """
        if unit is None:
            warnings.warn(
                "No unit provided, using default unit: Omega cm^2",
                stacklevel=2,
            )
            unit = r"$\Omega$ cm$^{2}$"
        if axis is None:
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
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
        axis.set_xlabel("Z' "+ unit)
        axis.set_ylabel("-Z'' "+ unit)
        return axis
    
    def plot_full_fit(
        self,
        freq: np.ndarray | None = None,
        axis: plt.Axes | None = None,
        unit: str | None = None,
    ):
        r"""Nyquist plot of the total fit plus decomposed branch arcs.

        Overlays the combined model (dashed black) and single-branch contributions
        with resistance labels on each semicircle.

        Parameters
        ----------
        freq, axis, unit
            Same as :meth:`plot_nyquist`.
        """
        if freq is None:
            freq = np.logspace(-5, 5, 1000)
        return _plot_full_fit(freq, self, axis, unit)


class FitResults(OptimizeResult):
    r"""Wrapper around a SciPy least-squares result for circuit fitting.

    Exposes optimizer fields (``success``, ``cost``, ``message``, ``nfev``, …) and
    Additional attributes:

    - ``parameters`` — :class:`CircuitParameters` rebuilt from the solution vector
    - ``residuals`` — complex residuals :math:`\hat{Z}_k - Z_k` per frequency
    - ``residuals_magnitude`` — :math:`|\hat{Z}_k - Z_k|`
    - ``n_groups`` — number of parallel branches in the fit

    The internal optimizer vector layout depends on ``complex_fit``:

    - ``complex_fit=False``:
      ``[R_s, L, R_1..R_n, \Delta\tau_1..\Delta\tau_n]``
    - ``complex_fit=True``:
      ``[R_s, L, n_L, R_1..R_n, \Delta\tau_1..\Delta\tau_n, e_1..e_n]``
    """

    def __init__(
        self,
        res_ls: OptimizeResult,
        n_groups: int,
        complex_fit: bool,
        tau_bounds: tuple[float, float] | None = None,
        initialize_rc_groups: int | None = None,
    ):
        r"""Populate from a converged :class:`scipy.optimize.OptimizeResult`."""
        # Parameter layout (must match impendance_fit least-squares vector):
        #   complex_fit=False:  [Rs, L,        R_1..R_n, C_1..C_n]
        #   complex_fit=True:   [Rs, L, n_L,   R_1..R_n, C_1..C_n, e_1..e_n]
        head = 3 if complex_fit else 2
        self.parameters = CircuitParameters(
            resistance_source=res_ls.x[0],
            inductance=res_ls.x[1],
            inductance_exponent=res_ls.x[2] if complex_fit else 1.0,
            resistances=res_ls.x[head : head + n_groups],
            capacitances=res_ls.x[head + n_groups : head + 2*n_groups],
            exponents=(
                res_ls.x[head + 2*n_groups : head + 3*n_groups]
                if complex_fit else np.ones(n_groups)
            ),
        )
        res_real = res_ls.fun[:res_ls.fun.size//2]
        res_imag = res_ls.fun[res_ls.fun.size//2:]
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
    

    def add_initial_rc_groups(self, rc_groups: int) -> None:
        """
        Add initial RC groups to the circuit parameters.
        """
        self.initialize_rc_groups = rc_groups