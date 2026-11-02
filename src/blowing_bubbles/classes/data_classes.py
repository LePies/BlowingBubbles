r"""Electrochemical impedance spectroscopy (EIS) data and fitting.

:class:`EISData` loads and processes measured spectra; :func:`fit_circuit` fits
the R–CPE ladder using iterative fast branch building.
"""

from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

import pandas as pd

from blowing_bubbles.classes.circuit_classes import CircuitParameters, FitResults


class EISData:
    """
    Class for handling Electrochemical Impedance Spectroscopy (EIS) data.

    Provides methods for loading, processing, visualizing, and fitting EIS
    measurements. Automatically detects frequency, real, and imaginary
    impedance columns from the data.

    See Also
    --------
    :meth:`plot_nyquist` : Nyquist plot on an existing matplotlib axes
    :meth:`fit` : Fit the equivalent-circuit model to this spectrum
    :class:`~blowing_bubbles.classes.circuit_classes.CircuitParameters`
        : Fitted branch parameters
    """

    def __init__(
            self,
            data_set: pd.DataFrame,
            area: float | None = None,
            unit: str = "Ohm",
            columns: tuple[str, str, str] | None = None,
            *,
            _validate: bool = True,
            _warn_stacklevel: int = 3,
        ) -> None:
        """
        Initialize EIS data.

        Parameters
        ----------
        data_set : pd.DataFrame
            DataFrame containing EIS data with frequency, real, and
            imaginary impedance columns
        area : float, optional
            Electrode area. If given, the impedance is normalized with
            :meth:`normalize` (multiplied by *area*, giving area-specific
            impedance). ``None`` (default) leaves the data unnormalized.
        unit : str
            Impedance unit used for plot labels (default ``"Ohm"``). When *area*
            is given and *unit* is the default ``"Ohm"``, the unit becomes
            ``"Ohm cm^2"``; any other *unit* is taken to already describe the
            normalized data and is kept as is.
        columns : tuple of str, optional
            Explicit ``(real, imag, freq)`` column names. If ``None`` they are
            detected with :meth:`find_keys`.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If frequency, real or imaginary values contain NaN/inf, or if any
            frequency is not strictly positive.

        Warns
        -----
        UserWarning
            If the spectrum contains duplicate frequencies. Rows are kept in
            their original order.

        See Also
        --------
        from_csv : Alternative constructor to load from file
        """
        self.data = data_set
        self.keys = tuple(columns) if columns is not None else self.find_keys()
        if _validate:
            self._check_input(stacklevel=_warn_stacklevel)
        self.unit = unit
        self.normalization_area: float | None = None
        self.normalized = False
        self._refresh_derived()
        if area is not None:
            self.normalize(area, unit if unit != "Ohm" else f"{unit} cm^2")

    @classmethod
    def from_csv(
        cls,
        path_to_data: str,
        area: float | None = None,
        unit: str = "Ohm",
        comment=None,
        delimiter=",",
        engine="python",
    ) -> EISData:
        """
        Load EIS data from CSV file.

        Convenience constructor that loads data directly from a CSV file.
        Optionally normalizes impedance by electrode area.

        Parameters
        ----------
        path_to_data : str
            Path to the CSV file
        area : float, optional
            Electrode area (e.g. in cm²). If given, the impedance is multiplied
            by *area* on load (see :meth:`normalize`); ``None`` (default) leaves
            it unnormalized.
        unit : str
            Unit for the data (default ``"Ohm"``). With *area* and the default
            unit, the normalized unit is ``"Ohm cm^2"``; a non-default *unit* is
            kept as is.

        Returns
        -------
        eis_data : EISData
            Loaded EIS data object

        See Also
        --------
        normalize : Apply normalization after loading
        """
        frame = pd.read_csv(
            path_to_data,
            comment=comment,
            delimiter=delimiter,
            engine=engine,
        )
        return EISData(
            frame,
            area=area,
            unit=unit,
            _warn_stacklevel=4,  # attribute the duplicate warning to the caller
        )

    def _with_impedance(self, z: np.ndarray) -> EISData:
        """New :class:`EISData` on this frequency grid with impedance *z*.

        Input validation is skipped: the frequencies come from an already
        validated object, and arithmetic results (e.g. after division by zero)
        are returned as computed, possibly non-finite.
        """
        r_key, i_key, f_key = self.keys
        frame = pd.DataFrame({f_key: self.Freq, r_key: z.real, i_key: z.imag})
        new = EISData(
            frame, unit=self.unit, columns=(r_key, i_key, f_key), _validate=False
        )
        new.normalization_area = self.normalization_area
        new.normalized = self.normalized
        return new

    def _check_input(self, stacklevel: int = 3) -> None:
        """Validate the frequency and impedance columns of :attr:`data`.

        *stacklevel* is passed to :func:`warnings.warn` for the
        duplicate-frequency warning so that it points at the user's call.
        """
        r_key, i_key, f_key = self.keys
        for key in (f_key, r_key, i_key):
            values = self.data[key].to_numpy(dtype=float)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Column {key!r} contains NaN or infinite values.")
        freq = self.data[f_key].to_numpy(dtype=float)
        if np.any(freq <= 0):
            raise ValueError(
                f"Column {f_key!r} must contain strictly positive frequencies."
            )
        if np.unique(freq).size != freq.size:
            warnings.warn(
                f"Column {f_key!r} contains duplicate frequencies.",
                UserWarning,
                stacklevel=stacklevel,
            )

    def _refresh_derived(self) -> None:
        """Re-sync Real/Imag/Freq and Rs/Rtot/Rp from :attr:`data`."""
        r_key, i_key, f_key = self.keys
        self.Real = self.data[r_key].to_numpy(dtype=float)
        self.Imag = self.data[i_key].to_numpy(dtype=float)
        self.Freq = self.data[f_key].to_numpy(dtype=float)
        self.Rs = self.find_Rs()[0]
        self.Rtot = self.find_Rtot()[0]
        self.Rp = self.find_Rp()[0]

    def _keep_rows(self, rows: slice | np.ndarray) -> None:
        """Keep only the positional *rows* of :attr:`data` and refresh."""
        subset = self.data.iloc[rows]
        if len(subset) == 0:
            raise ValueError("Selection would remove all data points.")
        self.data = subset.copy()
        self._refresh_derived()

    def _impedance_of(self, other: EISData | complex | float) -> np.ndarray | complex:
        if isinstance(other, EISData):
            if other.Freq.shape != self.Freq.shape or not np.allclose(
                other.Freq, self.Freq
            ):
                raise ValueError("EISData arithmetic requires identical frequencies.")
            return other.Real + 1j * other.Imag
        return other

    @property
    def Z(self) -> np.ndarray:
        """Complex impedance ``Real + 1j * Imag``."""
        return self.Real + 1j * self.Imag

    def __add__(self, other: EISData | complex | float) -> EISData:
        return self._with_impedance(self.Z + self._impedance_of(other))

    def __sub__(self, other: EISData | complex | float) -> EISData:
        return self._with_impedance(self.Z - self._impedance_of(other))

    def __mul__(self, other: EISData | complex | float) -> EISData:
        return self._with_impedance(self.Z * self._impedance_of(other))

    def __truediv__(self, other: EISData | complex | float) -> EISData:
        return self._with_impedance(self.Z / self._impedance_of(other))

    def __str__(self):
        """
        Print a summary of the EIS data.
        """
        print_string = "Eis data object"
        print_string += "\n" + "=" * 20
        print_string += "\n" + "Data:"
        print_string += "\n" + str(self.data)
        return print_string

    def save_csv(self, path: str) -> None:
        """
        Save EIS data to CSV file.

        Parameters
        ----------
        path : str
            Path to save the CSV file.
        """
        self.data.to_csv(path, index=False)

    def find_keys(self) -> tuple[str, str, str]:
        """
        Find keys for Real, Imaginary, and Frequency data.

        Returns
        -------
        R_key : str
            Key for real part.
        I_key : str
            Key for imaginary part.
        F_key : str
            Key for frequency.
        """
        categories = {
            "freq": ({"f", "freq", "frequency", "hz"}, "freq"),
            "real": ({"r", "re", "real", "zre", "zreal"}, "real"),
            "imag": ({"i", "im", "imag", "imaginary", "zim", "zimag"}, "imag"),
        }
        matches: dict[str, list[str]] = {name: [] for name in categories}
        for key in self.data.keys():
            tokens = [t for t in re.split(r"[^a-z]+", str(key).lower()) if t]
            for name, (exact, prefix) in categories.items():
                if any(t in exact or t.startswith(prefix) for t in tokens):
                    matches[name].append(key)
        for name, found in matches.items():
            if len(found) != 1:
                problem = "No" if not found else f"Ambiguous ({found})"
                raise ValueError(
                    f"{problem} column for {name!r} in {list(self.data.keys())}. "
                    "Pass columns=(real, imag, freq) explicitly."
                )
        return matches["real"][0], matches["imag"][0], matches["freq"][0]

    def limit_data(self, start: int, end: int) -> None:
        """
        Limit data to specified range.

        Parameters
        ----------
        start : int
            Start index.
        end : int
            End index.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the range selects no data points.

        Notes
        -----
        ``data``, ``Real``/``Imag``/``Freq`` and ``Rs``/``Rtot``/``Rp`` are all
        updated.
        """
        self._keep_rows(slice(start, end))

    def find_Rs(self) -> tuple[float, int]:
        """
        Find series resistance.

        Returns
        -------
        Rs : float
            Series resistance value.
        index : int
            Index of series resistance.
        """
        order = np.argsort(self.Freq)[::-1]  # high -> low frequency
        imag = self.Imag[order]
        # First sign change of Im(Z) coming from high frequency: the HF intercept
        # between the inductive tail and the first capacitive arc.
        crossings = np.nonzero(np.signbit(imag[:-1]) != np.signbit(imag[1:]))[0]
        if crossings.size and imag[0] > 0:
            k = int(crossings[0])
            i0, i1 = order[k], order[k + 1]
            t = imag[k] / (imag[k] - imag[k + 1])
            rs = self.Real[i0] + t * (self.Real[i1] - self.Real[i0])
            return float(rs), int(i0)
        i0 = int(order[0])
        return float(self.Real[i0]), i0

    def find_Rtot(self) -> tuple[float, int]:
        """
        Find total resistance.

        Returns
        -------
        Rtot : float
            Total resistance value.
        index : int
            Index of total resistance.
        """
        i = int(np.argmin(self.Freq))
        return float(self.Real[i]), i

    def find_Rp(self) -> tuple[float, float]:
        """
        Find polarization resistance.

        Returns
        -------
        Rp : float
            Polarization resistance value.
        nan : float
            NaN (not used).
        """
        return self.Rtot - self.Rs, np.nan

    def normalize(
        self, area: float, norm_unit: str = ""
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize impedance data by electrode area.

        Multiplies Real and Imag by the area factor to convert from
        absolute impedance (Ω) to area-specific impedance (Ω·cm²).
        Also recalculates Rs, Rtot, and Rp, and sets ``normalized`` to ``True``.
        Data can only be normalized once; this includes data that was
        normalized on construction via the ``area`` argument.

        Parameters
        ----------
        area : float
            Area normalization factor (e.g., electrode area in cm²)
        norm_unit : str
            Unit string for labels (e.g., "Ω·cm²")

        Returns
        -------
        Freq : array-like
            Frequency data (unchanged)
        Real : array-like
            Normalized real part
        Imag : array-like
            Normalized imaginary part

        Raises
        ------
        ValueError
            If *area* is not a positive finite number, or if the data is
            already normalized (e.g. by passing ``area`` to the constructor).

        See Also
        --------
        from_csv : Load with normalization in one step
        """
        if self.normalized:
            raise ValueError(
                "Data is already normalized (area="
                f"{self.normalization_area!r}); normalizing again would scale "
                "the impedance twice."
            )
        if not (np.isfinite(area) and area > 0):
            raise ValueError(f"area must be a positive finite number, got {area!r}.")
        # Copy so the caller's DataFrame is never modified in place.
        self.data = self.data.copy()
        self.data[self.keys[0]] = self.data[self.keys[0]] * area
        self.data[self.keys[1]] = self.data[self.keys[1]] * area
        self.unit = norm_unit
        self.normalization_area = area
        self._refresh_derived()

        self.normalized = True
        return self.Freq, self.Real, self.Imag

    def plot_nyquist(
        self,
        axes: plt.Axes | None = None,
        **plot_kwargs: Any,
    ) -> tuple[Figure, plt.Axes]:
        """Plot measured EIS on an existing subplot.

        This method reuses an existing axes instead of opening a new figure.
        """
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure

        if axes is None:
            fig, axes = plt.subplots()
        else:
            fig = axes.figure

        kwargs: dict[str, Any] = dict(plot_kwargs)
        kwargs.setdefault("label", "Nyquist")
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 5)
        kwargs.setdefault("linestyle", "")
        kwargs.setdefault("color", "black")
        kwargs.setdefault("markerfacecolor", "none")
   

        axes.plot(self.Real, -self.Imag, **kwargs)
        axes.set_xlabel(r"Z' [" + self.unit + "]")
        axes.set_ylabel(r"-Z'' [" + self.unit + "]")
        return cast(Figure, fig), axes

    def fit(
        self,
        n_restarts: int = 200,
        rc_groups: int = 5,
        initial_peaks: int = 2,
        verbose: bool = False,
        seed: int | None = None,
        tau_bounds: tuple[float, float] | None = None,
        resistance_source_bounds: tuple[float, float] | None = None,
        resistance_branch_bounds: tuple[float, float] | None = None,
        exponent_bounds: tuple[float, float] | None = None,
        inductance_exponent_bounds: tuple[float, float] | None = None,
        weighting: str = "modulus",
        n_jobs: int | None = 1,
        early_stop: int | None = None,
    ) -> FitResults:
        r"""Fit the parametric R–CPE ladder to this spectrum.

        Calls :func:`~blowing_bubbles.classes.data_classes.fit_circuit`
        (iterative fast branch building).

        Parameters
        ----------
        n_restarts : int
            Random restarts per nonlinear solve (default 200).
        rc_groups : int
            Maximum number of parallel R–CPE branches (default 5).
        verbose : bool
            Print progress during branch expansion.
        seed : int, optional
            RNG seed for fit restarts.
        tau_bounds : tuple of float, optional
            Bounds on branch time constants :math:`\tau_i = R_i C_i`.
        resistance_source_bounds : tuple of float, optional
            Bounds on source resistance :math:`R_s`.
        resistance_branch_bounds : tuple of float, optional
            Bounds on branch resistances :math:`R_i`.
        exponent_bounds : tuple of float, optional
            Bounds on CPE exponents :math:`\alpha_i` and :math:`\beta`.
        weighting : {"modulus", "unit"}
            Residual weighting (default ``"modulus"``, i.e. relative error).
        n_jobs : int, optional
            Worker processes for the restarts (``1``: sequential, default; ``-1``:
            all cores). Scripts using ``n_jobs != 1`` on Windows/macOS need an
            ``if __name__ == "__main__":`` guard.
        early_stop : int, optional
            Stop the restarts of each sub-fit once this many reached the best cost.
            ``None`` (default) runs all restarts.
    
        Returns
        -------
        FitResults
            Optimized
            :class:`~blowing_bubbles.classes.circuit_classes.CircuitParameters`
            in ``.parameters``.
        """
        res = fit_circuit(
            self,
            n_restarts=n_restarts,
            rc_groups=rc_groups,
            verbose=verbose,
            seed=seed,
            tau_bounds=tau_bounds,
            initial_peaks=initial_peaks,
            resistance_source_bounds=resistance_source_bounds,
            resistance_branch_bounds=resistance_branch_bounds,
            exponent_bounds=exponent_bounds,
            inductance_exponent_bounds=inductance_exponent_bounds,
            weighting=weighting,
            n_jobs=n_jobs,
            early_stop=early_stop,
        )

        return res

    def cut_frequencies_above(self, frequency: float) -> None:
        """
        Cut frequencies above a given frequency.

        Parameters
        ----------
        frequency : float
            Cutoff frequency in Hz.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If no frequency lies below *frequency*.
        """
        self._keep_rows(np.flatnonzero(self.Freq < frequency))

    def cut_frequencies_below(self, frequency: float) -> None:
        """
        Cut frequencies below a given frequency.

        Parameters
        ----------
        frequency : float
            Cutoff frequency in Hz.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If no frequency lies above *frequency*.
        """
        self._keep_rows(np.flatnonzero(self.Freq > frequency))

    def quick_fit(
        self,
        initial_parameters: CircuitParameters,
        complex_fit: bool = True,
        seed: int | None = None,
        sigma: float = 0.5,
        tau_bounds: tuple[float, float] | None = None,
        penalty_margin_weight: float = 1.0,
        penalty_weight: float | None = None,
        z_margin_frac: float = 0.0,
        z_margin_abs: float = 0.0,
        max_nfev: int | None = None,
        polish: bool = True,
        polish_maxiter: int = 50,
        constraint_tol: float = 1e-9,
        inductance_bounds: bool = False,
        inductance_margin: float = 2.0,
        resistance_source_bounds: tuple[float, float] | None = None,
        resistance_branch_bounds: tuple[float, float] | None = None,
        exponent_bounds: tuple[float, float] | None = None,
        inductance_exponent_bounds: tuple[float, float] | None = None,
        weighting: str = "modulus",
    ) -> FitResults:
        r"""Quick fit of the parametric R–CPE ladder to this spectrum.

        Calls :func:`~blowing_bubbles.classes.data_classes.quick_fit`
        (global optimization) with a single restart from *initial_parameters*.
        The remaining arguments are passed through to
        :func:`~blowing_bubbles.scripts.global_minimize.impendance_fit`.

        Notes
        -----
        The defaults now match ``impendance_fit``: ``z_margin_frac=0.0`` and
        ``inductance_bounds=False`` (previously ``0.05`` and ``True``, which
        could exclude the true circuit). Pass the old values explicitly to
        restore the previous behaviour. *weighting* (default ``"modulus"``,
        i.e. relative error) is new.
        """

        return quick_fit(
            data = self,
            n_restarts = 1,
            initial_parameters = initial_parameters,
            n_groups = len(initial_parameters.resistances),
            complex_fit = complex_fit,
            seed = seed,
            sigma = sigma,
            tau_bounds = tau_bounds,
            penalty_margin_weight = penalty_margin_weight,
            penalty_weight = penalty_weight,
            z_margin_frac = z_margin_frac,
            z_margin_abs = z_margin_abs,
            max_nfev = max_nfev,
            polish = polish,
            polish_maxiter = polish_maxiter,
            constraint_tol = constraint_tol,
            inductance_bounds = inductance_bounds,
            inductance_margin = inductance_margin,
            resistance_source_bounds = resistance_source_bounds,
            resistance_branch_bounds = resistance_branch_bounds,
            exponent_bounds = exponent_bounds,
            inductance_exponent_bounds = inductance_exponent_bounds,
            weighting = weighting,
        )


def fit_circuit(
    data: EISData,
    *,
    n_restarts: int,
    rc_groups: int = 5,
    initial_peaks: int = 2,
    verbose: bool = False,
    seed: int | None = None,
    tau_bounds: tuple[float, float] | None = None,
    resistance_source_bounds: tuple[float, float] | None = None,
    resistance_branch_bounds: tuple[float, float] | None = None,
    exponent_bounds: tuple[float, float] | None = None,
    inductance_exponent_bounds: tuple[float, float] | None = None,
    weighting: str = "modulus",
    n_jobs: int | None = 1,
    early_stop: int | None = None,
) -> FitResults:
    """Fit an R–CPE ladder to *data* using iterative fast branch building."""
    from blowing_bubbles.scripts.fit_circuit import fit_circuit as _run_fit

    return _run_fit(
        data,
        n_restarts,
        rc_groups=rc_groups,
        initial_peaks=initial_peaks,
        verbose=verbose,
        seed=seed,
        tau_bounds=tau_bounds,
        resistance_source_bounds=resistance_source_bounds,
        resistance_branch_bounds=resistance_branch_bounds,
        exponent_bounds=exponent_bounds,
        inductance_exponent_bounds=inductance_exponent_bounds,
        weighting=weighting,
        n_jobs=n_jobs,
        early_stop=early_stop,
    )


def quick_fit(
    data: EISData,
    n_groups: int = 2,
    complex_fit: bool = False,
    initial_parameters: CircuitParameters | None = None,
    n_restarts: int = 1,
    seed: int | None = None,
    sigma: float = 0.5,
    tau_bounds: tuple[float, float] | None = None,
    penalty_margin_weight: float = 1.0,
    penalty_weight: float | None = None,
    z_margin_frac: float = 0.0,
    z_margin_abs: float = 0.0,
    max_nfev: int | None = None,
    polish: bool = True,
    polish_maxiter: int = 50,
    constraint_tol: float = 1e-9,
    inductance_bounds: bool = False,
    inductance_margin: float = 2.0,
    resistance_source_bounds: tuple[float, float] | None = None,
    resistance_branch_bounds: tuple[float, float] | None = None,
    exponent_bounds: tuple[float, float] | None = None,
    inductance_exponent_bounds: tuple[float, float] | None = None,
    weighting: str = "modulus",
) -> FitResults:
    """Quick fit of an R–CPE ladder to *data* using default parameters.

    Thin wrapper around
    :func:`~blowing_bubbles.scripts.global_minimize.impendance_fit`; all
    arguments are passed through.

    Notes
    -----
    The defaults now match ``impendance_fit``: ``z_margin_frac=0.0`` and
    ``inductance_bounds=False`` (previously ``0.05`` and ``True``, which could
    exclude the true circuit). Pass the old values explicitly to restore the
    previous behaviour. *weighting* (default ``"modulus"``, i.e. relative
    error) is new.
    """
    from blowing_bubbles.scripts.global_minimize import impendance_fit as _run_quick_fit

    return _run_quick_fit(
        data = data,
        n_groups = n_groups,
        complex_fit = complex_fit,
        initial_parameters = initial_parameters,
        n_restarts = n_restarts,
        seed = seed,
        sigma = sigma,
        tau_bounds = tau_bounds,
        penalty_margin_weight = penalty_margin_weight,
        penalty_weight = penalty_weight,
        z_margin_frac = z_margin_frac,
        z_margin_abs = z_margin_abs,
        max_nfev = max_nfev,
        polish = polish,
        polish_maxiter = polish_maxiter,
        constraint_tol = constraint_tol,
        inductance_bounds = inductance_bounds,
        inductance_margin = inductance_margin,
        resistance_source_bounds = resistance_source_bounds,
        resistance_branch_bounds = resistance_branch_bounds,
        exponent_bounds = exponent_bounds,
        inductance_exponent_bounds = inductance_exponent_bounds,
        weighting = weighting,
    )
