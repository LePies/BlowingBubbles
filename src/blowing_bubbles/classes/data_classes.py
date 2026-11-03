r"""Electrochemical impedance spectroscopy (EIS) data and fitting.

:class:`EISData` loads, validates, normalizes, plots and fits measured spectra.
:func:`fit_circuit` fits the R–CPE ladder using iterative fast branch building,
and :func:`quick_fit` runs a single fixed-size least-squares fit.
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
    r"""Measured electrochemical impedance spectrum (EIS) with fitting helpers.

    Provides methods for loading, processing, visualizing and fitting EIS
    measurements. Frequency, real and imaginary impedance columns are detected
    automatically unless given explicitly.

    Parameters
    ----------
    data_set : pandas.DataFrame
        Table with frequency (Hz), real and imaginary impedance columns.
    area : float, optional
        Electrode area (e.g. cm²). If given, the impedance is normalized with
        :meth:`normalize` (multiplied by *area*, giving area-specific
        impedance). ``None`` (default) leaves the data unnormalized.
    unit : str, default "Ohm"
        Impedance unit used for plot labels. When *area* is given and *unit*
        is the default ``"Ohm"``, the unit becomes ``"Ohm cm^2"``; any other
        *unit* is taken to already describe the normalized data and is kept
        as is.
    columns : tuple of str, optional
        Explicit ``(real, imag, freq)`` column names. ``None`` (default)
        detects them with :meth:`find_keys`.

    Attributes
    ----------
    data : pandas.DataFrame
        The spectrum table (a copy once it has been normalized or trimmed).
    keys : tuple of str
        ``(real, imag, freq)`` column names in :attr:`data`.
    Real, Imag, Freq : numpy.ndarray
        Real part, imaginary part (same unit as :attr:`unit`) and frequency
        (Hz), in the row order of :attr:`data`.
    Rs, Rtot, Rp : float
        Series, total and polarization resistance from :meth:`find_Rs`,
        :meth:`find_Rtot` and :meth:`find_Rp`.
    unit : str
        Impedance unit used for plot labels.
    normalized : bool
        Whether the impedance has been multiplied by an area.
    normalization_area : float or None
        Area used for normalization, ``None`` if not normalized.

    Raises
    ------
    ValueError
        If the columns cannot be detected unambiguously (see
        :meth:`find_keys`), if frequency, real or imaginary values contain
        NaN/inf, if any frequency is not strictly positive, or if *area* is
        not a positive finite number.

    Warns
    -----
    UserWarning
        If the spectrum contains duplicate frequencies. Rows are kept in
        their original order.

    See Also
    --------
    from_csv : Alternative constructor that loads a CSV file.
    blowing_bubbles.classes.circuit_classes.CircuitParameters : Fitted
        branch parameters.

    Notes
    -----
    The sign convention of the input is kept: capacitive arcs have
    :math:`\mathrm{Im}(Z) < 0` and are plotted as :math:`-Z''`.

    Instances support ``+``, ``-``, ``*`` and ``/`` with another
    :class:`EISData` on the same frequency grid or with a scalar, returning a
    new :class:`EISData` (see :meth:`__add__`).
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
        """Validate the spectrum and derive the cached arrays and resistances."""
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
        """Load EIS data from a CSV file.

        Convenience constructor that reads the file with
        :func:`pandas.read_csv` and passes the table to :class:`EISData`
        (columns are detected automatically). Optionally normalizes the
        impedance by electrode area.

        Parameters
        ----------
        path_to_data : str
            Path to the CSV file.
        area : float, optional
            Electrode area (e.g. in cm²). If given, the impedance is multiplied
            by *area* on load (see :meth:`normalize`); ``None`` (default) leaves
            it unnormalized.
        unit : str, default "Ohm"
            Impedance unit for plot labels. With *area* and the default unit,
            the normalized unit is ``"Ohm cm^2"``; a non-default *unit* is kept
            as is.
        comment : str, optional
            Comment character, forwarded to :func:`pandas.read_csv`. ``None``
            (default) disables comment handling.
        delimiter : str, default ","
            Field delimiter, forwarded to :func:`pandas.read_csv`.
        engine : str, default "python"
            Parser engine, forwarded to :func:`pandas.read_csv`.

        Returns
        -------
        eis_data : EISData
            Loaded (and validated) EIS data object.

        See Also
        --------
        normalize : Apply normalization after loading.
        save_csv : Write the data back to CSV.
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
        """Return the impedance of *other* (scalars are returned unchanged).

        Raises ``ValueError`` if *other* is an :class:`EISData` on a different
        frequency grid.
        """
        if isinstance(other, EISData):
            if other.Freq.shape != self.Freq.shape or not np.allclose(
                other.Freq, self.Freq
            ):
                raise ValueError("EISData arithmetic requires identical frequencies.")
            return other.Real + 1j * other.Imag
        return other

    @property
    def Z(self) -> np.ndarray:
        """Complex impedance ``Real + 1j * Imag``, shape ``(n_freq,)``."""
        return self.Real + 1j * self.Imag

    def __add__(self, other: EISData | complex | float) -> EISData:
        """Return a new spectrum with impedance ``self.Z + other``.

        Parameters
        ----------
        other : EISData or complex or float
            Spectrum on the same frequency grid (compared with
            :func:`numpy.allclose`) or a scalar impedance.

        Returns
        -------
        result : EISData
            New object on this frequency grid, keeping this object's column
            names, unit and normalization state.

        Notes
        -----
        A mismatching frequency grid raises ``ValueError``. The result is not
        re-validated, so it may contain non-finite values (e.g. after division
        by zero). :meth:`__sub__`, :meth:`__mul__` and :meth:`__truediv__`
        behave the same way, operating element-wise on the complex impedance.
        """
        return self._with_impedance(self.Z + self._impedance_of(other))

    def __sub__(self, other: EISData | complex | float) -> EISData:
        """Return a new spectrum with impedance ``self.Z - other`` (see __add__)."""
        return self._with_impedance(self.Z - self._impedance_of(other))

    def __mul__(self, other: EISData | complex | float) -> EISData:
        """Return a new spectrum with impedance ``self.Z * other`` (see __add__)."""
        return self._with_impedance(self.Z * self._impedance_of(other))

    def __truediv__(self, other: EISData | complex | float) -> EISData:
        """Return a new spectrum with impedance ``self.Z / other`` (see __add__)."""
        return self._with_impedance(self.Z / self._impedance_of(other))

    def __str__(self):
        """Return a summary string containing the data table."""
        print_string = "Eis data object"
        print_string += "\n" + "=" * 20
        print_string += "\n" + "Data:"
        print_string += "\n" + str(self.data)
        return print_string

    def save_csv(self, path: str) -> None:
        """Save the data table to a CSV file.

        Parameters
        ----------
        path : str
            Output path; an existing file is overwritten. The table
            :attr:`data` is written without the index, so normalized or trimmed
            data is saved as currently held.
        """
        self.data.to_csv(path, index=False)

    def find_keys(self) -> tuple[str, str, str]:
        """Detect the column names of the real, imaginary and frequency data.

        Returns
        -------
        R_key : str
            Column name of the real part.
        I_key : str
            Column name of the imaginary part.
        F_key : str
            Column name of the frequency.

        Raises
        ------
        ValueError
            If no column, or more than one column, matches one of the three
            categories.

        Notes
        -----
        Each column name is lower-cased and split into alphabetic tokens. A
        column matches a category if any token is one of its names or starts
        with its prefix:

        - frequency: ``f``, ``freq``, ``frequency``, ``hz`` or prefix ``freq``;
        - real: ``r``, ``re``, ``real``, ``zre``, ``zreal`` or prefix ``real``;
        - imaginary: ``i``, ``im``, ``imag``, ``imaginary``, ``zim``,
          ``zimag`` or prefix ``imag``.
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
        """Keep only the rows ``start:end`` (positional slice), in place.

        Parameters
        ----------
        start : int
            First row to keep (positional, in the current row order).
        end : int
            Row after the last row to keep (exclusive, Python slice semantics;
            negative values count from the end).

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
        r"""Estimate the series resistance from the high-frequency intercept.

        Returns
        -------
        Rs : float
            Series resistance estimate (unit of :attr:`unit`).
        index : int
            Positional row index of the data point the estimate is based on
            (see Notes).

        Notes
        -----
        Coming from the highest frequency, if :math:`\mathrm{Im}(Z) > 0`
        there (inductive tail) and :math:`\mathrm{Im}(Z)` changes sign, the
        real-axis crossing is linearly interpolated between the two points
        bracketing the first sign change; *index* is the higher-frequency point
        of that pair. Otherwise the real part at the highest frequency is
        returned, with its index.
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
        """Estimate the total resistance as the real part at the lowest frequency.

        Returns
        -------
        Rtot : float
            Total resistance estimate (unit of :attr:`unit`).
        index : int
            Positional row index of the lowest-frequency point.
        """
        i = int(np.argmin(self.Freq))
        return float(self.Real[i]), i

    def find_Rp(self) -> tuple[float, float]:
        """Estimate the polarization resistance as ``Rtot - Rs``.

        Uses the cached attributes :attr:`Rtot` and :attr:`Rs`.

        Returns
        -------
        Rp : float
            Polarization resistance estimate (unit of :attr:`unit`).
        nan : float
            Always NaN (placeholder for symmetry with :meth:`find_Rs`).
        """
        return self.Rtot - self.Rs, np.nan

    def normalize(
        self, area: float, norm_unit: str = ""
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalize the impedance by electrode area, in place.

        Multiplies the real and imaginary columns by *area* to convert from
        absolute impedance (Ω) to area-specific impedance (Ω·cm²).
        Also recalculates Rs, Rtot and Rp, and sets ``normalized`` to ``True``.
        Data can only be normalized once; this includes data that was
        normalized on construction via the ``area`` argument.

        Parameters
        ----------
        area : float
            Area normalization factor (e.g. electrode area in cm²).
        norm_unit : str, default ""
            New value of :attr:`unit` for plot labels (e.g. ``"Ω·cm²"``). The
            default empty string leaves the axis labels without a unit.

        Returns
        -------
        Freq : numpy.ndarray
            Frequency data (Hz, unchanged).
        Real : numpy.ndarray
            Normalized real part.
        Imag : numpy.ndarray
            Normalized imaginary part.

        Raises
        ------
        ValueError
            If the data is already normalized (e.g. by passing ``area`` to the
            constructor), or if *area* is not a positive finite number.

        See Also
        --------
        from_csv : Load with normalization in one step.

        Notes
        -----
        :attr:`data` is replaced by a scaled copy, so the DataFrame passed to
        the constructor is never modified.
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
        """Plot the measured spectrum as a Nyquist plot (:math:`Z'` vs. :math:`-Z''`).

        Parameters
        ----------
        axes : matplotlib.axes.Axes, optional
            Axes to draw on. ``None`` (default) creates a new figure and axes.
        **plot_kwargs
            Forwarded to :meth:`matplotlib.axes.Axes.plot`. Defaults:
            ``label="Nyquist"``, ``marker="o"``, ``markersize=5``,
            ``linestyle=""``, ``color="black"``, ``markerfacecolor="none"``.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure containing *axes* (newly created if *axes* was ``None``).
        axes : matplotlib.axes.Axes
            The axes drawn on.

        Notes
        -----
        The axis labels include :attr:`unit`. The aspect ratio and limits are
        not changed.
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
        (iterative fast branch building) with all arguments passed through.

        Parameters
        ----------
        n_restarts : int, default 200
            Random restarts per nonlinear solve.
        rc_groups : int, default 5
            Maximum number of parallel R–CPE branches (values above 5 emit a
            warning).
        initial_peaks : int, default 2
            Maximum number of branches in the initial guess, taken from the
            peaks of :math:`-\mathrm{Im}(Z)`.
        verbose : bool, default False
            Print progress during branch expansion.
        seed : int, optional
            RNG seed for fit restarts. ``None`` (default) is not reproducible.
        tau_bounds : tuple of float, optional
            ``(tau_min, tau_max)`` bounds on :math:`\tau_i = R_i C_i`.
            ``None`` (default) leaves them unbounded.
        resistance_source_bounds : tuple of float, optional
            ``(R_s_min, R_s_max)`` bounds on the source resistance :math:`R_s`.
            ``None`` (default) keeps :math:`R_s` strictly positive.
        resistance_branch_bounds : tuple of float, optional
            ``(R_min, R_max)`` bounds on all branch resistances :math:`R_i`.
            ``None`` (default) keeps them strictly positive.
        exponent_bounds : tuple of float, optional
            ``(e_min, e_max)`` bounds on all CPE exponents :math:`e_i` (and on
            :math:`n_L` unless *inductance_exponent_bounds* is given).
            ``None`` (default) uses ``(0.2, 1.0)``.
        inductance_exponent_bounds : tuple of float, optional
            Bounds on the inductor exponent :math:`n_L`. ``None`` (default)
            uses *exponent_bounds* if given, otherwise ``(0.7, 1.0)``.
        weighting : {"modulus", "unit"}, default "modulus"
            Residual weighting: ``"modulus"`` divides each residual by
            :math:`|Z_k|` (relative error), ``"unit"`` is unweighted.
        n_jobs : int, optional
            Worker processes for the restarts (``1``: sequential, default; ``-1``:
            all cores). Scripts using ``n_jobs != 1`` on Windows/macOS need an
            ``if __name__ == "__main__":`` guard.
        early_stop : int, optional
            Stop the restarts of each sub-fit once this many reached the best cost.
            ``None`` (default) runs all restarts.

        Returns
        -------
        result : FitResults
            Fit result; the optimized
            :class:`~blowing_bubbles.classes.circuit_classes.CircuitParameters`
            are in ``result.parameters``.

        See Also
        --------
        quick_fit : Single fit with a fixed number of branches.
        blowing_bubbles.scripts.fit_circuit.fit_circuit : Underlying algorithm.
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
        """Remove all points at or above *frequency*, in place.

        Only rows with ``Freq < frequency`` are kept; ``data``, the cached
        arrays and ``Rs``/``Rtot``/``Rp`` are updated.

        Parameters
        ----------
        frequency : float
            Cutoff frequency (Hz).

        Raises
        ------
        ValueError
            If no frequency lies below *frequency*.
        """
        self._keep_rows(np.flatnonzero(self.Freq < frequency))

    def cut_frequencies_below(self, frequency: float) -> None:
        """Remove all points at or below *frequency*, in place.

        Only rows with ``Freq > frequency`` are kept; ``data``, the cached
        arrays and ``Rs``/``Rtot``/``Rp`` are updated.

        Parameters
        ----------
        frequency : float
            Cutoff frequency (Hz).

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
        r"""Refine *initial_parameters* with a single least-squares fit.

        Calls :func:`~blowing_bubbles.classes.data_classes.quick_fit` with one
        restart and the branch count of *initial_parameters*; the remaining
        arguments are passed through to
        :func:`~blowing_bubbles.scripts.global_minimize.impedance_fit`, whose
        documentation describes them in detail.

        Parameters
        ----------
        initial_parameters : CircuitParameters
            Starting point; its number of branches fixes the fitted
            ``n_groups``. It is not modified.
        complex_fit : bool, default True
            Fit the CPE exponents :math:`e_i` and :math:`n_L`; if ``False``
            they are fixed to 1.
        seed : int, optional
            RNG seed for the restart perturbation.
        sigma : float, default 0.5
            Log-normal scale of restart perturbations.
        tau_bounds : tuple of float, optional
            ``(tau_min, tau_max)`` bounds on :math:`\tau_i = R_i C_i`.
            ``None`` (default) leaves them unbounded.
        penalty_margin_weight : float, default 1.0
            Weight of the soft Nyquist margin penalty.
        penalty_weight : float, optional
            Overrides *penalty_margin_weight* when given.
        z_margin_frac, z_margin_abs : float, default 0.0
            Relative (to the :math:`Z'` span) and absolute soft margin on the
            Nyquist constraints; the larger of the two is used.
        max_nfev : int, optional
            Maximum function evaluations. ``None`` (default) uses
            ``500 * n_params``.
        polish : bool, default True
            Run a polishing step if hard constraints are violated.
        polish_maxiter : int, default 50
            Iteration cap for polishing.
        constraint_tol : float, default 1e-9
            Tolerance for declaring constraints satisfied.
        inductance_bounds : bool, default False
            If ``True``, bound :math:`L` around an estimate from the slope of
            :math:`\mathrm{Im}(Z)`; otherwise only :math:`L \ge 0`.
        inductance_margin : float, default 2.0
            Margin factor for the inductance bounds.
        resistance_source_bounds : tuple of float, optional
            ``(R_s_min, R_s_max)`` bounds on :math:`R_s`.
        resistance_branch_bounds : tuple of float, optional
            ``(R_min, R_max)`` bounds on all branch resistances.
        exponent_bounds : tuple of float, optional
            ``(e_min, e_max)`` bounds on all CPE exponents (only used with
            ``complex_fit=True``).
        inductance_exponent_bounds : tuple of float, optional
            Bounds on the inductor exponent :math:`n_L`.
        weighting : {"modulus", "unit"}, default "modulus"
            Residual weighting: ``"modulus"`` (relative error) or ``"unit"``
            (unweighted).

        Returns
        -------
        result : FitResults
            Fit result with the refined parameters in ``result.parameters``.

        See Also
        --------
        fit : Full fit with automatic branch building.

        Notes
        -----
        The defaults match ``impedance_fit``: ``z_margin_frac=0.0`` and
        ``inductance_bounds=False`` (previously ``0.05`` and ``True``, which
        could exclude the true circuit). Pass the old values explicitly to
        restore the previous behaviour. Unlike the module-level
        :func:`quick_fit`, ``complex_fit`` defaults to ``True`` here.
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
    r"""Fit an R–CPE ladder to *data* using iterative fast branch building.

    Thin keyword-only wrapper around
    :func:`blowing_bubbles.scripts.fit_circuit.fit_circuit` (imported lazily);
    all arguments are passed through. Its ``selection`` and ``min_improvement``
    options are not exposed here and keep their defaults.

    Parameters
    ----------
    data : EISData
        Measured spectrum.
    n_restarts : int
        Random restarts per nonlinear solve.
    rc_groups : int, default 5
        Maximum number of parallel R–CPE branches (values above 5 emit a
        warning).
    initial_peaks : int, default 2
        Maximum number of branches in the initial guess, taken from the peaks
        of :math:`-\mathrm{Im}(Z)`.
    verbose : bool, default False
        Print branch counts and cost changes during expansion.
    seed : int, optional
        RNG seed for fit restarts.
    tau_bounds : tuple of float, optional
        ``(tau_min, tau_max)`` bounds on :math:`\tau_i = R_i C_i`. ``None``
        (default) leaves them unbounded.
    resistance_source_bounds : tuple of float, optional
        ``(R_s_min, R_s_max)`` bounds on :math:`R_s`. ``None`` (default) keeps
        :math:`R_s` strictly positive.
    resistance_branch_bounds : tuple of float, optional
        ``(R_min, R_max)`` bounds on all branch resistances.
    exponent_bounds : tuple of float, optional
        ``(e_min, e_max)`` bounds on all CPE exponents. ``None`` (default)
        uses ``(0.2, 1.0)``.
    inductance_exponent_bounds : tuple of float, optional
        Bounds on :math:`n_L`. ``None`` (default) uses *exponent_bounds* if
        given, otherwise ``(0.7, 1.0)``.
    weighting : {"modulus", "unit"}, default "modulus"
        Residual weighting: ``"modulus"`` (relative error) or ``"unit"``
        (unweighted).
    n_jobs : int, optional
        Worker processes for the restarts (``1``: sequential, default; ``-1``:
        all cores). Scripts using ``n_jobs != 1`` on Windows/macOS need an
        ``if __name__ == "__main__":`` guard.
    early_stop : int, optional
        Stop the restarts of each sub-fit once this many reached the best
        cost. ``None`` (default) runs all restarts.

    Returns
    -------
    result : FitResults
        Best fit after branch expansion.

    See Also
    --------
    EISData.fit : Method form of this function.
    quick_fit : Single fit with a fixed number of branches.
    """
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
    r"""Fit an R–CPE ladder with a fixed number of branches to *data*.

    Thin wrapper around
    :func:`~blowing_bubbles.scripts.global_minimize.impedance_fit` (imported
    lazily); all arguments are passed through, and that function documents
    them in detail. Its ``n_jobs``, ``executor`` and ``early_stop`` options
    are not exposed here and keep their defaults (sequential restarts).

    Parameters
    ----------
    data : EISData
        Measured spectrum.
    n_groups : int, default 2
        Number of parallel R–CPE branches.
    complex_fit : bool, default False
        Fit the CPE exponents :math:`e_i` and :math:`n_L`; if ``False`` they
        are fixed to 1.
    initial_parameters : CircuitParameters, optional
        Starting point with ``n_groups`` branches (not modified). ``None``
        (default) uses a generic start vector.
    n_restarts : int, default 1
        Number of log-normal perturbation restarts (best cost is kept).
    seed : int, optional
        RNG seed for the restarts.
    sigma : float, default 0.5
        Log-normal scale of restart perturbations.
    tau_bounds : tuple of float, optional
        ``(tau_min, tau_max)`` bounds on :math:`\tau_i = R_i C_i`. ``None``
        (default) leaves them unbounded.
    penalty_margin_weight : float, default 1.0
        Weight of the soft Nyquist margin penalty.
    penalty_weight : float, optional
        Overrides *penalty_margin_weight* when given.
    z_margin_frac, z_margin_abs : float, default 0.0
        Relative (to the :math:`Z'` span) and absolute soft margin on the
        Nyquist constraints; the larger of the two is used.
    max_nfev : int, optional
        Maximum function evaluations per restart. ``None`` (default) uses
        ``500 * n_params``.
    polish : bool, default True
        Run a polishing step if hard constraints are violated.
    polish_maxiter : int, default 50
        Iteration cap for polishing.
    constraint_tol : float, default 1e-9
        Tolerance for declaring constraints satisfied.
    inductance_bounds : bool, default False
        If ``True``, bound :math:`L` around an estimate from the slope of
        :math:`\mathrm{Im}(Z)`; otherwise only :math:`L \ge 0`.
    inductance_margin : float, default 2.0
        Margin factor for the inductance bounds.
    resistance_source_bounds : tuple of float, optional
        ``(R_s_min, R_s_max)`` bounds on :math:`R_s`.
    resistance_branch_bounds : tuple of float, optional
        ``(R_min, R_max)`` bounds on all branch resistances.
    exponent_bounds : tuple of float, optional
        ``(e_min, e_max)`` bounds on all CPE exponents (only used with
        ``complex_fit=True``).
    inductance_exponent_bounds : tuple of float, optional
        Bounds on the inductor exponent :math:`n_L`.
    weighting : {"modulus", "unit"}, default "modulus"
        Residual weighting: ``"modulus"`` (relative error) or ``"unit"``
        (unweighted).

    Returns
    -------
    result : FitResults
        Optimized parameters, residuals and parameter uncertainties.

    See Also
    --------
    EISData.quick_fit : Method form starting from given parameters.
    fit_circuit : Fit with automatic branch building.

    Notes
    -----
    The defaults match ``impedance_fit``: ``z_margin_frac=0.0`` and
    ``inductance_bounds=False`` (previously ``0.05`` and ``True``, which could
    exclude the true circuit). Pass the old values explicitly to restore the
    previous behaviour.
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
