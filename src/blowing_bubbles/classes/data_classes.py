r"""Electrochemical impedance spectroscopy (EIS) data and fitting.

:class:`EISData` loads and processes measured spectra; :func:`fit_circuit` fits
the R–CPE ladder using iterative fast branch building.
"""

from __future__ import annotations

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
            unit: str = "Ohm"
        ) -> None:
        """
        Initialize EIS data.

        Parameters
        ----------
        data_set : pd.DataFrame
            DataFrame containing EIS data with frequency, real, and
            imaginary impedance columns
        area : float
            Electrode area for normalization (default: 1.0)

        Returns
        -------
        None

        See Also
        --------
        from_csv : Alternative constructor to load from file
        """
        self.data = data_set
        self.keys = self.find_keys()
        self.Real = data_set[self.keys[0]].to_numpy()
        self.Imag = data_set[self.keys[1]].to_numpy()
        self.Freq = data_set[self.keys[2]].to_numpy()
        self.Rs = self.find_Rs()[0]
        self.Rtot = self.find_Rtot()[0]
        self.Rp = self.find_Rp()[0]
        self.normalization_area = area
        self.unit = unit

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
        area : float
            Area normalization factor (e.g., cm²)
        unit : str
            Unit for the data (e.g., "Ohm")

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
            unit=unit
        )

    def __add__(self, other: EISData) -> EISData:
        data_temp = self.data.copy()
        data_temp["Real"] = data_temp[self.keys[0]] + other.data[other.keys[0]]
        data_temp["Imag"] = data_temp[self.keys[1]] + other.data[other.keys[1]]
        return EISData(data_temp)

    def __sub__(self, other: EISData) -> EISData:
        data_temp = self.data.copy()
        data_temp["Real"] = data_temp[self.keys[0]] - other.data[other.keys[0]]
        data_temp["Imag"] = data_temp[self.keys[1]] - other.data[other.keys[1]]
        return EISData(data_temp)

    def __mul__(self, other: EISData) -> EISData:
        return EISData(self.data * other.data)

    def __truediv__(self, other: EISData) -> EISData:
        return EISData(self.data / other.data)

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
        freq_names = ["f", "freq", "frequency"]
        real_names = ["r", "real", "real_part"]
        imag_names = ["i", "imag", "imaginary", "imaginary_part"]
        R_key: str | None = None
        I_key: str | None = None
        F_key: str | None = None
        for key in self.data.keys():
            if any(name in key.lower() for name in freq_names):
                F_key = key
            if any(name in key.lower() for name in real_names):
                R_key = key
            if any(name in key.lower() for name in imag_names):
                I_key = key
        if R_key is None or F_key is None or I_key is None:
            raise ValueError(
                "Could not find required keys in data. "
                "Expected keys containing 'R', 'F', and 'I' (case insensitive)."
            )
        return R_key, I_key, F_key

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
        """
        self.Real = self.Real[start:end]
        self.Imag = self.Imag[start:end]
        self.Freq = self.Freq[start:end]

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
        diff_last = np.diff(self.Imag)[-1]
        if np.isnan(diff_last) or diff_last == 0:
            # Handle NaN or zero case - default to finding minimum real part
            min_idx = np.nanargmin(np.abs(self.Imag))
            return float(self.Real[min_idx]), int(min_idx)
        elif diff_last > 0:
            sign = 1
        else:  # diff_last < 0
            sign = -1
        for i in range(1, len(self.Imag)):
            if sign * np.diff(self.Imag)[-i] < 0:
                break
        i = i - 1

        lower_limit = min(self.Imag[0:-i])
        upper_limit = max(self.Imag[0:-i])
        if upper_limit < 0:
            upper_limit = 0
        for j in range(len(self.Imag[-i:])):
            if self.Imag[-i:][j] > upper_limit or self.Imag[-i:][j] < lower_limit:
                break
        return self.Real[-i:][j], j - i

    def simpleRs(self) -> None:
        """
        Simple Rs calculation (placeholder).

        Returns
        None
        """
        pass

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
        return self.Real[0], 0

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
        Also recalculates Rs, Rtot, and Rp.

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

        See Also
        --------
        from_csv : Load with normalization in one step
        """
        self.Real = self.Real * area
        self.Imag = self.Imag * area
        self.data[self.keys[0]] = self.data[self.keys[0]] * area
        self.data[self.keys[1]] = self.data[self.keys[1]] * area
        self.unit = norm_unit
        self.normalization_area = area
        self.Rs = self.find_Rs()[0]
        self.Rtot = self.find_Rtot()[0]
        self.Rp = self.find_Rp()[0]

        self.normalized = "YES"
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
            exponent_bounds=exponent_bounds
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
        """
        print(f"Cutting frequencies above {frequency} Hz")
        cut_index = np.where(self.Freq < frequency)[0]
        self.Real = self.Real[cut_index]
        self.Imag = self.Imag[cut_index]
        self.Freq = self.Freq[cut_index]
        self.data = self.data.iloc[cut_index]

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
        """
        cut_index = np.where(self.Freq > frequency)[0]
        self.Real = self.Real[cut_index]
        self.Imag = self.Imag[cut_index]
        self.Freq = self.Freq[cut_index]
        self.data = self.data.iloc[cut_index]

    def quick_fit(
        self,
        initial_parameters: CircuitParameters,
        complex_fit: bool = True,
        seed: int | None = None,
        sigma: float = 0.5,
        tau_bounds: tuple[float, float] | None = None,
        penalty_margin_weight: float = 1.0,
        penalty_weight: float | None = None,
        z_margin_frac: float = 0.05,
        z_margin_abs: float = 0.0,
        max_nfev: int | None = None,
        polish: bool = True,
        polish_maxiter: int = 50,
        constraint_tol: float = 1e-9,
        inductance_bounds: bool = True,
        inductance_margin: float = 2.0,
        resistance_source_bounds: tuple[float, float] | None = None,
        resistance_branch_bounds: tuple[float, float] | None = None,
        exponent_bounds: tuple[float, float] | None = None
    ) -> FitResults:
        r"""Quick fit of the parametric R–CPE ladder to this spectrum.

        Calls :func:`~blowing_bubbles.classes.data_classes.quick_fit`
        (global optimization).
    
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
            exponent_bounds = exponent_bounds
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
        exponent_bounds=exponent_bounds
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
    z_margin_frac: float = 0.05,
    z_margin_abs: float = 0.0,
    max_nfev: int | None = None,
    polish: bool = True,
    polish_maxiter: int = 50,
    constraint_tol: float = 1e-9,
    inductance_bounds: bool = True,
    inductance_margin: float = 2.0,
    resistance_source_bounds: tuple[float, float] | None = None,
    resistance_branch_bounds: tuple[float, float] | None = None,
    exponent_bounds: tuple[float, float] | None = None,
) -> FitResults:
    """Quick fit of an R–CPE ladder to *data* using default parameters."""
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
        exponent_bounds = exponent_bounds
    )
