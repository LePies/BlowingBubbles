"""
Test synthetic EIS cases with known parameters.

This script:
1. Loads synthetic CSV files from research/data/synthetic/
2. Fits them using EISData.fit()
3. Compares fitted parameters to true parameters
4. Generates comparison plots
5. Saves results summary
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from blowing_bubbles.classes import CircuitParameters, EISData

# Configuration
DATA_DIR = Path("research/data/synthetic")
RESULTS_DIR = Path("research/results/synthetic")

# Create directories
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_true_parameters(json_path: Path) -> dict:
    """Load true parameters from JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


def fit_and_compare(
    csv_path: Path,
    json_path: Path,
    max_rc_groups: int = 5,
    n_restarts: int = 100,
    seed: int = 42,
) -> dict:
    """Fit a synthetic case and compare to true parameters.
    
    Uses the number of RC branches from true parameters, up to max_rc_groups.
    
    Returns:
        Dictionary with results and comparison metrics
    """
    print(f"\nProcessing: {csv_path.stem}")
    
    # Load data
    df = pd.read_csv(csv_path)
    eis_data = EISData(df)
    
    # Load true parameters
    true_params = load_true_parameters(json_path)
    
    # Determine number of RC groups from true parameters
    n_true_rc = len(true_params["resistances"])
    
    # Use the true number of RC branches, but don't exceed max_rc_groups
    use_rc_groups = min(n_true_rc, max_rc_groups)
    
    print(f"  Data points: {len(df)}")
    print(f"  True RC branches: {n_true_rc}")
    print(f"  Fitting with {use_rc_groups} RC groups (max={max_rc_groups})...")
    
    # Fit
    start_time = time.time()
    fit_result = eis_data.fit(
        n_restarts=n_restarts,
        rc_groups=use_rc_groups,
        verbose=False,
        seed=seed,
    )
    elapsed = time.time() - start_time
    
    # Get fitted parameters
    fitted_params = fit_result.parameters
    
    # Calculate comparison metrics
    metrics = calculate_comparison_metrics(true_params, fitted_params, use_rc_groups)
    
    print(f"  Fit completed in {elapsed:.2f} s")
    print(f"  Success: {fit_result.success}")
    print(f"  Final cost: {fit_result.cost:.6e}")
    
    return {
        "case_name": csv_path.stem,
        "success": fit_result.success,
        "cost": fit_result.cost,
        "elapsed_time": elapsed,
        "n_data_points": len(df),
        "true_rc_branches": n_true_rc,
        "fitted_rc_branches": use_rc_groups,
        "metrics": metrics,
        "true_parameters": true_params,
        "fitted_parameters": {
            "resistance_source": float(fitted_params.resistance_source),
            "inductance": float(fitted_params.inductance),
            "inductance_exponent": float(fitted_params.inductance_exponent),
            "resistances": fitted_params.resistances.tolist()[:use_rc_groups],
            "capacitances": fitted_params.capacitances.tolist()[:use_rc_groups],
            "exponents": fitted_params.exponents.tolist()[:use_rc_groups],
        },
    }


def calculate_comparison_metrics(
    true_params: dict,
    fitted_params: CircuitParameters,
    n_compare: int,
) -> dict:
    """Calculate error metrics between true and fitted parameters.
    
    Both true and fitted parameters are sorted by tau (R*C) before comparison
    to ensure consistent ordering.
    """
    import numpy as np
    
    metrics = {}
    
    # Series resistance error
    rs_true = true_params["resistance_source"]
    rs_fitted = fitted_params.resistance_source
    metrics["rs_error_abs"] = abs(rs_fitted - rs_true)
    metrics["rs_error_rel"] = abs(rs_fitted - rs_true) / rs_true * 100  # percentage
    
    # Inductance error
    l_true = true_params["inductance"]
    l_fitted = fitted_params.inductance
    metrics["l_error_abs"] = abs(l_fitted - l_true)
    metrics["l_error_rel"] = abs(l_fitted - l_true) / l_true * 100 if l_true > 0 else 0
    
    # Inductance exponent error
    le_true = true_params["inductance_exponent"]
    le_fitted = fitted_params.inductance_exponent
    metrics["le_error_abs"] = abs(le_fitted - le_true)
    
    # Sort both true and fitted parameters by tau = R*C for consistent comparison
    # True parameters (already sorted by tau in CircuitParameters)
    true_resistances = np.array(true_params["resistances"])
    true_capacitances = np.array(true_params["capacitances"])
    true_exponents = np.array(true_params["exponents"])
    true_taus = true_capacitances * true_resistances
    
    # Fitted parameters (need to sort by tau)
    fitted_resistances = fitted_params.resistances.copy()
    fitted_capacitances = fitted_params.capacitances.copy()
    fitted_exponents = fitted_params.exponents.copy()
    fitted_taus = fitted_capacitances * fitted_resistances
    
    # Sort fitted parameters by tau
    fitted_sort_idx = np.argsort(fitted_taus)
    fitted_resistances = fitted_resistances[fitted_sort_idx]
    fitted_capacitances = fitted_capacitances[fitted_sort_idx]
    fitted_exponents = fitted_exponents[fitted_sort_idx]
    
    # Now compare sorted parameters
    r_errors_abs = []
    r_errors_rel = []
    c_errors_abs = []
    c_errors_rel = []
    exp_errors_abs = []
    
    for i in range(n_compare):
        # Resistance
        r_true = true_resistances[i]
        r_fitted = fitted_resistances[i]
        r_errors_abs.append(abs(r_fitted - r_true))
        r_errors_rel.append(abs(r_fitted - r_true) / r_true * 100)
        
        # Capacitance
        c_true = true_capacitances[i]
        c_fitted = fitted_capacitances[i]
        c_errors_abs.append(abs(c_fitted - c_true))
        c_errors_rel.append(abs(c_fitted - c_true) / c_true * 100)
        
        # Exponent
        exp_true = true_exponents[i]
        exp_fitted = fitted_exponents[i]
        exp_errors_abs.append(abs(exp_fitted - exp_true))
    
    metrics["r_errors_abs"] = r_errors_abs
    metrics["r_errors_rel"] = r_errors_rel
    metrics["c_errors_abs"] = c_errors_abs
    metrics["c_errors_rel"] = c_errors_rel
    metrics["exp_errors_abs"] = exp_errors_abs
    
    # Average errors
    metrics["r_error_avg_rel"] = np.mean(r_errors_rel)
    metrics["c_error_avg_rel"] = np.mean(c_errors_rel)
    metrics["exp_error_avg_abs"] = np.mean(exp_errors_abs)
    
    return metrics


def plot_comparison(
    csv_path: Path,
    json_path: Path,
    result: dict,
    output_path: Path,
) -> None:
    """Generate Nyquist plot with data, true model, and fitted model."""
    import numpy as np
    
    print(f"  Generating plot...")
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Load true parameters and create model
    true_params_dict = load_true_parameters(json_path)
    true_params = CircuitParameters(
        resistance_source=true_params_dict["resistance_source"],
        inductance=true_params_dict["inductance"],
        inductance_exponent=true_params_dict["inductance_exponent"],
        resistances=np.array(true_params_dict["resistances"]),
        capacitances=np.array(true_params_dict["capacitances"]),
        exponents=np.array(true_params_dict["exponents"]),
    )
    
    # Create EISData object
    eis_data = EISData(df)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Plot data points
    eis_data.plot_nyquist(axes=ax, label="Data (noisy)", color="blue", alpha=0.5)
    
    # Plot true model
    freqs_plot = np.logspace(
        np.log10(df["Freq (Hz)"].min()),
        np.log10(df["Freq (Hz)"].max()),
        200
    )
    true_params.plot_full_fit(freqs_plot, ax, unit=r"$\Omega\cdot\mathrm{cm}^2$")
    
    # Plot fitted model - manually plot the impedance curve
    fitted_params = CircuitParameters(
        resistance_source=result["fitted_parameters"]["resistance_source"],
        inductance=result["fitted_parameters"]["inductance"],
        inductance_exponent=result["fitted_parameters"]["inductance_exponent"],
        resistances=np.array(result["fitted_parameters"]["resistances"]),
        capacitances=np.array(result["fitted_parameters"]["capacitances"]),
        exponents=np.array(result["fitted_parameters"]["exponents"]),
    )
    
    # Plot fitted model impedance
    z_fitted = fitted_params.get_impedance(freqs_plot)
    ax.plot(z_fitted.real, -z_fitted.imag, label="Fitted Model", color="red", linestyle="--", linewidth=2)
    
    # Plot true model impedance
    z_true = true_params.get_impedance(freqs_plot)
    ax.plot(z_true.real, -z_true.imag, label="True Model", color="green", linestyle="-", linewidth=2)
    
    # Customize plot
    ax.legend()
    ax.set_title(f"{csv_path.stem.replace('_', ' ').title()}")
    ax.set_xlabel("Zreal (Ohm.cm2)")
    ax.set_ylabel("-Zimag (Ohm.cm2)")
    ax.grid(True, alpha=0.3)
    
    # Add metrics text
    metrics = result["metrics"]
    metrics_text = (
        f"Cost: {result['cost']:.2e}\n"
        f"Rs err: {metrics['rs_error_rel']:.1f}%\n"
        f"R err: {metrics['r_error_avg_rel']:.1f}%\n"
        f"C err: {metrics['c_error_avg_rel']:.1f}%\n"
        f"Exp err: {metrics['exp_error_avg_abs']:.3f}"
    )
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
            verticalalignment="top", bbox=dict(boxstyle="round", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {output_path.name}")


def main():
    """Test all synthetic cases."""
    print("=" * 70)
    print("Testing Synthetic EIS Cases with Known Parameters")
    print("=" * 70)
    
    # Get all CSV files
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    
    if not csv_files:
        print("\nERROR: No CSV files found in {DATA_DIR}")
        print("Please run generate_synthetic_cases.py first.")
        return
    
    print(f"\nFound {len(csv_files)} synthetic cases to test")
    
    # Test each case
    results = []
    
    for csv_path in csv_files:
        json_path = csv_path.with_suffix(".csv").parent / (csv_path.stem + "_true_parameters.json")
        
        if not json_path.exists():
            print(f"\nWARNING: No parameter file found for {csv_path.name}")
            continue
        
        try:
            result = fit_and_compare(
                csv_path=csv_path,
                json_path=json_path,
                max_rc_groups=5,
                n_restarts=100,
                seed=42,
            )
            results.append(result)
            
            # Generate plot
            if result.get("success", False):
                plot_path = RESULTS_DIR / f"{csv_path.stem}_comparison.png"
                plot_comparison(csv_path, json_path, result, plot_path)
            
        except Exception as e:
            print(f"\nERROR processing {csv_path.name}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "case_name": csv_path.stem,
                "success": False,
                "error": str(e),
            })
    
    # Save results summary
    results_path = RESULTS_DIR / "synthetic_test_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate markdown summary
    generate_markdown_summary(results, RESULTS_DIR)
    
    # Print summary
    print_summary(results)


def generate_markdown_summary(results: list, output_dir: Path) -> None:
    """Generate a markdown summary of the test results."""
    md_path = output_dir / "test_results_summary.md"
    
    with open(md_path, "w") as f:
        f.write("# Synthetic EIS Test Results\n\n")
        f.write(f"Tested {len(results)} cases with known parameters.\n\n")
        
        successful = sum(1 for r in results if r.get("success", False))
        failed = sum(1 for r in results if not r.get("success", False))
        
        f.write(f"- **Successful fits**: {successful}\n")
        f.write(f"- **Failed fits**: {failed}\n\n")
        
        f.write("## Detailed Results\n\n")
        
        for r in results:
            if r.get("success", False):
                m = r["metrics"]
                f.write(f"### {r['case_name']}\n\n")
                f.write(f"- **Cost**: {r['cost']:.6e}\n")
                f.write(f"- **Time**: {r['elapsed_time']:.2f} s\n")
                f.write(f"- **Rs error**: {m['rs_error_rel']:.2f}%\n")
                f.write(f"- **R error (avg)**: {m['r_error_avg_rel']:.2f}%\n")
                f.write(f"- **C error (avg)**: {m['c_error_avg_rel']:.2f}%\n")
                f.write(f"- **Exp error (avg)**: {m['exp_error_avg_abs']:.3f}\n")
                f.write("\n")
            else:
                f.write(f"### {r.get('case_name', 'Unknown')} - FAILED\n")
                f.write(f"- Error: {r.get('error', 'Unknown')}\n\n")
    
    print(f"\nMarkdown summary saved to: {md_path}")


def print_summary(results: list) -> None:
    """Print a summary of the test results."""
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    successful = sum(1 for r in results if r.get("success", False))
    failed = sum(1 for r in results if not r.get("success", False))
    
    print(f"\nTotal cases: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if successful > 0:
        print("\n" + "-" * 70)
        print("{:<45} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
            "Case", "Cost", "Rs Err%", "R Err%", "C Err%", "Exp Err"))
        print("-" * 70)
        
        for r in results:
            if r.get("success", False):
                m = r["metrics"]
                print("{:<45} {:>10.2e} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.3f}".format(
                    r["case_name"][:45],
                    r["cost"],
                    m["rs_error_rel"],
                    m["r_error_avg_rel"],
                    m["c_error_avg_rel"],
                    m["exp_error_avg_abs"],
                ))
    
    if failed > 0:
        print("\nFailed cases:")
        for r in results:
            if not r.get("success", False):
                print(f"  {r.get('case_name', 'Unknown')}: {r.get('error', 'Unknown')}")
    
    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
