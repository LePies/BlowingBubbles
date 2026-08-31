"""
Analyze real EIS data from published datasets using BlowingBubbles.

This script:
1. Loads CSV files from the processed data directory
2. Fits them using the EISData.fit() method
3. Saves results and generates comparison plots
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from blowing_bubbles.classes import EISData

# Configuration
DATA_DIR = Path("research/data")
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = Path("research/results")

# Create directories
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_and_fit_csv(csv_path: Path, rc_groups: int = 3, n_restarts: int = 50, seed: int = 42) -> dict:
    """Load a CSV file and fit it using EISData.
    
    Args:
        csv_path: Path to CSV file
        rc_groups: Number of R-CPE branches
        n_restarts: Number of restarts for fitting
        seed: Random seed
        
    Returns:
        Dictionary with results
    """
    print(f"\nProcessing: {csv_path.name}")
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} data points")
    
    # Create EISData object
    eis_data = EISData(df)
    
    # Fit
    print(f"  Fitting with {rc_groups} R-CPE branches, {n_restarts} restarts...")
    start_time = time.time()
    
    try:
        fit_result = eis_data.fit(
            n_restarts=n_restarts,
            rc_groups=rc_groups,
            verbose=False,
            seed=seed,
        )
        elapsed = time.time() - start_time
        
        print(f"  Fit completed in {elapsed:.2f} s")
        print(f"  Success: {fit_result.success}")
        print(f"  Final cost: {fit_result.cost:.6e}")
        
        return {
            "file": csv_path.name,
            "success": fit_result.success,
            "cost": fit_result.cost,
            "elapsed_time": elapsed,
            "parameters": str(fit_result.parameters),
            "n_data_points": len(df),
            "rc_groups": rc_groups,
            "n_restarts": n_restarts,
        }
        
    except Exception as e:
        print(f"  ERROR during fitting: {e}")
        elapsed = time.time() - start_time
        return {
            "file": csv_path.name,
            "success": False,
            "error": str(e),
            "elapsed_time": elapsed,
            "n_data_points": len(df),
        }


def plot_comparison(csv_path: Path, fit_result, output_path: Path) -> None:
    """Generate a Nyquist plot comparing data and fit."""
    import numpy as np
    
    print(f"  Generating plot for {csv_path.name}...")
    
    # Load data
    df = pd.read_csv(csv_path)
    eis_data = EISData(df)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot data
    eis_data.plot_nyquist(axes=ax, label="Data", color="blue", alpha=0.5)
    
    # Plot fit
    freqs_plot = np.logspace(
        np.log10(df["Freq (Hz)"].min()),
        np.log10(df["Freq (Hz)"].max()),
        100
    )
    fit_result.parameters.plot_full_fit(freqs_plot, ax, unit=r"$\Omega\cdot\mathrm{cm}^2$")
    
    # Customize plot
    ax.legend()
    ax.set_title(f"{csv_path.stem.replace('_', ' ').title()}")
    ax.set_xlabel("Zreal (Ohm.cm2)")
    ax.set_ylabel("-Zimag (Ohm.cm2)")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {output_path.name}")


def main():
    """Analyze all processed CSV files."""
    print("=" * 70)
    print("Analyzing Real EIS Data from Published Datasets")
    print("=" * 70)
    
    # Get all CSV files
    csv_files = sorted(PROCESSED_DIR.glob("*.csv"))
    
    if not csv_files:
        print("\nERROR: No CSV files found in {PROCESSED_DIR}")
        print("Please run convert_all_datasets.py first.")
        return
    
    print(f"\nFound {len(csv_files)} CSV files to analyze")
    
    # Analyze each file
    results = []
    
    for csv_path in csv_files:
        try:
            result = load_and_fit_csv(csv_path, rc_groups=3, n_restarts=50, seed=42)
            results.append(result)
            
            # Generate plot if fitting was successful
            if result.get("success", False):
                plot_path = RESULTS_DIR / f"{csv_path.stem}_fit.png"
                # We need to re-load and fit to get the fit_result object for plotting
                df = pd.read_csv(csv_path)
                eis_data = EISData(df)
                fit_result = eis_data.fit(
                    n_restarts=50,
                    rc_groups=3,
                    verbose=False,
                    seed=42,
                )
                plot_comparison(csv_path, fit_result, plot_path)
            
        except Exception as e:
            print(f"\nERROR processing {csv_path.name}: {e}")
            results.append({
                "file": csv_path.name,
                "success": False,
                "error": str(e),
            })
    
    # Save results summary
    results_path = RESULTS_DIR / "real_data_analysis_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    # Print summary
    successful = sum(1 for r in results if r.get("success", False))
    failed = sum(1 for r in results if not r.get("success", False))
    
    print(f"\nTotal files: {len(results)}")
    print(f"Successful fits: {successful}")
    print(f"Failed fits: {failed}")
    
    if successful > 0:
        print("\nSuccessful fits:")
        for r in results:
            if r.get("success", False):
                print(f"  {r['file']}: cost={r['cost']:.6e}, time={r['elapsed_time']:.2f}s")
    
    if failed > 0:
        print("\nFailed fits:")
        for r in results:
            if not r.get("success", False):
                error = r.get("error", "Unknown error")
                print(f"  {r['file']}: {error[:50]}...")
    
    print(f"\nResults saved to: {results_path}")
    print("Plots saved to: {RESULTS_DIR}")
    
    # Also create a markdown summary
    md_path = RESULTS_DIR / "analysis_summary.md"
    with open(md_path, "w") as f:
        f.write("# Real EIS Data Analysis\n\n")
        f.write(f"Analyzed {len(results)} spectra from published datasets.\n\n")
        f.write(f"- **Successful fits**: {successful}\n")
        f.write(f"- **Failed fits**: {failed}\n\n")
        f.write("## Results\n\n")
        for r in results:
            if r.get("success", False):
                f.write(f"- **{r['file']}**: cost={r['cost']:.6e}, time={r['elapsed_time']:.2f}s\n")
            else:
                f.write(f"- **{r['file']}**: FAILED - {r.get('error', 'Unknown error')}\n")
    
    print(f"Markdown summary saved to: {md_path}")


if __name__ == "__main__":
    main()
