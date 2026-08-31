"""
Convert all original SOEC/SOFC EIS data to CSV format.

This script converts the raw numpy arrays from the Gorenc et al. datasets
(available at https://repo.ijs.si/e2pub/cnn_drt) into CSV files compatible
with the BlowingBubbles EISData class.

Datasets:
- Dataset 1 (SOFC): mat_Z.npy, f_all.npy (606 spectra, 210 frequencies)
- Dataset 2 (SOEC): Individual .npz files with Z_prbs, f_wt_site (293 spectra)
- Dataset 3 (SOEC): Individual .npz files with Z_prbs, f_wt_site (1248 spectra)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# Configuration
DATA_DIR = Path("research/data")
ORIGINAL_DIR = DATA_DIR / "original"
PROCESSED_DIR = DATA_DIR / "processed"

# Create directories
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def convert_sofc_dataset():
    """Convert Dataset 1 (SOFC stack monitoring) from .npy files."""
    print("\n" + "=" * 70)
    print("Converting Dataset 1: SOFC Stack Monitoring")
    print("=" * 70)
    
    npy_dir = ORIGINAL_DIR / "sofc_stack_monitoring_extracted"
    if not npy_dir.exists():
        print("  ERROR: Dataset 1 not found. Skipping.")
        return
    
    # Load data
    mat_Z = np.load(npy_dir / "mat_Z.npy", allow_pickle=True)
    f_all = np.load(npy_dir / "f_all.npy", allow_pickle=True)
    
    print(f"  Loaded {len(mat_Z)} spectra with {len(f_all)} frequency points")
    
    # Sample indices (every 100th spectrum to get a representative sample)
    sample_indices = list(range(0, len(mat_Z), 100))[:10]  # First 10, max 60
    
    for idx in sample_indices:
        # Get impedance data for this spectrum
        z_complex = mat_Z[idx]  # Shape (1, F)
        z_flat = z_complex.flatten()  # Shape (F,)
        
        # Create DataFrame
        df = pd.DataFrame({
            "Freq (Hz)": f_all,
            "Zreal (Ohm.cm2)": z_flat.real,
            "Zimag (Ohm.cm2)": z_flat.imag,
        })
        
        # Save CSV
        output_path = PROCESSED_DIR / f"sofc_stack_spectrum_{idx:04d}.csv"
        df.to_csv(output_path, index=False)
        print(f"  Saved: {output_path.name}")
    
    print(f"  Converted {len(sample_indices)} spectra from Dataset 1")


def convert_soec_npz_dataset(dataset_name: str, npz_dir: Path, sample_every: int = 50):
    """Convert Dataset 2 or 3 (SOEC) from individual .npz files.
    
    Args:
        dataset_name: Name for the dataset (e.g., 'soec_operating_conditions')
        npz_dir: Path to directory containing .npz files
        sample_every: Sample every Nth file (None = all)
    """
    print("\n" + "=" * 70)
    print(f"Converting Dataset: {dataset_name}")
    print("=" * 70)
    
    if not npz_dir.exists():
        print(f"  ERROR: {npz_dir} not found. Skipping.")
        return
    
    # Get all .npz files
    npz_files = sorted(npz_dir.glob("*.npz"))
    print(f"  Found {len(npz_files)} .npz files")
    
    # Determine which files to convert
    if sample_every:
        indices = list(range(0, len(npz_files), sample_every))[:20]  # Max 20 samples
    else:
        indices = list(range(len(npz_files)))
    
    converted_count = 0
    for i, npz_file in enumerate(npz_files):
        if i not in indices:
            continue
        
        try:
            # Load .npz file
            with np.load(npz_file) as data:
                # Check which keys are available
                if 'Z_prbs' in data and 'f_wt_site' in data:
                    z_complex = data['Z_prbs']
                    freqs = data['f_wt_site']
                elif 'Z' in data and 'f' in data:
                    z_complex = data['Z']
                    freqs = data['f']
                else:
                    print(f"  WARNING: Unknown format in {npz_file.name}")
                    continue
            
            # Create DataFrame
            df = pd.DataFrame({
                "Freq (Hz)": freqs,
                "Zreal (Ohm.cm2)": z_complex.real,
                "Zimag (Ohm.cm2)": z_complex.imag,
            })
            
            # Save CSV
            output_path = PROCESSED_DIR / f"{dataset_name}_{i:04d}.csv"
            df.to_csv(output_path, index=False)
            converted_count += 1
            print(f"  Saved: {output_path.name} (from {npz_file.name})")
            
        except Exception as e:
            print(f"  ERROR processing {npz_file.name}: {e}")
    
    print(f"  Converted {converted_count} spectra from {dataset_name}")


def main():
    """Convert all datasets."""
    print("\n" + "=" * 70)
    print("Converting All EIS Datasets to CSV")
    print("=" * 70)
    
    # Convert Dataset 1 (SOFC)
    convert_sofc_dataset()
    
    # Convert Dataset 2 (SOEC - Operating Conditions)
    soec_op_dir = ORIGINAL_DIR / "soec_operating_conditions_extracted" / "eis_data"
    if soec_op_dir.exists():
        convert_soec_npz_dataset(
            dataset_name="soec_operating_conditions",
            npz_dir=soec_op_dir,
            sample_every=50
        )
    
    # Convert Dataset 3 (SOEC - Degradation)
    # Check if there's another directory for dataset 3
    soec_deg_dir = ORIGINAL_DIR / "soec_degradation_extracted"
    if soec_deg_dir.exists():
        # Find eis_data subdirectory
        eis_dir = soec_deg_dir / "eis_data"
        if eis_dir.exists():
            convert_soec_npz_dataset(
                dataset_name="soec_degradation",
                npz_dir=eis_dir,
                sample_every=200
            )
        else:
            # Maybe .npz files are directly in the directory
            if any(soec_deg_dir.glob("*.npz")):
                convert_soec_npz_dataset(
                    dataset_name="soec_degradation",
                    npz_dir=soec_deg_dir,
                    sample_every=200
                )
    
    # List all generated files
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    csv_files = sorted(PROCESSED_DIR.glob("*.csv"))
    print(f"\nGenerated {len(csv_files)} CSV files in {PROCESSED_DIR}:")
    for f in csv_files:
        print(f"  {f.name}")
    
    print("\nConversion complete!")


if __name__ == "__main__":
    main()
