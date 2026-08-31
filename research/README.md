# Research Data and Results

This directory contains **EIS data** and **analysis results** for validating the BlowingBubbles fitting algorithm.

## Directory Structure

```
research/
├── data/
│   ├── original/              # ❌ Gitignored - Raw downloaded data
│   │   ├── sofc_stack_monitoring_extracted/
│   │   │   ├── mat_Z.npy      # Complex impedance (606 spectra × 210 freq)
│   │   │   └── f_all.npy      # Frequencies (210 points)
│   │   └── soec_operating_conditions_extracted/
│   │       └── eis_data/      # Individual .npz files (293 spectra)
│   ├── processed/              # ✅ Tracked - Real data converted to CSV
│   │   ├── sofc_stack_spectrum_*.csv
│   │   └── soec_operating_conditions_*.csv
│   └── synthetic/             # ✅ Tracked - Synthetic data with known parameters
│       ├── case1_*.csv
│       ├── case2a_*.csv
│       └── *_true_parameters.json
├── results/
│   ├── synthetic/             # ✅ Tracked - Synthetic data results
│   │   ├── *.png              # Nyquist plots: data vs true vs fitted
│   │   ├── test_results_summary.md
│   │   └── synthetic_test_results.json
│   └── *.png                  # ✅ Tracked - Real data results
├── *.py                      # ✅ Tracked - Scripts
└── README.md                  # ✅ Tracked
```

## Data Types

This directory contains **two types of data**:

### 1. Real EIS Data (from Published Papers)

These are actual EIS measurements from published datasets, used for **demonstration** that the algorithm works on real-world data.

#### Dataset 1: SOFC Stack Monitoring
- **Source**: Gorenc et al. 2026, https://repo.ijs.si/e2pub/cnn_drt
- **Type**: SOFC (Solid Oxide Fuel Cell)
- **Format**: mat_Z.npy (606 spectra × 210 frequencies), f_all.npy
- **Sampling**: 7 spectra (indices 0, 100, 200, 300, 400, 500, 600)
- **Description**: Stack monitoring with 210 frequency points per spectrum
- **Status**: ✅ Converted and analyzed
- **Note**: These datasets do NOT have known circuit parameters (they're DRT-focused)

#### Dataset 2: SOEC Operating Conditions
- **Source**: Gorenc et al. 2026, https://repo.ijs.si/e2pub/cnn_drt
- **Type**: SOEC (Solid Oxide Electrolyzer Cell)
- **Format**: Individual .npz files with keys:
  - `Z_prbs`: Complex impedance
  - `f_wt_site`: Frequency array
  - `new_rho_all`, `s1_all`, `s2_all`: DRT-related metadata
- **Total**: 293 spectra (324 .npz files in directory)
- **Sampling**: 7 spectra (indices 0, 50, 100, 150, 200, 250, 300)
- **Description**: Operating condition study with varying H₂/H₂O compositions
- **Status**: ✅ Converted and analyzed
- **Note**: These datasets do NOT have known circuit parameters

#### Dataset 3: SOEC Multi-Regime Degradation
- **Source**: Gorenc et al. 2026, https://repo.ijs.si/e2pub/cnn_drt
- **Type**: SOEC
- **Format**: Individual .npz files (similar to Dataset 2)
- **Total**: 1248 spectra
- **Status**: ⏳ Not yet downloaded (can be added following same pattern)

### 2. Synthetic EIS Data (with Known Parameters)

These are **computationally generated** EIS spectra based on published parameter ranges, used for **quantitative validation** of parameter recovery.

#### Case 1: Mutter et al. 2026 - Steam Electrolysis
- **Source**: Parameters based on published SOEC data
- **Conditions**: 750°C, 300 mA/cm², 90% H₂O, 10% H₂
- **Cell**: Commercial 4×4 cm² electrolyte-supported cell
- **Circuit**: R_s + L + 3×(R||CPE)
- **Parameters**: Modified from original to ensure good τ separation
- **Noise**: Gaussian, σ = 0.5 mΩ·cm²

#### Case 2: Boškoski et al. 2024 - Multiple Operating Conditions
- **Source**: Parameters based on published SOEC data
- **Conditions**: 800°C, industrial-scale 100 cm² SOEC
- **Three conditions**:
  - **2a**: 700 mA/cm², 20% H₂
  - **2b**: 700 mA/cm², 10% H₂
  - **2c**: 900 mA/cm², unstable steam supply
- **Circuit**: R_s + L + 2×(R||CPE)
- **Noise**: Gaussian, σ = 0.5 mΩ·cm² (0.7 mΩ·cm² for unstable condition)

## Scripts and Usage

### For Real Data (Demonstration)

#### Step 1: Convert Raw Data to CSV
```bash
python research/convert_all_datasets.py
```
- Reads raw numpy arrays from extracted datasets
- Converts to CSV with columns: `Freq (Hz)`, `Zreal (Ohm.cm2)`, `Zimag (Ohm.cm2)`
- Saves to `research/data/processed/`
- Samples spectra to keep repository size manageable

#### Step 2: Analyze with BlowingBubbles
```bash
python research/analyze_real_data.py
```
- Loads all CSV files from `research/data/processed/`
- Fits each spectrum with **3 R-CPE branches** using **50 restarts**
- Generates Nyquist plots with data + fit overlay
- Saves results to `research/results/`
- Creates summary files (JSON and Markdown)

### For Synthetic Data (Validation)

#### Step 1: Generate Synthetic Data
```bash
python research/generate_synthetic_cases.py
```
- Creates 4 test cases with known parameters from published literature
- Saves CSV files to `research/data/synthetic/`
- Saves true parameters as JSON files
- Adds realistic Gaussian noise (σ = 0.5-0.7 mΩ·cm²)

#### Step 2: Test and Validate
```bash
python research/test_synthetic_cases.py
```
- Loads synthetic CSV files and true parameters
- Fits each case using BlowingBubbles
- Compares fitted parameters to true parameters
- Calculates error metrics (Rs, R, C, exponents)
- Generates comparison plots: Data vs True Model vs Fitted Model
- Saves results to `research/results/synthetic/`

## Results Summary

### Real Data Results

✅ **All 14 real spectra successfully fitted!**

| Dataset | Spectra | Avg Cost | Avg Time |
|---------|---------|----------|----------|
| SOEC Operating Conditions | 7 | 2.3e-3 | 4.8 s |
| SOFC Stack Monitoring | 7 | 1.6e-6 | 5.2 s |

**Key Findings:**
- SOFC data fits extremely well (cost ~1e-7 to 1e-6)
- SOEC data fits well (cost ~1e-5 to 1e-3)
- All fits completed successfully with 3 R-CPE branches
- Higher frequency resolution in SOFC data (210 vs 88-110 points)
- **Note**: These datasets don't have known true parameters, so we can't validate parameter recovery

See `research/results/analysis_summary.md` for detailed results.

### Synthetic Data Results

✅ **All 4 synthetic cases successfully fitted with excellent parameter recovery!**

| Case | Cost | Rs Err | R Err | C Err | Exp Err |
|------|------|--------|-------|-------|---------|
| Case 1 (Mutter, 3 branches) | 1.35e-05 | 0.13% | 0.69% | 0.68% | 0.001 |
| Case 2a (Boškoski, 2 branches) | 1.38e-05 | 0.13% | 8.42% | 7.35% | 0.004 |
| Case 2b (Boškoski, 2 branches) | 2.07e-05 | 0.15% | 0.48% | 0.41% | 0.000 |
| Case 2c (Boškoski, 2 branches) | 4.09e-05 | 0.16% | 9.75% | 5.23% | 0.009 |

**Key Findings:**
- **Rs recovery**: Excellent across all cases (< 0.2% error)
- **R recovery**: Good to excellent (0.5-10% error)
- **C recovery**: Good to excellent (0.4-7% error)
- **Exponent recovery**: Excellent (0.001-0.009 error)
- **Note**: Case 1 parameters were modified to ensure good τ separation for identifiability

See `research/results/synthetic/test_results_summary.md` for detailed results.

## Data Format

### Input (Raw)
- **.npy files**: Complex impedance arrays and frequency arrays
- **.npz files**: Compressed numpy archives with multiple arrays

### Processed (CSV)
All CSV files have three columns:
```csv
Freq (Hz),Zreal (Ohm.cm2),Zimag (Ohm.cm2)
```

This format is directly compatible with `EISData(dataframe)` from BlowingBubbles.

## Git Tracking

- ✅ `research/data/processed/` - **Tracked** (CSV files, ~2 MB)
- ✅ `research/results/` - **Tracked** (plots and summaries)
- ✅ `research/*.py` - **Tracked** (scripts)
- ✅ `research/README.md` - **Tracked**
- ❌ `research/data/original/` - **Excluded** (large raw files, ~2 GB)

## Adding New Datasets

To add Dataset 3 or additional data:

1. Download zip from https://repo.ijs.si/e2pub/cnn_drt
2. Extract to `research/data/original/<name>_extracted/`
3. Update `convert_all_datasets.py` to include the new dataset
4. Run conversion script
5. Run analysis script

## References

1. Gorenc, J., et al. (2026). CNN-DRT: Convolutional Neural Network for Distribution of Relaxation Times Analysis. https://repo.ijs.si/e2pub/cnn_drt

2. Dataset documentation: https://repo.ijs.si/e2pub/cnn_drt

## Historical Note

This directory previously contained synthetic test data generated from published parameters (Mutter et al. 2026, Boškoski et al. 2024). That data has been removed and replaced with **real original EIS data** from the Gorenc et al. datasets, as requested by the user.
