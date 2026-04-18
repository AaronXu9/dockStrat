# Generalized ICM RMSD Calculator

This script (`generalized_icm_rmsd.py`) provides a generalized way to compute RMSD values using ICM for any docking method defined in `Approach.py`. It's designed to complement the PoseBusters analysis in `generate_pb_results.py`.

## Features

- **Multi-method support**: Works with all methods defined in `Approach.py` (ICM, DiffDock, Chai-1, Vina, GNINA, SurfDock, Boltz, UniDock2, etc.)
- **Automatic file discovery**: Finds prediction files using each method's specific file naming pattern
- **Score integration**: Extracts method-specific scores when available
- **Error handling**: Gracefully handles missing files, timeouts, and ICM errors
- **Flexible usage**: Can run all methods or individual methods

## Dependencies

- `pandas`
- `rdkit`
- `tqdm`
- ICM executable (`/home/aoxu/icm-3.9-4/icm64` by default)
- The `calculate_rmsd.icm` script (automatically discovered)

## Usage

### Run all methods:
```bash
python generalized_icm_rmsd.py
```

### Run a specific method:
```bash
python generalized_icm_rmsd.py --method boltz
python generalized_icm_rmsd.py --method vina --exp_name runsNposes_1
```

### From Python script:
```python
from generalized_icm_rmsd import run_single_method_rmsd, main

# Run single method
df_boltz = run_single_method_rmsd("boltz", exp_name="runsNposes_0")

# Run all methods
df_all = main()
```

## Configuration

The script can be configured by modifying the following variables in the `main()` function:

- `exp_name`: Experiment name (default: "runsNposes_0")
- `base_outdir`: Base output directory (default: `${PROJECT_ROOT}/forks`)
- `data_dir`: Data directory with reference files (default: `${PROJECT_ROOT}/data/runsNposes`)
- `BASE_DIRS`: Dictionary mapping method names to their output directories

## Output

The script generates:

1. **Individual CSV files**: `{method_name}_{exp_name}_rmsd_results.csv` for each method
2. **Combined CSV file**: `combined_{exp_name}_rmsd_results.csv` with all results
3. **Console output**: Summary statistics by method

### Output columns:
- `method`: Docking method name
- `protein`: Protein system name
- `idx`: Model index (0-4)
- `rank`: Rank of the prediction (1-5)
- `rmsd`: RMSD value calculated by ICM
- `score`: Method-specific score (if available)
- `file_path`: Path to the prediction file

## Integration with PoseBusters

This script complements `generate_pb_results.py` by providing RMSD calculations using ICM instead of PoseBusters. You can merge the results:

```python
import pandas as pd

# Load PoseBusters results
pb_results = pd.read_csv("posebusters_results.csv")

# Load ICM RMSD results
rmsd_results = pd.read_csv("combined_runsNposes_0_rmsd_results.csv")

# Merge on common columns
merged = pd.merge(
    pb_results, 
    rmsd_results[['method', 'protein', 'rank', 'rmsd', 'score']], 
    on=['method', 'protein', 'rank'], 
    suffixes=('_pb', '_icm')
)
```

## File Structure

```
notebooks/04_utils/
├── generalized_icm_rmsd.py        # Main script
├── Approach.py                    # Method definitions
├── generate_pb_results.py         # PoseBusters analysis
└── README_icm_rmsd.md            # This file

notebooks/01_method_analysis/boltz/development/
└── calculate_rmsd.icm            # ICM script for RMSD calculation
```

## Error Handling

The script handles various error conditions:

- **Missing files**: Logs missing reference or prediction files
- **ICM errors**: Captures ICM execution errors and continues
- **Timeouts**: Handles long-running ICM processes (300s timeout)
- **Conversion errors**: Handles SDF to PDB conversion failures

Error cases are recorded in the output with special values like "ICM_ERROR", "TIMEOUT", or "MISSING_FILE".

## Method-Specific Notes

- **Boltz & Chai-1**: These methods produce full protein-ligand complexes in PDB format. The script uses these complex files directly for protein RMSD calculation.
- **Vina, GNINA, DiffDock, etc.**: These methods produce ligand-only SDF files. **Protein RMSD cannot be calculated** for these methods as they don't predict protein conformations. The script will mark these as "LIGAND_ONLY_METHOD".
- **ICM/ICM_RTCNN**: Uses different directory naming patterns and produces ligand SDF files.
- **SurfDock**: Has custom output directory structure and produces ligand SDF files.

**Important**: Only methods that produce full protein-ligand complexes (Boltz, Chai-1) can have meaningful protein RMSD calculations. Other methods only dock ligands to fixed protein structures.

## Customization

To add support for new methods:

1. Define the method class in `Approach.py`
2. Add the method to `BASE_DIRS` dictionary
3. Add the approach class to the `approaches` list in `main()`
4. Update `METHOD_CLASSES` in `run_single_method_rmsd()`