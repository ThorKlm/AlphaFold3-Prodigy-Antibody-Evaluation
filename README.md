# AlphaFold Binding Evaluation

A tool for evaluating AlphaFold's performance in predicting protein-protein interactions, specifically for antibody-antigen binding. This project analyzes binding pose consistency and binding energy across multiple random seeds, creating a "confusion matrix" of binding entity-antigen combinations.

## Features

- Support for both **AlphaFold3 ZIP files** and **AlphaFold2-Multimer directory structure**
- Analyze binding pose consistency using antigen-aligned MSE
- Calculate binding energy using two methods:
  - PRODIGY binding energy prediction
  - Simple contact-based approximation
- Generate comprehensive visualizations including:
  - Pose consistency plots
  - Binding energy distributions
  - Confusion matrices
  - Combined matrix visualizations
- Scale binding energies for two-chain antibodies for better comparison with single-chain antibodies

## Installation

### Prerequisites

- Python 3.8+ (Python 3.9 recommended)
- Git
- PyCharm or another Python IDE

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/alphafold-binding-eval.git
   cd alphafold-binding-eval
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install numpy scipy matplotlib seaborn biopython pandas
   ```

4. **Install PRODIGY (optional but recommended):**
   ```bash
   pip install prodigy-protein
   ```

5. **Install the package in development mode:**
   ```bash
   pip install -e .
   ```

### Verifying Installation

You can verify that everything is installed correctly by running:

```bash
python -m alphafold3_eval.cli --check
```

This will check that PRODIGY is installed and working properly.

## Usage

### Command-line Interface

The package provides a command-line tool for running the analysis with different input formats:

#### AlphaFold3 ZIP Files (Default)

```bash
python -m alphafold3_eval.cli --input /path/to/alphafold3_zips --output /path/to/results
```

#### AlphaFold2-Multimer Directory Structure

```bash
python -m alphafold3_eval.cli --input /path/to/af2_multimer_dir --output /path/to/results --format alphafold2_multimer
```

### Additional Options

```bash
# With more options
python -m alphafold3_eval.cli \
  --input /path/to/input \
  --output /path/to/results \
  --format alphafold3 \            # or alphafold2_multimer
  --temp 25.0 \                    # Temperature for binding energy calculation
  --cutoff 5.0 \                   # Distance cutoff for interface detection
  --min-seeds 5 \                  # Minimum seeds required for analysis
  --no-scaling \                   # Disable scaling of two-chain antibodies
  --skip-prodigy \                 # Use only contact-based binding energy model
  --clean                          # Clean up intermediate files
```

### Expected Input Formats

#### AlphaFold3 ZIP Files

The tool expects ZIP files containing AlphaFold3 prediction structures in the following format:

```
prefix_binding-entity_antigen_*_seed_*.zip
```

Each ZIP file should contain CIF files with the protein complex structure.

#### AlphaFold2-Multimer Directory Structure

The tool supports directory structures with AlphaFold2-Multimer predictions:

```
/some/path/nbALB_8y9t_alb_seed0/top_model.pdb
/some/path/nbALB_8y9t_GFP_seed1/top_model.pdb
/some/path/nbGFP_6xzf_MCherry_seed2/top_model.pdb
...
```

The tool will automatically identify binding entities and antigens from the directory names.

### Output Files

The tool generates several output files:

- **CSV files:**
  - `center_position_mse_results.csv`: MSE results for each combination
  - `binding_energy_results.csv`: Unscaled binding energy for each model (both methods)
  - `binding_energy_results_scaled.csv`: Scaled binding energy for each model
  - `combined_analysis_results.csv`: Combined results (unscaled)
  - `combined_analysis_results_scaled.csv`: Combined results (scaled)

- **Visualization plots:**
  - MSE distribution plots
  - Binding energy boxplots (for both approximated and PRODIGY methods)
  - MSE vs. binding energy scatter plots
  - Binding energy confusion matrices
  - Comprehensive matrix visualizations

## How It Works

### Binding Entity and Antigen Detection

For AlphaFold2-Multimer predictions, the tool extracts:
- Binding entity (e.g., "nbALB_8y9t" or "nbGFP_6xzf")
- Antigen (e.g., "alb", "GFP", "SARS")

The tool identifies nanobodies (single-chain) and antibodies (two-chain) based on naming conventions:
- Names starting with "nb" are treated as single-chain antibodies
- Names starting with "ab" are treated as two-chain antibodies

### Binding Energy Calculation

The tool calculates binding energy using two methods:

1. **Simple Contact-Based Model**: Counts the number of contacts between atoms at the binding interface and applies a simple formula: `ΔG = -0.1 * contacts`

2. **PRODIGY Method**: Uses the PRODIGY tool to predict binding affinity based on interface properties, including the types of interactions (charged, polar, apolar) and the physicochemical properties of residues.

### Two-Chain Antibody Scaling

By default, the tool scales the binding energies of two-chain antibodies by a factor of 0.5 to make them more comparable with single-chain antibodies. This scaling accounts for the larger binding interface typical of two-chain antibodies. You can disable this scaling with the `--no-scaling` flag.

## Troubleshooting

### PRODIGY Issues

If you encounter issues with PRODIGY:

1. Verify PRODIGY installation:
   ```bash
   python -c "import prodigy; print('PRODIGY imported successfully!')"
   ```

2. Test PRODIGY on a sample PDB:
   ```bash
   python -m alphafold3_eval.cli --check
   ```

3. Try running with the simple contact-based model:
   ```bash
   python -m alphafold3_eval.cli --input /path/to/input --output /path/to/results --skip-prodigy
   ```

### File Loading Issues

If you encounter problems loading structure files:

1. Test the structure loading:
   ```bash
   # Create a test script (see repository for test_structure_loading.py)
   python test_structure_loading.py /path/to/your/files
   ```

2. Check file formats and ensure the files are valid PDB or CIF files

### Windows-Specific Issues

On Windows, the following might help:

1. Set UTF-8 encoding for the terminal:
   ```powershell
   $env:PYTHONIOENCODING = "utf-8"
   ```

2. Use full paths with quotes if paths contain spaces

## License

[MIT License](LICENSE)

## Acknowledgements

- [PRODIGY](https://github.com/haddocking/prodigy) for binding energy prediction
- [BioPython](https://biopython.org/) for structural biology tools