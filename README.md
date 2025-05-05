# AlphaFold3 Binding Evaluation

A tool for evaluating AlphaFold3's performance in predicting antibody-antigen binding structures. This project analyzes binding pose consistency and binding energy across multiple random seeds, creating a "confusion matrix" of binding entity-antigen combinations.

## Features

- Extract best models from AlphaFold3 prediction ZIP files
- Analyze binding pose consistency using antigen-aligned MSE
- Calculate binding energy using PRODIGY or simple contact-based model
- Generate visualizations of results, including:
  - Pose consistency plots
  - Binding energy distributions
  - Combined analysis plots

## Installation

### Prerequisites

- Python 3.8+ (Python 3.9 recommended)
- Git
- PyCharm or another Python IDE

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ThorKlm/AlphaFold3-Prodigy-Antibody-Evaluation.git
   cd alphafold3-binding-eval
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

4. **Install PRODIGY:**
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

The package provides a command-line tool for running the analysis:

```bash
# Basic usage
python -m alphafold3_eval.cli --input /path/to/alphafold_zips --output /path/to/results

# With more options
python -m alphafold3_eval.cli \
  --input /path/to/alphafold_zips \
  --output /path/to/results \
  --temp 25.0 \          # Temperature for binding energy calculation
  --cutoff 5.0 \         # Distance cutoff for interface detection
  --min-seeds 5 \        # Minimum seeds required for analysis
  --clean                # Clean up intermediate files
```

If you encounter issues with PRODIGY, you can use the simple contact-based model instead:

```bash
python -m alphafold3_eval.cli --input /path/to/alphafold_zips --output /path/to/results --skip-prodigy
```

### Expected Input Format

The tool expects ZIP files containing AlphaFold3 prediction structures in the following format:

```
prefix_binding-entity_antigen_*_seed_*.zip
```

Each ZIP file should contain one or more CIF files with the protein complex structure. The last chain in each structure is assumed to be the antigen, while all other chains belong to the binding entity (antibody or nanobody).

### Output

The tool generates several output files:

- **CSV files:**
  - `center_position_mse_results.csv`: MSE results for each combination
  - `binding_energy_results.csv`: Binding energy for each model
  - `combined_analysis_results.csv`: Combined results

- **Visualization plots:**
  - `center_position_mse_plot.png`: Binding pose consistency
  - `binding_energy_plot.png`: Binding energy distribution
  - `mse_vs_energy_plot.png`: Correlation between MSE and binding energy
  - `binding_energy_confusion_matrix.png`: Heatmap of binding energies
  - `combined_analysis.png`: Combined visualization

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
   python -m alphafold3_eval.cli --input /path/to/alphafold_zips --output /path/to/results --skip-prodigy
   ```

### Windows-Specific Issues

On Windows, the following might help:

1. Set UTF-8 encoding for the terminal:
   ```powershell
   $env:PYTHONIOENCODING = "utf-8"
   ```

2. Use the full path to the prodigy executable:
   ```powershell
   # Find where prodigy is installed
   pip show prodigy-protein
   ```

## License

[MIT License](LICENSE)

## Acknowledgements

- [PRODIGY](https://github.com/haddocking/prodigy) for binding energy prediction
- [BioPython](https://biopython.org/) for structural biology tools