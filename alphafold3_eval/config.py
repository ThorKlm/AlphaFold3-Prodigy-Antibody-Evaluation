"""
Configuration settings for the AlphaFold3 evaluation project.
"""
import os
from pathlib import Path
from typing import Dict, Optional


class Config:
    """Configuration for the AlphaFold3 evaluation project."""

    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize configuration with optional base directory.

        Args:
            base_dir: Base directory for the project. If None, the current working directory is used.
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()

        # Directory for temporary files
        self.temp_dir = self.base_dir / "temp_files"
        self.temp_dir.mkdir(exist_ok=True, parents=True)

        # Directory for extracted model files
        self.model_dir = self.base_dir / "model_directory"
        self.model_dir.mkdir(exist_ok=True, parents=True)

        # Directory for results
        self.results_dir = self.base_dir / "results"
        self.results_dir.mkdir(exist_ok=True, parents=True)

        # Directory for temporary PDB files for PRODIGY
        self.prodigy_temp_dir = self.temp_dir / "prodigy_pdbs"
        self.prodigy_temp_dir.mkdir(exist_ok=True, parents=True)

        # Output filenames
        self.output_files = {
            "mse_results": self.results_dir / "center_position_mse_results.csv",
            "per_seed_mse": self.results_dir / "per_seed_center_position_mse.csv",
            "binding_energy": self.results_dir / "binding_energy_results.csv",
            "combined_results": self.results_dir / "combined_analysis_results.csv",
        }

        # Plots
        self.plot_files = {
            "mse_plot": self.results_dir / "center_position_mse_plot.png",
            "energy_plot": self.results_dir / "binding_energy_plot.png",
            "combined_plot": self.results_dir / "mse_vs_energy_plot.png",
            "matrix_plot": self.results_dir / "matrix_visualization.png",
        }

        # PRODIGY settings
        self.prodigy_config = {
            "distance_cutoff": 5.5,  # Angstroms
            "use_temperature": True,
            "temperature": 25.0,     # Celsius
            "temp_dir": self.prodigy_temp_dir
        }

        # Analysis settings
        self.analysis_config = {
            "contact_cutoff": 5.0,   # Angstroms
            "min_seeds": 5,          # Minimum number of seeds required for analysis
            "use_prodigy": True,     # Whether to use PRODIGY for binding energy calculation
            "scale_two_chain": True, # Whether to scale binding energies of two-chain antibodies
        }

    def get_input_files(self, input_dir: Optional[str] = None) -> Dict[str, Path]:
        """
        Get dictionary of input files from the specified directory.

        Args:
            input_dir: Directory containing input files. If None, use base_dir.

        Returns:
            Dictionary mapping filenames to Paths.
        """
        search_dir = Path(input_dir) if input_dir else self.base_dir
        return {
            "zip_files": list(search_dir.glob("*.zip")),
        }