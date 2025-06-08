"""
AlphaFold3 Evaluation Package
"""

# Version
__version__ = "0.1.0"

# Import main components
from alphafold3_eval.config import Config
from alphafold3_eval.alignment import compute_antigen_aligned_mse
from alphafold3_eval.binding_analysis import calculate_binding_energies

# Don't import pipeline functions here to avoid circular imports
__all__ = [
    "Config",
    "compute_antigen_aligned_mse",
    "calculate_binding_energies"
]