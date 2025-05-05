"""
AlphaFold3 Binding Evaluation

A tool for evaluating AlphaFold3's performance in predicting protein-protein
interactions, specifically for antibody-antigen binding.
"""

__version__ = "0.1.0"

# Explicitly import and expose the run_pipeline function
from alphafold3_eval.pipeline import run_pipeline