#!/usr/bin/env python
"""
Main entry point for AlphaFold3 binding evaluation.
"""
import sys
import os

# Add the project directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alphafold3_eval.cli import main

if __name__ == "__main__":
    main()