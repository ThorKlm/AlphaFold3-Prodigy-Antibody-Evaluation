"""
Extract and process confidence metrics from AlphaFold3 predictions.
"""
import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Structure import Structure
import pandas as pd


def extract_af3_confidence_from_cif(cif_path: str) -> Dict[str, float]:
    """
    Extract AlphaFold3 confidence metrics from CIF file.

    Args:
        cif_path: Path to AlphaFold3 CIF file

    Returns:
        Dictionary with confidence metrics
    """
    metrics = {
        'ipTM': None,
        'pTM': None,
        'mean_pLDDT': None,
        'interface_pLDDT': None,
        'ranking_score': None
    }

    try:
        # Parse CIF file to extract metadata
        with open(cif_path, 'r') as f:
            lines = f.readlines()

        # Look for confidence metrics in CIF headers
        for line in lines:
            # AF3 stores metrics in specific fields
            if 'iptm' in line.lower() and '=' in line:
                value = line.split('=')[-1].strip()
                try:
                    metrics['ipTM'] = float(value)
                except:
                    pass

            elif 'ptm' in line.lower() and '=' in line and 'iptm' not in line.lower():
                value = line.split('=')[-1].strip()
                try:
                    metrics['pTM'] = float(value)
                except:
                    pass

            elif 'plddt' in line.lower() and 'mean' in line.lower():
                value = line.split('=')[-1].strip()
                try:
                    metrics['mean_pLDDT'] = float(value)
                except:
                    pass

            elif 'ranking_score' in line.lower():
                value = line.split('=')[-1].strip()
                try:
                    metrics['ranking_score'] = float(value)
                except:
                    pass

        # If metrics not found in header, try to extract from B-factors
        if metrics['mean_pLDDT'] is None:
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure('struct', cif_path)

            # Extract pLDDT from B-factors
            plddt_values = []
            for atom in structure.get_atoms():
                plddt_values.append(atom.get_bfactor())

            if plddt_values:
                metrics['mean_pLDDT'] = np.mean(plddt_values)

    except Exception as e:
        print(f"Error extracting confidence metrics from {cif_path}: {e}")

    return metrics


def extract_interface_plddt(structure: Structure, cutoff: float = 5.0) -> float:
    """
    Extract mean pLDDT for interface residues.

    Args:
        structure: Bio.PDB Structure
        cutoff: Distance cutoff for interface

    Returns:
        Mean pLDDT of interface residues
    """
    from alphafold3_eval.binding_analysis import extract_interface_residues

    binding_interface, antigen_interface = extract_interface_residues(structure, cutoff)

    # Get pLDDT values for interface residues
    interface_plddt = []
    for res in binding_interface + antigen_interface:
        for atom in res.get_atoms():
            interface_plddt.append(atom.get_bfactor())

    if interface_plddt:
        return np.mean(interface_plddt)
    return 0.0


def apply_confidence_calibration(metrics: Dict[str, float],
                                 calibration_factor: float = 0.85) -> Dict[str, float]:
    """
    Apply calibration to AlphaFold3 confidence metrics.

    Args:
        metrics: Raw confidence metrics
        calibration_factor: Factor to correct AF3 confidence inflation

    Returns:
        Calibrated metrics
    """
    calibrated = metrics.copy()

    # Apply calibration to specific metrics
    if calibrated['ipTM'] is not None:
        calibrated['calibrated_ipTM'] = calibrated['ipTM'] * calibration_factor

    if calibrated['pTM'] is not None:
        calibrated['calibrated_pTM'] = calibrated['pTM'] * calibration_factor

    # pLDDT typically doesn't need calibration
    calibrated['calibrated_pLDDT'] = calibrated.get('mean_pLDDT', 0.0)

    return calibrated