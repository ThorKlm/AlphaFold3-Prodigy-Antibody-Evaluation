"""
Utilities for handling protein structure files.
"""
import os
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from Bio.PDB import MMCIFParser
from Bio.PDB.Structure import Structure
from Bio.PDB.Chain import Chain


def extract_and_load_structure(zip_path: Union[str, Path], model_file: str,
                              extract_dir: Union[str, Path]) -> Optional[Structure]:
    """
    Extract a CIF file from a ZIP archive and load it as a Bio.PDB Structure.

    Args:
        zip_path: Path to the ZIP archive
        model_file: Name of the CIF file within the archive
        extract_dir: Directory to extract the CIF file to

    Returns:
        Loaded Bio.PDB Structure or None if extraction/loading fails
    """
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(exist_ok=True, parents=True)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extract(model_file, path=extract_dir)

        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(model_file, extract_dir / model_file)

        # Clean up the extracted file
        os.remove(extract_dir / model_file)

        return structure

    except Exception as e:
        print(f"Error extracting/loading {model_file} from {zip_path}: {e}")
        return None


def extract_best_model_from_zip(zip_path: Union[str, Path], output_dir: Union[str, Path],
                              output_filename: Optional[str] = None) -> Optional[str]:
    """
    Extract the best model (first in sorted order) from a ZIP archive.

    Args:
        zip_path: Path to the ZIP archive
        output_dir: Directory to save the extracted CIF file
        output_filename: Optional filename for the extracted model (default: based on ZIP name)

    Returns:
        Path to the extracted model file or None if extraction fails
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            cif_files = [f for f in zf.namelist() if f.endswith('.cif')]
            if not cif_files:
                print(f"No CIF files found in {zip_path}")
                return None

            # Select the best model (first in sorted order)
            best_model = sorted(cif_files)[0]

            # Generate output filename if not provided
            if output_filename is None:
                seed_name = Path(zip_path).stem
                output_filename = f"{seed_name}_model_0.cif"

            output_path = output_dir / output_filename

            # Extract and save the model
            with open(output_path, 'wb') as out_f:
                out_f.write(zf.read(best_model))

            return str(output_path)

    except Exception as e:
        print(f"Error extracting best model from {zip_path}: {e}")
        return None


def parse_model_filename(filename: Union[str, Path]) -> Optional[Dict[str, str]]:
    """
    Parse metadata from a model filename.

    Expected format: prefix_binding_entity_antigen_*_seed_*.cif

    Args:
        filename: Path to the model file

    Returns:
        Dictionary containing parsed metadata or None if parsing fails
    """
    filename = Path(filename).name
    parts = filename.split('_')

    if len(parts) < 6:
        return None

    try:
        # Extract components from filename
        binding_entity = parts[1] + '_' + parts[2]
        antigen = parts[3]

        # Find seed (look for the part after "seed")
        seed_idx = parts.index("seed") if "seed" in parts else -1
        if seed_idx >= 0 and seed_idx + 1 < len(parts):
            seed = parts[seed_idx + 1]
        else:
            # Alternative: assume seed is the 5th part
            seed = parts[5].split('.')[0]  # Remove file extension if present

        return {
            "binding_entity": binding_entity,
            "antigen": antigen,
            "seed": seed,
            "combination": f"{binding_entity}_{antigen}"
        }

    except Exception:
        return None


def split_chains_by_type(structure: Structure) -> Tuple[List[Chain], List[Chain]]:
    """
    Split chains in a structure into binding entity chains and antigen chains.

    By convention, the last chain is the antigen, and all others are part of the binding entity.

    Args:
        structure: Bio.PDB Structure to analyze

    Returns:
        Tuple of (binding_chains, antigen_chains)
    """
    chains = list(structure[0].get_chains())

    if len(chains) < 2:
        return [], []

    # By convention, the last chain is the antigen
    antigen_chains = [chains[-1]]

    # All other chains belong to the binding entity (antibody or nanobody)
    binding_chains = chains[:-1]

    return binding_chains, antigen_chains


def compute_chain_center(chain_list: List[Chain]) -> np.ndarray:
    """
    Compute the center of mass of a list of chains.

    Args:
        chain_list: List of Bio.PDB Chain objects

    Returns:
        Numpy array with the [x, y, z] coordinates of the center
    """
    coords = []
    for chain in chain_list:
        for atom in chain.get_atoms():
            coords.append(atom.get_coord())

    if not coords:
        return np.array([np.nan, np.nan, np.nan])

    return np.mean(coords, axis=0)