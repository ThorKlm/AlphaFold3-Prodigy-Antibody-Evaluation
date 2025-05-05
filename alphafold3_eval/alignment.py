"""
Functions for aligning protein structures.
"""
from typing import List, Optional, Tuple, Union

import numpy as np
from Bio.PDB import MMCIFParser
from Bio.PDB.Structure import Structure
from Bio.PDB.Chain import Chain
from Bio.SVDSuperimposer import SVDSuperimposer

from alphafold3_eval.structure_uitls import split_chains_by_type


def align_structure_to_reference(mobile_structure: Structure,
                               reference_structure: Structure) -> Optional[Structure]:
    """
    Align the antigen of mobile_structure to the antigen of reference_structure.

    Args:
        mobile_structure: Structure to be aligned
        reference_structure: Reference structure to align to

    Returns:
        Aligned structure or None if alignment fails
    """
    # Extract chains
    mobile_binding, mobile_antigen = split_chains_by_type(mobile_structure)
    ref_binding, ref_antigen = split_chains_by_type(reference_structure)

    # Check if we have valid chains
    if not mobile_binding or not mobile_antigen or not ref_binding or not ref_antigen:
        return None

    # Get coordinates of antigen atoms for alignment
    mobile_coords = np.array([atom.coord for atom in mobile_antigen[0].get_atoms()])
    ref_coords = np.array([atom.coord for atom in ref_antigen[0].get_atoms()])

    if len(mobile_coords) != len(ref_coords):
        # If the antigens have different numbers of atoms, try to align on the smaller set
        min_len = min(len(mobile_coords), len(ref_coords))
        mobile_coords = mobile_coords[:min_len]
        ref_coords = ref_coords[:min_len]

    # Perform superposition
    try:
        sup = SVDSuperimposer()
        sup.set(ref_coords, mobile_coords)  # Set reference and mobile coordinates
        sup.run()
        rot, tran = sup.get_rotran()

        # Apply transformation to all atoms in the mobile structure
        for atom in mobile_structure.get_atoms():
            atom.transform(rot, tran)

        return mobile_structure

    except Exception as e:
        print(f"Error during structure alignment: {e}")
        return None


def compute_antigen_aligned_mse(model_paths: List[str]) -> float:
    """
    Compute the Mean Squared Error of binding entity centers after aligning antigens.

    This is a measure of binding pose consistency across different seeds.

    Args:
        model_paths: List of paths to model files

    Returns:
        MSE value or np.nan if computation fails
    """
    parser = MMCIFParser(QUIET=True)

    coords_antigen = []
    coords_binding_region = []

    for cif_path in model_paths:
        try:
            structure = parser.get_structure(cif_path, cif_path)
            binding_chains, antigen_chains = split_chains_by_type(structure)

            if not binding_chains or not antigen_chains:
                continue

            binding_center = compute_combined_chain_center(binding_chains)
            antigen_center = compute_combined_chain_center(antigen_chains)

            coords_binding_region.append(binding_center)
            coords_antigen.append(antigen_center)

        except Exception as e:
            print(f"Error processing {cif_path}: {e}")

    if len(coords_antigen) < 2:
        return np.nan

    # Align binding positions based on antigen reference
    antigen_ref = coords_antigen[0]
    aligned_binding_positions = [
        binding + (antigen_ref - antigen)
        for binding, antigen in zip(coords_binding_region, coords_antigen)
    ]

    # Compute MSE of aligned binding positions
    aligned_binding_positions = np.array(aligned_binding_positions)
    mse = np.mean(np.var(aligned_binding_positions, axis=0))

    return mse


def compute_combined_chain_center(chain_list: List[Chain]) -> np.ndarray:
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