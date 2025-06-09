"""
Functions for aligning protein structures.
"""
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Structure import Structure
from Bio.PDB.Chain import Chain
from Bio.SVDSuperimposer import SVDSuperimposer

from alphafold3_eval.structure_uitls import (
    split_chains_by_type,
    load_structure,
    alternative_structure_loading,
    compute_chain_center,
    parse_model_filename,  # ADD THIS
    split_chains_by_type,
    compute_chain_center
)

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


def compute_antigen_aligned_mse_(model_paths: List[str]) -> float:
    """
    Compute the Mean Squared Error of binding entity centers after aligning antigens.

    This is a measure of binding pose consistency across different seeds.

    Args:
        model_paths: List of paths to model files

    Returns:
        MSE value or np.nan if computation fails
    """
    coords_antigen = []
    coords_binding_region = []

    successful_models = 0

    for cif_path in model_paths:
        try:
            # Try to load the structure with our robust methods
            structure = load_structure(cif_path)

            # If standard loading fails, try alternative methods
            if structure is None:
                print(f"Attempting alternative loading for {cif_path}")
                structure = alternative_structure_loading(cif_path)

            # If still unable to load, skip this model
            if structure is None:
                print(f"Skipping {cif_path} due to loading failure")
                continue

            # Split chains
            binding_chains, antigen_chains = split_chains_by_type(structure)

            if not binding_chains or not antigen_chains:
                print(f"Warning: Could not identify binding chains and antigen chains in {cif_path}")
                continue

            # Compute centers
            # binding_center = compute_chain_center(binding_chains)
            binding_center = compute_chain_center(binding_chains)
            antigen_center = compute_chain_center(antigen_chains)

            # Check for valid coordinates
            if np.isnan(binding_center).any() or np.isnan(antigen_center).any():
                print(f"Warning: Invalid coordinates in {cif_path}")
                continue

            # Add to lists
            coords_binding_region.append(binding_center)
            coords_antigen.append(antigen_center)
            successful_models += 1

        except Exception as e:
            print(f"Error processing {cif_path}: {e}")

    # Check if we have enough successful models
    if successful_models < 2:
        print(
            f"Warning: Only {successful_models} models were processed successfully. Need at least 2 for MSE calculation.")
        return np.nan

    # Align binding positions based on antigen reference
    antigen_ref = coords_antigen[0]
    aligned_binding_positions = [
        binding + (antigen_ref - antigen)
        for binding, antigen in zip(coords_binding_region, coords_antigen)
    ]

    # Compute MSE of aligned binding positions
    aligned_binding_positions = np.array(aligned_binding_positions)
    # Current
    mse = np.mean(np.var(aligned_binding_positions, axis=0))

    # Size-normalized
    antigen_radius = compute_antigen_radius_of_gyration(coords_antigen)
    normalized_mse = mse / (antigen_radius ** 2)

    print(f"  Computed MSE from {successful_models} models: {mse:.4f}")

    return mse


def compute_antigen_aligned_mse(model_paths: List[str]) -> float:
    """
    Compute MSE of nanobody centers after Kabsch alignment of antigens.

    This measures binding pose consistency by:
    1. Kabsch-aligning all antigens to the first structure
    2. Computing variance of nanobody centers after alignment
    3. Normalizing by antigen size for fair comparison

    Args:
        model_paths: List of paths to model files

    Returns:
        Size-normalized MSE value or np.nan if computation fails
    """
    structures_data = []

    # Load all structures and extract coordinates
    for model_path in model_paths:
        try:
            structure = load_structure(model_path)
            if not structure:
                continue

            binding_chains, antigen_chains = split_chains_by_type(structure)
            if not binding_chains or not antigen_chains:
                continue

            # Get antigen coordinates for alignment (CA atoms only for stability)
            antigen_coords = []
            for chain in antigen_chains:
                for residue in chain.get_residues():
                    for atom in residue.get_atoms():
                        if atom.name == 'CA':  # Use CA atoms for more stable alignment
                            antigen_coords.append(atom.coord)

            if len(antigen_coords) < 3:  # Need minimum atoms for alignment
                continue

            antigen_coords = np.array(antigen_coords)

            # Get nanobody center
            nanobody_center = compute_chain_center(binding_chains)

            if not np.isnan(nanobody_center).any():
                structures_data.append({
                    'antigen_coords': antigen_coords,
                    'nanobody_center': nanobody_center,
                    'structure': structure
                })

        except Exception as e:
            print(f"Error processing {model_path}: {e}")
            continue

    if len(structures_data) < 2:
        print(f"Warning: Only {len(structures_data)} valid structures for MSE calculation")
        return np.nan

    # Use first structure as reference
    ref_antigen = structures_data[0]['antigen_coords']
    aligned_nanobody_centers = [structures_data[0]['nanobody_center']]

    # Align each structure to reference using Kabsch algorithm
    successful_alignments = 1  # Count reference

    for i in range(1, len(structures_data)):
        mobile_antigen = structures_data[i]['antigen_coords']
        mobile_nanobody = structures_data[i]['nanobody_center']

        try:
            # Handle different number of atoms by using common subset
            min_len = min(len(ref_antigen), len(mobile_antigen))
            if min_len < 3:
                continue

            # Use first min_len atoms (typically covers most of structure)
            ref_subset = ref_antigen[:min_len]
            mobile_subset = mobile_antigen[:min_len]

            # Compute Kabsch alignment
            rotation, translation = kabsch_alignment(mobile_subset, ref_subset)

            # Apply transformation to nanobody center
            aligned_nanobody = np.dot(mobile_nanobody, rotation.T) + translation
            aligned_nanobody_centers.append(aligned_nanobody)
            successful_alignments += 1

        except Exception as e:
            print(f"Alignment failed for structure {i}: {e}")
            continue

    if successful_alignments < 2:
        return np.nan

    # Compute MSE of aligned nanobody centers
    aligned_centers = np.array(aligned_nanobody_centers)
    mse = np.mean(np.var(aligned_centers, axis=0))

    # Normalize by antigen size (radius of gyration)
    antigen_size = compute_antigen_radius_of_gyration(ref_antigen)
    normalized_mse = mse / (antigen_size ** 2) if antigen_size > 0 else mse

    print(f"  Computed MSE from {successful_alignments} models: {mse:.4f} (normalized: {normalized_mse:.6f})")

    return mse  # Return raw MSE for now, can switch to normalized later


def compute_antigen_radius_of_gyration(coords: np.ndarray) -> float:
    """
    Compute radius of gyration of antigen for size normalization.

    Args:
        coords: Antigen coordinates (N, 3)

    Returns:
        Radius of gyration in Angstroms
    """
    if len(coords) == 0:
        return 0.0

    # Center coordinates
    center = np.mean(coords, axis=0)
    centered_coords = coords - center

    # Compute radius of gyration
    rg_squared = np.mean(np.sum(centered_coords ** 2, axis=1))
    rg = np.sqrt(rg_squared)

    return rg


def compute_weighted_mse_all_models(model_paths: List[str]) -> Dict[str, float]:
    """
    Compute both top-model MSE and weighted MSE across all models.

    Args:
        model_paths: List of all model file paths for one combination

    Returns:
        Dictionary with 'top_model_mse' and 'weighted_mse' keys
    """
    # Group models by seed
    seed_groups = {}
    for model_path in model_paths:
        # Extract seed info from filename
        meta = parse_model_filename(model_path)
        if meta:
            seed = meta['seed']
            if seed not in seed_groups:
                seed_groups[seed] = []
            seed_groups[seed].append(model_path)

    # Sort models within each seed by rank (model_0 is best)
    for seed in seed_groups:
        seed_groups[seed].sort(key=lambda x: int(x.split('_model_')[1].split('.')[0]))

    # Calculate top-model MSE (rank 1 only)
    top_models = [models[0] for models in seed_groups.values() if models]
    top_model_mse = compute_antigen_aligned_mse(top_models)

    # Calculate weighted MSE (all models)
    model_weights = [1.0, 0.8, 0.6, 0.4, 0.2]  # Weights for ranks 1-5

    all_centers = []
    all_weights = []

    # Process each seed
    for seed, models in seed_groups.items():
        try:
            # Load first model as reference for alignment
            ref_structure = load_structure(models[0])
            if not ref_structure:
                continue

            ref_binding_chains, ref_antigen_chains = split_chains_by_type(ref_structure)
            ref_antigen_coords = np.array([atom.coord for chain in ref_antigen_chains
                                           for atom in chain.get_atoms() if atom.name == 'CA'])

            # Process all models in this seed
            for rank, model_path in enumerate(models[:5]):  # Up to 5 models
                structure = load_structure(model_path)
                if not structure:
                    continue

                binding_chains, antigen_chains = split_chains_by_type(structure)

                # Align antigen to reference (first model of first seed)
                if rank > 0:  # Align non-reference models
                    antigen_coords = np.array([atom.coord for chain in antigen_chains
                                               for atom in chain.get_atoms() if atom.name == 'CA'])

                    min_len = min(len(ref_antigen_coords), len(antigen_coords))
                    if min_len >= 3:
                        rotation, translation = kabsch_alignment(
                            antigen_coords[:min_len],
                            ref_antigen_coords[:min_len]
                        )

                        # Apply transformation to binding center
                        binding_center = compute_chain_center(binding_chains)
                        aligned_center = np.dot(binding_center, rotation.T) + translation
                    else:
                        continue
                else:
                    # Reference model doesn't need alignment
                    # aligned_center = compute_chain_center(binding_chains)
                    aligned_center = compute_chain_center(binding_chains)

                if not np.isnan(aligned_center).any():
                    all_centers.append(aligned_center)
                    all_weights.append(model_weights[rank])

        except Exception as e:
            print(f"Error processing seed {seed}: {e}")
            continue

    # Calculate weighted MSE
    if len(all_centers) >= 2:
        centers_array = np.array(all_centers)
        weights_array = np.array(all_weights)

        # Weighted mean
        weighted_mean = np.average(centers_array, weights=weights_array, axis=0)

        # Weighted variance
        weighted_var = np.average((centers_array - weighted_mean) ** 2, weights=weights_array, axis=0)
        weighted_mse = np.mean(weighted_var)
    else:
        weighted_mse = np.nan

    return {
        'top_model_mse': top_model_mse,
        'weighted_mse': weighted_mse
    }


def compute_antigen_radius_of_gyration(coords: Union[np.ndarray, List[np.ndarray]]) -> float:
    """
    Compute radius of gyration of antigen for size normalization.

    Args:
        coords: Antigen coordinates (N, 3) or list of coordinate arrays

    Returns:
        Radius of gyration in Angstroms
    """
    # Handle both single array and list of arrays
    if isinstance(coords, list):
        if len(coords) == 0:
            return 0.0
        # Use first structure as reference
        coords = coords[0]

    if len(coords) == 0:
        return 0.0

    # Center coordinates
    center = np.mean(coords, axis=0)
    centered_coords = coords - center

    # Compute radius of gyration
    rg_squared = np.mean(np.sum(centered_coords ** 2, axis=1))
    rg = np.sqrt(rg_squared)

    return rg

def kabsch_alignment(mobile_coords: np.ndarray, ref_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute optimal rotation and translation using Kabsch algorithm.

    Args:
        mobile_coords: Coordinates to be aligned (N, 3)
        ref_coords: Reference coordinates (N, 3)

    Returns:
        Tuple of (rotation_matrix, translation_vector)
    """
    assert len(mobile_coords) == len(ref_coords), "Coordinate arrays must have same length"

    # Center coordinates
    mobile_center = np.mean(mobile_coords, axis=0)
    ref_center = np.mean(ref_coords, axis=0)

    mobile_centered = mobile_coords - mobile_center
    ref_centered = ref_coords - ref_center

    # Compute covariance matrix
    H = np.dot(mobile_centered.T, ref_centered)

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation
    rotation = np.dot(Vt.T, U.T)

    # Ensure proper rotation (det = 1, not reflection)
    if np.linalg.det(rotation) < 0:
        Vt[-1, :] *= -1
        rotation = np.dot(Vt.T, U.T)

    # Compute translation
    translation = ref_center - np.dot(mobile_center, rotation.T)

    return rotation, translation