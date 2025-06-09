"""
Functions for analyzing protein-protein binding interactions.
"""
"""
Functions for analyzing protein-protein binding interactions.
"""
import os
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from Bio.PDB import NeighborSearch
from Bio.PDB.Structure import Structure
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.PDBIO import PDBIO
# Remove this line: from scipy.spatial.distance import pdist, squareform

from alphafold3_eval.structure_uitls import (
    split_chains_by_type,
    load_structure,
    alternative_structure_loading
)

def extract_interface_residues(structure: Structure,
                             cutoff: float = 5.0) -> Tuple[List[Residue],
                                                          List[Residue]]:
    """
    Extract interface residues from a protein-protein complex.

    Args:
        structure: Bio.PDB Structure to analyze
        cutoff: Distance cutoff for interface residues in Angstroms

    Returns:
        Tuple of (binding_interface_residues, antigen_interface_residues)
    """
    binding_chains, antigen_chains = split_chains_by_type(structure)

    if not binding_chains or not antigen_chains:
        return [], []

    binding_atoms = [atom for chain in binding_chains for atom in chain.get_atoms()]
    antigen_atoms = [atom for chain in antigen_chains for atom in chain.get_atoms()]

    # Create neighbor search
    ns = NeighborSearch(binding_atoms + antigen_atoms)

    # Find interface atoms in binding entity
    binding_interface_atoms = [
        atom for atom in binding_atoms
        if any(neigh in antigen_atoms for neigh in ns.search(atom.coord, cutoff))
    ]

    # Find interface atoms in antigen
    antigen_interface_atoms = [
        atom for atom in antigen_atoms
        if any(neigh in binding_atoms for neigh in ns.search(atom.coord, cutoff))
    ]

    # Get unique residues
    binding_interface_residues = list(set(atom.get_parent() for atom in binding_interface_atoms))
    antigen_interface_residues = list(set(atom.get_parent() for atom in antigen_interface_atoms))

    return binding_interface_residues, antigen_interface_residues


def extract_interface_centroid(structure: Structure,
                             cutoff: float = 5.0) -> Optional[np.ndarray]:
    """
    Extract the centroid of the binding interface.

    Args:
        structure: Bio.PDB Structure to analyze
        cutoff: Distance cutoff for interface in Angstroms

    Returns:
        Numpy array with the [x, y, z] coordinates of the interface centroid
        or None if extraction fails
    """
    binding_chains, antigen_chains = split_chains_by_type(structure)

    if not binding_chains or not antigen_chains:
        return None

    binding_atoms = [atom for chain in binding_chains for atom in chain.get_atoms()]
    antigen_atoms = [atom for chain in antigen_chains for atom in chain.get_atoms()]

    # Create neighbor search
    ns = NeighborSearch(binding_atoms + antigen_atoms)

    # Find interface atoms in binding entity
    interface_atoms = [
        atom for atom in binding_atoms
        if any(neigh in antigen_atoms for neigh in ns.search(atom.coord, cutoff))
    ]

    if not interface_atoms:
        return None

    # Compute centroid
    coords = np.array([atom.coord for atom in interface_atoms])
    centroid = coords.mean(axis=0)

    return centroid


def estimate_binding_energy_simple(structure: Structure,
                                 cutoff: float = 5.0) -> Tuple[float, int]:
    """
    Estimate binding energy using a simple contact-based model.

    Args:
        structure: Bio.PDB Structure to analyze
        cutoff: Distance cutoff for contacts in Angstroms

    Returns:
        Tuple of (estimated_binding_energy, number_of_contacts)
    """
    binding_chains, antigen_chains = split_chains_by_type(structure)

    if not binding_chains or not antigen_chains:
        return 0.0, 0

    binding_atoms = [atom for chain in binding_chains for atom in chain.get_atoms()]
    antigen_atoms = [atom for chain in antigen_chains for atom in chain.get_atoms()]

    # Create neighbor search
    ns = NeighborSearch(binding_atoms + antigen_atoms)

    # Count contacts
    contacts = sum(
        1 for atom in binding_atoms
        for neighbor in ns.search(atom.coord, cutoff)
        if neighbor in antigen_atoms
    )

    # Estimate binding energy (very simple model)
    delta_g = -0.1 * contacts

    return delta_g, contacts


def detect_antibody_type(binding_entity: str) -> int:
    """
    Detect whether a binding entity is a single-chain or two-chain antibody.

    Args:
        binding_entity: Name of the binding entity

    Returns:
        Number of chains (1 or 2)
    """
    # Use naming convention to detect type
    # Assuming nanobodies typically start with "nb" and antibodies with "ab"
    if binding_entity.lower().startswith("nb"):
        return 1  # Single-chain
    elif binding_entity.lower().startswith("ab"):
        return 2  # Two-chain
    else:
        # For unknown naming patterns, default to two-chain
        return 2  # Default to two-chain


def convert_structure_to_pdb(structure: Structure, temp_dir: Optional[str] = None) -> str:
    """Convert a BioPython structure to PDB format and save to a temporary file."""

    # Fix path handling
    if temp_dir:
        # Normalize path to avoid doubling
        temp_dir = os.path.normpath(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        # Create temporary file in the specified directory
        temp_file = tempfile.NamedTemporaryFile(
            suffix='.pdb',
            delete=False,
            dir=temp_dir  # Use dir parameter correctly
        )
    else:
        temp_file = tempfile.NamedTemporaryFile(suffix='.pdb', delete=False)

    temp_file.close()
    print(f"    DEBUG: Created temp file: {temp_file.name}")

    # Write structure to PDB file
    try:
        io = PDBIO()
        io.set_structure(structure)
        io.save(temp_file.name)
        print(f"    DEBUG: Successfully wrote PDB to: {temp_file.name}")

        # Verify file exists
        if os.path.exists(temp_file.name):
            print(f"    DEBUG: File exists and size: {os.path.getsize(temp_file.name)} bytes")
        else:
            print(f"    DEBUG: ERROR - File does not exist after writing!")

    except Exception as e:
        print(f"    DEBUG: Error writing PDB file: {e}")
        return None

    return temp_file.name



def run_prodigy_on_pdb(pdb_path: str, temp: float = 25.0) -> Optional[float]:
    """
    Run PRODIGY on a PDB file.

    Args:
        pdb_path: Path to the PDB file
        temp: Temperature in Celsius for binding energy calculation

    Returns:
        Estimated binding energy in kcal/mol or None if calculation fails
    """
    try:
        # Run PRODIGY command
        cmd = ["prodigy", f"--temperature={temp}", pdb_path]

        # Set UTF-8 encoding for subprocess
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        # Run with full error capture
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            encoding='utf-8',
            errors='replace'
        )

        # Check for errors
        if result.returncode != 0:
            print(f"PRODIGY error (return code {result.returncode})")
            if result.stderr:
                print(f"Error details: {result.stderr}")
            return None

        # Parse output for binding energy
        for line in result.stdout.split('\n'):
            if "Predicted binding affinity" in line:
                parts = line.split(':')
                if len(parts) >= 2:
                    try:
                        # Extract the numeric value (first token after the colon)
                        binding_energy = float(parts[1].strip().split()[0])
                        return binding_energy
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing binding energy: {e}")
                        print(f"Line: {line}")

        print("No binding energy found in PRODIGY output")
        return None

    except Exception as e:
        print(f"Error running PRODIGY: {e}")
        return None


def run_prodigy_binding_energy(structure: Structure,
                               temp: float = 25.0,
                               temp_dir: Optional[str] = None) -> Optional[float]:
    """Calculate binding energy using PRODIGY."""

    print(f"    DEBUG: PRODIGY called with temp_dir={temp_dir}")

    try:
        # Convert structure to PDB format
        pdb_path = convert_structure_to_pdb(structure, temp_dir)
        print(f"    DEBUG: PDB conversion returned: {pdb_path}")

        if not pdb_path:
            print(f"    DEBUG: PDB conversion failed")
            return None

        # Verify file exists before calling PRODIGY
        if not os.path.exists(pdb_path):
            print(f"    DEBUG: ERROR - PDB file does not exist: {pdb_path}")
            return None

        # Run PRODIGY on the PDB file
        binding_energy = run_prodigy_on_pdb(pdb_path, temp)
        print(f"    DEBUG: PRODIGY returned: {binding_energy}")

        return binding_energy

    except Exception as e:
        print(f"    DEBUG: Exception in run_prodigy_binding_energy: {e}")
        return None

    finally:
        # Clean up temporary file
        if 'pdb_path' in locals() and pdb_path and os.path.exists(pdb_path):
            os.remove(pdb_path)
            print(f"    DEBUG: Cleaned up temp file: {pdb_path}")

def run_prodigy_binding_energy_direct(structure: Structure,
                                    temp: float = 25.0,
                                    temp_dir: Optional[str] = None) -> Optional[float]:
    """
    Calculate binding energy using PRODIGY's Python API directly.

    Args:
        structure: Bio.PDB Structure to analyze
        temp: Temperature in Celsius for binding energy calculation
        temp_dir: Directory to save temporary files (default: system temp dir)

    Returns:
        Estimated binding energy in kcal/mol or None if calculation fails
    """
    try:
        # Import PRODIGY modules
        from prodigy.predict_IC import predict_ic
        from prodigy.lib.parsers import parse_structure

        # Convert structure to PDB format
        pdb_path = convert_structure_to_pdb(structure, temp_dir)

        # Parse the structure with PRODIGY's parser
        parsed_structure = parse_structure(pdb_path)

        # Extract binding and antigen chains
        binding_chains, antigen_chains = split_chains_by_type(structure)

        if not binding_chains or not antigen_chains:
            print("Not enough chains for binding analysis")
            return None

        # Get chain IDs
        binding_chain_ids = [chain.id for chain in binding_chains]
        antigen_chain_ids = [chain.id for chain in antigen_chains]

        # Run PRODIGY prediction
        result = predict_ic(parsed_structure,
                          binding_chain_ids,
                          antigen_chain_ids,
                          temp=temp)

        # Extract binding energy
        if result and 'binding_energy' in result:
            return result['binding_energy']

        return None

    except ImportError:
        print("PRODIGY Python API not available. Falling back to command-line interface.")
        return run_prodigy_binding_energy(structure, temp, temp_dir)

    except Exception as e:
        print(f"Error using PRODIGY API: {e}")
        return None

    finally:
        # Clean up temporary file
        if 'pdb_path' in locals() and os.path.exists(pdb_path):
            os.remove(pdb_path)


def calculate_binding_energies(structure: Structure,
                               config: Dict,
                               use_prodigy: bool = True) -> Dict[str, float]:
    """
    Calculate binding energies using both methods: contact-based and PRODIGY.

    Args:
        structure: Bio.PDB Structure to analyze
        config: Configuration dictionary with analysis settings
        use_prodigy: Whether to use PRODIGY for binding energy calculation

    Returns:
        Dictionary with binding energy results
    """
    # Get binding entity name for antibody type detection
    binding_chains, _ = split_chains_by_type(structure)
    binding_entity = None
    if hasattr(structure, 'binding_entity'):
        binding_entity = structure.binding_entity

    # Calculate contact-based binding energy
    approx_energy, contacts = estimate_binding_energy_simple(
        structure,
        cutoff=config.get('contact_cutoff', 5.0)
    )

    # Initialize results with contact-based energy
    results = {
        "approx_energy": approx_energy,
        "contacts": contacts,
        "prodigy_energy": None
    }

    # Add antibody type if binding entity is known
    if binding_entity:
        antibody_type = detect_antibody_type(binding_entity)
        results["antibody_type"] = antibody_type

    # Calculate PRODIGY binding energy if requested
    if use_prodigy:
        # Create temp directory if needed
        temp_dir = None
        if 'temp_dir' in config:
            temp_dir = config['temp_dir']
            os.makedirs(temp_dir, exist_ok=True)

        print("Calculating PRODIGY binding energy...")
        try:
            # Use command-line interface directly (skip Python API attempt)
            prodigy_energy = run_prodigy_binding_energy(
                structure,
                temp=config.get('temperature', 25.0),
                temp_dir=temp_dir
            )

            results["prodigy_energy"] = prodigy_energy

        except Exception as e:
            print(f"Error calculating PRODIGY binding energy: {e}")

    return results


def estimate_binding_energy_simple(structure: Structure,
                                   cutoff: float = 5.0) -> Tuple[float, int]:
    """
    Estimate binding energy using a vectorized contact-based model.

    Args:
        structure: Bio.PDB Structure to analyze
        cutoff: Distance cutoff for contacts in Angstroms

    Returns:
        Tuple of (estimated_binding_energy, number_of_contacts)
    """
    binding_chains, antigen_chains = split_chains_by_type(structure)

    if not binding_chains or not antigen_chains:
        return 0.0, 0

    # Get coordinates as numpy arrays (vectorized approach)
    binding_coords = np.array([atom.coord for chain in binding_chains
                               for atom in chain.get_atoms()])
    antigen_coords = np.array([atom.coord for chain in antigen_chains
                               for atom in chain.get_atoms()])

    if len(binding_coords) == 0 or len(antigen_coords) == 0:
        return 0.0, 0

    # Vectorized distance calculation using broadcasting
    # Shape: (n_binding, 1, 3) - (1, n_antigen, 3) = (n_binding, n_antigen, 3)
    # diff = binding_coords[:, np.newaxis, :] - antigen_coords[np.newaxis, :, :]
    # distances_sq = np.sum(diff ** 2, axis=2)
    # distances = np.sqrt(distances_sq)

    # Count contacts using boolean indexing
    # contacts = np.sum(distances <= cutoff)
    if len(binding_coords) * len(antigen_coords) > 1000000:  # Use chunked approach for large complexes
        contacts = 0
        chunk_size = 1000
        for i in range(0, len(binding_coords), chunk_size):
            chunk = binding_coords[i:i + chunk_size]
            diff = chunk[:, np.newaxis, :] - antigen_coords[np.newaxis, :, :]
            distances_sq = np.sum(diff ** 2, axis=2)
            distances = np.sqrt(distances_sq)
            contacts += np.sum(distances <= cutoff)
    else:
        # Original vectorized approach for smaller complexes
        diff = binding_coords[:, np.newaxis, :] - antigen_coords[np.newaxis, :, :]
        distances_sq = np.sum(diff ** 2, axis=2)
        distances = np.sqrt(distances_sq)
        contacts = np.sum(distances <= cutoff)
    # Estimate binding energy (same simple model)
    delta_g = -0.1 * contacts

    return float(delta_g), int(contacts)


def calculate_interface_rmsd(structures: List[Structure],
                             cutoff: float = 5.0) -> float:
    """
    Calculate RMSD of interface centroids across multiple structures.

    Args:
        structures: List of Bio.PDB Structure objects
        cutoff: Distance cutoff for interface in Angstroms

    Returns:
        RMSD of interface centroids or np.nan if calculation fails
    """
    centroids = []

    for structure in structures:
        centroid = extract_interface_centroid(structure, cutoff=cutoff)
        if centroid is not None:
            centroids.append(centroid)

    if len(centroids) < 2:
        return np.nan

    # Calculate RMSD using numpy only
    centroids = np.array(centroids)

    # Calculate pairwise distances manually
    n = len(centroids)
    distances = []

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            distances.append(dist)

    mean_rmsd = np.mean(distances)
    return mean_rmsd