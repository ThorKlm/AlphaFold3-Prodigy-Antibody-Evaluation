"""
Utilities for handling protein structure files.
"""
import os
import zipfile
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
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


def load_alphafold2_multimer_model(model_dir: Union[str, Path], subfolder_pattern: str,
                                  model_file: str = 'top_model.pdb') -> Optional[str]:
    """
    Load AlphaFold2-Multimer models from directory structure like "/nbSARS_7f5h_MCherry_seed0/top_model.pdb".

    Args:
        model_dir: Base directory containing AlphaFold2-Multimer predictions
        subfolder_pattern: Pattern to match subfolders (e.g., "*_seed*")
        model_file: Name of the model file in each subfolder

    Returns:
        Dictionary mapping combinations to lists of model paths
    """
    model_dir = Path(model_dir)

    # Find all subfolders matching the pattern
    subfolders = [d for d in model_dir.glob(subfolder_pattern) if d.is_dir()]

    # Look for model files
    model_paths = []
    for subfolder in subfolders:
        model_path = subfolder / model_file
        if model_path.exists():
            model_paths.append(str(model_path))

    return model_paths


def group_af2_multimer_models(model_paths: List[str]) -> Dict[str, List[str]]:
    """
    Group AlphaFold2-Multimer models by antibody-antigen combination.

    Args:
        model_paths: List of paths to model files

    Returns:
        Dictionary mapping combinations to lists of model paths
    """
    # First, group all models by binding entity
    binding_entity_to_models = {}

    for model_path in model_paths:
        # Extract folder name (e.g., "nbALB_8y9t_alb_seed6")
        folder_name = Path(model_path).parent.name

        # Find the binding entity (everything before '_seed')
        binding_parts = folder_name.split('_seed')[0].split('_')

        # Handle different naming conventions
        if len(binding_parts) >= 3:
            # Format like "nbALB_8y9t_alb_seed6" or "nbGFP_6xzf_MCherry_seed1"
            binding_entity = f"{binding_parts[0]}_{binding_parts[1]}"
            antigen = binding_parts[2]
        elif len(binding_parts) == 2:
            # Format like "nbALB_alb_seed6"
            binding_entity = binding_parts[0]
            antigen = binding_parts[1]
        else:
            # Unknown format, use the whole thing as binding entity
            binding_entity = '_'.join(binding_parts)
            antigen = "unknown"

        combination = f"{binding_entity}_{antigen}"

        if combination not in binding_entity_to_models:
            binding_entity_to_models[combination] = []

        binding_entity_to_models[combination].append(model_path)

    return binding_entity_to_models


def parse_model_filename(filename: Union[str, Path]) -> Optional[Dict[str, str]]:
    """
    Parse metadata from a model filename.

    Expected formats:
    - prefix_binding_entity_antigen_*_seed_*.cif  (AlphaFold3)
    - /some/path/nbALB_8y9t_alb_seed6/top_model.pdb  (AlphaFold2-Multimer)
    - /some/path/nbGFP_6xzf_MCherry_seed1/top_model.pdb  (AlphaFold2-Multimer)

    Args:
        filename: Path to the model file

    Returns:
        Dictionary containing parsed metadata or None if parsing fails
    """
    filename = str(filename)

    # Check if it's an AlphaFold2-Multimer path with "seed" in the path
    if ('seed' in filename) and ('top_model.pdb' in filename or 'ranked_0.pdb' in filename):
        try:
            # Extract folder name (e.g., "nbALB_8y9t_alb_seed6" or "nbGFP_6xzf_MCherry_seed1")
            folder_name = Path(filename).parent.name

            # Split at "_seed" to get the prefix and seed number
            if '_seed' in folder_name:
                prefix_parts = folder_name.split('_seed')[0].split('_')
                seed = folder_name.split('_seed')[1]
            else:
                # Handle other naming patterns
                prefix_parts = folder_name.split('_')[:-1]  # Assume last part is seed
                seed = folder_name.split('_')[-1]

            # Extract binding entity and antigen based on the number of parts
            if len(prefix_parts) >= 3:
                # Format like "nbALB_8y9t_alb_seed6"
                binding_entity = f"{prefix_parts[0]}_{prefix_parts[1]}"
                antigen = prefix_parts[2]
            elif len(prefix_parts) == 2:
                # Format like "nbALB_alb_seed6"
                binding_entity = prefix_parts[0]
                antigen = prefix_parts[1]
            else:
                # Unknown format, use the whole prefix as binding entity
                binding_entity = '_'.join(prefix_parts)
                antigen = "unknown"

            return {
                "binding_entity": binding_entity,
                "antigen": antigen,
                "seed": seed,
                "combination": f"{binding_entity}_{antigen}"
            }
        except Exception as e:
            print(f"Error parsing AlphaFold2-Multimer filename {filename}: {e}")
            return None

    # Regular AlphaFold3 format
    parts = Path(filename).name.split('_')

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


def load_structure(file_path: Union[str, Path]) -> Optional[Structure]:
    """
    Load a structure from a PDB or CIF file with enhanced error handling.

    Args:
        file_path: Path to the structure file

    Returns:
        Loaded Bio.PDB Structure or None if loading fails
    """
    file_path = str(file_path)

    try:
        if file_path.endswith('.pdb'):
            # Use PDBParser with more permissive options
            from Bio.PDB import PDBParser
            parser = PDBParser(QUIET=True, PERMISSIVE=True)

            # Read file content to check for specific issues
            with open(file_path, 'r') as f:
                content = f.read()

            # Load structure
            structure = parser.get_structure("structure", file_path)

            # Verify that structure has chains and atoms
            model = structure[0]
            chains = list(model.get_chains())
            if not chains:
                print(f"Warning: No chains found in {file_path}")
                return None

            # Check if we have at least two chains (for binding interface)
            if len(chains) < 2:
                print(f"Warning: Need at least 2 chains for binding analysis, found {len(chains)} in {file_path}")
                return None

            # Count atoms
            atom_count = sum(1 for _ in structure.get_atoms())
            if atom_count == 0:
                print(f"Warning: No atoms found in {file_path}")
                return None

            return structure

        elif file_path.endswith('.cif'):
            # Use MMCIFParser
            from Bio.PDB import MMCIFParser
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure("structure", file_path)
            return structure
        else:
            # Try to guess file format
            if file_path.endswith('.ent'):
                # PDB format
                from Bio.PDB import PDBParser
                parser = PDBParser(QUIET=True, PERMISSIVE=True)
                structure = parser.get_structure("structure", file_path)
                return structure
            else:
                print(f"Unrecognized file format for {file_path}")
                return None

    except Exception as e:
        print(f"Error loading structure from {file_path}: {e}")
        return None


def load_structure_with_pymol(file_path: Union[str, Path], output_dir: Optional[str] = None) -> Optional[str]:
    """
    Load a structure using PyMOL as a fallback method for problematic PDB files.

    Args:
        file_path: Path to the structure file
        output_dir: Directory to save the cleaned PDB file

    Returns:
        Path to the cleaned PDB file or None if loading fails
    """
    try:
        # Check if PyMOL is available
        import pymol
        from pymol import cmd

        # Initialize PyMOL
        pymol.finish_launching(['pymol', '-q'])

        # Create output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = os.path.dirname(file_path)

        # Generate output filename
        output_path = os.path.join(output_dir, f"cleaned_{os.path.basename(file_path)}")

        # Load and save structure
        cmd.load(file_path, "structure")
        cmd.save(output_path, "structure")
        cmd.delete("all")

        # Load the cleaned structure with BioPython
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True, PERMISSIVE=True)
        structure = parser.get_structure("structure", output_path)

        return structure

    except ImportError:
        print("PyMOL not available for structure conversion")
        return None
    except Exception as e:
        print(f"Error using PyMOL to convert {file_path}: {e}")
        return None


def load_alphafold2_multimer_model(model_dir: Union[str, Path],
                                   subfolder_pattern: str = "*_seed*",
                                   model_file: str = "top_model.pdb") -> List[str]:
    """
    Load AlphaFold2-Multimer models from directory structure.

    Args:
        model_dir: Base directory containing AlphaFold2-Multimer predictions
        subfolder_pattern: Pattern to match subfolders (e.g., "*_seed*")
        model_file: Name of the model file in each subfolder

    Returns:
        List of paths to model files
    """
    model_dir = Path(model_dir)

    # Verify directory exists
    if not model_dir.exists() or not model_dir.is_dir():
        print(f"Error: Directory not found: {model_dir}")
        return []

    # Find all subfolders matching the pattern
    subfolders = [d for d in model_dir.glob(subfolder_pattern) if d.is_dir()]

    if not subfolders:
        print(f"Warning: No subfolders matching '{subfolder_pattern}' found in {model_dir}")
        return []

    # Look for model files
    model_paths = []
    for subfolder in subfolders:
        model_path = subfolder / model_file
        if model_path.exists():
            # Check if the file is readable
            try:
                with open(model_path, 'r') as f:
                    # Read a few bytes to verify
                    f.read(10)
                model_paths.append(str(model_path))
            except Exception as e:
                print(f"Warning: Could not read {model_path}: {e}")

    if not model_paths:
        print(f"Warning: No {model_file} files found in subfolders")

    return model_paths


def alternative_structure_loading(file_path: Union[str, Path], temp_dir: Optional[str] = None) -> Optional[Structure]:
    """
    Attempt to load structure using alternative methods when the primary method fails.

    Args:
        file_path: Path to the structure file
        temp_dir: Directory for temporary files

    Returns:
        Loaded structure or None if all methods fail
    """
    import tempfile

    if temp_dir is None:
        temp_dir = tempfile.gettempdir()

    # Method 1: Try BioPython with PERMISSIVE=True
    try:
        structure = load_structure(file_path)
        if structure:
            return structure
    except Exception as e:
        print(f"Standard loading failed for {file_path}: {e}")

    # Method 2: Try reading the file line by line to find and fix issues
    try:
        fixed_file = os.path.join(temp_dir, f"fixed_{os.path.basename(file_path)}")
        with open(file_path, 'r') as f_in, open(fixed_file, 'w') as f_out:
            for line in f_in:
                # Fix common issues
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    # Ensure proper formatting of ATOM/HETATM records
                    if len(line) < 80:
                        line = line.ljust(80)
                    # Ensure valid atom names
                    if line[12:16].strip() == '':
                        line = line[:12] + ' X   ' + line[16:]
                f_out.write(line)

        # Try loading the fixed file
        structure = load_structure(fixed_file)
        if structure:
            return structure
    except Exception as e:
        print(f"Failed to fix and load {file_path}: {e}")

    # Method 3: Try PyMOL if available
    try:
        structure = load_structure_with_pymol(file_path, temp_dir)
        if structure:
            return structure
    except Exception as e:
        print(f"PyMOL loading failed for {file_path}: {e}")

    # If all methods fail
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