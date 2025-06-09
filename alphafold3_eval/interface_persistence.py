"""
Interface persistence analysis across multiple structure predictions.
"""
import numpy as np
from typing import Dict, List, Tuple, Set
from Bio.PDB.Structure import Structure
from Bio.PDB import NeighborSearch
from collections import defaultdict


def identify_interface_contacts(structure: Structure,
                                cutoff: float = 5.0) -> Set[Tuple[str, str]]:
    """
    Identify all atom-atom contacts at the interface.

    Args:
        structure: Bio.PDB Structure
        cutoff: Distance cutoff for contacts

    Returns:
        Set of contact pairs (chain1:resid1:atom1, chain2:resid2:atom2)
    """
    from alphafold3_eval.structure_uitls import split_chains_by_type

    binding_chains, antigen_chains = split_chains_by_type(structure)

    if not binding_chains or not antigen_chains:
        return set()

    # Get atoms from each group
    binding_atoms = []
    for chain in binding_chains:
        for atom in chain.get_atoms():
            binding_atoms.append(atom)

    antigen_atoms = []
    for chain in antigen_chains:
        for atom in chain.get_atoms():
            antigen_atoms.append(atom)

    # Find contacts
    contacts = set()
    ns = NeighborSearch(binding_atoms + antigen_atoms)

    for atom1 in binding_atoms:
        neighbors = ns.search(atom1.coord, cutoff)
        for atom2 in neighbors:
            if atom2 in antigen_atoms:
                # Create unique identifiers for atoms
                id1 = f"{atom1.parent.parent.id}:{atom1.parent.id[1]}:{atom1.name}"
                id2 = f"{atom2.parent.parent.id}:{atom2.parent.id[1]}:{atom2.name}"
                contacts.add((id1, id2))

    return contacts


def calculate_interface_persistence(structures: List[Structure],
                                    cutoff: float = 5.0) -> Dict[str, float]:
    """
    Calculate interface persistence across multiple structures.

    Args:
        structures: List of aligned structures
        cutoff: Distance cutoff for contacts

    Returns:
        Dictionary with persistence metrics
    """
    if len(structures) < 2:
        return {
            'mean_persistence': 0.0,
            'contact_conservation': 0.0,
            'persistent_contacts': 0,
            'total_unique_contacts': 0
        }

    # Get contacts for each structure
    all_contacts = []
    for struct in structures:
        contacts = identify_interface_contacts(struct, cutoff)
        all_contacts.append(contacts)

    # Count persistence of each contact
    contact_counts = defaultdict(int)
    for contacts in all_contacts:
        for contact in contacts:
            contact_counts[contact] += 1

    # Calculate metrics
    n_structures = len(structures)
    total_contacts = len(contact_counts)

    if total_contacts == 0:
        return {
            'mean_persistence': 0.0,
            'contact_conservation': 0.0,
            'persistent_contacts': 0,
            'total_unique_contacts': 0
        }

    # Persistence scores
    persistence_scores = [count / n_structures for count in contact_counts.values()]
    mean_persistence = np.mean(persistence_scores)

    # Highly persistent contacts (present in >80% of structures)
    persistent_threshold = 0.8
    persistent_contacts = sum(1 for score in persistence_scores if score >= persistent_threshold)

    # Contact conservation (fraction of contacts that are persistent)
    contact_conservation = persistent_contacts / total_contacts if total_contacts > 0 else 0.0

    return {
        'mean_persistence': mean_persistence,
        'contact_conservation': contact_conservation,
        'persistent_contacts': persistent_contacts,
        'total_unique_contacts': total_contacts,
        'persistence_distribution': persistence_scores
    }


# Complete the calculate_residue_persistence function in interface_persistence.py

def calculate_residue_persistence(structures: List[Structure],
                                  cutoff: float = 5.0) -> Dict[str, Dict[str, float]]:
    """
    Calculate persistence at residue level.

    Args:
        structures: List of aligned structures
        cutoff: Distance cutoff for interface

    Returns:
        Dictionary mapping residue IDs to persistence scores
    """
    from alphafold3_eval.binding_analysis import extract_interface_residues

    # Count how often each residue is at the interface
    residue_counts = defaultdict(int)
    n_structures = len(structures)

    for struct in structures:
        binding_interface, antigen_interface = extract_interface_residues(struct, cutoff)

        for res in binding_interface + antigen_interface:
            res_id = f"{res.parent.id}:{res.id[1]}"
            residue_counts[res_id] += 1

    # Calculate persistence scores
    binding_persistence = {}
    antigen_persistence = {}

    for res_id, count in residue_counts.items():
        persistence_score = count / n_structures

        # Determine if it's binding or antigen based on chain ID
        chain_id = res_id.split(':')[0]

        # You may need to adjust this logic based on your chain naming convention
        # For now, assume last chain is antigen (consistent with split_chains_by_type)
        all_chain_ids = list(set(res_id.split(':')[0] for res_id in residue_counts.keys()))
        all_chain_ids.sort()

        if chain_id == all_chain_ids[-1]:  # Last chain is antigen
            antigen_persistence[res_id] = persistence_score
        else:
            binding_persistence[res_id] = persistence_score

    return {
        'binding_entity': binding_persistence,
        'antigen': antigen_persistence
    }