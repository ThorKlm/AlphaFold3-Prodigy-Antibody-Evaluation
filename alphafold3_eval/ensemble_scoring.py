"""
Ensemble-based scoring combining multiple metrics.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
from Bio.PDB.Structure import Structure


def normalize_score(values: np.ndarray, inverse: bool = False) -> np.ndarray:
    """
    Normalize scores to 0-1 range.

    Args:
        values: Array of values to normalize
        inverse: If True, lower values get higher scores

    Returns:
        Normalized scores
    """
    if len(values) == 0 or np.all(np.isnan(values)):
        return np.zeros_like(values)

    # Remove NaN values for calculation
    valid_mask = ~np.isnan(values)
    if not np.any(valid_mask):
        return np.zeros_like(values)

    min_val = np.min(values[valid_mask])
    max_val = np.max(values[valid_mask])

    if min_val == max_val:
        return np.ones_like(values) * 0.5

    normalized = (values - min_val) / (max_val - min_val)

    if inverse:
        normalized = 1.0 - normalized

    return normalized


def calculate_ensemble_score(
    metrics: Dict[str, float],
    weights: Optional[Dict[str, float]] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate weighted ensemble score from multiple metrics.

    Args:
        metrics: Dictionary of metric values
        weights: Optional weights for each metric

    Returns:
        Tuple of (ensemble_score, normalized_scores)
    """
    if weights is None:
        weights = {
            'interface_persistence': 0.40,
            'confidence': 0.30,
            'structural_consistency': 0.20,
            'binding_energy': 0.10
        }

    # Normalize each metric
    normalized = {}

    # Interface persistence (0-1, higher is better)
    if 'interface_persistence' in metrics:
        normalized['interface_persistence'] = metrics['interface_persistence']

    # Confidence (0-1, higher is better)
    if 'confidence' in metrics:
        normalized['confidence'] = metrics['confidence']

    # Structural consistency (MSE - lower is better, so inverse)
    if 'structural_consistency' in metrics:
        # Convert to 0-1 where higher is better
        mse = metrics['structural_consistency']
        # Use exponential decay for MSE to score conversion
        normalized['structural_consistency'] = np.exp(-mse / 10.0)

    # Binding energy (more negative is better)
    if 'binding_energy' in metrics:
        # Normalize to 0-1 where more negative gets higher score
        energy = metrics['binding_energy']
        # Sigmoid transformation
        normalized['binding_energy'] = 1.0 / (1.0 + np.exp(energy / 5.0))

    # Calculate weighted sum
    ensemble_score = 0.0
    total_weight = 0.0

    for metric, weight in weights.items():
        if metric in normalized:
            ensemble_score += normalized[metric] * weight
            total_weight += weight

    # Normalize by total weight
    if total_weight > 0:
        ensemble_score /= total_weight

    return ensemble_score, normalized


def calculate_statistical_significance(
    binding_matrix: np.ndarray,
    n_permutations: int = 1000,
    random_seed: int = 42
) -> Dict[str, float]:
    """
    Test if diagonal values are significantly higher than off-diagonal.

    Args:
        binding_matrix: Square matrix of binding scores
        n_permutations: Number of permutations for test
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with statistical test results
    """
    np.random.seed(random_seed)

    # Extract diagonal and off-diagonal values
    diagonal_mask = np.eye(binding_matrix.shape[0], dtype=bool)
    diagonal_values = binding_matrix[diagonal_mask]
    off_diagonal_values = binding_matrix[~diagonal_mask]

    # Remove NaN values
    diagonal_values = diagonal_values[~np.isnan(diagonal_values)]
    off_diagonal_values = off_diagonal_values[~np.isnan(off_diagonal_values)]

    if len(diagonal_values) == 0 or len(off_diagonal_values) == 0:
        return {
            'observed_difference': np.nan,
            'p_value': np.nan,
            'effect_size': np.nan,
            'diagonal_mean': np.nan,
            'off_diagonal_mean': np.nan
        }

    # Observed difference
    observed_diff = np.mean(diagonal_values) - np.mean(off_diagonal_values)

    # Permutation test
    all_values = np.concatenate([diagonal_values, off_diagonal_values])
    n_diagonal = len(diagonal_values)

    permuted_diffs = []
    for _ in range(n_permutations):
        np.random.shuffle(all_values)
        perm_diagonal = all_values[:n_diagonal]
        perm_off_diagonal = all_values[n_diagonal:]
        perm_diff = np.mean(perm_diagonal) - np.mean(perm_off_diagonal)
        permuted_diffs.append(perm_diff)

    # Calculate p-value
    permuted_diffs = np.array(permuted_diffs)
    p_value = np.mean(permuted_diffs >= observed_diff)

    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt(
        (np.var(diagonal_values) + np.var(off_diagonal_values)) / 2
    )
    effect_size = observed_diff / pooled_std if pooled_std > 0 else np.nan

    # Additional statistics
    results = {
        'observed_difference': observed_diff,
        'p_value': p_value,
        'effect_size': effect_size,
        'diagonal_mean': np.mean(diagonal_values),
        'diagonal_std': np.std(diagonal_values),
        'off_diagonal_mean': np.mean(off_diagonal_values),
        'off_diagonal_std': np.std(off_diagonal_values),
        'n_diagonal': len(diagonal_values),
        'n_off_diagonal': len(off_diagonal_values)
    }

    # Mann-Whitney U test as additional check
    try:
        u_stat, u_pvalue = stats.mannwhitneyu(
            diagonal_values,
            off_diagonal_values,
            alternative='greater'
        )
        results['mann_whitney_u'] = u_stat
        results['mann_whitney_p'] = u_pvalue
    except:
        results['mann_whitney_u'] = np.nan
        results['mann_whitney_p'] = np.nan

    return results