"""
Cache manager for storing and retrieving computed binding energies.
"""
import os
import json
import hashlib
import time
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union, List
import pandas as pd

"""
Cache manager for storing and retrieving computed binding energies.
"""
import os
import json
import hashlib
import time
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union, List
import pandas as pd


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class BindingEnergyCache:
    """Manager for caching binding energy and MSE calculations."""

    def __init__(self, cache_dir: Union[str, Path]):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        # Separate cache files for different methods
        self.cache_files = {
            'simple': self.cache_dir / 'simple_binding_energy_cache.json',
            'prodigy': self.cache_dir / 'prodigy_binding_energy_cache.json',
            'mse': self.cache_dir / 'mse_cache.json'
        }

        # Load existing caches with robust error handling
        self.caches = {}
        for method, cache_file in self.cache_files.items():
            if cache_file.exists():
                try:
                    # Check if file is empty first
                    if cache_file.stat().st_size == 0:
                        print(f"  Warning: Empty {method} cache file, creating new cache")
                        cache_file.unlink()  # Delete empty file
                        self.caches[method] = {}
                        continue

                    # Try to load the cache
                    with open(cache_file, 'r') as f:
                        self.caches[method] = json.load(f)
                    print(f"  Loaded {len(self.caches[method])} cached {method} entries")

                except (json.JSONDecodeError, ValueError) as e:
                    print(f"  Warning: Corrupted {method} cache file, creating new cache")
                    print(f"    Error: {e}")

                    # Create unique backup filename with timestamp
                    timestamp = int(time.time())
                    backup_file = cache_file.with_suffix(f'.json.backup.{timestamp}')

                    try:
                        # Move corrupted file to backup
                        cache_file.rename(backup_file)
                        print(f"    Corrupted cache backed up to: {backup_file}")
                    except Exception as backup_error:
                        print(f"    Could not backup corrupted cache: {backup_error}")
                        # If backup fails, just delete the corrupted file
                        try:
                            cache_file.unlink()
                            print(f"    Deleted corrupted cache file")
                        except Exception as delete_error:
                            print(f"    Could not delete corrupted cache: {delete_error}")

                    # Initialize empty cache
                    self.caches[method] = {}

                except Exception as e:
                    print(f"  Warning: Could not load {method} cache: {e}")
                    self.caches[method] = {}
            else:
                self.caches[method] = {}

    def _get_structure_hash(self, structure_path: str) -> str:
        """
        Generate a unique hash for a structure file.

        Args:
            structure_path: Path to structure file

        Returns:
            MD5 hash of the file
        """
        # Use file path and modification time for hash
        stat = os.stat(structure_path)
        hash_string = f"{structure_path}_{stat.st_mtime}_{stat.st_size}"
        return hashlib.md5(hash_string.encode()).hexdigest()

    def _get_combination_hash(self, model_paths: List[str]) -> str:
        """
        Generate a unique hash for a combination of model files.

        Args:
            model_paths: List of paths to model files

        Returns:
            MD5 hash of the combination
        """
        # Sort paths to ensure consistent hashing
        sorted_paths = sorted(model_paths)

        # Create hash from file paths, sizes, and modification times
        hash_components = []
        for path in sorted_paths:
            if os.path.exists(path):
                stat = os.stat(path)
                hash_components.append(f"{path}_{stat.st_mtime}_{stat.st_size}")

        hash_string = "|".join(hash_components)
        return hashlib.md5(hash_string.encode()).hexdigest()

    def _save_cache_safely(self, method: str):
        """
        Safely save cache to file with atomic write operation.

        Args:
            method: Cache method ('simple', 'prodigy', 'mse')
        """
        cache_file = self.cache_files[method]
        temp_file = cache_file.with_suffix('.json.tmp')

        try:
            # Write to temporary file first
            with open(temp_file, 'w') as f:
                json.dump(self.caches[method], f, indent=2, cls=NumpyEncoder)

            # Atomic rename (only on successful write)
            temp_file.replace(cache_file)

        except Exception as e:
            print(f"Warning: Failed to save {method} cache: {e}")
            # Clean up temp file if it exists
            if temp_file.exists():
                temp_file.unlink()

    def get_cached_energy(self, structure_path: str, method: str = 'simple',
                          temperature: float = 25.0, cutoff: float = 5.0) -> Optional[Dict]:
        """
        Retrieve cached binding energy if available.

        Args:
            structure_path: Path to structure file
            method: 'simple' or 'prodigy'
            temperature: Temperature for PRODIGY
            cutoff: Distance cutoff for contacts

        Returns:
            Cached results or None
        """
        struct_hash = self._get_structure_hash(structure_path)
        cache_key = f"{struct_hash}_{temperature}_{cutoff}"

        if method in self.caches and cache_key in self.caches[method]:
            return self.caches[method][cache_key]

        return None

    def get_cached_mse(self, combination: str, model_paths: List[str],
                       cutoff: float = 5.0) -> Optional[float]:
        """
        Retrieve cached MSE if available.

        Args:
            combination: Combination name (e.g., "nbGFP_6xzf_gfp")
            model_paths: List of model file paths
            cutoff: Distance cutoff used

        Returns:
            Cached MSE value or None
        """
        combo_hash = self._get_combination_hash(model_paths)
        cache_key = f"{combination}_{combo_hash}_{cutoff}"

        if cache_key in self.caches['mse']:
            cached_result = self.caches['mse'][cache_key]
            print(f"    Using cached MSE for {combination}: {cached_result['mse']:.4f}")
            return cached_result['mse']

        return None

    def cache_energy(self, structure_path: str, results: Dict, method: str = 'simple',
                     temperature: float = 25.0, cutoff: float = 5.0):
        """
        Cache binding energy results.

        Args:
            structure_path: Path to structure file
            results: Dictionary with energy results
            method: 'simple' or 'prodigy'
            temperature: Temperature for PRODIGY
            cutoff: Distance cutoff for contacts
        """
        struct_hash = self._get_structure_hash(structure_path)
        cache_key = f"{struct_hash}_{temperature}_{cutoff}"

        # Store results
        self.caches[method][cache_key] = results

        # Save to file safely
        self._save_cache_safely(method)

    def cache_mse(self, combination: str, model_paths: List[str],
                  mse_value: float, cutoff: float = 5.0):
        """
        Cache MSE result.

        Args:
            combination: Combination name
            model_paths: List of model file paths
            mse_value: Computed MSE value
            cutoff: Distance cutoff used
        """
        combo_hash = self._get_combination_hash(model_paths)
        cache_key = f"{combination}_{combo_hash}_{cutoff}"

        # Store result with metadata
        self.caches['mse'][cache_key] = {
            'mse': mse_value,
            'num_models': len(model_paths),
            'combination': combination,
            'cutoff': cutoff,
            'timestamp': pd.Timestamp.now().isoformat()
        }

        # Save to file safely
        self._save_cache_safely('mse')

    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about cached entries."""
        return {
            method: len(cache)
            for method, cache in self.caches.items()
        }