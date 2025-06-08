"""
Cache manager for storing and retrieving computed binding energies.
"""
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Optional, Union
import pandas as pd


class BindingEnergyCache:
    """Manager for caching binding energy calculations."""

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
            'prodigy': self.cache_dir / 'prodigy_binding_energy_cache.json'
        }

        # Load existing caches
        self.caches = {}
        for method, cache_file in self.cache_files.items():
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    self.caches[method] = json.load(f)
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

        # Save to file
        with open(self.cache_files[method], 'w') as f:
            json.dump(self.caches[method], f, indent=2)

    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about cached entries."""
        return {
            method: len(cache)
            for method, cache in self.caches.items()
        }