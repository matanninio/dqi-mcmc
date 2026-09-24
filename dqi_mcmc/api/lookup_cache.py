# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The DQI-MCMC Authors

"""
Lookup table caching utilities.

This module provides functions to load and save pre-computed lookup tables from disk.
Tables are stored in HDF5 format, one file per m value.
"""

from pathlib import Path

import h5py
import numpy as np


def get_cache_dir(problem_type: str = "maxxorsat") -> Path:
    """
    Get the cache directory for lookup tables.

    Args:
        problem_type: Type of problem - "maxxorsat" or "opi"

    Returns:
        Path to the cache directory for the specified problem type

    """
    cache_dir = Path(__file__).parent.parent / "data" / "lookup_tables" / problem_type
    return cache_dir


def get_cache_file(m: int, p: int = 2, r: int = 1) -> Path:
    """
    Get the cache file path for a given m value and problem type.

    Args:
        m: Number of constraints
        p: Field size (default: 2 for maxXorSat)
        r: Number of satisfied values per constraint (default: 1 for maxXorSat)

    Returns:
        Path to the cache file

    Notes:
        - For maxXorSat (p=2, r=1): maxxorsat/lookup_m{m}.h5
        - For OPI: opi/lookup_opi_m{m}_p{p}_r{r}.h5

    """
    if p == 2 and r == 1:
        # Standard maxXorSat naming in maxxorsat subdirectory
        return get_cache_dir("maxxorsat") / f"lookup_m{m}.h5"
    else:
        # OPI naming with parameters in opi subdirectory
        return get_cache_dir("opi") / f"lookup_opi_m{m}_p{p}_r{r}.h5"


def load_lookup_table(
    m: int, n: int, ell: int, p: int = 2, r: int = 1
) -> np.ndarray | None:
    """
    Load a pre-computed lookup table from disk.

    Args:
        m: Number of constraints
        n: Number of variables
        ell: Polynomial degree
        p: Field size (default: 2 for maxXorSat)
        r: Number of satisfied values per constraint (default: 1 for maxXorSat)

    Returns:
        lookup_log_polynomial_squared array (normalized) or None if not found

    """
    cache_file = get_cache_file(m, p, r)
    if not cache_file.exists():
        return None

    dataset_name = f"n{n}_ell{ell}"

    try:
        with h5py.File(cache_file, "r") as f:
            if dataset_name not in f:
                return None

            grp = f[dataset_name]
            lookup_log_poly_sq = np.array(grp["lookup_log_polynomial_squared"])  # type: ignore[index]

            # Normalize so that min(finite values) == 0.  All tables saved by
            # compute_optimal_lookup_table_xorsat / compute_optimal_lookup_table_opi
            # are already normalized before saving, so this subtraction is always
            # a no-op for well-formed files (min == 0 → subtract 0).  It is kept
            # here as a defensive measure so that any table saved via
            # save_lookup_table() without prior normalization still loads correctly.
            # NOTE: callers must not rely on the absolute scale of the returned
            # values — only differences (i.e. log-ratios) are meaningful.
            finite_mask = np.isfinite(lookup_log_poly_sq)
            if np.any(finite_mask):
                min_log = np.min(lookup_log_poly_sq[finite_mask])
                lookup_log_poly_sq = lookup_log_poly_sq.copy()
                lookup_log_poly_sq[finite_mask] -= min_log

            return lookup_log_poly_sq
    except Exception:
        return None


def save_lookup_table(
    m: int, n: int, ell: int, lookup_log_poly_sq: np.ndarray, p: int = 2, r: int = 1
) -> Path:
    """
    Save a pre-computed lookup table to disk.

    Args:
        m: Number of constraints
        n: Number of variables
        ell: Polynomial degree
        lookup_log_poly_sq: The lookup table array to save
        p: Field size (default: 2 for maxXorSat)
        r: Number of satisfied values per constraint (default: 1 for maxXorSat)

    Returns:
        Path to the saved cache file

    """
    cache_file = get_cache_file(m, p, r)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    dataset_name = f"n{n}_ell{ell}"

    with h5py.File(cache_file, "a") as f:
        if dataset_name in f:
            del f[dataset_name]
        grp = f.create_group(dataset_name)
        grp.create_dataset("lookup_log_polynomial_squared", data=lookup_log_poly_sq)

    return cache_file


def is_cached(m: int, n: int, ell: int, p: int = 2, r: int = 1) -> bool:
    """
    Check if a lookup table is cached on disk.

    Args:
        m: Number of constraints
        n: Number of variables
        ell: Polynomial degree
        p: Field size (default: 2 for maxXorSat)
        r: Number of satisfied values per constraint (default: 1 for maxXorSat)

    Returns:
        True if the lookup table exists in cache, False otherwise

    """
    cache_file = get_cache_file(m, p, r)
    if not cache_file.exists():
        return False

    dataset_name = f"n{n}_ell{ell}"

    try:
        with h5py.File(cache_file, "r") as f:
            return dataset_name in f
    except Exception:
        return False
