# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The DQI-MCMC Authors

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
Simplified fast Cython implementation using incremental formula with C arrays.

Key insight: Since we compute ALL values upfront in one pass, we don't need
complex caching logic or recursion. Just fill the array iteratively.
"""

cimport cython
from libc.stdlib cimport malloc, free

# Use Python's arbitrary precision integers for the actual values


cdef class FastElementaryPolyCache:
    """
    Fast cache for elementary polynomial values using pure iterative computation.

    Stores e_k(num_ones) for all k in [0, ell] and num_ones in [0, m].
    Uses incremental formula to compute values efficiently in one pass.

    Supports two formulas:
    - Standard (maxXorSat): e_k(a) = e_k(a-1) + e_{k-1}(a-1) + e_{k-1}(a)
    - Generalized (OPI): e_k(a) = e_k(a-1) + e_{k-1}(a-1)*g_plus - e_{k-1}(a)*g_minus
      where g_plus = 2p-2r and g_minus = -2r
    """
    cdef int m
    cdef int ell
    cdef object g_plus  # Python object for arbitrary precision
    cdef object g_minus  # Python object for arbitrary precision
    cdef object[:, :] cache  # 2D memoryview of Python objects (arbitrary precision ints)

    @property
    def ell_max(self):
        """Maximum polynomial degree stored in this cache."""
        return self.ell

    def __init__(self, int m, int ell, int p=2, int r=1):
        """
        Initialize cache for problem size m and maximum degree ell.

        Parameters
        ----------
        m : int
            Number of constraints
        ell : int
            Maximum polynomial degree to cache
        p : int, optional
            Field size (default: 2 for maxXorSat)
        r : int, optional
            Number of satisfied values per constraint (default: 1 for maxXorSat)

        Notes
        -----
        The cache will store e_k(num_ones) for all k in [0, ell] and
        num_ones in [0, m]. You can then use get_value() to retrieve
        values for any k <= ell, allowing you to build lookup tables
        for multiple polynomial degrees from a single cache.

        When p=2 and r=1 (default), uses the standard maxXorSat formula:
            e_k(a) = e_k(a-1) + e_{k-1}(a-1) + e_{k-1}(a)

        For other values, uses the generalized OPI formula:
            e_k(a) = e_k(a-1) + e_{k-1}(a-1)*g_plus - e_{k-1}(a)*g_minus
            where g_plus = p-r and g_minus = -r

        Example
        -------
        # Build cache once for ell_max=100 (maxXorSat)
        cache = build_elementary_poly_cache_fast(m=1000, ell=100)

        # Build cache for OPI with p=3, r=1
        cache_opi = build_elementary_poly_cache_fast(m=1000, ell=100, p=3, r=1)

        # Use for multiple ell values
        for ell in [20, 50, 100]:
            for num_ones in range(m + 1):
                for k in range(ell + 1):
                    e_k = cache.get_value(num_ones, k)
        """
        self.m = m
        self.ell = ell

        # Compute g_plus and g_minus for generalized formula
        # For maxXorSat (p=2, r=1): g_plus = 1, g_minus = -1
        # For OPI: g_plus = p - r, g_minus = -r
        self.g_plus = p - r
        self.g_minus = -r

        # Allocate cache as 2D array of Python objects
        # cache[num_ones, k] = e_k(num_ones)
        import numpy as np
        self.cache = np.empty((m + 1, ell + 1), dtype=object)

        # Initialize base case: e_0(num_ones) = 1 for all num_ones
        for num_ones in range(m + 1):
            self.cache[num_ones, 0] = 1

    cdef object get_direct(self, int num_ones, int k):
        """
        Compute e_k(0) directly using the closed-form formula.

        Only called for num_ones=0 (the seed row in compute_all).

            e_k(0) = C(m, k) * g_minus^k
        """
        from math import comb

        if num_ones == 0:
            if k > self.m:
                return 0
            return comb(self.m, k) * (self.g_minus ** k)
        else:
            raise NotImplementedError("only support num_ones==0")

    def compute_all(self):
        """
        Compute all e_k(num_ones) values iteratively using dynamic programming.

        This is the ONLY way to fill the cache - no on-demand computation.

        Uses the generalized incremental formula:
            e_k(a) = e_k(a-1) + g_plus*e_{k-1}(a-1) - g_minus*e_{k-1}(a)

        For MAX-XOR-SAT (p=2, r=1), this simplifies to:
            e_k(a) = e_k(a-1) + e_{k-1}(a-1) + e_{k-1}(a)

        because g_plus=1 and g_minus=-1, so subtracting g_minus gives a plus sign.
        """
        cdef int num_ones, k

        # Base case already initialized: e_0(num_ones) = 1

        # For num_ones = 0, compute all k using direct formula
        for k in range(1, self.ell + 1):
            self.cache[0, k] = self.get_direct(0, k)

        # For each num_ones > 0, compute all k using incremental formula
        # This is pure dynamic programming - no recursion, no checks
        for num_ones in range(1, self.m + 1):
            for k in range(1, self.ell + 1):
                # Generalized formula: e_k(a) = e_k(a-1) + g_plus*e_{k-1}(a-1) - g_minus*e_{k-1}(a)
                # For MAX-XOR-SAT (p=2, r=1): g_plus=1, g_minus=-1
                # This simplifies to: e_k(a) = e_k(a-1) + e_{k-1}(a-1) + e_{k-1}(a)
                # Note: The apparent "+" for the last term is correct because we subtract g_minus=-1
                # All three values on RHS are guaranteed to be computed already
                self.cache[num_ones, k] = (
                    self.cache[num_ones - 1, k] +
                    self.cache[num_ones - 1, k - 1] * self.g_plus -
                    self.cache[num_ones, k - 1] * self.g_minus
                )

    def get_value(self, int num_ones, int k):
        """
        Get cached value. Must call compute_all() first!

        This is just a simple array lookup - no computation.
        """
        return self.cache[num_ones, k]

    def get_all_for_num_ones(self, int num_ones):
        """Get all e_k values for a specific num_ones."""
        return [self.cache[num_ones, k] for k in range(self.ell + 1)]

    def compute_log_cache(self, normalize=False):
        """
        Pre-compute log(abs(e_k)) and sign(e_k) for all cached values.

        Parameters
        ----------
        normalize : bool, optional
            If True, normalize log values by subtracting the minimum non-infinite value.
            (default: False)

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            - log_cache[num_ones, k] = log(abs(e_k(num_ones))) or -inf if e_k == 0
            - sign_cache[num_ones, k] = sign(e_k(num_ones))
        """
        import numpy as np
        import math

        log_cache = np.full((self.m + 1, self.ell + 1), -np.inf, dtype=np.float64)
        sign_cache = np.zeros((self.m + 1, self.ell + 1), dtype=np.int8)

        for num_ones in range(self.m + 1):
            for k in range(self.ell + 1):
                e_k = self.cache[num_ones, k]
                if e_k != 0:
                    log_cache[num_ones, k] = math.log(abs(e_k))
                    sign_cache[num_ones, k] = 1 if e_k > 0 else -1

        # Normalize if requested
        if normalize:
            # Find minimum non-infinite value
            finite_mask = ~np.isinf(log_cache)
            if np.any(finite_mask):
                min_log = np.min(log_cache[finite_mask])
                # Subtract minimum from all finite values
                log_cache[finite_mask] -= min_log

        return log_cache, sign_cache


def build_elementary_poly_cache_fast(int m, int ell, int p=2, int r=1):
    """
    Build complete cache of elementary polynomial values.

    Parameters
    ----------
    m : int
        Number of constraints
    ell : int
        Maximum polynomial degree to cache
    p : int, optional
        Field size (default: 2 for maxXorSat)
    r : int, optional
        Number of satisfied values per constraint (default: 1 for maxXorSat)

    Returns
    -------
    FastElementaryPolyCache
        Cache object with all values computed
    """
    cdef FastElementaryPolyCache cache = FastElementaryPolyCache(m, ell, p, r)
    cache.compute_all()
    return cache
