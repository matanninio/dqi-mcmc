# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 IBM Corporation

"""
CLI scripts for DQI-MCMC experiments.

Shared utilities used across multiple scripts live here to avoid duplication.
"""


def make_per_rhs_seed(p: int, rhs_idx: int, seed_offset: int = 0) -> int:
    """
    Compute the deterministic per-(p, rhs_idx) seed used throughout the paper experiments.

    The formula ``(hash((p, rhs_idx)) & 0x7FFFFFFF) + seed_offset`` gives a stable
    non-negative integer suitable for ``numpy.random.default_rng``.

    IMPORTANT: Python's built-in ``hash()`` is randomised unless ``PYTHONHASHSEED`` is
    fixed.  For fully reproducible runs set ``PYTHONHASHSEED=0`` in the environment
    (or pass an explicit ``--seed-offset`` to override).

    Args:
        p: Prime modulus of the problem instance.
        rhs_idx: Index of the right-hand side vector (0-based).
        seed_offset: Optional offset added to the base seed (default: 0).

    Returns:
        A non-negative integer seed.

    """
    return (hash((p, rhs_idx)) & 0x7FFFFFFF) + seed_offset
