# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 IBM Corporation

"""
Pre-compute and cache the XOR-SAT lookup table for a given m.

This is a one-time step before running run_xorsat_sampling.py for a given m.
For small m (<=500) this completes in seconds; for m=3000 it may take a few minutes.

Usage:
    python precompute_xorsat_lookup.py --m 250
    python precompute_xorsat_lookup.py --m 1000 --ell 78
    python precompute_xorsat_lookup.py --m 250 --force  # overwrite existing
"""

import argparse
import sys
from pathlib import Path

# Add paper-release to path so we can import dqi_mcmc
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dqi_mcmc.api.lookup_cache import is_cached, save_lookup_table
from dqi_mcmc.api.maxxorsat.max_xor_p_sat_problem import (
    compute_optimal_lookup_table_xorsat,
)
from dqi_mcmc.api.maxxorsat.xorsat_problem_reader import (
    get_problem_filepath,
    read_matrix_from_tsv,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Pre-compute and cache the XOR-SAT lookup table for a given m.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-m",
        "--m",
        type=int,
        required=True,
        help="Problem size (number of constraints). Must be one of: 100, 250, 500, 750, 1000, 3000",
    )

    parser.add_argument(
        "-l",
        "--ell",
        type=int,
        default=None,
        help="Polynomial degree (default: (n+1)//2 where n is number of variables)",
    )

    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing cached table"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    m = args.m

    # Load the constraint matrix to determine n
    matrix_path = get_problem_filepath(m)
    print(f"Loading constraint matrix from: {matrix_path}")
    b = read_matrix_from_tsv(matrix_path)
    n = b.shape[1]
    print(f"  m={m}, n={n}")

    # Determine ell
    ell = args.ell if args.ell is not None else (n + 1) // 2
    print(f"  ell={ell}")

    # Check if already cached
    if not args.force and is_cached(m, n, ell):
        print(f"Already cached: m={m}, n={n}, ell={ell}. Use --force to overwrite.")
        return

    # Compute the lookup table
    print(f"Computing lookup table for m={m}, n={n}, ell={ell}...")
    table = compute_optimal_lookup_table_xorsat(n, m, ell)
    print("  Done computing.")

    # Save to cache
    cache_file = save_lookup_table(m, n, ell, table)
    print(f"Saved to: {cache_file}")
    print("Lookup table pre-computed and cached successfully.")


if __name__ == "__main__":
    main()
