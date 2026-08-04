# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The DQI-MCMC Authors

"""
Generate random OPI right-hand side vectors.

Generates random RHS vectors for OPI problems and saves them as NumPy arrays.

Output Format:
    - Shape: (n_samples, p-1, r)
    - Values: Random integers in range [0, p) WITHOUT repeats within each row.

Usage Examples:
    # Generate 100 samples with p=13 (default r=p//2=6)
    python generate_opi_rhs.py --p 13

    # Generate with custom r and seed
    python generate_opi_rhs.py --p 13 --r 6 --seed 42 --n-samples 50
"""

import argparse
from pathlib import Path

import numpy as np


def dtype_for_p(p: int) -> type:
    if p <= 2**8:
        return np.uint8
    if p <= 2**16:
        return np.uint16
    return np.uint32


def generate_and_save(path: str | Path, n_samples: int, r: int, p: int, seed: int = 123) -> None:
    """
    Generate and save random OPI right-hand side vectors.

    Args:
        path: Output file path
        n_samples: Number of independent RHS vectors to generate
        r: Number of entries per component (typically p//2, must be <= p)
        p: Prime field size
        seed: Random seed for reproducibility (default: 123)

    """
    if r > p:
        raise ValueError(f"Cannot sample {r} unique values from range [0, {p}). Got r={r}, p={p}")

    rng = np.random.default_rng(seed)
    dt = dtype_for_p(p)

    total_rows = n_samples * (p - 1)

    rhs = np.empty((n_samples, p - 1, r), dtype=dt)

    # Generate samples without replacement for each row
    for i in range(total_rows):
        sample_idx = i // (p - 1)
        row_idx = i % (p - 1)
        rhs[sample_idx, row_idx, :] = rng.choice(p, size=r, replace=False)

    # Ensure parent directory exists
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_path, rhs)
    print(f"Generated {n_samples} RHS vectors with shape ({p-1}, {r})")
    print(f"Saved to: {output_path}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate random OPI right-hand side vectors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("-p", "--p", type=int, required=True, help="Prime field size (required)")

    parser.add_argument(
        "-r",
        "--r",
        type=int,
        default=None,
        help="Number of entries per component (default: p//2)",
    )

    parser.add_argument(
        "-n",
        "--n-samples",
        type=int,
        default=100,
        help="Number of independent RHS vectors to generate (default: 100)",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducibility (default: 123)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file path (default: problems/opi/rhs_nsamples{n}_p{p}_r{r}.npy)",
    )

    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_args()

    # Set default r if not provided
    r = args.r if args.r is not None else args.p // 2

    # Determine output path
    if args.output:
        filename = Path(args.output)
    else:
        # Default path relative to this script's grandparent (paper-release/)
        BASE_DIR = Path(__file__).resolve().parents[1]
        filename = (
            BASE_DIR / "problems" / "opi" / f"rhs_nsamples{args.n_samples}_p{args.p}_r{r}.npy"
        )

    generate_and_save(
        path=filename,
        n_samples=args.n_samples,
        r=r,
        p=args.p,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
