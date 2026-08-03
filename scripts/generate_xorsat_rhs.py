# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The DQI-MCMC Authors

"""
Generate random XOR-SAT right-hand side vectors.

Generates random RHS vectors for MAX-XOR-SAT problems and saves them as NumPy arrays.

Output Format:
    - Shape: (n_samples, m, 1)
    - Values: Random integers in {0, 1} (binary)

Usage Examples:
    # Generate 100 RHS vectors for m=250
    python generate_xorsat_rhs.py --m 250

    # With custom seed and output path
    python generate_xorsat_rhs.py --m 1000 --seed 42 --output /path/to/rhs.npy
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


def generate_and_save(path: str | Path, n_samples: int, m: int, seed: int = 123) -> None:
    """
    Generate and save random XOR-SAT right-hand side vectors.

    Args:
        path: Output file path
        n_samples: Number of independent RHS vectors to generate
        m: Number of constraints
        seed: Random seed for reproducibility (default: 123)

    """
    rng = np.random.default_rng(seed)
    # p=2, r=1 for XOR-SAT: binary values in {0, 1}
    rhs = rng.integers(0, 2, size=(n_samples, m, 1), dtype=np.uint8)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_path, rhs)
    print(f"Generated {n_samples} RHS vectors with shape ({m}, 1)")
    print(f"Saved to: {output_path}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate random XOR-SAT right-hand side vectors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-m", "--m", type=int, required=True, help="Number of constraints (required)"
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
        help="Output file path (default: problems/maxxorsat/rhs_nsamples{n}_m{m}_p2_r1.npy)",
    )

    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_args()

    # Determine output path
    if args.output:
        filename = Path(args.output)
    else:
        BASE_DIR = Path(__file__).resolve().parents[1]
        filename = (
            BASE_DIR
            / "problems"
            / "maxxorsat"
            / f"rhs_nsamples{args.n_samples}_m{args.m}_p2_r1.npy"
        )

    generate_and_save(
        path=filename,
        n_samples=args.n_samples,
        m=args.m,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
