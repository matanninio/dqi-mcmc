# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The DQI-MCMC Authors

r"""
Run XOR-SAT Block Gibbs sampling for one (m, rhs-index) combination.

Appends one JSONL line to the output file with the sampling results.

Usage:
    python run_xorsat_sampling.py --m 250 --rhs-index 0 --num-samples 100000 --ell 3 \\
        --output results_m250.jsonl

    # With burn-in and non-default block size:
    python run_xorsat_sampling.py --m 250 --rhs-index 0 --num-samples 100000 --ell 3 \\
        --block-size 10 --num-burn-in 1000 --output results.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add paper-release to path so we can import dqi_mcmc
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dqi_mcmc.api.block_gibbs_sampler import BlockGibbsSamplerOptimized
from dqi_mcmc.api.maxxorsat.max_xor_p_sat_problem import MaxXorPSatProblem
from dqi_mcmc.api.maxxorsat.xorsat_problem_reader import read_problem
from scripts import make_per_rhs_seed


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run XOR-SAT Block Gibbs sampling for one (m, rhs-index) combination.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("-m", "--m", type=int, required=True, help="Problem size")
    parser.add_argument("-i", "--rhs-index", type=int, required=True, help="RHS index (0-99)")
    parser.add_argument(
        "-N", "--num-samples", type=int, required=True, help="Number of Gibbs steps"
    )
    parser.add_argument("-l", "--ell", type=int, required=True, help="Polynomial degree")
    parser.add_argument("--block-size", type=int, default=10, help="Gibbs block size (default: 10)")
    parser.add_argument("--num-burn-in", type=int, default=0, help="Burn-in samples (default: 0)")
    parser.add_argument(
        "-s",
        "--seed-offset",
        type=int,
        default=0,
        help="Seed offset for reproducibility (default: 0)",
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="JSONL output file (appends)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    m = args.m
    rhs_index = args.rhs_index
    p = 2  # XOR-SAT is binary

    # Deterministic per-RHS seed formula (same as paper)
    seed = make_per_rhs_seed(p, rhs_index, args.seed_offset)

    print(f"XOR-SAT sampling: m={m}, rhs_index={rhs_index}, ell={args.ell}, seed={seed}")

    # Load problem
    b, v = read_problem(m, v_index=rhs_index)
    n = b.shape[1]
    print(f"  Loaded: m={m}, n={n}")

    # Construct problem (set_optimal_lookup_table is called automatically in constructor)
    problem = MaxXorPSatProblem(b=b, v=v, p=p, ell=args.ell)

    # Construct sampler
    sampler = BlockGibbsSamplerOptimized(problem)

    # Run sampling
    print(
        f"  Sampling {args.num_samples} steps (burn-in={args.num_burn_in}, block_size={args.block_size})..."
    )
    samples, values = sampler.sample_with_values(
        num_samples=args.num_samples,
        num_burn_in_samples=args.num_burn_in,
        seed=seed,
        block_size=args.block_size,
        block_strategy="random",
    )

    # Extract f(x) values (column 1)
    fx_values = values[:, 1].astype(int).tolist()
    max_fx = int(np.max(fx_values))
    mean_fx = float(np.mean(fx_values))

    print(f"  max_fx={max_fx}, mean_fx={mean_fx:.2f}")

    # Build result dict
    result = {
        "m": m,
        "v_index": rhs_index,
        "n": n,
        "num_samples": args.num_samples,
        "ell": args.ell,
        "block_size": args.block_size,
        "seed": seed,
        "seed_offset": args.seed_offset,
        "max_fx": max_fx,
        "mean_fx": mean_fx,
        "fx_values": fx_values,
    }

    # Append to JSONL output file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a") as f:
        f.write(json.dumps(result) + "\n")

    print(f"  Result appended to: {output_path}")


if __name__ == "__main__":
    main()
