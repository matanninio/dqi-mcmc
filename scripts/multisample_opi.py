# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The DQI-MCMC Authors

r"""
Collect N good OPI samples and print a table of results.

Supports both sampling algorithms:
  Algorithm 1 (-C / --continuous): single continuous Gibbs run; records each
      new unique solution as it is found.
  Algorithm 2 (-R / --restart):    independent restart per sample; each
      restart has its own seed (base_seed + sample_number).

Output columns (tab-separated):
  algorithm  idx  sample_number  tau_dqi  f_value  x

Usage:
    python scripts/multisample_opi.py -C --p 13 --idx 1 --num-samples 6
    python scripts/multisample_opi.py -R --p 13 --idx 1 --num-samples 10
    python scripts/multisample_opi.py --algorithm 1 --p 13 --idx 1 --num-samples 6
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dqi_mcmc.api.maxlinsat.block_gibbs_sampler import BlockGibbsSampler
from scripts import make_per_rhs_seed
from scripts.run_opi_resampling import initialize_state, load_problem


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect N good OPI samples and print a results table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    alg = parser.add_mutually_exclusive_group(required=True)
    alg.add_argument(
        "-C",
        "--continuous",
        dest="algorithm",
        action="store_const",
        const=1,
        help="Algorithm 1: continuous Gibbs run",
    )
    alg.add_argument(
        "-R",
        "--restart",
        dest="algorithm",
        action="store_const",
        const=2,
        help="Algorithm 2: independent restart per sample",
    )
    alg.add_argument(
        "--algorithm", dest="algorithm", type=int, choices=[1, 2], help="Algorithm number (1 or 2)"
    )

    parser.add_argument("--p", type=int, required=True, help="Prime modulus")
    parser.add_argument("--idx", type=int, required=True, help="RHS index (0-based)")
    parser.add_argument(
        "--num-samples", type=int, required=True, help="Number of good samples to collect"
    )
    parser.add_argument(
        "--num-bit-flips", type=int, default=3, help="Gibbs block size (default: 3)"
    )
    parser.add_argument(
        "--seed-offset", type=int, default=0, help="Added to the deterministic seed (default: 0)"
    )
    parser.add_argument("--n-factor", type=int, default=2, help="n = p // n_factor (default: 2)")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10_000_000,
        help="Max Gibbs steps per attempt (default: 10 000 000)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    p, rhs_idx = args.p, args.idx
    base_dir = Path(__file__).resolve().parents[1]

    base_seed = make_per_rhs_seed(p, rhs_idx, args.seed_offset)
    problem, solutions_vectors, m, _, predN = load_problem(p, args.n_factor, rhs_idx, base_dir)
    block_size = min(args.num_bit_flips, problem.num_variables, 3)
    sampler = BlockGibbsSampler(problem)

    # Print header
    print("\t".join(["algorithm", "idx", "sample_number", "tau_dqi", "f_value", "x"]))

    if args.algorithm == 1:
        # ── Algorithm 1: single continuous run ────────────────────────────────
        rng = np.random.default_rng(base_seed)
        x = rng.integers(0, p, size=problem.num_variables, dtype=np.int64)
        x, poly_vals, value, fx = initialize_state(sampler, problem, x, rng)

        unique_solutions = set()
        total_steps = 0

        while len(unique_solutions) < args.num_samples and total_steps < args.max_steps:
            x, poly_vals, value, fx = sampler._one_step(
                rng=rng,
                x=x,
                poly_vals=poly_vals,
                value=value,
                fx_old=fx,
                block_size=block_size,
                block_strategy="permutation",
            )
            total_steps += 1
            n_sat = (int(fx) + m) // 2

            if n_sat > predN:
                x_tuple = tuple(x.tolist())
                if x_tuple not in unique_solutions:
                    unique_solutions.add(x_tuple)
                    sample_num = len(unique_solutions)
                    print(f"1\t{rhs_idx}\t{sample_num}\t{total_steps}\t{n_sat}\t{list(x_tuple)}")

    else:
        # ── Algorithm 2: independent restart per sample ───────────────────────
        # Algorithm 2 design:
        # - Each restart uses a fresh independent RNG seeded with base_seed + iteration + 1.
        # - The full batch_size=50_000 steps are always run to completion — the permutation
        #   state (_permutation, _permutation_idx) is NOT reset between restarts; it carries
        #   over from the end of the previous full batch, exactly as sample_with_values does.
        # - tau_dqi is the step index of the first hit within the batch.
        batch_size = 50_000
        for iteration in range(args.num_samples):
            iteration_seed = base_seed + iteration + 1
            rng = np.random.default_rng(iteration_seed)

            x = rng.integers(0, p, size=problem.num_variables, dtype=np.int64)
            x, poly_vals, value, fx = initialize_state(sampler, problem, x, rng)

            first_hit = None
            for step in range(1, batch_size + 1):
                x, poly_vals, value, fx = sampler._one_step(
                    rng=rng,
                    x=x,
                    poly_vals=poly_vals,
                    value=value,
                    fx_old=fx,
                    block_size=block_size,
                    block_strategy="permutation",
                )
                n_sat = (int(fx) + m) // 2
                if first_hit is None and n_sat > predN:
                    first_hit = (step, n_sat, x.copy())
                # always continue to step 50_000 to advance permutation state

            if first_hit is not None:
                step, n_sat, x_hit = first_hit
                print(f"2\t{rhs_idx}\t{iteration + 1}\t{step}\t{n_sat}\t{x_hit.tolist()}")
            else:
                print(f"2\t{rhs_idx}\t{iteration + 1}\tNOT_FOUND\t-\t-", file=sys.stderr)


if __name__ == "__main__":
    main()
