# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The DQI-MCMC Authors

r"""
Run one OPI Gibbs-sampling trial and print a human-readable summary.

Given a prime p and an RHS index, runs Algorithm 1 (continuous sampling) until
one good sample is found (n_satisfied > predN), then prints:

  - Problem parameters (p, n, r, m, predN)
  - tau_dqi   — Gibbs steps taken to find the first good sample
  - f_value   — number of satisfied constraints at that sample
  - best_x    — solution vector that achieved f_value
  - trajectory — every step at which a new best n_satisfied was reached

Usage:
    python scripts/sample_opi.py --p 13 --idx 8
    python scripts/sample_opi.py --p 13 --idx 12
    python scripts/sample_opi.py --p 43 --idx 3 --max-steps 10000000

Optional flags:
    --seed-offset INT    Added to the deterministic seed (default: 0)
    --n-factor INT       n = p // n_factor (default: 2)
    --max-steps INT      Safety cap on Gibbs steps (default: 10 000 000)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dqi_mcmc.api.maxlinsat.block_gibbs_sampler import BlockGibbsSampler
from dqi_mcmc.api.maxlinsat.max_opi_problem import MaxOPIProblem
from scripts import make_per_rhs_seed
from scripts.run_opi_resampling import initialize_state


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one OPI Gibbs trial and print trajectory + best solution.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--p", type=int, required=True, help="Prime modulus")
    parser.add_argument("--idx", type=int, required=True, help="RHS index (0-based)")
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Added to the deterministic per-(p, idx) seed (default: 0)",
    )
    parser.add_argument(
        "--n-factor",
        type=int,
        default=2,
        help="n = p // n_factor, sets number of variables (default: 2)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10_000_000,
        help="Maximum Gibbs steps before giving up (default: 10 000 000)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    p, rhs_idx = args.p, args.idx
    n = p // args.n_factor
    r = p // 2

    # Locate the RHS file
    base_dir = Path(__file__).resolve().parents[1]
    rhs_file = base_dir / "problems" / "opi" / f"rhs_nsamples100_p{p}_r{r}.npy"
    if not rhs_file.exists():
        sys.exit(
            f"RHS file not found for p={p}, r={r}.\n"
            f"Run:  python scripts/generate_opi_rhs.py --p {p}"
        )

    solutions_vectors = np.load(rhs_file)
    if rhs_idx >= len(solutions_vectors):
        sys.exit(f"idx={rhs_idx} out of range — RHS file has {len(solutions_vectors)} entries.")

    v = solutions_vectors[rhs_idx]
    problem = MaxOPIProblem(p=p, v=v, num_variables=n)
    m = problem.num_constraints
    predicted = problem.n_predicted()
    predN = int(np.floor(predicted * m))
    block_size = min(3, n)

    # Deterministic seed: same formula as run_opi_resampling.py
    seed = make_per_rhs_seed(p, rhs_idx, args.seed_offset)
    rng = np.random.default_rng(seed)

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"p={p}  idx={rhs_idx}  n={n}  r={r}  m={m}  predN={predN}  seed={seed}")
    print(
        f"Running Algorithm 1 (continuous) — block_size={block_size}, max_steps={args.max_steps:,}"
    )
    print()

    sampler = BlockGibbsSampler(problem)

    # Initialise from a random starting point
    x = rng.integers(0, p, size=n, dtype=np.int64)
    x, poly_vals, value, fx = initialize_state(sampler, problem, x, rng)

    # ── Sampling loop ─────────────────────────────────────────────────────────
    best_n_sat = (int(fx) + m) // 2
    best_x = x.copy()
    trajectory = []  # [[step, n_satisfied], ...]

    # Record the initial state as step 0 entry in the trajectory
    trajectory.append([0, best_n_sat])

    wall_start = time.perf_counter()
    tau_dqi = None
    step = 0

    while step < args.max_steps:
        x, poly_vals, value, fx = sampler._one_step(
            rng=rng,
            x=x,
            poly_vals=poly_vals,
            value=value,
            fx_old=fx,
            block_size=block_size,
            block_strategy="permutation",
        )
        step += 1
        n_sat = (int(fx) + m) // 2

        if n_sat > best_n_sat:
            best_n_sat = n_sat
            best_x = x.copy()
            trajectory.append([step, n_sat])

        if n_sat > predN and tau_dqi is None:
            tau_dqi = step
            break  # one good sample is enough

    wall_elapsed = time.perf_counter() - wall_start

    # ── Results ───────────────────────────────────────────────────────────────
    print("─" * 52)
    if tau_dqi is not None:
        print(f"tau_dqi   : {tau_dqi}")
        print(f"f_value   : {best_n_sat} / {m} satisfied (predN = {predN})")
    else:
        print(f"NOT FOUND in {args.max_steps:,} steps (best = {best_n_sat} / {m})")
    print(f"best_x    : {best_x.tolist()}")
    print(f"wall time : {wall_elapsed:.2f}s")
    print()
    print("Trajectory  [step, n_satisfied]:")
    for entry in trajectory:
        marker = " ← first above predN" if entry[1] > predN and entry[0] == tau_dqi else ""
        print(f"  step {entry[0]:>8,} :  n_sat = {entry[1]}{marker}")
    print("─" * 52)


if __name__ == "__main__":
    main()
