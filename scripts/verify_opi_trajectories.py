# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The DQI-MCMC Authors

r"""
Run one batch of Gibbs sampling and emit a single rhs_data JSONL line matching
the format used in data/opi/gibbs3_p*_no_warmup.jsonl.

The output line contains:
  - trajectory: [[tau_k, n_satisfied, runtime_s], ...] — each time a new best is reached
 - resume_state.tau_k: step index of first solution above predN
 - resume_state.best_x: x vector (length n) at the best n_satisfied found,
   or null if no steps were executed (steps_run == 0)
 - samples_at_surpass: total_samples when predN was first surpassed

Output goes to stdout (one JSONL line) or --output file (appends).
Progress goes to stderr.

Seed formula (matching reference data): (hash((p, rhs_idx)) & 0x7FFFFFFF) + seed_offset

Usage:
    # p=13, rhs=0, batch-size=500000 -> stdout
    python verify_opi_trajectories.py --p 13 --rhs 0 --batch-size 500000

    # p=13, rhs=0, write to file
    python verify_opi_trajectories.py --p 13 --rhs 0 --output out.jsonl
"""

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dqi_mcmc.api.maxlinsat.block_gibbs_sampler import BlockGibbsSampler
from dqi_mcmc.api.maxlinsat.max_opi_problem import MaxOPIProblem
from scripts import make_per_rhs_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one OPI sampling batch and emit an rhs_data JSONL line.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--p", type=int, required=True, help="Prime modulus")
    parser.add_argument("--rhs", type=int, required=True, help="RHS index (0-based)")
    parser.add_argument(
        "--block-size",
        type=int,
        default=3,
        help="Number of variables updated per Gibbs step (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500_000,
        help="Number of Gibbs steps per batch (default: 500000)",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Added to the deterministic per-RHS seed (default: 0)",
    )
    parser.add_argument(
        "--n-factor",
        type=int,
        default=2,
        help="n = p // n_factor (default: 2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Append result line to this file. Omit to write to stdout.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    p, rhs_idx = args.p, args.rhs
    n = p // args.n_factor
    r = p // 2

    rhs_file = base_dir / "problems" / "opi" / f"rhs_nsamples100_p{p}_r{r}.npy"
    if not rhs_file.exists():
        sys.exit(
            f"RHS file not found: {rhs_file}\n"
            f"Run: python scripts/generate_opi_rhs.py --p {p}"
        )

    solutions_vectors = np.load(rhs_file)
    v = solutions_vectors[rhs_idx]

    problem = MaxOPIProblem(p=p, v=v, num_variables=n)
    m = problem.num_constraints
    predicted = problem.n_predicted()
    predN = int(np.floor(predicted * m))

    # Seed formula matching the reference data files
    seed = make_per_rhs_seed(p, rhs_idx, args.seed_offset)
    rng = np.random.default_rng(seed)

    block_size = min(args.block_size, n, 3)
    sampler = BlockGibbsSampler(problem)

    print(
        f"p={p}, rhs={rhs_idx}, n={n}, r={r}, m={m}, "
        f"predN={predN}, block_size={block_size}, batch_size={args.batch_size}, seed={seed}",
        file=sys.stderr,
    )

    # Run sampling batch, stopping as soon as predN is surpassed
    wall_start = time.perf_counter()
    cpu_start = time.process_time()

    samples, values = sampler.sample_with_values(
        args.batch_size,
        x_0=None,
        block_size=block_size,
        rng=rng,
        stop_above=predN,
    )

    wall_elapsed = time.perf_counter() - wall_start
    cpu_elapsed = time.process_time() - cpu_start
    steps_run = len(samples)
    runtime_per_sample = wall_elapsed / steps_run

    # Reconstruct trajectory: each time n_satisfied improves, record (tau_k, n_sat, runtime)
    # values[:,1] stores fx = 2*n_satisfied - m; convert back to n_satisfied
    best = -1
    best_x = None
    trajectory = []
    tau_k_at_surpass = 0
    samples_at_surpass = steps_run  # default: end of batch if never surpassed
    surpassed_predN = False

    for idx in range(steps_run):
        fx_stored = int(values[idx, 1])
        n_sat = (fx_stored + m) // 2

        if n_sat > best:
            best = n_sat
            best_x = samples[idx].copy()
            runtime_k = runtime_per_sample * (idx + 1)
            trajectory.append([idx + 1, int(n_sat), round(runtime_k, 2)])

            print(f"  step {idx+1}: new best n_satisfied={n_sat}/{m}", file=sys.stderr)

            if n_sat > predN and not surpassed_predN:
                surpassed_predN = True
                tau_k_at_surpass = idx + 1
                samples_at_surpass = idx + 1

    if not surpassed_predN:
        print(
            f"  WARNING: predN={predN} not surpassed in {args.batch_size} steps "
            f"(best={best})",
            file=sys.stderr,
        )

    # Build output line matching gibbs3_p*_no_warmup.jsonl format
    resume_state = {
        "total_samples": steps_run,
        "tau_k": tau_k_at_surpass,
        "x_0": samples[-1].tolist(),
        "best": int(best),
        "best_x": best_x.tolist() if best_x is not None else None,
        "runtime": round(wall_elapsed, 2),
        "cpu_time": round(cpu_elapsed, 2),
        "wall_cpu_ratio": (
            round(wall_elapsed / cpu_elapsed, 3) if cpu_elapsed > 0 else 1.0
        ),
        "trajectory": trajectory,
    }

    timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    line_dict = {
        "type": "rhs_data",
        "rhs_idx": rhs_idx,
        "trajectory": trajectory,
        "resume_state": resume_state,
        "surpassed_predN": surpassed_predN,
        "samples_at_surpass": samples_at_surpass,
        "timestamp": timestamp,
    }

    line = json.dumps(line_dict)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "a") as f:
            f.write(line + "\n")
        print(f"Result appended to {out_path}", file=sys.stderr)
    else:
        print(line)


if __name__ == "__main__":
    main()
