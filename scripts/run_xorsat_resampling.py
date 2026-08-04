# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The DQI-MCMC Authors

r"""
Run XOR-SAT resampling experiments for one (m, rhs-index, algorithm) combination.

Implements two resampling algorithms:
  Algorithm 1 (Continuous): Keep sampling from x_0=random until the predicted threshold is
      first reached, recording τ\_DQI and three observables at that state.
  Algorithm 2 (Restart):    Collect N good samples; each attempt starts fresh from a random
      state and each successive attempt seeds its RNG with base_seed + attempt + 1.

A "good sample" is a state x where n_satisfied(x) / m ≥ predicted_fraction.

Appends one result line per call to a JSONL output file.

Usage:
    # Algorithm 1, m=250, rhs-index=0
    python run_xorsat_resampling.py --m 250 --rhs-index 0 --algorithm 1 \\
        --ell 12 --block-size 3 --output results_m250_alg1.jsonl

    # Algorithm 2, m=100, rhs-index=3 (10 good samples)
    python run_xorsat_resampling.py --m 100 --rhs-index 3 --algorithm 2 \\
        --ell 4 --block-size 3 --num-good-samples 10 --output results_m100_alg2.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter as now

import numpy as np

# Allow running as a script directly from within paper-release
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dqi_mcmc.api.block_gibbs_sampler import BlockGibbsSamplerOptimized
from dqi_mcmc.api.maxxorsat.max_xor_p_sat_problem import MaxXorPSatProblem
from dqi_mcmc.api.maxxorsat.xorsat_problem_reader import read_problem
from scripts import make_per_rhs_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run XOR-SAT resampling to collect τ_DQI values.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-m", "--m", type=int, required=True, help="Problem size")
    parser.add_argument("-i", "--rhs-index", type=int, required=True, help="RHS index (0-99)")
    parser.add_argument(
        "-a",
        "--algorithm",
        type=int,
        choices=[1, 2],
        required=True,
        help="Algorithm: 1=Continuous, 2=Restart",
    )
    parser.add_argument("-l", "--ell", type=int, required=True, help="Polynomial degree")
    parser.add_argument("--block-size", type=int, default=3, help="Gibbs block size (default: 3)")
    parser.add_argument(
        "-g",
        "--num-good-samples",
        type=int,
        default=10,
        help="(Alg 2 only) number of good samples to collect per call (default: 10)",
    )
    parser.add_argument(
        "-s",
        "--seed-offset",
        type=int,
        default=0,
        help="Seed offset for reproducibility (default: 0)",
    )
    parser.add_argument(
        "-x",
        "--max-samples",
        type=int,
        default=100_000_000,
        help="Safety limit: maximum Gibbs steps (default: 100M)",
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="JSONL output file (appended)"
    )
    return parser.parse_args()


def _run_until_predicted(
    problem: MaxXorPSatProblem,
    rng: np.random.Generator,
    block_size: int,
    predicted: float,
    max_samples: int,
    batch: int = 100_000,
) -> tuple[int, dict, list] | None:
    """
    Run Gibbs sampling until n_satisfied/m >= predicted.

    Returns ``(tau_dqi, observables, trajectory)`` or ``None`` if ``max_samples`` is exhausted.
    The *trajectory* is a list of [tau, n_satisfied, runtime] records at every new-best step.
    """
    m = problem.num_constraints
    sampler = BlockGibbsSamplerOptimized(problem)

    x = rng.integers(2, size=sampler.x_length, dtype=np.int8)
    bx = ((x @ problem.b.T) % problem.p).astype(np.int64)
    n_sat = int(np.sum(bx == problem.v))
    fx_val = 2 * n_sat - m
    value = problem.log_poly_p2(fx_val)

    tau = 0
    best = n_sat
    trajectory = [[0, n_sat, 0.0]]
    t0 = now()

    while tau < max_samples:
        run = min(batch, max_samples - tau)
        for _ in range(run):
            x, bx, value, fx_val = sampler._one_step_optimized(
                rng, x, bx, value, fx_val, block_size=block_size, block_strategy="random"
            )
            tau += 1
            n_sat = int((fx_val + m) // 2)
            if n_sat > best:
                best = n_sat
                trajectory.append([tau, n_sat, round(now() - t0, 6)])
            if n_sat / m >= predicted:
                obs = {
                    "fx": float(problem.f(x)),
                    "wt_x": int(np.sum(x)),
                    "wt_bx": int(np.sum(problem.bx(x))),
                }
                return tau, obs, trajectory

    return None


def main():
    args = parse_args()
    m = args.m
    rhs_idx = args.rhs_index
    p = 2  # XOR-SAT is binary

    base_seed = make_per_rhs_seed(p, rhs_idx, args.seed_offset)
    print(
        f"XOR-SAT resampling: m={m}, rhs_idx={rhs_idx}, "
        f"algorithm={args.algorithm}, ell={args.ell}, seed={base_seed}"
    )

    b, v = read_problem(m, v_index=rhs_idx)
    problem = MaxXorPSatProblem(b=b, v=v, p=p, ell=args.ell)
    predicted = problem.n_predicted()
    n = problem.num_variables

    print(f"  n={n}, m={m}, predicted={predicted:.6f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.algorithm == 1:
        rng = np.random.default_rng(base_seed)
        result = _run_until_predicted(problem, rng, args.block_size, predicted, args.max_samples)
        if result is None:
            print(f"  WARNING: max_samples={args.max_samples} exhausted without reaching predicted")
            return

        tau_dqi, obs, trajectory = result
        print(f"  tau_dqi={tau_dqi}, fx={obs['fx']}, wt_x={obs['wt_x']}, wt_bx={obs['wt_bx']}")

        record = {
            "type": "rhs_data",
            "rhs_idx": rhs_idx,
            "algorithm": 1,
            "ell": args.ell,
            "block_size": args.block_size,
            "seed": base_seed,
            "seed_offset": args.seed_offset,
            "tau_dqi": tau_dqi,
            "observables": obs,
            "trajectory": trajectory,
        }

    else:  # algorithm 2
        records_obs = []
        records_traj = []
        for attempt in range(args.num_good_samples):
            iter_seed = base_seed + attempt + 1
            rng = np.random.default_rng(iter_seed)
            result = _run_until_predicted(
                problem, rng, args.block_size, predicted, args.max_samples
            )
            if result is None:
                print(f"  WARNING: attempt {attempt + 1} exhausted max_samples without success")
                records_obs.append(None)
                records_traj.append([])
            else:
                tau_dqi_i, obs_i, traj_i = result
                print(
                    f"  attempt {attempt + 1}/{args.num_good_samples}: "
                    f"tau_dqi={tau_dqi_i}, fx={obs_i['fx']}"
                )
                records_obs.append(obs_i)
                records_traj.append(traj_i)

        record = {
            "type": "rhs_data",
            "rhs_idx": rhs_idx,
            "algorithm": 2,
            "ell": args.ell,
            "block_size": args.block_size,
            "seed": base_seed,
            "seed_offset": args.seed_offset,
            "num_good_samples": args.num_good_samples,
            "observables": records_obs,
            "trajectories": records_traj,
        }

    with open(output_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"  Result appended to: {output_path}")


if __name__ == "__main__":
    main()
