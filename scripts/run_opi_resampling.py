# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The DQI-MCMC Authors

r"""
Run OPI resampling experiments for one (p, rhs-index, algorithm) combination.

Implements two resampling algorithms:
  Algorithm 1 (Continuous): Keep sampling until collecting N unique good samples.
  Algorithm 2 (Restart):    For each of N iterations, restart from scratch until 1 good sample.

A "good sample" is one where f(x) > predN = floor(n_predicted() * num_constraints).

Appends result lines to a JSONL output file. Writes a metadata line if the output is new.

Usage:
    # Algorithm 1, p=13, rhs-index=0
    python run_opi_resampling.py --p 13 --rhs-index 0 --algorithm 1 \\
        --num-bit-flips 10 --num-good-samples 10 --output results_p13_alg1.jsonl

    # Algorithm 2, p=13, rhs-index=3
    python run_opi_resampling.py --p 13 --rhs-index 3 --algorithm 2 \\
        --num-bit-flips 10 --num-good-samples 10 --output results_p13_alg2.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Add paper-release to path so we can import dqi_mcmc
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dqi_mcmc.api.maxlinsat.block_gibbs_sampler import BlockGibbsSampler
from dqi_mcmc.api.maxlinsat.max_opi_problem import MaxOPIProblem
from scripts import make_per_rhs_seed


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run OPI resampling experiments to collect tau_dqi values.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("-p", "--p", type=int, required=True, help="Prime modulus")
    parser.add_argument(
        "-i", "--rhs-index", type=int, required=True, help="Single RHS index (0-9)"
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        type=int,
        choices=[1, 2],
        required=True,
        help="Algorithm: 1=Continuous, 2=Restart",
    )
    parser.add_argument(
        "--num-bit-flips", type=int, default=10, help="Gibbs block size (default: 10)"
    )
    parser.add_argument(
        "-g",
        "--num-good-samples",
        type=int,
        default=10,
        help="Number of good samples to collect (default: 10)",
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
        "--max-samples-per-attempt",
        type=int,
        default=10_000_000,
        help="Maximum samples per attempt before giving up (default: 10M)",
    )
    parser.add_argument(
        "-F", "--n-factor", type=int, default=2, help="n = p // n_factor (default: 2)"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="JSONL output file (appends). Defaults to stdout if not given.",
    )

    return parser.parse_args()


def load_problem(p: int, n_factor: int, rhs_idx: int, base_dir: Path) -> tuple:
    """Load problem and RHS from disk. Returns (problem, sampler, solutions_vectors, predN)."""
    n = p // n_factor
    r = p // 2

    # Load RHS file
    rhs_file = base_dir / "problems" / "opi" / f"rhs_nsamples100_p{p}_r{r}.npy"
    if not rhs_file.exists():
        raise FileNotFoundError(
            f"RHS file not found for p={p}, r={r}.\n"
            f"Run: python scripts/generate_opi_rhs.py --p {p}"
        )
    solutions_vectors = np.load(rhs_file)

    # Create problem instance with the first RHS to set up the structure
    v = solutions_vectors[rhs_idx]
    problem = MaxOPIProblem(p=p, v=v, num_variables=n)
    m = problem.num_constraints

    # Compute predicted fraction and threshold
    predicted = problem.n_predicted()
    predN = int(np.floor(predicted * m))

    return problem, solutions_vectors, m, predicted, predN


def initialize_state(
    sampler: BlockGibbsSampler, problem: MaxOPIProblem, x: np.ndarray, rng=None
):
    """Initialize sampling state variables."""
    poly_vals = sampler._compute_poly_vals(x)
    n_satisfied = np.sum(sampler.v_mask[np.arange(sampler.m), poly_vals])
    fx = 2 * n_satisfied - sampler.m
    value = float(problem.log_poly_p2(fx=fx))
    return x, poly_vals, value, fx


def run_algorithm_1_continuous(
    rhs_idx: int,
    problem: MaxOPIProblem,
    solutions_vectors: np.ndarray,
    sampler: BlockGibbsSampler,
    rng: np.random.Generator,
    predN: int,
    num_good_samples: int,
    block_size: int,
    max_samples_per_attempt: int,
) -> list[dict]:
    """
    Algorithm 1 (Continuous): Keep sampling until collecting N UNIQUE good samples.

    Returns:
        List of dicts with tau_dqi, trajectory, and best_x for each unique good sample.

    """
    # Update problem with this RHS
    problem.v = solutions_vectors[rhs_idx]
    # Update sampler's cached v and v_mask
    sampler.v = problem.v
    sampler.v_mask = np.zeros((sampler.m, sampler.p), dtype=np.int8)
    for i in range(sampler.m):
        sampler.v_mask[i, problem.v[i]] = 1

    results = []
    total_samples = 0
    unique_solutions = set()  # Track unique solutions as tuples

    # Trajectory tracking (shared across the whole continuous run)
    best_n_sat = -1
    best_x = None
    trajectory: list[list] = []  # [[step, n_satisfied, runtime_s], ...]

    # Initialize state (no warmup — use fresh random start)
    x = rng.integers(0, sampler.p, size=sampler.x_length, dtype=np.int64)
    x, poly_vals, value, fx = initialize_state(sampler, problem, x, rng)

    print(
        f"  Algorithm 1 (Continuous): Collecting {num_good_samples} UNIQUE good samples..."
    )

    # Start timing from the beginning - DO NOT RESET during the run
    cpu_time_start = time.process_time()
    wall_time_start = time.time()

    while (
        len(unique_solutions) < num_good_samples
        and total_samples < max_samples_per_attempt
    ):
        # Take one Gibbs step
        x, poly_vals, value, fx = sampler._one_step(
            rng=rng,
            x=x,
            poly_vals=poly_vals,
            value=value,
            fx_old=fx,
            block_size=block_size,
            block_strategy="permutation",
        )
        total_samples += 1

        # Check if this is a good sample.
        # fx = 2*n_satisfied - m, so n_satisfied = (fx + m) // 2.
        # The threshold predN is in units of n_satisfied, so convert before comparing.
        n_satisfied = (fx + sampler.m) // 2

        # Track trajectory: record whenever n_satisfied reaches a new best
        if n_satisfied > best_n_sat:
            best_n_sat = n_satisfied
            best_x = x.copy()
            wall_time_elapsed = time.time() - wall_time_start
            trajectory.append(
                [total_samples, int(n_satisfied), round(wall_time_elapsed, 2)]
            )

        if n_satisfied > predN:
            x_tuple = tuple(x.tolist())

            # Only record if this is a NEW unique solution
            if x_tuple not in unique_solutions:
                unique_solutions.add(x_tuple)

                # Record CUMULATIVE timing (from start of run, not since last good sample)
                cpu_time_elapsed = time.process_time() - cpu_time_start
                wall_time_elapsed = time.time() - wall_time_start

                tau_dqi = total_samples
                assert best_x is not None  # always set before n_satisfied > predN

                results.append(
                    {
                        "rhs_idx": int(rhs_idx),
                        "algorithm": 1,
                        "tau_dqi": int(tau_dqi),
                        "f_value": int(n_satisfied),
                        "sample_number": len(unique_solutions),
                        "trajectory": list(trajectory),
                        "best_x": best_x.tolist(),
                        "x": x.tolist(),
                        "cpu_time": float(cpu_time_elapsed),
                        "wall_time": float(wall_time_elapsed),
                    }
                )

                print(
                    f"    Unique good sample {len(unique_solutions)}/{num_good_samples}: "
                    f"tau_dqi={tau_dqi}, f={n_satisfied}, cpu={cpu_time_elapsed:.3f}s"
                )

    if len(unique_solutions) < num_good_samples:
        print(
            f"  WARNING: Only collected {len(unique_solutions)}/{num_good_samples} "
            f"unique good samples after {total_samples} samples"
        )

    return results


def run_algorithm_2_restart(
    rhs_idx: int,
    problem: MaxOPIProblem,
    solutions_vectors: np.ndarray,
    sampler: BlockGibbsSampler,
    base_seed: int,
    predN: int,
    num_good_samples: int,
    block_size: int,
    max_samples_per_attempt: int,
) -> list[dict]:
    """
    Algorithm 2 (Restart): For each iteration, restart from scratch until getting 1 good sample.

    Each iteration uses an independent RNG seeded with ``base_seed + iteration + 1``.

    Returns:
        List of dicts with tau_dqi, trajectory, and best_x for each good sample.

    """
    # Update problem with this RHS
    problem.v = solutions_vectors[rhs_idx]
    # Update sampler's cached v and v_mask
    sampler.v = problem.v
    sampler.v_mask = np.zeros((sampler.m, sampler.p), dtype=np.int8)
    for i in range(sampler.m):
        sampler.v_mask[i, problem.v[i]] = 1

    results = []

    print(
        f"  Algorithm 2 (Restart): Collecting {num_good_samples} good samples with restarts..."
    )

    # Algorithm 2 design:
    # - Each restart uses a fresh RNG seeded with base_seed + iteration + 1.
    # - The sampler's _permutation/_permutation_idx is NOT reset between restarts;
    #   it carries over from the previous full-batch run.
    # - We run all batch_size steps unconditionally; tau_dqi is the step index of
    #   the first good sample found in the batch output.
    batch_size = 50_000

    for iteration in range(num_good_samples):
        # Each restart gets its own independent RNG.
        iteration_seed = base_seed + iteration + 1
        rng = np.random.default_rng(iteration_seed)
        # DO NOT reset _permutation — it must carry over from the previous batch.

        # Restart: new random initialization (no warmup)
        x = rng.integers(0, sampler.p, size=sampler.x_length, dtype=np.int64)

        # Initialize state variables
        x, poly_vals, value, fx = initialize_state(sampler, problem, x, rng)

        # Per-restart trajectory tracking
        best_n_sat = -1
        best_x = None
        trajectory: list[list] = []  # [[step, n_satisfied, runtime_s], ...]

        # Start timing for this restart iteration
        cpu_time_start = time.process_time()
        wall_time_start = time.time()

        first_hit: dict | None = None  # recorded at the first step that exceeds predN

        # Run the full batch unconditionally (matches reference batch_size=50_000 behaviour).
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

            # fx = 2*n_satisfied - m, so n_satisfied = (fx + m) // 2.
            n_satisfied = (fx + sampler.m) // 2

            # Track trajectory: record whenever n_satisfied reaches a new best
            if n_satisfied > best_n_sat:
                best_n_sat = n_satisfied
                best_x = x.copy()
                wall_time_elapsed = time.time() - wall_time_start
                trajectory.append([step, int(n_satisfied), round(wall_time_elapsed, 2)])

            # Record only the FIRST hit; keep running to advance permutation state.
            if first_hit is None and n_satisfied > predN:
                cpu_time_elapsed = time.process_time() - cpu_time_start
                wall_time_elapsed = time.time() - wall_time_start
                assert best_x is not None
                first_hit = {
                    "rhs_idx": int(rhs_idx),
                    "algorithm": 2,
                    "tau_dqi": int(step),
                    "f_value": int(n_satisfied),
                    "sample_number": iteration + 1,
                    "trajectory": list(trajectory),
                    "best_x": best_x.tolist(),
                    "x": x.tolist(),
                    "cpu_time": float(cpu_time_elapsed),
                    "wall_time": float(wall_time_elapsed),
                }

        if first_hit is not None:
            results.append(first_hit)
            print(
                f"    Restart {iteration + 1}/{num_good_samples}: "
                f"tau_dqi={first_hit['tau_dqi']}, f={first_hit['f_value']}, "
                f"cpu={first_hit['cpu_time']:.3f}s"
            )
        else:
            print(
                f"  WARNING: Restart {iteration + 1} failed to find good sample "
                f"in {batch_size} samples"
            )

    return results


def main():
    """Main entry point."""
    args = parse_args()

    p = args.p
    rhs_idx = args.rhs_index
    base_dir = Path(__file__).resolve().parents[1]

    # Deterministic per-RHS seed formula (same as paper)
    base_seed = make_per_rhs_seed(p, rhs_idx, args.seed_offset)
    rng = np.random.default_rng(base_seed)

    print(
        f"OPI resampling: p={p}, rhs_idx={rhs_idx}, algorithm={args.algorithm}, seed={base_seed}"
    )

    # Load problem
    problem, solutions_vectors, m, predicted, predN = load_problem(
        p, args.n_factor, rhs_idx, base_dir
    )
    n = problem.num_variables
    r = problem.r

    print(f"  p={p}, n={n}, r={r}, m={m}, predicted={predicted:.4f}, predN={predN}")

    # Effective block size: cannot exceed n (num_variables) or 3 (max supported by sampler)
    effective_block_size = min(args.num_bit_flips, n, 3)
    if effective_block_size != args.num_bit_flips:
        print(
            f"  Note: block_size capped at {effective_block_size} (num_bit_flips={args.num_bit_flips}, n={n}, max=3)"
        )

    # Build sampler
    sampler = BlockGibbsSampler(problem)

    # Determine output destination: file or stdout
    output_path = Path(args.output) if args.output is not None else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    def write_line(line: str) -> None:
        if output_path is not None:
            with open(output_path, "a") as f:
                f.write(line + "\n")
        else:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    # Write metadata line if this is a new file (or always for stdout)
    is_new_file = output_path is None or not output_path.exists()
    if is_new_file:
        metadata = {
            "type": "metadata",
            "p": p,
            "n": n,
            "r": r,
            "num_constraints": m,
            "predicted_fraction": predicted,
            "predN": predN,
            "num_bit_flips": args.num_bit_flips,
            "sampler_type": "BlockGibbsSampler",
            "algorithm": args.algorithm,
            "seed_formula": "(hash((p, rhs_idx)) & 0x7FFFFFFF) + seed_offset",
        }
        write_line(json.dumps(metadata))
        if output_path is not None:
            print(f"  Wrote metadata to: {output_path}")

    # Run algorithm
    if args.algorithm == 1:
        results = run_algorithm_1_continuous(
            rhs_idx=rhs_idx,
            problem=problem,
            solutions_vectors=solutions_vectors,
            sampler=sampler,
            rng=rng,
            predN=predN,
            num_good_samples=args.num_good_samples,
            block_size=effective_block_size,
            max_samples_per_attempt=args.max_samples_per_attempt,
        )
    else:
        results = run_algorithm_2_restart(
            rhs_idx=rhs_idx,
            problem=problem,
            solutions_vectors=solutions_vectors,
            sampler=sampler,
            base_seed=base_seed,
            predN=predN,
            num_good_samples=args.num_good_samples,
            block_size=effective_block_size,
            max_samples_per_attempt=args.max_samples_per_attempt,
        )

    # Write all result lines
    for result in results:
        write_line(json.dumps(result))

    if output_path is not None:
        print(f"  {len(results)} result(s) appended to: {output_path}")
    else:
        print(f"  {len(results)} result(s) written to stdout", file=sys.stderr)


if __name__ == "__main__":
    main()
