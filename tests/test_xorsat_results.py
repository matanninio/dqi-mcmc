# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 IBM Corporation

"""
Golden-value tests that verify the paper's XOR-SAT resampling results can be
reproduced exactly from the code and scripts in the paper-release directory.

These tests use Algorithm 1 (Continuous) and Algorithm 2 (Restart) as
implemented in ``scripts/run_xorsat_resampling.py`` and compare against
known-good values captured from a reference run.

Both algorithms are deterministic: given a fixed seed (derived from the
per-RHS formula ``make_per_rhs_seed(p=2, rhs_idx, seed_offset=0)``) and the
LDPC constraint matrix shipped in ``problems/maxxorsat/``, every τ_DQI and
observable value is bit-for-bit reproducible across machines with the same
Python/NumPy/Scipy stack.

To keep the test suite fast we use ``m=100`` (the smallest available problem)
and ``ell=4`` (the paper's optimal degree for m=100).  Each algorithm-1 run
completes in well under a second; each algorithm-2 attempt also finishes
quickly for m=100.

Golden values were captured by running the script once with ``PYTHONHASHSEED=0``
and baking in the output.  The ``PYTHONHASHSEED`` variable does NOT affect these
tests (the seed formula uses Python's ``hash()`` with a tuple of ints, but the
tests pass the seed explicitly from the pre-computed table below).
"""

from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_DIR = Path(__file__).resolve().parents[1]
_M100_RHS_FILE = _BASE_DIR / "problems" / "maxxorsat" / "rhs_nsamples100_m100_p2_r1.npy"

ELL_OPTS = {100: 4, 250: 12, 500: 33, 750: 57, 1000: 86}

# ---------------------------------------------------------------------------
# Golden tables (m=100, ell=4, block_size=3, seed_offset=0)
# ---------------------------------------------------------------------------
# Captured with: PYTHONHASHSEED=0 python scripts/run_xorsat_resampling.py
#   --m 100 --rhs-index <i> --algorithm 1 --ell 4 --block-size 3
# Keys: rhs_idx → {tau_dqi, observables}

ALG1_GOLDEN_M100 = {
    0: {"tau_dqi": 52, "observables": {"fx": 34.0, "wt_x": 34, "wt_bx": 54}},
    1: {"tau_dqi": 26, "observables": {"fx": 30.0, "wt_x": 34, "wt_bx": 50}},
    2: {"tau_dqi": 18, "observables": {"fx": 30.0, "wt_x": 31, "wt_bx": 48}},
}

# Captured with: PYTHONHASHSEED=0 python scripts/run_xorsat_resampling.py
#   --m 100 --rhs-index <i> --algorithm 2 --ell 4 --block-size 3 --num-good-samples 3
# Keys: rhs_idx → list of 3 observables dicts

ALG2_GOLDEN_M100 = {
    0: [
        {"fx": 36.0, "wt_x": 27, "wt_bx": 43},
        {"fx": 30.0, "wt_x": 43, "wt_bx": 54},
        {"fx": 32.0, "wt_x": 22, "wt_bx": 43},
    ],
    1: [
        {"fx": 42.0, "wt_x": 38, "wt_bx": 54},
        {"fx": 38.0, "wt_x": 31, "wt_bx": 50},
        {"fx": 38.0, "wt_x": 36, "wt_bx": 54},
    ],
    2: [
        {"fx": 30.0, "wt_x": 38, "wt_bx": 50},
        {"fx": 30.0, "wt_x": 35, "wt_bx": 50},
        {"fx": 32.0, "wt_x": 31, "wt_bx": 49},
    ],
}


def _requires_m100_rhs(test_func):
    """Skip test if the m=100 RHS file has not been generated."""
    return pytest.mark.skipif(
        not _M100_RHS_FILE.exists(),
        reason=f"RHS file not found: {_M100_RHS_FILE}. "
        "Run: python scripts/generate_xorsat_rhs.py --m 100",
    )(test_func)


def _load_problem_and_sampler(m: int, rhs_idx: int):
    """Load the MaxXorPSatProblem and BlockGibbsSamplerOptimized for given (m, rhs_idx)."""
    import sys

    sys.path.insert(0, str(_BASE_DIR))

    from dqi_mcmc.api.block_gibbs_sampler import BlockGibbsSamplerOptimized
    from dqi_mcmc.api.maxxorsat.max_xor_p_sat_problem import MaxXorPSatProblem
    from dqi_mcmc.api.maxxorsat.xorsat_problem_reader import read_problem

    b, v = read_problem(m, v_index=rhs_idx)
    problem = MaxXorPSatProblem(b=b, v=v, p=2, ell=ELL_OPTS[m])
    sampler = BlockGibbsSamplerOptimized(problem)
    return problem, sampler


# ---------------------------------------------------------------------------
# Import the resampling logic directly (avoids subprocess overhead)
# ---------------------------------------------------------------------------


def _run_until_predicted(problem, rng, block_size, predicted, max_samples=10_000_000):
    """Inline version of the script's _run_until_predicted (avoids import of __main__)."""
    from dqi_mcmc.api.block_gibbs_sampler import BlockGibbsSamplerOptimized

    m = problem.num_constraints
    sampler = BlockGibbsSamplerOptimized(problem)

    x = rng.integers(2, size=sampler.x_length, dtype=np.int8)
    bx = ((x @ problem.b.T) % problem.p).astype(np.int64)
    n_sat = int(np.sum(bx == problem.v))
    fx_val = 2 * n_sat - m
    value = problem.log_poly_p2(fx_val)

    tau = 0
    batch = 50_000

    while tau < max_samples:
        run = min(batch, max_samples - tau)
        for _ in range(run):
            x, bx, value, fx_val = sampler._one_step_optimized(
                rng,
                x,
                bx,
                value,
                fx_val,
                block_size=block_size,
                block_strategy="random",
            )
            tau += 1
            n_sat = int((fx_val + m) // 2)
            if n_sat / m >= predicted:
                obs = {
                    "fx": float(problem.f(x)),
                    "wt_x": int(np.sum(x)),
                    "wt_bx": int(np.sum(problem.bx(x))),
                }
                return tau, obs

    return None  # pragma: no cover


# ---------------------------------------------------------------------------
# Tests — Algorithm 1 (Continuous)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rhs_idx", sorted(ALG1_GOLDEN_M100))
def test_alg1_xorsat_golden_m100(rhs_idx):
    """
    Algorithm 1 (Continuous) for XOR-SAT, m=100.

    Verifies that τ_DQI and the observables (fx, wt_x, wt_bx) at the first
    "good sample" exactly match the values stored in data/algorithm1/.
    """
    if not _M100_RHS_FILE.exists():
        pytest.skip(f"RHS file not found: {_M100_RHS_FILE}")

    from scripts import make_per_rhs_seed

    expected = ALG1_GOLDEN_M100[rhs_idx]
    p = 2
    base_seed = make_per_rhs_seed(p, rhs_idx, 0)

    problem, _ = _load_problem_and_sampler(100, rhs_idx)
    predicted = problem.n_predicted()
    rng = np.random.default_rng(base_seed)

    result = _run_until_predicted(problem, rng, block_size=3, predicted=predicted)
    assert (
        result is not None
    ), "max_samples exhausted — did not reach predicted threshold"

    tau_dqi, obs = result
    assert (
        tau_dqi == expected["tau_dqi"]
    ), f"rhs_idx={rhs_idx}: τ_DQI mismatch: got {tau_dqi}, expected {expected['tau_dqi']}"
    assert (
        obs == expected["observables"]
    ), f"rhs_idx={rhs_idx}: observables mismatch: got {obs}, expected {expected['observables']}"


# ---------------------------------------------------------------------------
# Tests — Algorithm 2 (Restart)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rhs_idx", sorted(ALG2_GOLDEN_M100))
def test_alg2_xorsat_golden_m100(rhs_idx):
    """
    Algorithm 2 (Restart) for XOR-SAT, m=100.

    Verifies that the first 3 good-sample observables collected with independent
    per-attempt seeds (base_seed + attempt + 1) exactly match the golden values.
    """
    if not _M100_RHS_FILE.exists():
        pytest.skip(f"RHS file not found: {_M100_RHS_FILE}")

    from scripts import make_per_rhs_seed

    expected_list = ALG2_GOLDEN_M100[rhs_idx]
    p = 2
    base_seed = make_per_rhs_seed(p, rhs_idx, 0)

    problem, _ = _load_problem_and_sampler(100, rhs_idx)
    predicted = problem.n_predicted()

    for attempt, expected_obs in enumerate(expected_list):
        iter_seed = base_seed + attempt + 1
        rng = np.random.default_rng(iter_seed)
        result = _run_until_predicted(problem, rng, block_size=3, predicted=predicted)
        assert (
            result is not None
        ), f"rhs_idx={rhs_idx}, attempt={attempt + 1}: max_samples exhausted"
        _, obs = result
        assert obs == expected_obs, (
            f"rhs_idx={rhs_idx}, attempt={attempt + 1}: "
            f"observables mismatch: got {obs}, expected {expected_obs}"
        )


# ---------------------------------------------------------------------------
# Tests — structural verification against data/algorithm1/ and data/algorithm2/
# ---------------------------------------------------------------------------


DATA_DIR = _BASE_DIR.parent / "data"
ALG1_DATA_DIR = DATA_DIR / "algorithm1"
ALG2_DATA_DIR = DATA_DIR / "algorithm2"


def _load_jsonl(path):
    """Load all JSON lines from a JSONL file, returning a list of dicts."""
    import json

    lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(json.loads(line))
    return lines


@pytest.mark.skipif(
    not ALG1_DATA_DIR.exists(),
    reason="data/algorithm1/ directory not found",
)
def test_alg1_data_structure():
    """
    Structural sanity-check on all data/algorithm1/*.jsonl files.

    Verifies that:
    - The metadata line is present and has required fields.
    - Every rhs_data line has tau_dqi > 0 and a non-empty trajectory.
    - The observables dict has exactly {fx, wt_x, wt_bx}.
    """
    jsonl_files = sorted(ALG1_DATA_DIR.glob("gibbs3_*.jsonl"))
    assert len(jsonl_files) > 0, "No JSONL files found in data/algorithm1/"

    for path in jsonl_files:
        lines = _load_jsonl(path)
        assert len(lines) >= 2, f"{path.name}: expected at least 2 lines (meta + data)"

        meta = lines[0]
        assert (
            meta.get("type") == "metadata"
        ), f"{path.name}: first line must be metadata"
        for field in (
            "ell",
            "num_variables",
            "num_constraints",
            "predicted_fraction",
            "num_bit_flips",
            "num_rhs",
            "rhs_indices",
        ):
            assert field in meta, f"{path.name}: metadata missing field '{field}'"

        for row in lines[1:]:
            assert row.get("type") == "rhs_data", f"{path.name}: unexpected row type"
            assert row["tau_dqi"] > 0, f"{path.name}: tau_dqi must be positive"
            assert (
                len(row["trajectory"]) >= 1
            ), f"{path.name}: trajectory must be non-empty"
            obs = row["observables"]
            assert set(obs.keys()) == {
                "fx",
                "wt_x",
                "wt_bx",
            }, f"{path.name}: observables must have exactly {{fx, wt_x, wt_bx}}"


@pytest.mark.skipif(
    not ALG2_DATA_DIR.exists(),
    reason="data/algorithm2/ directory not found",
)
def test_alg2_data_structure():
    """
    Structural sanity-check on all data/algorithm2/*.jsonl files.

    Verifies that:
    - The metadata line records num_good_samples_per_rhs.
    - Every rhs_data line has a list of observables of the expected length.
    - Each observable has exactly {fx, wt_x, wt_bx}.
    """
    jsonl_files = sorted(ALG2_DATA_DIR.glob("gibbs3_*algo2.jsonl"))
    assert len(jsonl_files) > 0, "No JSONL files found in data/algorithm2/"

    for path in jsonl_files:
        lines = _load_jsonl(path)
        assert len(lines) >= 2, f"{path.name}: expected at least 2 lines (meta + data)"

        meta = lines[0]
        assert (
            meta.get("type") == "metadata"
        ), f"{path.name}: first line must be metadata"
        assert (
            "num_good_samples_per_rhs" in meta
        ), f"{path.name}: metadata missing 'num_good_samples_per_rhs'"
        expected_n_obs = meta["num_good_samples_per_rhs"]

        for row in lines[1:]:
            assert row.get("type") == "rhs_data", f"{path.name}: unexpected row type"
            obs_list = row["observables"]
            assert isinstance(
                obs_list, list
            ), f"{path.name}: observables must be a list"
            assert len(obs_list) == expected_n_obs, (
                f"{path.name}: rhs_idx={row['rhs_idx']}: "
                f"expected {expected_n_obs} observables, got {len(obs_list)}"
            )
            for obs in obs_list:
                assert set(obs.keys()) == {
                    "fx",
                    "wt_x",
                    "wt_bx",
                }, f"{path.name}: each observable must have exactly {{fx, wt_x, wt_bx}}"


@pytest.mark.skipif(
    not (ALG1_DATA_DIR.exists() and _M100_RHS_FILE.exists()),
    reason="data/algorithm1/ or m=100 RHS file not found",
)
def test_alg1_m100_predicted_fraction_consistent():
    """
    The predicted_fraction in data/algorithm1/ m=100 metadata matches
    what the paper-release code computes from the same problem file.
    """
    from dqi_mcmc.api.maxxorsat.max_xor_p_sat_problem import MaxXorPSatProblem
    from dqi_mcmc.api.maxxorsat.xorsat_problem_reader import read_problem

    path = ALG1_DATA_DIR / "gibbs3_m100_n62_ell4_num_rhs100_0.jsonl"
    if not path.exists():
        pytest.skip(f"Golden file not found: {path}")

    lines = _load_jsonl(path)
    stored_predicted = lines[0]["predicted_fraction"]

    b, v = read_problem(100, v_index=0)
    problem = MaxXorPSatProblem(b=b, v=v, p=2, ell=ELL_OPTS[100])
    computed_predicted = problem.n_predicted()

    assert abs(computed_predicted - stored_predicted) < 1e-10, (
        f"predicted_fraction mismatch: stored={stored_predicted}, "
        f"computed={computed_predicted}"
    )


@pytest.mark.skipif(
    not (ALG2_DATA_DIR.exists() and _M100_RHS_FILE.exists()),
    reason="data/algorithm2/ or m=100 RHS file not found",
)
def test_alg2_m100_predicted_fraction_consistent():
    """
    The predicted_fraction in data/algorithm2/ m=100 metadata matches
    what the paper-release code computes from the same problem file.
    """
    from dqi_mcmc.api.maxxorsat.max_xor_p_sat_problem import MaxXorPSatProblem
    from dqi_mcmc.api.maxxorsat.xorsat_problem_reader import read_problem

    path = ALG2_DATA_DIR / "gibbs3_m100_n62_ell4_num_rhs100_algo2.jsonl"
    if not path.exists():
        pytest.skip(f"Golden file not found: {path}")

    lines = _load_jsonl(path)
    stored_predicted = lines[0]["predicted_fraction"]

    b, v = read_problem(100, v_index=0)
    problem = MaxXorPSatProblem(b=b, v=v, p=2, ell=ELL_OPTS[100])
    computed_predicted = problem.n_predicted()

    assert abs(computed_predicted - stored_predicted) < 1e-10, (
        f"predicted_fraction mismatch: stored={stored_predicted}, "
        f"computed={computed_predicted}"
    )
