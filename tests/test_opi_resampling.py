# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The DQI-MCMC Authors

"""Tests for OPI resampling algorithms (Algorithm 1 and Algorithm 2)."""

import numpy as np
import pytest

# Golden tau_dqi values for Algorithm 1 and 2, reproduced from the paper data files:
#   data/opi/multisampling_gibbs3_p11_alg1_no_warmup.jsonl  (rhs_idx=0, samples 1-3)
#   data/opi/multisampling_gibbs3_p11_alg2_no_warmup.jsonl  (rhs_idx=0, samples 1-3)
#
# Parameters: p=11, n=5, r=5, m=10, predN=9, rhs_idx=0, block_size=3.
# Seed: make_per_rhs_seed(11, 0, 0) == 757692519  (requires PYTHONHASHSEED=0;
#       confirmed by iteration_seed=757692520 in the Alg2 data file).
_P = 11
_N_FACTOR = 2
_N = _P // _N_FACTOR  # 5
_R = _P // 2  # 5
_BASE_SEED = 757692519  # make_per_rhs_seed(11, 0, 0) with PYTHONHASHSEED=0

EXPECTED_ALG1_TAU_DQI = [11, 21, 46]
EXPECTED_ALG2_TAU_DQI = [129, 17, 31]


def _make_problem_and_sampler(p11_rhs_file, rhs_idx=0):
    """Helper: load p=11 problem from disk."""
    from dqi_mcmc.api.maxlinsat.block_gibbs_sampler import BlockGibbsSampler
    from dqi_mcmc.api.maxlinsat.max_opi_problem import MaxOPIProblem

    solutions_vectors = np.load(p11_rhs_file)
    v = solutions_vectors[rhs_idx]
    problem = MaxOPIProblem(p=_P, v=v, num_variables=_N, use_cache=False)
    m = problem.num_constraints
    predicted = problem.n_predicted()
    predN = int(np.floor(predicted * m))
    sampler = BlockGibbsSampler(problem)
    return problem, sampler, solutions_vectors, predN


class TestOpiResamplingSmoke:
    """Smoke tests: call the algorithms with tiny inputs and verify result shape."""

    def test_alg1_smoke(self, p11_rhs_file):
        """Algorithm 1 returns a list with the expected keys."""
        from scripts.run_opi_resampling import run_algorithm_1_continuous

        problem, sampler, solutions_vectors, predN = _make_problem_and_sampler(p11_rhs_file)

        results = run_algorithm_1_continuous(
            rhs_idx=0,
            problem=problem,
            solutions_vectors=solutions_vectors,
            sampler=sampler,
            rng=np.random.default_rng(1),
            predN=predN,
            num_good_samples=2,
            block_size=3,
            max_samples_per_attempt=500_000,
        )
        assert isinstance(results, list)
        if results:
            r = results[0]
            assert "tau_dqi" in r
            assert "f_value" in r
            assert r["algorithm"] == 1

    def test_alg2_smoke(self, p11_rhs_file):
        """Algorithm 2 returns a list with the expected keys."""
        from scripts.run_opi_resampling import run_algorithm_2_restart

        problem, sampler, solutions_vectors, predN = _make_problem_and_sampler(p11_rhs_file)

        results = run_algorithm_2_restart(
            rhs_idx=0,
            problem=problem,
            solutions_vectors=solutions_vectors,
            sampler=sampler,
            base_seed=2,
            predN=predN,
            num_good_samples=2,
            block_size=3,
            max_samples_per_attempt=500_000,
        )
        assert isinstance(results, list)
        if results:
            r = results[0]
            assert "tau_dqi" in r
            assert r["algorithm"] == 2


class TestOpiResamplingGolden:
    """Golden-value tests: reproduce the first three entries from the paper data files."""

    @pytest.mark.requires_p11_rhs_file
    def test_alg1_golden(self, p11_rhs_file):
        """Algorithm 1 tau_dqi values must reproduce data/opi/multisampling_gibbs3_p11_alg1_no_warmup.jsonl."""
        from scripts.run_opi_resampling import run_algorithm_1_continuous

        problem, sampler, solutions_vectors, predN = _make_problem_and_sampler(p11_rhs_file)
        results = run_algorithm_1_continuous(
            rhs_idx=0,
            problem=problem,
            solutions_vectors=solutions_vectors,
            sampler=sampler,
            rng=np.random.default_rng(_BASE_SEED),
            predN=predN,
            num_good_samples=3,
            block_size=3,
            max_samples_per_attempt=500,
        )
        tau_dqi_values = [r["tau_dqi"] for r in results]
        assert tau_dqi_values == EXPECTED_ALG1_TAU_DQI

    @pytest.mark.requires_p11_rhs_file
    def test_alg2_golden(self, p11_rhs_file):
        """Algorithm 2 tau_dqi values must reproduce data/opi/multisampling_gibbs3_p11_alg2_no_warmup.jsonl."""
        from scripts.run_opi_resampling import run_algorithm_2_restart

        problem, sampler, solutions_vectors, predN = _make_problem_and_sampler(p11_rhs_file)
        results = run_algorithm_2_restart(
            rhs_idx=0,
            problem=problem,
            solutions_vectors=solutions_vectors,
            sampler=sampler,
            base_seed=_BASE_SEED,
            predN=predN,
            num_good_samples=3,
            block_size=3,
            max_samples_per_attempt=500_000,
        )
        tau_dqi_values = [r["tau_dqi"] for r in results]
        assert tau_dqi_values == EXPECTED_ALG2_TAU_DQI
