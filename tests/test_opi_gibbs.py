# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The DQI-MCMC Authors

"""Tests for the OPI Block Gibbs sampler."""

import numpy as np

# Golden f(x) trajectory for 20 samples, seed=42, block_size=1.
# Captured by running BlockGibbsSampler.sample_with_values on tiny_opi_problem
# (p=7, num_variables=3, ell=2, v built with rng seed=1).
# Updated after the diagonal fix (diag = np.arange(ell+1)*d instead of np.full(...,d) with diag[0]=0).
EXPECTED_OPI_FX = [2, 2, 2, 2, 2, 2, 0, 2, 0, 2, 0, 2, 2, 2, 4, 4, 2, 2, 2, 4]


class TestOpiGibbsSmoke:
    def test_smoke(self, tiny_opi_problem):
        """50 samples → correct shapes."""
        from dqi_mcmc.api.maxlinsat.block_gibbs_sampler import BlockGibbsSampler

        sampler = BlockGibbsSampler(tiny_opi_problem)
        samples, values = sampler.sample_with_values(
            50,
            rng=np.random.default_rng(0),
            block_size=1,
            block_strategy="permutation",
        )
        n = tiny_opi_problem.num_variables
        assert samples.shape == (50, n), f"Expected (50, {n}), got {samples.shape}"
        assert values.shape == (50, 2), f"Expected (50, 2), got {values.shape}"

    def test_seed_consistency(self, tiny_opi_problem):
        """Same rng seed → identical f(x) trajectories."""
        from dqi_mcmc.api.maxlinsat.block_gibbs_sampler import BlockGibbsSampler

        sampler = BlockGibbsSampler(tiny_opi_problem)
        _, v1 = sampler.sample_with_values(
            30,
            rng=np.random.default_rng(42),
            block_size=1,
            block_strategy="permutation",
        )
        _, v2 = sampler.sample_with_values(
            30,
            rng=np.random.default_rng(42),
            block_size=1,
            block_strategy="permutation",
        )
        np.testing.assert_array_equal(v1[:, 1], v2[:, 1])

    def test_golden_fx_values(self, tiny_opi_problem):
        """20 samples seed=42 → exact golden f(x) values."""
        from dqi_mcmc.api.maxlinsat.block_gibbs_sampler import BlockGibbsSampler

        sampler = BlockGibbsSampler(tiny_opi_problem)
        _, values = sampler.sample_with_values(
            20,
            rng=np.random.default_rng(42),
            block_size=1,
            block_strategy="permutation",
        )
        np.testing.assert_array_equal(values[:, 1], EXPECTED_OPI_FX)
