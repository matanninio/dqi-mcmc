# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The DQI-MCMC Authors

"""Tests for the XOR-SAT Block Gibbs sampler."""

import numpy as np

# Golden f(x) trajectory for 20 samples, seed=42, block_size=1.
# Captured by running: sampler.sample_with_values(20, seed=42, block_size=1, num_burn_in_samples=0)
# on the tiny 10×6 problem defined in conftest.py (rng seed=0 for b and v).
EXPECTED_XORSAT_FX = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]


class TestXorSatGibbsSmoke:
    def test_smoke(self, tiny_xorsat_problem):
        """50 samples → correct shapes."""
        from dqi_mcmc.api.block_gibbs_sampler import BlockGibbsSamplerOptimized

        sampler = BlockGibbsSamplerOptimized(tiny_xorsat_problem)
        samples, values = sampler.sample_with_values(
            50, seed=0, block_size=1, num_burn_in_samples=0
        )
        n = tiny_xorsat_problem.num_variables
        assert samples.shape == (50, n), f"Expected (50, {n}), got {samples.shape}"
        assert values.shape == (50, 2), f"Expected (50, 2), got {values.shape}"

    def test_seed_consistency(self, tiny_xorsat_problem):
        """Same seed → identical f(x) trajectories."""
        from dqi_mcmc.api.block_gibbs_sampler import BlockGibbsSamplerOptimized

        sampler = BlockGibbsSamplerOptimized(tiny_xorsat_problem)
        _, v1 = sampler.sample_with_values(30, seed=42, block_size=1, num_burn_in_samples=0)
        _, v2 = sampler.sample_with_values(30, seed=42, block_size=1, num_burn_in_samples=0)
        np.testing.assert_array_equal(v1[:, 1], v2[:, 1])

    def test_golden_fx_values(self, tiny_xorsat_problem):
        """20 samples seed=42 → exact golden f(x) values."""
        from dqi_mcmc.api.block_gibbs_sampler import BlockGibbsSamplerOptimized

        sampler = BlockGibbsSamplerOptimized(tiny_xorsat_problem)
        _, values = sampler.sample_with_values(20, seed=42, block_size=1, num_burn_in_samples=0)
        np.testing.assert_array_equal(values[:, 1], EXPECTED_XORSAT_FX)
