# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The DQI-MCMC Authors

"""
Block Gibbs sampler for MaxOPIProblem.

This module provides a Block Gibbs sampler for OPI problems.

"""

# Configure Numba parallelization based on environment
# This must be done BEFORE importing numba to take effect
import os

# Check if we're in test mode or if user explicitly set num processes to 1
_in_test_mode = (
    "PYTEST_CURRENT_TEST" in os.environ  # pytest is running
    or os.environ.get("DQI_MCMC_NUM_PROCESSES", "").strip() == "1"  # explicit override
)

if _in_test_mode and "NUMBA_NUM_THREADS" not in os.environ:
    # Set Numba to use only 1 thread for tests to avoid parallel execution issues
    os.environ["NUMBA_NUM_THREADS"] = "1"

from typing import TYPE_CHECKING, Any  # noqa: E402

import numpy as np  # noqa: E402
from numba import njit, prange  # noqa: E402

if TYPE_CHECKING:
    from dqi_mcmc.api.maxlinsat.max_opi_problem import MaxOPIProblem


@njit
def _compute_k1_step(
    poly_vals: np.ndarray,
    indices: np.ndarray,
    gamma_powers_subset: np.ndarray,
    v_mask: np.ndarray,
    p: int,
    m: int,
    lookup_log_poly_sq: np.ndarray,
    random_value: float,
) -> tuple[int, float, int]:
    """
    Optimized k=1 case: single variable update.

    This function assumes the variable at indices[0] has been set to 0 in the input state,
    so we directly enumerate values 0 to p-1 without computing offsets.

    Args:
        poly_vals: Polynomial evaluations with indices[0] set to 0
        indices: Single index to update
        gamma_powers_subset: gamma_powers[:, indices]
        v_mask: Membership mask
        p: Prime modulus
        m: Number of constraints
        lookup_log_poly_sq: Lookup table
        random_value: Random value for sampling

    Returns:
        Tuple of (chosen_value, log_weight, n_satisfied)

    """
    num_configs = p
    log_weights = np.zeros(num_configs, dtype=np.float64)
    gamma_col = gamma_powers_subset[:, 0]

    # Loop over all possible values (0 to p-1)
    poly_vals_prop = poly_vals.copy()
    for val in range(p):
        # Apply modulo before checking satisfaction
        poly_vals_prop %= p

        # Count satisfied constraints
        n_satisfied = 0
        for i in range(m):
            n_satisfied += v_mask[i, poly_vals_prop[i]]

        log_weights[val] = lookup_log_poly_sq[n_satisfied]

        # Update for next iteration (always add)
        poly_vals_prop += gamma_col

    # Sample from distribution
    max_log_weight = np.max(log_weights)
    weights = np.exp(log_weights - max_log_weight)
    weights_sum = np.sum(weights)

    cumsum = 0.0
    chosen_idx = 0
    target = random_value * weights_sum
    for i in range(num_configs):
        cumsum += weights[i]
        if target < cumsum:
            chosen_idx = i
            break

    # Compute final satisfaction count for chosen value
    poly_vals_final = poly_vals + chosen_idx * gamma_col
    poly_vals_final %= p
    n_satisfied_final = 0
    for i in range(m):
        n_satisfied_final += v_mask[i, poly_vals_final[i]]

    return chosen_idx, float(lookup_log_poly_sq[n_satisfied_final]), n_satisfied_final


@njit
def _compute_k2_step(
    poly_vals: np.ndarray,
    indices: np.ndarray,
    gamma_powers_subset: np.ndarray,
    v_mask: np.ndarray,
    p: int,
    m: int,
    lookup_log_poly_sq: np.ndarray,
    random_value: float,
) -> tuple[int, int, float, int]:
    """
    Optimized k=2 case: two variable update.

    This function assumes the variables at indices[0] and indices[1] have been set to 0,
    so we directly enumerate all p^2 combinations without computing offsets.

    Returns:
        Tuple of (first_val, second_val, log_weight, n_satisfied)

    """
    num_configs = p * p
    log_weights = np.zeros(num_configs, dtype=np.float64)
    gamma_col0 = gamma_powers_subset[:, 0]
    gamma_col1 = gamma_powers_subset[:, 1]

    # Loop over first position (0 to p-1)
    for first_val in range(p):
        # Compute base polynomial values with first position set
        poly_vals_base = poly_vals + first_val * gamma_col0

        # Base offset in global weights array
        config_offset = first_val * p

        # Loop over second position (0 to p-1)
        poly_vals_prop = poly_vals_base.copy()
        for second_val in range(p):
            # Apply modulo before checking satisfaction
            poly_vals_prop %= p

            # Count satisfied constraints
            n_satisfied = 0
            for i in range(m):
                n_satisfied += v_mask[i, poly_vals_prop[i]]

            log_weights[config_offset + second_val] = lookup_log_poly_sq[n_satisfied]

            # Update for next iteration (always add)
            poly_vals_prop += gamma_col1

    # Sample from distribution
    max_log_weight = np.max(log_weights)
    weights = np.exp(log_weights - max_log_weight)
    weights_sum = np.sum(weights)

    cumsum = 0.0
    chosen_idx = 0
    target = random_value * weights_sum
    for i in range(num_configs):
        cumsum += weights[i]
        if target < cumsum:
            chosen_idx = i
            break

    # Decode chosen configuration
    first_val_chosen = chosen_idx // p
    second_val_chosen = chosen_idx % p

    # Compute final satisfaction count
    poly_vals_final = (
        poly_vals + first_val_chosen * gamma_col0 + second_val_chosen * gamma_col1
    )
    poly_vals_final %= p
    n_satisfied_final = 0
    for i in range(m):
        n_satisfied_final += v_mask[i, poly_vals_final[i]]

    return (
        first_val_chosen,
        second_val_chosen,
        float(lookup_log_poly_sq[n_satisfied_final]),
        n_satisfied_final,
    )


@njit(parallel=True)
def _compute_k3_step_parallel_2d(
    poly_vals: np.ndarray,
    indices: np.ndarray,
    gamma_powers_subset: np.ndarray,
    v_mask: np.ndarray,
    p: int,
    m: int,
    lookup_log_poly_sq: np.ndarray,
    random_value: float,
    log_weights_2d: np.ndarray | None = None,
) -> tuple[int, int, int, float, int]:
    """
    Optimized k=3 case with 2D log_weights array to eliminate cache line contention.

    OPTIMIZATION: Uses a 2D log_weights array of shape (p, p^2) instead of 1D array of
    shape (p^3,). Each parallel thread (first_val) writes to its own row, eliminating
    false sharing and cache line contention between threads.

    Returns:
        Tuple of (first_val, second_val, third_val, log_weight, n_satisfied)

    """
    # Allocate or reuse 2D log_weights array
    p_squared = p * p
    if log_weights_2d is None:
        log_weights_2d = np.zeros((p, p_squared), dtype=np.float64)
    else:
        log_weights_2d[:, :] = 0.0

    gamma_col0 = gamma_powers_subset[:, 0]
    gamma_col1 = gamma_powers_subset[:, 1]
    gamma_col2 = gamma_powers_subset[:, 2]

    # Parallel loop over first position (0 to p-1)
    for first_val in prange(p):
        poly_vals_base = poly_vals + first_val * gamma_col0

        for second_val in range(p):
            poly_vals_prop = poly_vals_base + second_val * gamma_col1

            for third_val in range(p):
                poly_vals_prop %= p

                n_satisfied = 0
                for i in range(m):
                    n_satisfied += v_mask[i, poly_vals_prop[i]]

                config_idx = second_val * p + third_val
                log_weights_2d[first_val, config_idx] = lookup_log_poly_sq[n_satisfied]

                poly_vals_prop += gamma_col2

    # Flatten the 2D array for sampling
    log_weights_flat = log_weights_2d.ravel()

    # Sample from distribution
    max_log_weight = np.max(log_weights_flat)
    weights = np.exp(log_weights_flat - max_log_weight)
    weights_sum = np.sum(weights)

    cumsum = 0.0
    chosen_idx = 0
    num_configs = p * p * p
    target = random_value * weights_sum
    for i in range(num_configs):
        cumsum += weights[i]
        if target < cumsum:
            chosen_idx = i
            break

    # Decode chosen configuration
    first_val_chosen = chosen_idx // (p * p)
    remaining_idx = chosen_idx % (p * p)
    second_val_chosen = remaining_idx // p
    third_val_chosen = remaining_idx % p

    # Compute final satisfaction count
    poly_vals_final = (
        poly_vals
        + first_val_chosen * gamma_col0
        + second_val_chosen * gamma_col1
        + third_val_chosen * gamma_col2
    )
    poly_vals_final %= p
    n_satisfied_final = 0
    for i in range(m):
        n_satisfied_final += v_mask[i, poly_vals_final[i]]

    return (
        first_val_chosen,
        second_val_chosen,
        third_val_chosen,
        float(lookup_log_poly_sq[n_satisfied_final]),
        n_satisfied_final,
    )


@njit(parallel=False)
def _compute_block_gibbs_step_core(
    x: np.ndarray,
    poly_vals: np.ndarray,
    indices: np.ndarray,
    gamma_powers: np.ndarray,
    v_mask: np.ndarray,
    p: int,
    m: int,
    lookup_log_poly_sq: np.ndarray,
    random_value: float,
    log_weights_2d: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    """
    Main dispatcher for block Gibbs sampling with support for k=1, 2, 3, and k>3.

    Returns:
        Tuple of (x_new, poly_vals_new, value_new, fx_new, chosen_idx)

    """
    k = len(indices)

    # Normalize state: copy x and set the k variables to 0
    x_normalized = x.copy()
    poly_vals_normalized = poly_vals.copy()

    # Set k variables to 0 and update poly_vals accordingly
    gamma_powers_subset = gamma_powers[:, indices]
    for idx_pos in range(k):
        old_val = x[indices[idx_pos]]
        if old_val != 0:
            x_normalized[indices[idx_pos]] = 0
            # Subtract old_val * gamma_powers to get poly_vals at 0
            offset = (p - old_val) % p  # This is -old_val mod p
            for i in range(m):
                poly_vals_normalized[i] = (
                    poly_vals_normalized[i] + offset * gamma_powers_subset[i, idx_pos]
                ) % p

    # Dispatch based on k value (conditionals outside loops for performance)
    if k == 1:
        # Call k=1 specialized function
        chosen_val, value_new, n_satisfied = _compute_k1_step(
            poly_vals_normalized,
            indices,
            gamma_powers_subset,
            v_mask,
            p,
            m,
            lookup_log_poly_sq,
            random_value,
        )

        # Apply chosen value to normalized state
        x_new = x_normalized.copy()
        x_new[indices[0]] = chosen_val

        # Update poly_vals
        poly_vals_new = poly_vals_normalized.copy()
        if chosen_val != 0:
            for i in range(m):
                poly_vals_new[i] = (
                    poly_vals_new[i] + chosen_val * gamma_powers_subset[i, 0]
                ) % p

        fx_new = 2 * n_satisfied - m
        chosen_idx = chosen_val

    elif k == 2:
        # Call k=2 specialized function
        first_val, second_val, value_new, n_satisfied = _compute_k2_step(
            poly_vals_normalized,
            indices,
            gamma_powers_subset,
            v_mask,
            p,
            m,
            lookup_log_poly_sq,
            random_value,
        )

        # Apply chosen values to normalized state
        x_new = x_normalized.copy()
        x_new[indices[0]] = first_val
        x_new[indices[1]] = second_val

        # Update poly_vals
        poly_vals_new = poly_vals_normalized.copy()
        if first_val != 0:
            for i in range(m):
                poly_vals_new[i] = (
                    poly_vals_new[i] + first_val * gamma_powers_subset[i, 0]
                ) % p
        if second_val != 0:
            for i in range(m):
                poly_vals_new[i] = (
                    poly_vals_new[i] + second_val * gamma_powers_subset[i, 1]
                ) % p

        fx_new = 2 * n_satisfied - m
        chosen_idx = first_val * p + second_val

    elif k == 3:
        # Call k=3 specialized function with parallel processing
        (
            first_val,
            second_val,
            third_val,
            value_new,
            n_satisfied,
        ) = _compute_k3_step_parallel_2d(
            poly_vals_normalized,
            indices,
            gamma_powers_subset,
            v_mask,
            p,
            m,
            lookup_log_poly_sq,
            random_value,
            log_weights_2d,
        )

        # Apply chosen values to normalized state
        x_new = x_normalized.copy()
        x_new[indices[0]] = first_val
        x_new[indices[1]] = second_val
        x_new[indices[2]] = third_val

        # Update poly_vals
        poly_vals_new = poly_vals_normalized.copy()
        if first_val != 0:
            for i in range(m):
                poly_vals_new[i] = (
                    poly_vals_new[i] + first_val * gamma_powers_subset[i, 0]
                ) % p
        if second_val != 0:
            for i in range(m):
                poly_vals_new[i] = (
                    poly_vals_new[i] + second_val * gamma_powers_subset[i, 1]
                ) % p
        if third_val != 0:
            for i in range(m):
                poly_vals_new[i] = (
                    poly_vals_new[i] + third_val * gamma_powers_subset[i, 2]
                ) % p

        fx_new = 2 * n_satisfied - m
        chosen_idx = first_val * p * p + second_val * p + third_val

    else:
        raise NotImplementedError(
            f"Block size k={k} not yet supported. Currently supports k=1, 2, 3."
        )

    return x_new, poly_vals_new, value_new, float(fx_new), chosen_idx


class BlockGibbsSampler:
    """
    Block Gibbs sampler for MaxOPIProblem with full enumeration.

    This sampler performs exact Gibbs sampling by enumerating all p^k
    configurations for k variables. Only feasible for small p or small k.

    Example:
        >>> problem = MaxOPIProblem(p=5, v=v, num_variables=10, ell=3)
        >>> sampler = BlockGibbsSampler(problem)
        >>> samples, values = sampler.sample_with_values(
        ...     num_samples=10000,
        ...     num_burn_in_samples=1000,
        ...     block_size=2
        ... )

    """

    def __init__(self, problem: "MaxOPIProblem") -> None:
        """
        Initialize the Block Gibbs sampler.

        Args:
            problem: A MaxOPIProblem instance

        """
        from dqi_mcmc.api.maxlinsat.max_opi_problem import MaxOPIProblem

        if not isinstance(problem, MaxOPIProblem):
            raise TypeError("BlockGibbsSampler requires MaxOPIProblem")

        self.problem = problem
        self.x_length = problem.num_variables

        # Pre-compute and cache problem parameters
        self.v = problem.v
        self.p = problem.p
        self.m = problem.num_constraints
        self.gamma = problem.gamma
        self.r = problem.r

        # Pre-compute all powers of gamma needed for polynomial evaluation
        # gamma_powers[i][j] = gamma^(i*j) mod p for i in [0, m), j in [0, n)
        self.gamma_powers = np.zeros((self.m, self.x_length), dtype=np.int64)
        t = 1  # gamma^0
        for i in range(self.m):
            power = 1  # t^0
            for j in range(self.x_length):
                self.gamma_powers[i, j] = power
                power = (power * t) % self.p
            t = (t * self.gamma) % self.p

        # Pre-compute membership masks for faster constraint checking
        self.v_mask = np.zeros((self.m, self.p), dtype=np.int8)
        for i in range(self.m):
            self.v_mask[i, self.v[i]] = 1

        # Cache lookup table
        self.lookup_float64 = problem.lookup_log_polynomial_squared.astype(np.float64)

        # Pre-allocate 2D log_weights array for k=3 optimization
        self._log_weights_2d_k3 = np.zeros((self.p, self.p * self.p), dtype=np.float64)

        # State for permutation-based block selection
        self._permutation = None
        self._permutation_idx = 0

    def get_permutation_state(self) -> dict:
        """
        Get the current permutation state for checkpoint/resume.

        Returns:
            Dictionary with 'permutation' and 'permutation_idx' keys.

        """
        return {
            "permutation": (
                self._permutation.tolist() if self._permutation is not None else None
            ),
            "permutation_idx": self._permutation_idx,
        }

    def set_permutation_state(self, state: dict) -> None:
        """
        Restore permutation state from checkpoint.

        Args:
            state: Dictionary with 'permutation' and 'permutation_idx' keys

        """
        if state["permutation"] is not None:
            self._permutation = np.array(state["permutation"], dtype=np.int64)
        else:
            self._permutation = None
        self._permutation_idx = state["permutation_idx"]

    def _compute_poly_vals(self, x: np.ndarray) -> np.ndarray:
        """
        Compute Q(gamma^i) for all i in [0, m) using pre-computed powers.

        Args:
            x: Coefficient vector of length n

        Returns:
            Array of length m with Q(gamma^i) mod p for each constraint

        """
        poly_vals = np.sum(x * self.gamma_powers, axis=1) % self.p
        return poly_vals.astype(np.int64)

    def find_best_random_start(
        self,
        num_random_starts: int,
        seed: int | None = None,
        num_warmup_samples: int = 0,
        block_size: int = 1,
        block_strategy: str = "permutation",
    ) -> np.ndarray:
        """
        Generate num_random_starts random samples and return the one with the highest score.

        Args:
            num_random_starts: Number of random samples to try
            seed: Random seed for reproducibility
            num_warmup_samples: If > 0, run warmup and select based on best f(x) achieved
            block_size: Size of blocks for warmup
            block_strategy: Strategy for selecting blocks ("random", "sequential" or "permutation").

        Returns:
            The vector with the highest score among the tries

        """
        rng = np.random.default_rng(seed=seed)

        if num_warmup_samples == 0:
            # Select based on initial f(x)
            best_x = rng.integers(0, self.p, size=self.x_length, dtype=np.int64)
            best_fx = self.problem.f(best_x)

            for _ in range(num_random_starts - 1):
                x = rng.integers(0, self.p, size=self.x_length, dtype=np.int64)
                fx = self.problem.f(x)

                if fx > best_fx:
                    best_fx = fx
                    best_x = x.copy()

            return best_x
        else:
            # Run warmup and select based on best f(x) achieved
            x = rng.integers(0, self.p, size=self.x_length, dtype=np.int64)
            poly_vals = self._compute_poly_vals(x)
            n_satisfied = np.sum(self.v_mask[np.arange(self.m), poly_vals])
            fx = 2 * n_satisfied - self.m
            value = float(self.problem.log_poly_p2(fx=fx))
            best_fx = fx

            # Run warmup for first try
            for _ in range(num_warmup_samples):
                x, poly_vals, value, fx = self._one_step(
                    rng,
                    x,
                    poly_vals,
                    value,
                    fx,
                    block_size=block_size,
                    block_strategy=block_strategy,
                )
                if fx > best_fx:
                    best_fx = fx

            best_final_x = x.copy()

            # Try remaining samples
            for _ in range(num_random_starts - 1):
                x = rng.integers(0, self.p, size=self.x_length, dtype=np.int64)
                poly_vals = self._compute_poly_vals(x)
                n_satisfied = np.sum(self.v_mask[np.arange(self.m), poly_vals])
                fx = 2 * n_satisfied - self.m
                value = float(self.problem.log_poly_p2(fx=fx))
                max_fx = fx

                for _ in range(num_warmup_samples):
                    x, poly_vals, value, fx = self._one_step(
                        rng,
                        x,
                        poly_vals,
                        value,
                        fx,
                        block_size=block_size,
                        block_strategy=block_strategy,
                    )
                    if fx > max_fx:
                        max_fx = fx

                if max_fx > best_fx:
                    best_fx = max_fx
                    best_final_x = x.copy()

            return best_final_x

    def sample_with_values(
        self,
        num_samples: int,
        num_burn_in_samples: int = 0,
        block_size: int = 1,
        block_strategy: str = "permutation",
        l_dim: int | None = None,
        x_0: np.ndarray | None = None,
        seed: int | None = None,
        num_random_starts: int | None = None,
        num_warmup_samples: int = 0,
        rng: np.random.Generator | None = None,
        stop_above: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample using Block Gibbs sampling with full enumeration.

        Args:
            num_samples: Number of samples to generate
            num_burn_in_samples: Number of initial samples to discard
            block_size: Number of variables to update simultaneously
            block_strategy: Block selection strategy:
                - "random": Randomly select block_size variables (no coverage guarantee).
                - "sequential": Select consecutive variables, starting from idx = 0.
                - "permutation": At each epoch, draw a fresh random permutation of all variables
                  and sweep through it in blocks of ``block_size``. Every variable is updated
                  exactly once per epoch, which improves mixing for OPI problems. This is the
                  default, and matches the Markov chain construction described in Section 6 of
                  the paper (Algorithm A2).
            l_dim: Not implemented (for compatibility)
            x_0: Initial state (if None, determined by num_random_starts)
            seed: Random seed (ignored if rng is provided)
            rng: External RNG instance to use (for checkpoint/resume). If provided, seed is ignored.
            stop_above: If set, stop sampling as soon as n_satisfied > stop_above.
                The returned arrays are truncated to the actual number of steps taken.

        Returns:
            samples: Array of shape (num_samples, x_length) — may be shorter than num_samples
                if stop_above is reached first.
            values: Array of shape (num_samples, 2) with [log(P^2(f(x))), f(x)] — same shape
                as samples.

        """
        if l_dim is not None:
            raise NotImplementedError("l_dim parameter is not yet implemented")

        assert block_size <= self.x_length, "block_size cannot be larger than x_length"
        assert block_strategy in [
            "random",
            "sequential",
            "permutation",
        ], "block_strategy must be 'random', 'sequential', or 'permutation'"

        if block_strategy == "sequential":
            # set the permutation to 0,1,2,...,n-1
            self._permutation = np.arange(self.x_length, dtype=np.int64)

        # Check if enumeration is feasible
        max_configs = self.p**block_size
        if max_configs > 10_000_000:
            raise ValueError(
                f"Full enumeration too large: {self.p}^{block_size} = {max_configs:,} configurations. "
                f"Reduce block_size."
            )

        # Use provided RNG or create new one from seed
        if rng is None:
            rng = np.random.default_rng(seed=seed)
        samples = np.empty((num_samples, self.x_length), dtype=np.int64)
        values = np.empty((num_samples, 2), dtype=np.float64)

        # Initialize state
        if x_0 is not None:
            x = np.array(x_0, dtype=np.int64)
        elif num_random_starts is not None and num_random_starts > 1:
            x = self.find_best_random_start(
                num_random_starts=num_random_starts,
                seed=seed,
                num_warmup_samples=num_warmup_samples,
                block_size=block_size,
                block_strategy=block_strategy,
            )
        else:
            x = rng.integers(0, self.p, size=self.x_length, dtype=np.int64)

        # Compute initial polynomial values and objective
        poly_vals = self._compute_poly_vals(x)
        n_satisfied = np.sum(self.v_mask[np.arange(self.m), poly_vals])
        fx = 2 * n_satisfied - self.m
        value = self.problem.log_poly_p2(fx=fx)

        # Burn-in phase
        for _ in range(num_burn_in_samples):
            x, poly_vals, value, fx = self._one_step(
                rng,
                x,
                poly_vals,
                value,
                fx,
                block_size=block_size,
                block_strategy=block_strategy,
            )

        # Sampling phase
        for t in range(num_samples):
            x, poly_vals, value, fx = self._one_step(
                rng,
                x,
                poly_vals,
                value,
                fx,
                block_size=block_size,
                block_strategy=block_strategy,
            )
            samples[t] = x.copy()
            values[t, :] = [value, fx]

            if stop_above is not None:
                n_satisfied = (int(fx) + self.m) // 2
                if n_satisfied > stop_above:
                    return samples[: t + 1], values[: t + 1]

        return samples, values

    def _one_step(
        self,
        rng,
        x,
        poly_vals,
        value,
        fx_old,
        block_size,
        block_strategy,
    ) -> tuple[Any, Any, float, float]:
        """
        Take one Block Gibbs step.

        Args:
            rng: Random number generator
            x: Current state vector
            poly_vals: Current polynomial evaluations
            value: Current log(P^2(f(x)))
            fx_old: Current f(x)
            block_size: Number of variables to update
            block_strategy: "random", "sequential", or "permutation"

        Returns:
            x_new, poly_vals_new, value_new, fx_new

        """
        # Select indices to update
        if block_strategy == "random":
            indices = rng.choice(self.x_length, size=block_size, replace=False).astype(
                np.int64
            )
        elif block_strategy == "sequential":
            # Extract next block from permutation
            end_idx = min(self._permutation_idx + block_size, self.x_length)
            indices = self._permutation[self._permutation_idx : end_idx].copy()
            # if we don't have enough variables left in this epoch, wrap around to the beginning
            if len(indices) < block_size:
                remaining = block_size - len(indices)
                indices = np.concatenate([indices, self._permutation[:remaining]])
                self._permutation_idx = remaining
            else:
                self._permutation_idx = end_idx
        elif block_strategy == "permutation":
            # Create new permutation if needed
            if self._permutation is None or self._permutation_idx >= self.x_length:
                self._permutation = rng.permutation(self.x_length).astype(np.int64)
                self._permutation_idx = 0

            # Extract next block from permutation
            end_idx = min(self._permutation_idx + block_size, self.x_length)
            indices = self._permutation[self._permutation_idx : end_idx].copy()

            # If we don't have enough variables left in this epoch simply sample a smaller block
            self._permutation_idx = end_idx
        else:
            raise NotImplementedError(
                f"Block strategy {block_strategy} not implemented"
            )

        indices.sort()

        # Generate random value for sampling
        random_value = float(rng.random())

        # Call JIT-compiled core function
        x_new, poly_vals_new, value_new, fx_new, _ = _compute_block_gibbs_step_core(
            x=x,
            poly_vals=poly_vals,
            indices=indices,
            gamma_powers=self.gamma_powers,
            v_mask=self.v_mask,
            p=self.p,
            m=self.m,
            lookup_log_poly_sq=self.lookup_float64,
            random_value=random_value,
            log_weights_2d=self._log_weights_2d_k3,
        )

        return x_new, poly_vals_new, value_new, fx_new
