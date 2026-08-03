# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The DQI-MCMC Authors

from typing import Any

import numpy as np
from numba import njit

from dqi_mcmc.api.maxxorsat.max_xor_p_sat_problem import MaxXorPSatProblem


@njit
def _bit_length(n):
    """Numba-compatible implementation of bit_length()."""
    if n == 0:
        return 0
    length = 0
    while n:
        length += 1
        n >>= 1
    return length


# @njit(cache=True)
def _compute_one_step_optimized_core(
    x: np.ndarray,
    bx: np.ndarray,
    value: float,
    indices: np.ndarray,
    b_cols_array: np.ndarray,
    v: np.ndarray,
    p: int,
    m: int,
    lookup_log_poly_sq: np.ndarray,
    random_value: float,
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    """
    Core computational logic for one optimized MCMC step, JIT-compiled with numba.

    This function performs the heavy lifting of computing conditional probabilities
    and sampling a new state using Gray code enumeration and incremental updates.

    Args:
        x: Current state vector
        bx: Current b @ x.T % p (cached)
        value: Current log P^2(f(x)) value
        indices: Sorted indices of bits to update
        b_cols_array: 2D array of b matrix columns for the selected indices
        v: Target vector
        p: Prime modulus
        m: Number of constraints
        lookup_log_poly_sq: Lookup table for log(P^2(f(x)))
        random_value: Random value in [0, 1) for sampling

    Returns:
        Tuple of (x_new, bx_new, value_new, fx_new, chosen_g)

    """
    block_size = len(indices)
    num_configurations = 2**block_size
    log_weights = np.zeros(num_configurations, dtype=np.float64)

    # First configuration: no changes to x (all indices=0)
    log_weights[0] = value

    # Track which bits are flipped relative to original x
    flipped = np.zeros(block_size, dtype=np.bool_)
    bx_current = bx.copy()

    # Iterate over all 2^{block_size} - 1 remaining possibilities using Gray codes
    prev = 0
    for t in range(1, num_configurations):
        # Gray code value
        g = t ^ (t >> 1)

        # Find which bit flipped
        diff = g ^ prev
        j = _bit_length(diff) - 1  # which bit flipped (0..block_size-1)

        # Update bx incrementally
        b_col = b_cols_array[j]
        if not flipped[j]:
            # Flipping this bit from original x value
            if x[indices[j]] == 0:
                bx_current = (bx_current + b_col) % p
            else:
                bx_current = (bx_current - b_col) % p
            flipped[j] = True
        else:
            # Flipping this bit back to original x value
            if x[indices[j]] == 0:
                bx_current = (bx_current - b_col) % p
            else:
                bx_current = (bx_current + b_col) % p
            flipped[j] = False

        # Compute log probability using cached lookup
        n_satisfied = np.sum(bx_current == v)
        fx_current = 2 * n_satisfied - m
        lookup_idx = (fx_current + m) // 2
        log_weights[g] = lookup_log_poly_sq[lookup_idx]

        prev = g

    # Convert log-probs to normalized probabilities
    # Subtract max log-prob for numerical stability before exponentiating
    max_log_weight = np.max(log_weights)
    weights = np.exp(log_weights - max_log_weight)
    weights_sum = np.sum(weights)
    conditionals = weights / weights_sum

    # Sample using the provided random value
    # Manual implementation of categorical sampling
    cumsum = 0.0
    chosen_g = 0
    for i in range(num_configurations):
        cumsum += conditionals[i]
        if random_value < cumsum:
            chosen_g = i
            break

    # Transform the Gray code to a new bitstring
    x_new = x.copy()
    bx_new = bx.copy()
    for j in range(block_size):
        # If the j-th bit of the chosen Gray code is 1,
        # it means that bit in 'indices' must be flipped.
        if (chosen_g >> j) & 1:
            x_new[indices[j]] ^= 1
            # Update bx incrementally
            b_col = b_cols_array[j]
            if x[indices[j]] == 0:
                bx_new = (bx_new + b_col) % p
            else:
                bx_new = (bx_new - b_col) % p

    # Compute final values
    n_satisfied_new = np.sum(bx_new == v)
    fx_new = 2 * n_satisfied_new - m
    lookup_idx_new = (fx_new + m) // 2
    value_new = float(lookup_log_poly_sq[lookup_idx_new])

    return x_new, bx_new, value_new, float(fx_new), chosen_g


class BlockGibbsSamplerOptimized:
    """
    Optimized Block Gibbs algorithm for sampler specifically for MaxXorPSatProblem.

    This sampler avoids repeated calls to the expensive target function by:
    1. Computing incremental updates when flipping a single bit
    2. Caching intermediate results (bx values)
    3. Using efficient numpy operations

    This is significantly faster than the generic MetropolisHastingsSampler
    for MaxXorPSatProblem instances.

    Like MetropolisHastingsSampler, this sampler supports three methods for
    selecting the initial state:
    1. Provide a specific vector via x_0 parameter
    2. Use a single random vector (default when x_0=None and num_random_starts=None)
    3. Try multiple random vectors and select the best one (via num_random_starts parameter)
    """

    def __init__(self, problem: MaxXorPSatProblem) -> None:
        """
        Initialize the optimized sampler with a MaxXorPSatProblem.

        Args:
            problem: A MaxXorPSatProblem instance with the problem definition.
            x_length: The length of the binary vector (inferred from problem if None).

        """
        x_length = problem.num_variables

        # Ensure x_length is an int (type narrowing for type checker)
        assert isinstance(x_length, int), "x_length must be an integer"
        self.x_length: int = x_length

        self.problem = problem
        self.target = problem.logp2fx_fx  # Returns both P^2(f(x)) and f(x)

        # Pre-compute and cache problem parameters for optimized version
        self.v = problem.v
        self.p = problem.p
        self.m = problem.num_constraints

        # Pre-compute all columns of b for efficient access during sampling
        # This trades memory for speed: O(n*m) space for O(1) column access
        # For sparse matrices, this is much more efficient than repeated getcol() calls
        if hasattr(problem.b, "getcol"):
            # Sparse matrix: pre-extract all columns as dense arrays
            self.b_columns = [problem.b.getcol(i).toarray().ravel() for i in range(x_length)]  # type: ignore[attr-defined]
        else:
            # Dense matrix: pre-extract columns as views (no memory copy)
            self.b_columns = [problem.b[:, i] for i in range(x_length)]

        # Pre-convert arrays to consistent types to reduce typeof overhead
        # This ensures Numba sees the same types on every call, allowing it to reuse cached type information
        self.b_columns_int64 = [col.astype(np.int64) for col in self.b_columns]
        self.v_int64 = self.v.astype(np.int64)
        self.lookup_float64 = problem.lookup_log_polynomial_squared.astype(np.float64)

        # Keep original b for initial bx computation (uses efficient @ operator)
        self.b = problem.b

    def find_best_random_start(
        self,
        num_random_starts: int,
        num_bitflips: int = 1,
        seed: int | None = None,
        num_warmup_samples: int = 0,
        block_size: int = 1,
        block_strategy: str = "random",
    ) -> np.ndarray:
        """
        Generate num_random_starts random samples and return the one with the highest score.

        Args:
            num_random_starts: Number of random samples to try.
            seed: Random seed for reproducibility.
            num_warmup_samples: If > 0, run each random start through this many MCMC steps
                and select based on the best f(x) achieved during warmup.

        Returns:
            The binary vector with the highest score among the tries.

        """
        assert num_bitflips <= self.x_length, "num_bitflips cannot be larger than x_length"
        rng = np.random.default_rng(seed=seed)

        if num_warmup_samples == 0:
            best_x = rng.integers(2, size=self.x_length).astype(np.int8)
            best_fx = self.problem.f(best_x)

            for _ in range(num_random_starts - 1):
                x = rng.integers(2, size=self.x_length).astype(np.int8)
                fx = self.problem.f(x)

                if fx > best_fx:
                    best_fx = fx
                    best_x = x.copy()

            return best_x
        else:
            x = rng.integers(2, size=self.x_length).astype(np.int8)
            bx = (x @ self.b.T) % self.p
            n_satisfied = np.sum(bx == self.v)
            fx = 2 * n_satisfied - self.m
            value = float(self.problem.log_poly_p2(fx=fx))
            best_fx = fx

            # Run warmup for first try
            for _ in range(num_warmup_samples):
                x, bx, value, fx = self._one_step_optimized(
                    rng,
                    x,
                    bx,
                    value,
                    fx,
                    block_size=block_size,
                    block_strategy=block_strategy,
                )
                if fx > best_fx:
                    best_fx = fx

            # Store final state from first warmup
            best_final_x = x.copy()

            # Try remaining samples
            for _ in range(num_random_starts - 1):
                x = rng.integers(2, size=self.x_length).astype(np.int8)
                bx = (x @ self.b.T) % self.p
                n_satisfied = np.sum(bx == self.v)
                fx = 2 * n_satisfied - self.m
                value = float(self.problem.log_poly_p2(fx=fx))
                max_fx = fx

                # Run warmup steps
                for _ in range(num_warmup_samples):
                    x, bx, value, fx = self._one_step_optimized(
                        rng,
                        x,
                        bx,
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
        block_strategy="random",
        l_dim: int | None = None,
        x_0: np.ndarray | None = None,
        seed: int | None = None,
        num_random_starts: int | None = None,
        num_warmup_samples: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Optimized sampling using incremental updates.

        Args:
            num_samples (int): The number of samples to generate.
            num_burn_in_samples (int): Number of initial samples to discard.
                Defaults to 0.
            block_size (int): size of block over which we sample from the conditional distribution
            block_strategy (str): Strategy to choose which bits to update. Must be either "random"
                or "sequential". The XOR-SAT sampler uses ``"random"`` (the default): at each step
                a fresh uniformly-random subset of ``block_size`` bits is chosen. This matches the
                Markov chain construction described in Section 5 of the paper (Algorithm A1).
            l_dim (int | None): The dimension of the latent space. Defaults to None (not implemented).
            x_0: The initial state of the chain. If None, will be determined by
                num_random_starts parameter.
            seed (int | None): The seed for the random number generator. Defaults to None.
            num_random_starts: If provided and x_0 is None, will try this many
                random starting points and select the best one.
            num_warmup_samples: If num_random_starts > 1, run each random start through
                this many MCMC steps and select based on best f(x) achieved.

            The function returns a tuple containing the generated samples and values.
            values is a 2D array of shape (num_samples, 2) where each row contains [P^2(f(x)), f(x)].

        """
        if l_dim is not None:
            raise NotImplementedError("l_dim parameter is not yet implemented")

        assert block_size <= self.x_length, "block_size cannot be larger than x_length"

        assert block_strategy in [
            "random",
            "sequential",
        ], "block_strategy must be either random or sequential"

        # if seed is given, create random number generator with it.
        rng = np.random.default_rng(seed=seed)
        samples = np.empty((num_samples, self.x_length), dtype=np.int8)
        values = np.empty((num_samples, 2), dtype=np.float64)

        # Initialize state with correct types from the start
        if x_0 is not None:
            x = np.asarray(x_0, dtype=np.int8)
        elif num_random_starts is not None and num_random_starts > 1:
            # Find best starting point from multiple random tries
            x = self.find_best_random_start(
                num_random_starts=num_random_starts,
                seed=seed,
                num_warmup_samples=num_warmup_samples,
            ).astype(np.int8)
        else:
            x = rng.integers(2, size=self.x_length).astype(np.int8)

        # Compute initial bx and value with correct types
        bx = ((x @ self.b.T) % self.p).astype(np.int64)
        n_satisfied = np.sum(bx == self.v)
        fx = 2 * n_satisfied - self.m
        value = self.problem.log_poly_p2(fx=fx)

        # Burn-in phase
        for _ in range(num_burn_in_samples):
            x, bx, value, fx = self._one_step_optimized(
                rng=rng,
                x=x,
                bx=bx,
                value=value,
                fx_old=fx,
                block_size=block_size,
                block_strategy=block_strategy,
            )
        for t in range(num_samples):
            x, bx, value, fx = self._one_step_optimized(
                rng=rng,
                x=x,
                bx=bx,
                value=value,
                fx_old=fx,
                block_size=block_size,
                block_strategy=block_strategy,
            )
            samples[t] = x.copy()
            values[t, :] = [value, fx]

        return samples, values

    def _one_step_optimized(
        self, rng, x, bx, value, fx_old, block_size, block_strategy
    ) -> tuple[Any, Any, float, float]:
        """
        Take a single optimized step in the Markov chain using incremental updates.

        Args:
            rng: Random number generator
            x: Current state vector (int8)
            bx: Current b @ x.T % p (cached, int64)
            value: Current log P^2(f(x)) value (cached, float)
            fx_old: Current f(x) value (cached, float)
            block_size: Size of block to update
            block_strategy: Strategy for selecting block indices

        Returns:
            x_new, bx_new, value_new, fx_new: Updated state

        """
        # ====================================================
        # 1. Choose the bit indices to update
        # ====================================================
        if block_strategy == "random":
            indices = rng.choice(self.x_length, size=block_size, replace=False).astype(np.int64)
        elif block_strategy == "sequential":
            start_idx = rng.integers(0, self.x_length)
            indices = np.array(
                [(start_idx + i) % self.x_length for i in range(block_size)],
                dtype=np.int64,
            )
        else:
            raise NotImplementedError(f"Block strategy {block_strategy} not implemented.")
        # sort the indices
        indices.sort()

        # Pre-extract the columns we'll need for incremental updates
        b_cols = [self.b_columns_int64[idx] for idx in indices]
        b_cols_array = np.array(b_cols, dtype=np.int64)

        # Generate random value for sampling
        random_value = float(rng.random())

        # Call the JIT-compiled core function
        x_new, bx_new, value_new, fx_new, _ = _compute_one_step_optimized_core(
            x=x,
            bx=bx,
            value=value,
            indices=indices,
            b_cols_array=b_cols_array,
            v=self.v_int64,
            p=self.p,
            m=self.m,
            lookup_log_poly_sq=self.lookup_float64,
            random_value=random_value,
        )

        return x_new, bx_new, value_new, fx_new
