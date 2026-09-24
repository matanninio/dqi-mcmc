# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The DQI-MCMC Authors

r"""
Problem input: B, v, p
Algorithmic input:
    * num samples
    * ell (depends on the decoder if given)
    * x_0 (opt)
    * classical decoder for B (opt)

S: num of satisfied constraints
Algorithmic output: average S, max S, max x, histogram over S, all samples (opt)

Algorithm steps:
compute the weights of the polynomial P
For _ in range(num samples):
Mcmc: sample x~P^2(f(x))
Evaluate f(x)
Compute average S, max S
"""

import warnings
from collections.abc import Sequence
from typing import Any, overload

import numpy as np
import numpy.typing as npt
from numpy import signedinteger
from numpy.polynomial import Polynomial
from scipy.linalg import eigh_tridiagonal
from scipy.sparse import csr_matrix
from scipy.special import logsumexp

# Import the fast cache implementation (the optimal method)
from dqi_mcmc.api._elementary_poly_fast import build_elementary_poly_cache_fast


def compute_optimal_lookup_table_xorsat(
    num_variables: int,
    num_constraints: int,
    ell: int,
    normalize: bool = True,
) -> np.ndarray:
    r"""
    Compute lookup table for the optimal polynomial P using fast Cython cache.

    This function computes the lookup table for log(P^2(f(x)))
    for all possible objective function values f(x) in {-m, -m+2, ..., m-2, m}.

    Args:
        num_variables (int): The number of variables (n).
        num_constraints (int): The number of constraints (m).
        ell (int): The degree of the polynomial.
        normalize (bool): If True, normalize lookup_log_polynomial_squared so minimum is 0.
                         This is valid when only relative probabilities matter. (default: True)

    Returns:
        np.ndarray: lookup_log_polynomial_squared - Array of log(P^2(f(x))) values, shape (m+1,)

    """
    m = num_constraints
    n = num_variables

    # =============================================
    # 1. Compute the weights w_k
    # =============================================
    ks = np.arange(1, ell + 1, dtype=np.float64)
    off_diag = np.sqrt(ks * (m - ks + 1.0))
    diag = np.zeros(
        ell + 1, dtype=np.float64
    )  # for maxxorsat the diagonal is all zeroes
    # get only the largest eigenpair
    _, w_k = eigh_tridiagonal(diag, off_diag, select="i", select_range=(ell, ell))
    w_k = w_k[:, 0]

    # Pre-compute log values for w_k (optimization)
    log_w_k = np.log(np.abs(w_k))
    sign_w_k = np.sign(w_k)

    # =============================================
    # 2. Pre-compute all elementary polynomial values with fast cache
    # =============================================
    fast_cache = build_elementary_poly_cache_fast(m, ell)
    log_cache, sign_cache = fast_cache.compute_log_cache()

    # =============================================
    # 3. Initialize the lookup table
    # =============================================
    lookup_log_polynomial_squared = np.zeros(m + 1)

    # =============================================
    # 4. Compute log(P^2(f(x))) using pre-computed log cache
    # =============================================
    # Pre-compute log(binom(m, k)) for all k (optimization)
    log_binom_m = np.zeros(ell + 1)
    for k in range(1, ell + 1):
        log_binom_m[k] = log_binom_m[k - 1] + np.log(m - k + 1) - np.log(k)

    log_2 = np.log(2)
    n_log_2 = n * log_2

    for num_ones in range(0, m + 1):
        log_p_elem_k, signs = [], []
        for k in range(ell + 1):
            # Use pre-computed log and sign values
            log_ek = log_cache[num_ones, k]
            if np.isinf(log_ek):  # e_k was zero
                continue
            sign_ek = sign_cache[num_ones, k]

            signs.append(sign_ek * sign_w_k[k])
            log_denom_k = log_binom_m[k] / 2
            log_p_elem_k.append(log_ek + log_w_k[k] - log_denom_k)

        log_pfx, _ = logsumexp(a=log_p_elem_k, b=signs, return_sign=True)
        # Store log[P^2(f(x))] = 2*log(P(f(x))) - n*log(2)
        lookup_log_polynomial_squared[num_ones] = 2 * log_pfx - n_log_2

    # Optionally normalize lookup_log_polynomial_squared so minimum is 0
    # This is valid since we only care about relative probabilities
    if normalize:
        # Only consider finite values for normalization (exclude -inf from zeros)
        finite_mask = np.isfinite(lookup_log_polynomial_squared)
        if np.any(finite_mask):
            min_log = np.min(lookup_log_polynomial_squared[finite_mask])
            lookup_log_polynomial_squared[finite_mask] -= min_log

    return lookup_log_polynomial_squared


class MaxXorPSatProblem:
    r"""
    A class representing linear expressions for Satisfiability (LAN-SAT) problems over a
        Galois Field (GF(p)).

    [1] Gil-Fuster, E., Ninio, M., Bittel, L., Shimoni, Y., Eisert, J., Woerner, S., and Carrera Vázquez, A.
        Approximate sampling from decoded quantum interferometry via Markov chain Monte Carlo methods.
        arXiv (2026). https://arxiv.org/abs/2607.28120

    """

    def __init__(
        self,
        b: np.ndarray | csr_matrix,
        v: np.ndarray,
        p: int = 2,
        ell: int | None = None,
        polynomial_coef: Sequence[float] | None = None,
        use_cache: bool = True,
    ) -> None:
        """
        Initializes the MaxXorPSatProblem.

        Args:
            b (np.ndarray): A 2D NumPy array representing the Boolean function's coefficient matrix.
            v (np.ndarray): A 1D NumPy array representing the target output vector.
            p (int): An integer representing the prime modulus for the Galois Field (default is 2).
            ell (int | None): The degree of the polynomial. Required if `polynomial_coef` is not
                provided.
            polynomial_coef (Sequence[float] | None): Custom polynomial coefficients. If provided,
                takes priority over `ell`.
            use_cache (bool): If True, attempt to load pre-computed lookup tables from disk before
                computing them. Default is True.

        """
        self.b: np.ndarray | csr_matrix = b
        self.v = v
        self.p = p

        # Check if 'b' is two-dimensional
        if b.ndim != 2:
            raise ValueError("Array 'b' must be two-dimensional.")
        assert b.shape is not None, "Shape should exist after ndim check"
        self.num_variables: int = b.shape[1]
        self.num_constraints: int = b.shape[0]
        assert self.num_constraints == v.shape[0], "v's length should match b"

        # Determine polynomial degree 'ell' and construct lookup table
        if polynomial_coef is not None:
            computed_ell = len(polynomial_coef) - 1
            if ell is not None:
                assert ell == computed_ell, (
                    f"Provided 'ell' ({ell}) does not match the polynomial degree "
                    f"inferred from 'polynomial_coef' ({computed_ell}). "
                    f"Expected ell = len(polynomial_coef) - 1 = {computed_ell}."
                )
                warnings.warn(
                    "'ell' is ignored because 'polynomial_coef' was provided; "
                    "the polynomial degree is inferred from 'polynomial_coef'.",
                    UserWarning,
                    stacklevel=2,
                )
            self.ell = computed_ell
            self.set_custom_lookup_table(polynomial_coef)
        elif ell is not None:
            self.ell = ell
            self.set_optimal_lookup_table(use_cache=use_cache)
        else:
            raise ValueError(
                "Either 'ell' or 'polynomial_coef' must be provided to determine "
                "the polynomial degree."
            )

    def set_custom_lookup_table(self, polynomial_coef: Sequence[float]) -> None:
        """Set the lookup table based on a custom polynomial."""
        m = self.num_constraints
        fx_range: np.typing.NDArray[signedinteger[Any]] = np.arange(-m, m + 1, 2)
        custom_polynomial: Polynomial = Polynomial(coef=polynomial_coef)
        lookup_polynomial_temp = np.array(
            [custom_polynomial(fx) for fx in fx_range], dtype=np.float64
        )
        self.lookup_log_polynomial_squared = 2 * np.log(np.abs(lookup_polynomial_temp))

    def set_optimal_lookup_table(self, use_cache: bool = True) -> None:
        r"""
        Construct a lookup table for the polynomial P.

        This method first attempts to load pre-computed tables from disk cache.
        If not found, it computes them using compute_optimal_lookup_table_xorsat.

        Args:
            use_cache: If True, attempt to load from disk cache before computing

        """
        if use_cache:
            try:
                from ..lookup_cache import load_lookup_table

                cached = load_lookup_table(
                    self.num_constraints, self.num_variables, self.ell, p=2, r=1
                )
                if cached is not None:
                    self.lookup_log_polynomial_squared = cached
                    return
            except ImportError:
                pass

        self.lookup_log_polynomial_squared = compute_optimal_lookup_table_xorsat(
            self.num_variables,
            self.num_constraints,
            self.ell,
        )

    def bx(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the dot product of the coefficient matrix 'b' and input vector 'x', then applies
            modulo 'p' operation.

        Args:
            x (np.ndarray): A 1D NumPy array representing the input vector.

        Returns:
            np.ndarray: The result of the dot product of 'b' and 'x', modulo 'p'.

        """
        assert x.shape[-1] == self.num_variables
        bx = x @ self.b.T
        bx_mod_p = bx % self.p
        return bx_mod_p

    def match_vector(self, x: np.ndarray) -> np.ndarray:
        """
        Checks where the result of bx(x) equals the target vector 'v' mode p.

        Args:
            x (np.ndarray): A 1D NumPy array representing the input vector.

        Returns:
            np.ndarray: A list of boolean values indicating whether the corresponding bits match.

        """
        return self.bx(x) == self.v

    def f(self, x: np.ndarray) -> np.ndarray | int:
        """
        Return the value of the function for a vector x.
        The value of the function for a vector x is computed as:
        sum(bx == v)-sum(bx != v)

        Args:
            x: Binary vector (1D array) or batch of vectors (2D array)

        Returns:
            f(x) value(s) - int for single vector, ndarray for batch

        """
        n_match = self.n_satisfied(x)
        fx = 2 * n_match - self.num_constraints
        if isinstance(fx, np.ndarray):
            return (
                int(fx) if fx.ndim == 0 or (fx.ndim == 1 and fx.shape[0] == 1) else fx
            )
        else:
            return int(fx)

    @overload
    def log_poly_p2(self, fx: int | float) -> float: ...

    @overload
    def log_poly_p2(self, fx: np.ndarray) -> np.ndarray: ...

    def log_poly_p2(self, fx: int | float | np.ndarray) -> float | np.ndarray:
        """
        Return log(P^2(f)) for given f(x) value(s).

        Args:
            fx: The f(x) value (scalar) or values (array)

        Returns:
            log(P^2(f(x))) - float for scalar input, ndarray for array input

        """
        if isinstance(self.lookup_log_polynomial_squared, np.ndarray):
            is_scalar = isinstance(fx, (int, float)) or (
                isinstance(fx, np.ndarray) and fx.ndim == 0
            )

            if is_scalar:
                lookup_idx = int((fx + self.num_constraints) // 2)
                logp2fx = self.lookup_log_polynomial_squared[lookup_idx]
                return float(logp2fx)
            else:
                fx_array = np.asarray(fx)
                lookup_idx = ((fx_array + self.num_constraints) // 2).astype(int)
                logp2fx = self.lookup_log_polynomial_squared[lookup_idx]
                return logp2fx
        else:
            raise ValueError("Optimal polynomial not set.")

    @overload
    def p2(self, fx: int | float) -> float: ...

    @overload
    def p2(self, fx: np.ndarray) -> np.ndarray: ...

    def p2(self, fx: int | float | np.ndarray) -> float | np.ndarray:
        """Return log(P^2(f)) for backward compatibility."""
        return self.log_poly_p2(fx)

    def logp2fx_fx(self, x: npt.NDArray[np.int_]) -> tuple[float, float]:
        """Return log(P^2(f(x))) and f(x) as Python floats for a single vector (1D array)"""
        fx_val = self.f(x)
        assert isinstance(fx_val, int)
        logp2fx = self.log_poly_p2(fx_val)
        return logp2fx, float(fx_val)

    def logp2fx(self, x: np.ndarray) -> float | np.ndarray:
        """
        Return log(P^2(f(x))) for binary vector(s) x.

        Args:
            x: Binary vector (1D) or batch of vectors (2D)

        Returns:
            log(P^2(f(x))) - float for single vector, ndarray for batch

        """
        fx = self.f(x)
        return self.log_poly_p2(fx)

    def n_satisfied(self, x: np.ndarray) -> int | np.ndarray:
        """
        Return the number of satisfied constraints for binary vector(s) x.

        Args:
            x: Binary vector (1D) or batch of vectors (2D)

        Returns:
            Number of satisfied constraints - int for single vector, ndarray for batch

        """
        match = self.match_vector(x)
        n_match = (match).sum(axis=-1)
        if isinstance(n_match, np.ndarray):
            return (
                int(n_match)
                if n_match.ndim == 0 or (n_match.ndim == 1 and n_match.shape[0] == 1)
                else n_match
            )
        else:
            return int(n_match)

    def n_predicted(self):
        """Return the predicted fraction of satisfied constraints"""
        m = self.num_constraints
        ks = np.arange(1, self.ell + 1, dtype=np.float64)
        off_diag = np.sqrt(ks * (m - ks + 1.0))
        diag = np.zeros(self.ell + 1, dtype=np.float64)
        _, w_k = eigh_tridiagonal(
            diag, off_diag, select="i", select_range=(self.ell, self.ell)
        )
        w_k = w_k[:, 0]
        inner = np.dot(off_diag, w_k[:-1] * w_k[1:])
        return 0.5 + inner / m
