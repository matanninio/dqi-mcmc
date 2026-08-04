# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The DQI-MCMC Authors

import math
from math import comb
from typing import overload

import numpy as np
import numpy.typing as npt
from scipy.linalg import eigh_tridiagonal
from scipy.special import logsumexp

from dqi_mcmc.api._elementary_poly_fast import build_elementary_poly_cache_fast


def evaluate_elementary_poly_pure_python(
    m: int, num_ones: int, degree: int, g_plus: float, g_minus: float
) -> float:
    """
    Cached computation of elementary polynomial of degree `k=degree`, e_k, evaluated on
    the given number of ones.

    Args:
        m (int): The number of constraints.
        num_ones (int): The number of ones (satisfied constraints).
        degree (int): The degree of the elementary polynomial.
        g_plus, g_minus (float): The variables on which e_k is evaluated.

    Returns:
        int: The value of the polynomial e_k(num_ones).

    """
    # Valid extremal values for i where both binomials are nonzero
    i_lo = max(0, degree - (m - num_ones))
    i_hi = min(degree, num_ones)
    if i_lo > i_hi:
        return 0

    # we compute binom(n, k+1) from binom(n,k)
    # Start values
    c1 = comb(num_ones, i_lo)  # C(sx, i)
    j = degree - i_lo
    c2 = comb(m - num_ones, j)  # C(n - sx, m - i)
    # Start powers: g_plus^i * g_minus^j
    gp_pow = g_plus**i_lo
    gm_pow = g_minus**j
    inv_gm = 1.0 / g_minus

    e_k = 0
    for i in range(i_lo, i_hi + 1):
        e_k += (c1 * c2) * (gp_pow * gm_pow)

        # advance i -> i+1, j -> j-1
        if i < i_hi:
            c1 = (c1 * (num_ones - i)) // (i + 1)  # C(sx, i+1)
            c2 = (c2 * j) // (m - num_ones - j + 1)  # C(n-sx, j-1)
            j -= 1
            gp_pow *= g_plus  # g_plus^(i+1)
            gm_pow *= inv_gm  # g_minus^(j-1)

    return e_k


def compute_optimal_lookup_table_opi(
    num_variables: int,
    num_constraints: int,
    ell: int,
    r: int,
    p: int,
    eps: float = 1e-300,
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
        r (int): Size of the subsets F_i
        p (int): The prime number defining the problem
        eps (float): Small value to avoid numerical issues (default: 1e-300).
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
    # Diagonal formula: d= (p - 2*r) / sqrt(r * (p-r))
    # For OPI, the diagonal should be 0,d,2d,3d,...
    d = (p - 2 * r) / np.sqrt(r * (p - r))
    diag = np.arange(ell + 1, dtype=np.float64) * d
    # get only the largest eigenpair
    _, w_k = eigh_tridiagonal(diag, off_diag, select="i", select_range=(ell, ell))
    w_k = w_k[:, 0]

    # Pre-compute log values for w_k (optimization)
    log_w_k = np.log(np.abs(w_k))
    sign_w_k = np.sign(w_k)

    # =============================================
    # 2. Pre-compute all elementary polynomial values with fast cache
    # =============================================
    fast_cache = build_elementary_poly_cache_fast(m, ell, p=p, r=r)
    log_cache, sign_cache = fast_cache.compute_log_cache()

    # =============================================
    # 3. Initialize the lookup table
    # =============================================
    lookup_log_polynomial_squared = np.zeros(m + 1)

    # =============================================
    # 4. Compute log(P^2(f(x))) using pre-computed log cache
    # =============================================
    # Compute constants for OPI
    # Note that P(f) = \sum_k u_k e_k(f, fbar)
    # where u_k = w_k / sqrt{2^n * binom(m, k) * varphi^k}
    # varphi = 2 * r * (p-r) / p, so log(1/varphi) = log(p) - log(2) - log(r) - log(p-r)
    log_2 = np.log(2)
    log_weight = (log_2 - np.log(r) - np.log(p - r) - np.log(p)) / 2

    # Pre-compute log(binom(m, k)) for all k (optimization)
    log_binom_m = np.zeros(ell + 1)
    for k in range(1, ell + 1):
        log_binom_m[k] = log_binom_m[k - 1] + np.log(m - k + 1) - np.log(k)

    n_log_2 = n * log_2

    for num_ones in range(0, m + 1):
        log_p_elem_k, signs = [], []
        log_k_weight = 0
        for k in range(ell + 1):
            # Use pre-computed log and sign values
            log_ek = log_cache[num_ones, k]
            if np.isinf(log_ek):  # e_k was zero
                continue
            sign_ek = sign_cache[num_ones, k]

            # Compute the contribution to P(f(x))
            log_binom_k = log_binom_m[k] / 2
            log_w_k_weight = log_k_weight - log_binom_k
            log_p_elem_k.append(log_ek + log_w_k[k] + log_w_k_weight)
            signs.append(sign_ek * sign_w_k[k])

            # update numerators and denominators
            log_k_weight += log_weight

        log_pfx, _ = logsumexp(a=log_p_elem_k, b=signs, return_sign=True)
        # Store log[P^2(f(x))] = 2*log(P(f(x))) - n*log(2)
        lookup_log_polynomial_squared[num_ones] = 2 * log_pfx - n_log_2

    # Optionally normalize lookup_log_polynomial_squared so minimum is 0
    if normalize:
        finite_mask = np.isfinite(lookup_log_polynomial_squared)
        if np.any(finite_mask):
            min_log = np.min(lookup_log_polynomial_squared[finite_mask])
            lookup_log_polynomial_squared[finite_mask] -= min_log

    return lookup_log_polynomial_squared


def distinct_prime_factors(n: int) -> list[int]:
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        factors.append(n)
    return factors


def smallest_primitive_root(p: int) -> int:
    r"""Compute the smallest primitive root of F_p."""
    if p == 2:
        return 1
    phi = p - 1
    primes = distinct_prime_factors(phi)

    for g in range(2, p):
        if all(pow(g, phi // q, p) != 1 for q in primes):
            return g
    raise RuntimeError("No primitive root found; is p prime?")


class MaxOPIProblem:
    r"""
    Optimal Polynomial Intersection (OPI).

    Given a prime number p and an integer n < p-1, let
    F_1, ..., F_{p-1} be subsets of the finite field F_p. The Optimal Polynomial
    Intersection problem consists of finding a polynomial Q in F_p[y] of degree at
    most n-1 that maximizes f_OPI(Q).

    [1] Gil-Fuster, E., Ninio, M., Bittel, L., Shimoni, Y., Eisert, J., Woerner, S., and Carrera Vázquez, A.
        Approximate sampling from decoded quantum interferometry via Markov chain Monte Carlo methods.
        arXiv (2026). https://arxiv.org/abs/2607.28120

    """

    def __init__(
        self,
        p: int,
        v: np.ndarray,
        num_variables: int,
        ell: int | None = None,
        gamma: int | None = None,
        use_cache: bool = True,
    ) -> None:
        r"""
        Initializes the OPI problem with the given parameters.

        Args:
            p (int): An integer representing the prime modulus for the finite field.
            v (np.ndarray): A 2D NumPy array of shape (n, r) whose rows encode the sets F_i.
            num_variables (int) : The number of variables.
            ell (int): The degree of the optimal polynomial. By default is (n + 1) // 2.
            gamma (int | None): A primitive element of F_p. If None, computed automatically.

        """
        self.v = v
        self.p = p
        self.num_variables = num_variables

        # Check if 'v' is two-dimensional
        if v.ndim != 2:
            raise ValueError("Array 'v' must be two-dimensional.")
        self.num_constraints = p - 1
        if v.shape[0] != self.num_constraints:
            raise ValueError("The length of 'v' should be 'p-1'.")
        self.r: int = v.shape[1]

        # Determine polynomial degree 'ell' and construct lookup table
        if ell is not None:
            self.ell = ell
        else:
            self.ell = (self.num_variables + 1) // 2

        # Set the gamma primitive of F_p
        if gamma is not None:
            self.gamma = gamma
        else:
            self.gamma = smallest_primitive_root(self.p)

        # construct the lookup table
        self.set_optimal_lookup_table(use_cache=use_cache)

        # Pre-compute all powers of gamma needed for polynomial evaluation
        self.gamma_powers = np.zeros((self.num_constraints, self.num_variables), dtype=np.int64)
        t = 1  # gamma^0
        for i in range(self.num_constraints):
            power = 1  # t^0
            for j in range(self.num_variables):
                self.gamma_powers[i, j] = power
                power = (power * t) % self.p
            t = (t * self.gamma) % self.p

        self.v_mask = np.zeros((self.num_constraints, self.p), dtype=bool)
        for i in range(self.num_constraints):
            self.v_mask[i, self.v[i]] = True

    def set_optimal_lookup_table(self, eps=1e-300, use_cache: bool = True) -> None:
        r"""
        Construct a lookup table for the polynomial P.

        This method first attempts to load pre-computed tables from disk cache.
        If not found, it computes them using compute_optimal_lookup_table_opi.

        Args:
            eps: float is 1e-300 by default.
            use_cache: If True, attempt to load from disk cache before computing

        """
        if use_cache:
            try:
                from ..lookup_cache import load_lookup_table

                cached = load_lookup_table(
                    self.num_constraints,
                    self.num_variables,
                    self.ell,
                    p=self.p,
                    r=self.r,
                )
                if cached is not None:
                    self.lookup_log_polynomial_squared = cached
                    return
            except ImportError:
                pass

        self.lookup_log_polynomial_squared = compute_optimal_lookup_table_opi(
            self.num_variables, self.num_constraints, self.ell, self.r, self.p, eps
        )

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

    def n_satisfied(self, x: np.ndarray) -> int:
        """Return the value of the function for a vector x"""
        poly_vals = np.sum(x * self.gamma_powers, axis=1) % self.p
        num_satisfied = np.sum(self.v_mask[np.arange(self.num_constraints), poly_vals])

        return num_satisfied

    def n_predicted(self):
        """Return the predicted fraction of satisfied constraints"""
        ell_over_em = self.ell / self.num_constraints
        r_over_p = self.r / self.p

        first_term = math.sqrt(ell_over_em * (1 - r_over_p))
        second_term = math.sqrt(r_over_p * (1 - ell_over_em))

        return (first_term + second_term) ** 2
