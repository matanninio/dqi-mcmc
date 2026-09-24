# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 IBM Corporation

"""Shared pytest fixtures and markers for the DQI-MCMC paper-release test suite."""

from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BASE_DIR = Path(__file__).resolve().parents[1]
_P7_RHS_FILE = _BASE_DIR / "problems" / "opi" / "rhs_nsamples100_p7_r3.npy"
_P11_RHS_FILE = _BASE_DIR / "problems" / "opi" / "rhs_nsamples100_p11_r5.npy"


# ---------------------------------------------------------------------------
# Custom markers
# ---------------------------------------------------------------------------
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_p7_rhs_file: skip if problems/opi/rhs_nsamples100_p7_r3.npy is absent",
    )
    config.addinivalue_line(
        "markers",
        "requires_p11_rhs_file: skip if problems/opi/rhs_nsamples100_p11_r5.npy is absent",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tiny_xorsat_problem():
    """
    A tiny MaxXorPSatProblem built from an in-memory 10×6 binary matrix.

    m=10 constraints, n=6 variables, ell=3, p=2 (XOR-SAT).
    """
    from dqi_mcmc.api.maxxorsat.max_xor_p_sat_problem import MaxXorPSatProblem

    rng = np.random.default_rng(0)
    b = rng.integers(0, 2, size=(10, 6)).astype(np.int8)
    v = rng.integers(0, 2, size=10).astype(np.int8)
    return MaxXorPSatProblem(b=b, v=v, p=2, ell=3, use_cache=False)


@pytest.fixture(scope="session")
def tiny_opi_problem():
    """
    A tiny MaxOPIProblem with p=7, v of shape (6, 3), num_variables=3, ell=2.

    p=7 → num_constraints = p-1 = 6, r=3 (p//2).
    """
    from dqi_mcmc.api.maxlinsat.max_opi_problem import MaxOPIProblem

    p = 7
    r = p // 2  # 3
    rng = np.random.default_rng(1)
    # v shape: (p-1, r) = (6, 3), values in [0, p)
    v = np.array(
        [rng.choice(p, size=r, replace=False) for _ in range(p - 1)],
        dtype=np.uint8,
    )
    return MaxOPIProblem(p=p, v=v, num_variables=3, ell=2, use_cache=False)


@pytest.fixture
def p7_rhs_file():
    """Path to the p=7 RHS file; skips if absent."""
    if not _P7_RHS_FILE.exists():
        pytest.skip(f"RHS file not found: {_P7_RHS_FILE}")
    return _P7_RHS_FILE


@pytest.fixture
def p11_rhs_file():
    """Path to the p=11 RHS file; skips if absent."""
    if not _P11_RHS_FILE.exists():
        pytest.skip(f"RHS file not found: {_P11_RHS_FILE}")
    return _P11_RHS_FILE
