# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The DQI-MCMC Authors

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix

if TYPE_CHECKING:
    from numpy.typing import NDArray

# set a constant to the problems/maxxorsat dir relative to this file
# paper-release/dqi_mcmc/api/maxxorsat/ -> up 3 levels -> paper-release/ -> problems/maxxorsat/
PROBLEM_DIR = Path(__file__).parents[3] / "problems" / "maxxorsat"

_problem_files = {
    100: "irreg_LDPC_maxdeg1000_n62_m100.txt",
    250: "irreg_LDPC_maxdeg1000_n156_m250.txt",
    500: "irreg_LDPC_maxdeg1000_n312_m500.txt",
    750: "irreg_LDPC_maxdeg1000_n468_m750.txt",
    1000: "irreg_LDPC_maxdeg1000_n624_m1000.txt",
    3000: "irreg_LDPC_maxdeg1000_n1873_m3000.txt",
}

_rhs_files = {
    100: "rhs_nsamples100_m100_p2_r1.npy",
    250: "rhs_nsamples100_m250_p2_r1.npy",
    500: "rhs_nsamples100_m500_p2_r1.npy",
    750: "rhs_nsamples100_m750_p2_r1.npy",
    1000: "rhs_nsamples100_m1000_p2_r1.npy",
    3000: "rhs_nsamples100_m3000_p2_r1.npy",
}


def get_problem_keys():
    """Returns the available output sizes for problems."""
    return list(_problem_files.keys())


def get_problem_filepath(key: int) -> Path:
    return PROBLEM_DIR / _problem_files[key]


def get_rhs_filepath(key: int) -> Path:
    return PROBLEM_DIR / _rhs_files[key]


def read_matrix_from_tsv(file_path: str | Path, dtype=int) -> csr_matrix:
    data = []
    i = []
    j = []

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            x, y, v = map(int, line.strip().split("\t"))
            data.append(v)
            i.append(x)
            j.append(y)

    sparse_array = csr_matrix(
        (data, (i, j)), shape=(max(i) + 1, max(j) + 1), dtype=dtype
    )
    return sparse_array


def read_rhs_table(file_path: str | Path) -> "NDArray[np.int_]":
    """
    Read the RHS (right hand side, v vector) table from a file.
    the returned array holds (typically) 100 vectors the size of the output dim of the problem

    Args:
        file_path (str | Path): The path to the RHS table file.

    Returns:
        np.ndarray: The RHS table as a NumPy array of numpy arrays

    """
    return np.load(file_path).squeeze()  # type: ignore[return-value]


def read_problem(m: int, v_index: int = 0) -> tuple[csr_matrix, "NDArray[np.int_]"]:
    """
    Read a MAX-XOR-SAT problem from file.

    Args:
        m: The number of constraints (problem size). Must be one of the available sizes.
        v_index: Index of the right-hand side vector to use (0-99 for most problems).

    Returns:
        tuple: (b, v) where:
            - b: The constraint matrix (sparse CSR format)
            - v: The right-hand side vector

    Raises:
        KeyError: If m is not an available problem size.
        FileNotFoundError: If the RHS file has not been generated yet.
        IndexError: If v_index is out of range for the problem.

    Example:
        >>> b, v = read_problem(m=250, v_index=0)
        >>> print(f"Problem size: {b.shape}")
        Problem size: (250, 156)

    """
    if m not in _problem_files:
        available = sorted(_problem_files.keys())
        raise KeyError(
            f"Problem size m={m} not available. " f"Available sizes: {available}"
        )

    # Read the constraint matrix
    matrix_path = get_problem_filepath(m)
    b = read_matrix_from_tsv(matrix_path)

    # Read the RHS vectors
    rhs_path = get_rhs_filepath(m)
    if not rhs_path.exists():
        raise FileNotFoundError(
            f"RHS file not found: {rhs_path}\n" f"Run: generate-xorsat-rhs --m {m}"
        )
    rhs_table = read_rhs_table(rhs_path)

    # Select the specified vector
    if v_index < 0 or v_index >= len(rhs_table):
        raise IndexError(
            f"v_index={v_index} out of range. "
            f"Available indices: 0-{len(rhs_table)-1}"
        )

    v = rhs_table[v_index]

    return b, v
