# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The DQI-MCMC Authors

"""Protocol interface for SAT-like problems that can be sampled with MCMC."""

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


@runtime_checkable
class SATLikeProblem(Protocol):
    """
    Protocol defining the interface for SAT-like problems.

    This protocol defines the minimal interface required for a problem to be
    compatible with the MCMC samplers.

    Both MaxXorPSatProblem and MaxOPIProblem implement this protocol.
    """

    num_variables: int
    """Number of variables in the problem."""

    p: int
    """Field size (2 for binary XOR-SAT, larger primes for OPI problems)."""

    def f(self, x: npt.NDArray[np.int_]) -> int | npt.NDArray[np.int_]:
        """
        Compute the objective function value for a given state.

        Args:
            x: Binary vector representing a state (1D) or batch of states (2D)

        Returns:
            Objective function value (int for single state, array for batch)
            Typically 2*satisfied - m

        """
        ...

    def logp2fx_fx(self, x: npt.NDArray[np.int_]) -> tuple[float, float]:
        """
        Compute log(P^2(f(x))) and f(x) for a given state.

        Args:
            x: Binary vector representing a state

        Returns:
            Tuple of (log(P^2(f(x))), f(x))

        """
        ...
