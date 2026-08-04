# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The DQI-MCMC Authors

"""
RNG state serialization utilities for numpy.random.Generator.

This module provides utilities to save and restore the state of numpy's
modern Generator RNG, enabling checkpoint/resume functionality without
managing explicit seeds.
"""

import json
from typing import Any

import numpy as np


def serialize_rng_state(rng: np.random.Generator) -> dict[str, Any]:
    """
    Serialize the state of a numpy.random.Generator to a JSON-compatible dict.

    Args:
        rng: A numpy.random.Generator instance

    Returns:
        A JSON-serializable dict containing the RNG state

    """
    state = rng.bit_generator.state

    serialized = {
        "bit_generator": state["bit_generator"],
        "state": {},
        "has_uint32": state.get("has_uint32", 0),
        "uinteger": state.get("uinteger", 0),
    }

    for key, value in state["state"].items():
        if isinstance(value, (int, np.integer)):
            serialized["state"][key] = str(value)
        elif isinstance(value, np.ndarray):
            serialized["state"][key] = value.tolist()
        elif isinstance(value, dict):
            serialized["state"][key] = _serialize_nested_dict(value)
        else:
            serialized["state"][key] = value

    return serialized


def _serialize_nested_dict(d: dict) -> dict:
    """Helper to recursively serialize nested dictionaries."""
    result = {}
    for key, value in d.items():
        if isinstance(value, (int, np.integer)):
            result[key] = str(value)
        elif isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, dict):
            result[key] = _serialize_nested_dict(value)
        else:
            result[key] = value
    return result


def deserialize_rng_state(rng: np.random.Generator, state_dict: dict[str, Any]) -> None:
    """
    Restore the state of a numpy.random.Generator from a serialized dict.

    This modifies the RNG in-place to restore it to the saved state.

    Args:
        rng: A numpy.random.Generator instance to restore
        state_dict: A dict previously created by serialize_rng_state()

    """
    restored_state = {
        "bit_generator": state_dict["bit_generator"],
        "state": {},
        "has_uint32": state_dict.get("has_uint32", 0),
        "uinteger": state_dict.get("uinteger", 0),
    }

    for key, value in state_dict["state"].items():
        if isinstance(value, str) and value.isdigit():
            restored_state["state"][key] = int(value)
        elif isinstance(value, list):
            restored_state["state"][key] = np.array(value)
        elif isinstance(value, dict):
            restored_state["state"][key] = _deserialize_nested_dict(value)
        else:
            restored_state["state"][key] = value

    rng.bit_generator.state = restored_state


def _deserialize_nested_dict(d: dict) -> dict:
    """Helper to recursively deserialize nested dictionaries."""
    result = {}
    for key, value in d.items():
        if isinstance(value, str) and value.isdigit():
            result[key] = int(value)
        elif isinstance(value, list):
            result[key] = np.array(value)
        elif isinstance(value, dict):
            result[key] = _deserialize_nested_dict(value)
        else:
            result[key] = value
    return result


def save_rng_state_to_file(rng: np.random.Generator, filepath: str) -> None:
    """
    Save RNG state to a JSON file.

    Args:
        rng: A numpy.random.Generator instance
        filepath: Path to save the state file

    """
    state = serialize_rng_state(rng)
    with open(filepath, "w") as f:
        json.dump(state, f, indent=2)


def load_rng_state_from_file(filepath: str) -> np.random.Generator:
    """
    Load RNG state from a JSON file and create a new Generator with that state.

    Args:
        filepath: Path to the saved state file

    Returns:
        A new Generator instance with the restored state

    """
    with open(filepath) as f:
        state_dict = json.load(f)

    rng = np.random.default_rng()
    deserialize_rng_state(rng, state_dict)
    return rng


def capture_checkpoint_state(rng: np.random.Generator, sampler) -> dict[str, Any]:
    """
    Capture complete checkpoint state including both RNG and sampler state.

    Args:
        rng: A numpy.random.Generator instance
        sampler: A sampler instance with get_permutation_state() method

    Returns:
        A dictionary containing both 'random_state' and 'permutation_state' keys

    """
    return {
        "random_state": serialize_rng_state(rng),
        "permutation_state": sampler.get_permutation_state(),
    }


def restore_checkpoint_state(
    rng: np.random.Generator, sampler, checkpoint_state: dict[str, Any]
) -> None:
    """
    Restore complete checkpoint state including both RNG and sampler state.

    Args:
        rng: A numpy.random.Generator instance to restore
        sampler: A sampler instance with set_permutation_state() method
        checkpoint_state: Dictionary with 'random_state' and 'permutation_state' keys

    """
    deserialize_rng_state(rng, checkpoint_state["random_state"])
    sampler.set_permutation_state(checkpoint_state["permutation_state"])
