"""Shared fixtures for the test suite."""

import numpy as np
import pytest

from hummingbird_rl.sim import BirdDynamics, BirdParams


@pytest.fixture
def params() -> BirdParams:
    return BirdParams()


@pytest.fixture
def stable_params() -> BirdParams:
    """Same bird without the hover instability (k_u = 0)."""
    return BirdParams().with_changes(k_u_pitch=0.0, k_u_roll=0.0)


@pytest.fixture
def dyn(params) -> BirdDynamics:
    return BirdDynamics(params)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)
