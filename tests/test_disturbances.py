"""Tests for disturbances.py: every regime draws conditions within its stated ranges."""

import numpy as np
import pytest

from hummingbird_rl.envs.disturbances import REGIMES, grid_conditions, sample_conditions

N = 300


def draw(regime, params, rng):
    return [sample_conditions(regime, rng, params) for _ in range(N)]


def test_calm_is_calm(params, rng):
    c = sample_conditions("calm", rng, params)
    assert not c.mean_wind.any() and c.gust_sigma == 0 and c.action_noise == 0 and c.obs_noise == 0


def test_normal_ranges(params, rng):
    for c in draw("normal", params, rng):
        assert c.mean_wind[1] == 0.0                                  # horizontal
        assert np.linalg.norm(c.mean_wind) <= 0.05 * params.weight + 1e-12
        assert c.gust_sigma == pytest.approx(0.02 * params.weight)
        assert c.mass_factor == 1.0 and c.damage == (0.0, 0.0) and c.failure_time is None


def test_randomised_ranges(params, rng):
    cs = draw("randomised", params, rng)
    winds = np.array([np.linalg.norm(c.mean_wind) for c in cs]) / params.weight
    assert winds.max() <= 0.30 and winds.max() > 0.25
    assert all(0.75 <= c.mass_factor <= 1.25 for c in cs)
    damaged = [max(c.damage) for c in cs if max(c.damage) > 0]
    assert all(min(c.damage) == 0.0 for c in cs)                       # at most one wing
    assert max(damaged) <= 0.4
    assert 0.6 < len(damaged) / N < 0.8                                # about 70% of the episodes


def test_failure_regime(params, rng):
    cs = draw("failure", params, rng)
    assert all(1.5 <= c.failure_time <= 3.0 and c.failure_amount == 0.4 for c in cs)
    left = sum(c.failure_wing == 0 for c in cs) / N
    assert 0.4 < left < 0.6


def test_grid_cell(params, rng):
    c = grid_conditions(rng, params, wind_fraction=0.3, damage=0.2)
    assert np.linalg.norm(c.mean_wind) == pytest.approx(0.3 * params.weight)
    assert c.gust_sigma == pytest.approx(0.1 * params.weight)
    assert c.damage == (0.2, 0.0)
    assert grid_conditions(rng, params, 0.0, 0.0).gust_sigma == pytest.approx(0.02 * params.weight)


def test_unknown_regime_is_rejected(params, rng):
    with pytest.raises(ValueError):
        sample_conditions("stormy", rng, params)
    assert set(REGIMES) == {"calm", "normal", "randomised", "failure"}
