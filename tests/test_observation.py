"""Tests for observation.py: layout, heading frame, empty slots and noise."""

import numpy as np
import pytest

from hummingbird_rl.envs import observation as ob
from hummingbird_rl.sim import BirdState
from hummingbird_rl.world.scene import START, Obstacle

R_BIRD = 0.02
ZERO_ACTION = np.zeros(5)


def observe(state, targets=(), obstacles=(), **kw):
    return ob.build_observation(state, list(targets), list(obstacles), ZERO_ACTION, 0.0, R_BIRD, **kw)


def test_layout_covers_the_whole_vector():
    assert ob.OBSTACLES.stop == ob.HOLD == ob.OBS_SIZE - 1
    assert ob.TARGETS.stop - ob.TARGETS.start == 4 * ob.N_TARGETS
    assert ob.OBSTACLES.stop - ob.OBSTACLES.start == 5 * ob.N_OBSTACLES


def test_target_ahead_looks_the_same_whatever_the_heading():
    """A target 20 cm in front of the beak must give the same numbers for any yaw."""
    seen = []
    for yaw in np.radians([0, 90, -135]):
        state = BirdState.from_euler(pos=START, yaw=yaw)
        forward = np.array([np.cos(yaw), 0.0, -np.sin(yaw)])     # where the beak points in the world
        seen.append(observe(state, [START + 0.2 * forward])[ob.TARGETS][:4])
    for s in seen:
        assert np.allclose(s, [0.4, 0.0, 0.0, 1.0], atol=1e-6)


def test_velocity_in_heading_frame():
    state = BirdState.from_euler(pos=START, vel=(0.0, 0.0, 0.3), yaw=np.pi / 2)  # flying along world z
    # Facing -z after a 90 degree yaw to the left, so world +z is straight behind.
    assert np.allclose(observe(state)[ob.VELOCITY], [-0.3, 0.0, 0.0], atol=1e-6)


def test_empty_slots_are_zero():
    obs = observe(BirdState.from_euler(pos=START), [START + [0.1, 0, 0]])
    assert obs[ob.TARGETS][7] == 0.0 and not obs[ob.TARGETS][4:].any()      # no second target
    assert not obs[ob.OBSTACLES].any()


def test_obstacles_closest_first_and_capped_at_five():
    obstacles = [Obstacle(START + [0.1 * (i + 2), 0.0, 0.0], 0.05) for i in range(7)]
    obs = observe(BirdState.from_euler(pos=START), obstacles=obstacles[::-1])
    blocks = obs[ob.OBSTACLES].reshape(ob.N_OBSTACLES, 5)
    assert np.all(blocks[:, 4] == 1.0)
    assert np.all(np.diff(blocks[:, 0]) > 0)                    # dx grows: closest first
    assert blocks[0, 0] == pytest.approx(0.2 / ob.POS_SCALE)
    assert blocks[0, 3] == pytest.approx(0.05 / ob.RADIUS_SCALE)


def test_noise_only_on_measured_entries(rng):
    state = BirdState.from_euler(pos=START)
    clean = observe(state, [START + [0.1, 0, 0]])
    noisy = observe(state, [START + [0.1, 0, 0]], noise=0.01, rng=rng)
    diff = noisy - clean
    assert np.all(diff[:ob.RATES.stop] != 0) and np.all(diff[ob.TARGETS][:3] != 0)
    assert not diff[ob.PREV_ACTION].any() and diff[ob.TARGETS][3] == 0.0 and diff[ob.HOLD] == 0.0
    assert np.abs(diff).max() < 0.06


def test_values_are_clipped():
    obs = observe(BirdState.from_euler(pos=START, vel=(10.0, 0, 0)))
    assert obs.dtype == np.float32 and obs.max() <= ob.CLIP
