"""Tests for the tasks: reward signs, the hold rule, success and waypoint switching.

These use hand-made states rather than the simulator, so that each rule is
checked on its own.
"""

import numpy as np
import pytest

from hummingbird_rl.sim import BirdState, hover_action
from hummingbird_rl.tasks import GotoTask, HoverTask, NavigationTask
from hummingbird_rl.tasks.base import HOLD_STEPS
from hummingbird_rl.world.scene import START, Scene


def still_at(pos) -> BirdState:
    return BirdState.from_euler(pos=pos)


@pytest.fixture
def hover_a(params):
    return hover_action(params)


def test_episode_lengths(params):
    assert HoverTask(params).n_steps == 300
    assert NavigationTask(params).n_steps == 900


def test_hover_reward_is_highest_at_the_target(params, hover_a):
    task = HoverTask(params)
    scene = Scene(targets=[START.copy()])
    rewards = []
    for offset in (0.0, 0.05, 0.20):
        state = still_at(START + [offset, 0, 0])
        task.reset(scene, state)
        rewards.append(task.evaluate(state, hover_a, hover_a, params.hover_power()).reward)
    assert rewards[0] > rewards[1] > rewards[2]
    assert rewards[0] == pytest.approx(1.0 - 0.01, abs=1e-6)    # only the energy term at hover power


def test_hover_success_judged_on_the_last_two_seconds(params, hover_a):
    task = HoverTask(params)
    scene = Scene(targets=[START.copy()])
    far, near = still_at(START + [0.2, 0, 0]), still_at(START + [0.01, 0, 0])
    task.reset(scene, far)
    for i in range(task.n_steps):
        out = task.evaluate(far if i < 150 else near, hover_a, hover_a, 0.0)
        assert out.success == (i == task.n_steps - 1)     # far for 2.5 s, then near: a success
        assert not out.terminated


def test_smoothness_and_energy_penalties(params, hover_a):
    task = HoverTask(params)
    task.reset(Scene(targets=[START.copy()]), still_at(START))
    base = task.common_terms(hover_a, hover_a, 0.0)
    assert base == 0.0
    assert task.common_terms(hover_a + 0.5, hover_a, 0.0) < base
    assert task.common_terms(hover_a, hover_a, 2 * params.hover_power()) == pytest.approx(-0.02)


def test_goto_rewards_progress(params, hover_a):
    task = GotoTask(params)
    target = START + np.array([0.3, 0.0, 0.0])
    task.reset(Scene(targets=[target]), still_at(START))
    closer = task.evaluate(still_at(START + [0.01, 0, 0]), hover_a, hover_a, 0.0).reward
    back = task.evaluate(still_at(START), hover_a, hover_a, 0.0).reward
    assert closer > 0.9 and back < -0.9           # about 1 point per centimetre


def test_goto_succeeds_only_after_a_full_hold(params, hover_a):
    task = GotoTask(params)
    target = START + np.array([0.3, 0.0, 0.0])
    task.reset(Scene(targets=[target]), still_at(target))
    for _ in range(HOLD_STEPS - 1):
        assert not task.evaluate(still_at(target), hover_a, hover_a, 0.0).terminated
    out = task.evaluate(still_at(target), hover_a, hover_a, 0.0)
    assert out.terminated and out.success and out.reward > 50


def test_flying_through_the_target_resets_the_hold(params, hover_a):
    task = GotoTask(params)
    target = START + np.array([0.3, 0.0, 0.0])
    task.reset(Scene(targets=[target]), still_at(target))
    fast = BirdState.from_euler(pos=target, vel=(0.5, 0, 0))
    for _ in range(3 * HOLD_STEPS):
        assert not task.evaluate(fast, hover_a, hover_a, 0.0).terminated
    assert task.hold_progress() == 0.0


def test_navigation_switches_waypoints_in_order(params, hover_a):
    task = NavigationTask(params)
    wps = [START + [0.3, 0, 0], START + [0.3, 0, 0.3], START + [0.0, 0, 0.3]]
    task.reset(Scene(targets=wps), still_at(START))
    assert np.allclose(task.obs_targets()[0], wps[0]) and len(task.obs_targets()) == 2

    out = task.evaluate(still_at(wps[0]), hover_a, hover_a, 0.0)          # pass waypoint 1
    assert task.index == 1 and out.info["waypoints_reached"] == 1 and out.reward > 20
    # The progress reference was reset: standing still at the next step gives no big penalty.
    assert task.evaluate(still_at(wps[0]), hover_a, hover_a, 0.0).reward > -0.1

    task.evaluate(still_at(wps[1]), hover_a, hover_a, 0.0)                # pass waypoint 2
    assert task.is_last and len(task.obs_targets()) == 1

    # Passing the last one is not enough: it has to be held.
    outs = [task.evaluate(still_at(wps[2]), hover_a, hover_a, 0.0) for _ in range(HOLD_STEPS)]
    assert not any(o.terminated for o in outs[:-1])
    assert outs[-1].terminated and outs[-1].success and outs[-1].info["waypoints_reached"] == 3
