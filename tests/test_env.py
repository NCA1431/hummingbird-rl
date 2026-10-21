"""Tests for the Gymnasium environment as a whole."""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from hummingbird_rl.envs import ENV_IDS, Conditions, HummingbirdEnv, make_env
from hummingbird_rl.sim import BirdState
from hummingbird_rl.world.scene import START, Obstacle, Scene

TASK_NAMES = ["hover", "goto", "obstacles", "navigation"]


@pytest.mark.parametrize("task", TASK_NAMES)
def test_passes_gymnasium_checks(task):
    check_env(HummingbirdEnv(task=task), skip_render_check=True)


@pytest.mark.parametrize("task", TASK_NAMES)
def test_registered_ids(task):
    env = gym.make(ENV_IDS[task])
    obs, _ = env.reset(seed=0)
    assert obs.shape == (48,)
    env.close()


def test_same_seed_same_episode():
    def rollout(seed):
        env = make_env("goto", seed=seed)
        out = []
        for _ in range(50):
            obs, r, *_ = env.step(env.hover)
            out.append(r)
        return np.array(out), obs
    (r1, o1), (r2, o2), (r3, _) = rollout(3), rollout(3), rollout(4)
    assert np.array_equal(r1, r2) and np.array_equal(o1, o2)
    assert not np.array_equal(r1, r3)


def test_constant_hover_action_does_not_solve_hover():
    """Hovering is unstable: replaying the hover action open-loop must fail almost always."""
    env = make_env("hover", seed=0)
    successes, episodes = 0, 0
    while episodes < 30:
        _, _, term, trunc, info = env.step(env.hover)
        if term or trunc:
            successes += info["is_success"]
            episodes += 1
            env.reset()
    assert successes / episodes < 0.1


def test_falling_ends_with_a_crash():
    env = make_env("hover", regime="calm", seed=0)
    env.reset(options={"state": BirdState.from_euler(pos=START)})
    wings_stopped = -np.ones(5)
    for _ in range(200):
        _, r, term, trunc, info = env.step(wings_stopped)
        if term:
            break
    assert term and info["reason"] == "crash" and r == -10.0


def test_hitting_an_obstacle_ends_with_a_collision():
    env = make_env("obstacles", regime="calm", seed=0)
    wall = Obstacle(START + np.array([0.04, 0.0, 0.0]), 0.03)       # just in front of the bird
    scene = Scene(targets=[START + np.array([0.6, 0.0, 0.0])], obstacles=[wall])
    env.reset(options={"scene": scene, "state": BirdState.from_euler(pos=START, vel=(0.5, 0, 0))})
    _, r, term, _, info = env.step(env.hover)
    assert term and info["reason"] == "collision"


def test_hover_episode_is_truncated_at_five_seconds():
    """With the instability switched off the hover action survives, and time runs out at 300 steps."""
    from hummingbird_rl.sim import BirdParams
    p = BirdParams().with_changes(k_u_pitch=0.0, k_u_roll=0.0)
    env = HummingbirdEnv("hover", regime="calm", params=p)
    env.reset(seed=0, options={"state": BirdState.from_euler(pos=START)})
    for n in range(1, 301):
        _, _, term, trunc, info = env.step(env.hover)
        if term or trunc:
            break
    assert n == 300 and trunc and not term
    assert info["reason"] == "time_limit" and info["is_success"]
    assert info["time"] == pytest.approx(5.0)


def test_sudden_failure_happens_on_time():
    env = make_env("hover", regime="calm", seed=0)
    failing = Conditions(failure_time=0.5, failure_wing=1, failure_amount=0.4)
    env.reset(options={"conditions": failing, "state": BirdState.from_euler(pos=START)})
    for _ in range(29):                        # 29 decisions = 0.483 s
        env.step(env.hover)
    assert env.dyn.damage == (0.0, 0.0)
    env.step(env.hover)
    env.step(env.hover)
    assert env.dyn.damage == (0.0, 0.4)


def test_fixed_conditions_are_used(params):
    env = make_env("hover", seed=0)
    heavy = Conditions(mass_factor=1.2, damage=(0.1, 0.0))
    _, info = env.reset(options={"conditions": heavy})
    assert env.dyn.mass == pytest.approx(1.2 * params.mass)
    assert info["conditions"]["damage"] == (0.1, 0.0)


def test_failure_reward_can_be_changed():
    env = HummingbirdEnv("hover", regime="calm", failure_reward=-50.0)
    env.reset(seed=0, options={"state": BirdState.from_euler(pos=START)})
    for _ in range(200):
        _, r, term, _, info = env.step(-np.ones(5))      # wings stopped: the bird falls
        if term:
            break
    assert info["reason"] == "crash" and r == -50.0


def test_number_of_obstacles_reaches_the_sampler():
    env = HummingbirdEnv("navigation", regime="calm", n_obstacles=(3, 3))
    for seed in range(3):
        env.reset(seed=seed)
        assert len(env.scene.obstacles) == 3
