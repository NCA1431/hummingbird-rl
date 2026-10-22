"""Tests for the web exports: recorded episodes, policy networks, golden files."""

import json

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO, SAC, TD3

from hummingbird_rl.controllers import LQRController, PIDController
from hummingbird_rl.envs import HummingbirdEnv
from hummingbird_rl.envs.observation import build_observation
from hummingbird_rl.evaluation import TEST_SEED, run_episode
from hummingbird_rl.export import (
    build_index,
    build_policy_index,
    golden_dynamics,
    golden_episodes,
    golden_observations,
    golden_policies,
    lqr_dict,
    mlp_forward,
    policy_to_dict,
    record_episode,
    save_json,
)
from hummingbird_rl.sim import DEFAULT_PARAMS, BirdDynamics, BirdParams, BirdState, hover_action
from hummingbird_rl.world.scene import Obstacle


@pytest.mark.parametrize("task", ["goto", "navigation"])
def test_recorded_episode_is_the_evaluated_one(task):
    """Same seed, same loop: the viewer must show the episode that the evaluation counted."""
    ep = record_episode(HummingbirdEnv(task, "normal"), PIDController(), TEST_SEED)
    ref = run_episode(HummingbirdEnv(task, "normal"), PIDController(), TEST_SEED)
    assert ep["result"]["success"] == ref["success"] and ep["result"]["reason"] == ref["reason"]
    assert ep["result"]["length"] == ref["length"] and ep["result"]["return"] == pytest.approx(ref["return"])
    frames = ep["frames"]
    assert all(len(v) == ref["length"] + 1 for v in frames.values())     # the start + one frame per decision
    assert frames["t"][0] == 0.0 and np.allclose(frames["pos"][0], ep["scene"]["start"], atol=0.03)
    assert all(a <= b for a, b in zip(frames["target"], frames["target"][1:]))   # targets only move forward
    json.dumps(ep)                                                       # plain JSON, nothing numpy left


def test_index_lists_every_episode(tmp_path):
    ep = record_episode(HummingbirdEnv("hover", "normal"), PIDController(), TEST_SEED)
    for seed in (1, 2):
        save_json({**ep, "seed": seed, "policy": "pid", "suite": "test"}, tmp_path / "pid" / "hover" / f"{seed}.json")
    save_json({"episodes": []}, tmp_path / "index.json")                  # an old index is not an episode
    index = build_index(tmp_path)
    assert [e["file"] for e in index] == ["pid/hover/1.json", "pid/hover/2.json"]
    assert index[0]["policy"] == "pid" and index[0]["task"] == "hover"


@pytest.mark.parametrize("algo", [PPO, SAC, TD3])
def test_exported_network_gives_the_same_actions(algo):
    env = HummingbirdEnv("goto", "normal")
    kwargs = {} if algo is PPO else {"buffer_size": 100}
    model = algo("MlpPolicy", env, policy_kwargs={"net_arch": [32, 32]}, seed=0, device="cpu", **kwargs)
    with torch.no_grad():        # bigger random weights, so that clipping and tanh saturation do occur
        for w in model.policy.parameters():
            w.add_(0.3 * torch.randn_like(w))
    policy = json.loads(json.dumps(policy_to_dict(model, "test")))
    rng = np.random.default_rng(0)
    for _ in range(20):
        obs = rng.uniform(-2.0, 2.0, 48).astype(np.float32)
        expected, _ = model.predict(obs, deterministic=True)
        assert np.allclose(mlp_forward(policy, obs), expected, atol=1e-5)


def test_golden_dynamics_replay_from_the_file():
    """Replaying the stored inputs gives the stored outputs: the file is self-consistent."""
    data = json.loads(json.dumps(golden_dynamics(n_cases=3, n_decisions=20)))
    assert data == json.loads(json.dumps(golden_dynamics(n_cases=3, n_decisions=20)))     # deterministic
    for case in data["cases"]:
        p = BirdParams(**{k: tuple(v) if isinstance(v, list) else v for k, v in case["params"].items()})
        dyn = BirdDynamics(p, case["mass_factor"], tuple(case["damage"]))
        s = case["start"]
        state = BirdState(*(np.array(s[k]) for k in ("pos", "vel", "quat", "omega")), s["time"], s["phase"])
        for k, (action, wind) in enumerate(zip(case["actions"], case["wind"])):
            for _ in range(p.action_repeat):
                state, _ = dyn.step(state, np.array(action), np.array(wind))
            assert np.allclose(state.pos, case["expected"]["pos"][k], rtol=1e-10, atol=1e-12)
            assert np.allclose(state.quat, case["expected"]["quat"][k], rtol=1e-10, atol=1e-12)


def test_golden_observations_replay_from_the_file():
    data = json.loads(json.dumps(golden_observations(n_cases=10)))
    for case in data["cases"]:
        s = case["state"]
        state = BirdState(np.array(s["pos"]), np.array(s["vel"]), np.array(s["quat"]), np.array(s["omega"]))
        obstacles = [Obstacle(np.array(o["center"]), o["radius"]) for o in case["obstacles"]]
        obs = build_observation(state, [np.array(t) for t in case["targets"]], obstacles,
                                np.array(case["prev_action"]), case["hold"], case["r_bird"])
        assert np.array_equal(obs, np.array(case["expected"], dtype=np.float32))


def test_lqr_export_is_the_controllers_gain():
    data = json.loads(json.dumps(lqr_dict()))
    assert np.array_equal(np.array(data["K"]), LQRController().K)          # exact, through JSON
    assert data["u0"] == hover_action(DEFAULT_PARAMS).tolist()


def test_policy_index_lists_every_network(tmp_path):
    model = PPO("MlpPolicy", HummingbirdEnv("hover", "normal"), policy_kwargs={"net_arch": [8, 8]}, device="cpu")
    for name in ("b_net", "a_net"):
        save_json({**policy_to_dict(model, name), "task": "hover", "which": "final"}, tmp_path / f"{name}.json")
    save_json({"policies": []}, tmp_path / "index.json")                    # an old index is not a network
    index = build_policy_index(tmp_path)
    assert [e["name"] for e in index] == ["a_net", "b_net"]
    assert index[0] == {"file": "a_net.json", "name": "a_net", "algo": "ppo", "task": "hover", "which": "final"}


def test_golden_policies_are_self_consistent():
    for case in golden_policies(n_obs=3)["cases"]:
        for obs, expected in zip(case["observations"], case["expected"]):
            assert np.array_equal(mlp_forward(case["policy"], np.array(obs, dtype=np.float32)), expected)


def test_golden_episodes_are_complete():
    """One episode per task and controller, each with one action, reward and position per decision."""
    cases = golden_episodes()["cases"]
    assert {(c["task"], c["controller"]) for c in cases} == {
        (t, k) for t in ("hover", "goto", "obstacles", "navigation") for k in ("pid", "lqr")}
    for c in cases:
        n = c["result"]["length"]
        assert len(c["actions"]) == len(c["rewards"]) == len(c["pos"]) == n
        assert c["conditions"]["action_noise"] == c["conditions"]["obs_noise"] == c["conditions"]["gust_sigma"] == 0.0
