"""End-to-end check of the training script on a tiny budget (about 20 s)."""

import numpy as np
import pytest
import torch

from hummingbird_rl.envs import make_env
from hummingbird_rl.sim import DEFAULT_PARAMS, hover_action
from hummingbird_rl.training import train


def test_load_hyperparams_with_overrides():
    hp = train.load_hyperparams("ppo", ["gamma=0.995", "n_epochs=3", "policy_kwargs.net_arch=[64, 64]"])
    assert hp["gamma"] == 0.995 and hp["n_epochs"] == 3
    assert hp["policy_kwargs"]["net_arch"] == [64, 64] and hp["policy_kwargs"]["activation_fn"] == "tanh"


@pytest.mark.parametrize("algo", ["ppo", "sac", "td3"])
def test_untrained_policy_starts_at_hover(algo):
    small = ["buffer_size=1000"] if algo != "ppo" else []          # no 1M-transition buffer in a test
    hp = train.load_hyperparams(algo, ["policy_kwargs={net_arch: [32, 32]}"] + small)
    env = train.make_train_env("hover", "calm", 1, 0, False, False, hp["gamma"])
    model = train.build_model(algo, hp, env, seed=0, device="cpu", tb_dir=None)
    train.start_at_hover(model, algo)
    obs = make_env("hover", seed=0).reset(seed=0)[0]
    action, _ = model.predict(obs, deterministic=True)
    assert np.allclose(action, hover_action(DEFAULT_PARAMS), atol=0.02)


def test_tiny_ppo_run_writes_its_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "ROOT", tmp_path)
    (tmp_path / "configs" / "algos").mkdir(parents=True)
    (tmp_path / "configs" / "algos" / "ppo.yaml").write_text((train.Path(train.__file__).resolve().parents[2]
                                                               / "configs" / "algos" / "ppo.yaml").read_text())
    torch.manual_seed(0)
    out = train.main(["--algo", "ppo", "--task", "goto", "--steps", "1024", "--val-every", "512",
                      "--val-episodes", "2", "--set", "n_envs=2", "n_steps=256", "batch_size=64", "--name", "tiny"])
    for f in ["config.yaml", "model.zip", "best_model.zip", "vecnormalize.pkl"]:
        assert (out / f).exists(), f


def test_linear_schedule_goes_to_zero():
    lr = train.linear_schedule(3e-4)
    assert lr(1.0) == pytest.approx(3e-4) and lr(0.5) == pytest.approx(1.5e-4) and lr(0.0) == 0.0


def test_init_from_copies_the_networks(tmp_path, monkeypatch):
    """A model saved under results/runs/<name>_<date> can be found by name and its weights copied."""
    hp = train.load_hyperparams("ppo", ["policy_kwargs.net_arch={pi: [16], vf: [16]}"])
    monkeypatch.setattr(train, "ROOT", tmp_path)              # look for runs in a temporary folder
    env = train.make_train_env("goto", "calm", 1, 0, False, False, hp["gamma"])
    source = train.build_model("ppo", hp, env, seed=1, device="cpu", tb_dir=None)
    run_dir = tmp_path / "results" / "runs" / "ppo_obstacles_normal_s0_2026-10-21_010203"
    run_dir.mkdir(parents=True)
    source.save(run_dir / "best_model")

    target = train.build_model("ppo", hp, env, seed=2, device="cpu", tb_dir=None)
    path = train.resolve_model("ppo_obstacles_normal_s0")
    assert path == run_dir / "best_model.zip"
    train.copy_weights(target, path, "ppo", "cpu")
    for a, b in zip(source.policy.state_dict().values(), target.policy.state_dict().values()):
        assert torch.equal(a, b)
