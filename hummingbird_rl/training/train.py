"""Train one agent on one task, under one disturbance regime.

    uv run python -m hummingbird_rl.training.train --algo ppo --task hover --steps 1e6 --wandb

Each run writes into results/runs/<name>_<date>_<time>/ :
    config.yaml         everything needed to reproduce the run
    model.zip           the final policy
    best_model.zip      the policy with the best validation score
    checkpoints/        a snapshot every 250k steps, in case a long run dies
    vecnormalize.pkl    reward-normalisation statistics (PPO only; only needed to resume training)
    tb/                 TensorBoard logs (also synced to W&B with --wandb)

Hyperparameters come from configs/algos/<algo>.yaml; any of them can be
overridden from the command line, e.g. --set gamma=0.995 policy_kwargs.log_std_init=0,
which is how the ablations are run. Options of the environment itself go through
--env, e.g. --env failure_reward=-50.

--init-from starts from the weights of an earlier model instead of the hover
initialisation: for transfer (Obstacles -> Navigation) and fine-tuning (normal ->
randomised). It accepts a .zip file or a run name, in which case the best model
of the most recent run with that name is used.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from hummingbird_rl.envs import HummingbirdEnv
from hummingbird_rl.evaluation.rollout import VALIDATION_SEED
from hummingbird_rl.sim import DEFAULT_PARAMS, hover_action
from hummingbird_rl.training.callbacks import EpisodeStatsCallback, ValidationCallback

ROOT = Path(__file__).resolve().parents[2]
ALGOS = {"ppo": PPO, "sac": SAC, "td3": TD3}
ACTIVATIONS = {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU}


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algo", choices=sorted(ALGOS), required=True)
    ap.add_argument("--task", choices=["hover", "goto", "obstacles", "navigation"], required=True)
    ap.add_argument("--regime", default="normal", help="normal or randomised (see envs/disturbances.py)")
    ap.add_argument("--steps", type=float, default=1e6, help="total environment steps (decisions)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu or cuda; PPO is always best on cpu")
    ap.add_argument("--threads", type=int, default=1, help="torch threads; keep 1 when running several jobs")
    ap.add_argument("--subproc", action="store_true", help="run the parallel envs in separate processes")
    ap.add_argument("--val-every", type=float, default=50_000)
    ap.add_argument("--val-episodes", type=int, default=50)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override hyperparameters")
    ap.add_argument("--env", nargs="*", default=[], metavar="KEY=VALUE", help="options of HummingbirdEnv")
    ap.add_argument("--init-from", default=None, help="model .zip or run name to start from")
    ap.add_argument("--no-hover-init", action="store_true", help="ablation: keep the default initialisation")
    ap.add_argument("--name", default=None, help="run name (default: algo_task_regime_sSEED)")
    ap.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    ap.add_argument("--project", default="hummingbird-rl")
    ap.add_argument("--group", default=None, help="W&B group, e.g. the experiment name")
    return ap.parse_args(argv)


def load_hyperparams(algo: str, overrides: list[str]) -> dict:
    hp = yaml.safe_load((ROOT / "configs" / "algos" / f"{algo}.yaml").read_text())
    for item in overrides:
        key, value = item.split("=", 1)
        *parents, leaf = key.split(".")          # "policy_kwargs.log_std_init=0" reaches inside a dict
        node = hp
        for k in parents:
            node = node.setdefault(k, {})
        node[leaf] = yaml.safe_load(value)       # "0.995" -> float, "[64, 64]" -> list, "true" -> bool
    return hp


def parse_pairs(items: list[str]) -> dict:
    """["a=1", "b=[2, 3]"] -> {"a": 1, "b": [2, 3]}, values read as YAML."""
    return {k: yaml.safe_load(v) for k, v in (item.split("=", 1) for item in items)}


def monitored_env(task: str, regime: str, env_kwargs: dict | None = None) -> Monitor:
    """Monitor records each episode's return and length, which SB3 logs as rollout/ep_rew_mean."""
    return Monitor(HummingbirdEnv(task, regime, **(env_kwargs or {})))


def make_train_env(task: str, regime: str, n_envs: int, seed: int, subproc: bool, normalize_reward: bool, gamma: float,
                   env_kwargs: dict | None = None):
    fns = [partial(monitored_env, task, regime, env_kwargs) for _ in range(n_envs)]
    venv = SubprocVecEnv(fns) if subproc else DummyVecEnv(fns)
    venv.seed(seed)                              # env i gets seed + i at its first reset
    if normalize_reward:
        # Only the rewards are normalised (divided by a running std of the return);
        # the observations are already scaled by hand, which keeps the policy usable
        # without any saved statistics.
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=gamma)
    return venv


def linear_schedule(initial: float):
    """Learning rate decreasing linearly to 0. SB3 calls it with the fraction of training left (1 -> 0)."""
    return lambda progress_remaining: initial * progress_remaining


def build_model(algo: str, hp: dict, env, seed: int, device: str, tb_dir: Path):
    hp = dict(hp)
    hp.pop("n_envs")
    hp.pop("normalize_reward", None)
    if hp.pop("anneal_lr", False):
        hp["learning_rate"] = linear_schedule(hp["learning_rate"])
    pk = dict(hp.pop("policy_kwargs", {}))
    if "activation_fn" in pk:
        pk["activation_fn"] = ACTIVATIONS[pk["activation_fn"]]
    noise_std = hp.pop("action_noise_std", None)
    if noise_std is not None:
        hp["action_noise"] = NormalActionNoise(np.zeros(5), noise_std * np.ones(5))
    return ALGOS[algo]("MlpPolicy", env, policy_kwargs=pk, seed=seed, device=device,
                       tensorboard_log=str(tb_dir) if tb_dir else None, verbose=0, **hp)


def start_at_hover(model, algo: str) -> None:
    """Make the untrained policy output the hover action instead of zeros.

    With the raw initialisation the mean action is 0, i.e. both wings at half
    amplitude: not enough lift, so every early episode is a fall and the agent
    first spends a lot of samples just discovering how hard to flap. Setting the
    bias of the last layer to the hover action (and shrinking its weights) starts
    learning from the trim point, as one would with a classical controller.
    SAC and TD3 squash their output with tanh, so their bias is atanh(hover).
    """
    hover = torch.as_tensor(hover_action(DEFAULT_PARAMS), dtype=torch.float32)
    with torch.no_grad():
        if algo == "ppo":
            layers = [model.policy.action_net]
            bias = hover
        else:
            last = model.actor.mu if algo == "sac" else model.actor.mu[-2]   # TD3: Linear before the Tanh
            layers = [last]
            bias = torch.atanh(hover.clamp(-0.99, 0.99))
        for layer in layers:
            layer.weight.mul_(0.01)
            layer.bias.copy_(bias)
        if algo == "td3":
            model.actor_target.load_state_dict(model.actor.state_dict())


def resolve_model(spec: str) -> Path:
    """A .zip path as given, or the best model of the most recent run called `spec`."""
    path = Path(spec)
    if path.suffix == ".zip":
        return path
    runs = sorted((ROOT / "results" / "runs").glob(f"{spec}_*/best_model.zip"))   # names end with the date
    if not runs:
        raise FileNotFoundError(f"no run called {spec!r} with a best_model.zip in results/runs")
    return runs[-1]


def copy_weights(model, source: Path, algo: str, device: str) -> None:
    """Start from the networks of an earlier model. Only the weights are copied: the optimiser
    starts afresh (new Adam statistics), and so do the replay buffer and the reward normalisation."""
    model.policy.load_state_dict(ALGOS[algo].load(source, device=device).policy.state_dict())


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def main(argv=None) -> Path:
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    steps = int(args.steps)
    hp = load_hyperparams(args.algo, args.set)
    name = args.name or f"{args.algo}_{args.task}_{args.regime}_s{args.seed}"
    # The date and time make every run folder unique, so a rerun never overwrites older models.
    out = ROOT / "results" / "runs" / f"{name}_{time.strftime('%Y-%m-%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)

    env_kwargs = parse_pairs(args.env)
    init_from = resolve_model(args.init_from) if args.init_from else None
    config = {"algo": args.algo, "task": args.task, "regime": args.regime, "steps": steps, "seed": args.seed,
              "device": args.device, "hyperparams": hp, "env": env_kwargs,
              "init_from": str(init_from) if init_from else None, "hover_init": not (args.no_hover_init or init_from),
              "commit": git_commit(),
              "bird": json.loads(json.dumps(asdict(DEFAULT_PARAMS)))}     # tuples -> lists, for YAML
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.project, name=name, group=args.group, config=config,
                         tags=[args.algo, args.task, args.regime], sync_tensorboard=True, dir=str(out))

    env = make_train_env(args.task, args.regime, hp["n_envs"], args.seed, args.subproc,
                         hp.get("normalize_reward", False), hp["gamma"], env_kwargs)
    model = build_model(args.algo, hp, env, args.seed, args.device, out / "tb")
    if init_from:
        copy_weights(model, init_from, args.algo, args.device)
    elif not args.no_hover_init:
        start_at_hover(model, args.algo)

    # Validation always uses the same regime as training; robustness is measured later.
    val_env = HummingbirdEnv(args.task, args.regime, **env_kwargs)
    callbacks = CallbackList([
        EpisodeStatsCallback(),
        ValidationCallback(val_env, int(args.val_every), args.val_episodes, out, VALIDATION_SEED),
        CheckpointCallback(max(250_000 // hp["n_envs"], 1), str(out / "checkpoints"), name_prefix="model"),
    ])

    print(f"[{name}] training {steps:,} steps on {args.device} -> {out}")
    t0 = time.time()
    model.learn(total_timesteps=steps, callback=callbacks, tb_log_name="run", progress_bar=False)
    model.save(out / "model")
    if isinstance(env, VecNormalize):
        env.save(str(out / "vecnormalize.pkl"))
    minutes = (time.time() - t0) / 60
    print(f"[{name}] done in {minutes:.1f} min ({steps / (60 * minutes):.0f} steps/s)")

    env.close()
    if run is not None:
        run.finish()
    return out


if __name__ == "__main__":
    main()
