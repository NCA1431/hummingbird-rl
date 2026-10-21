"""Creating environments by name, from our own scripts or with Gymnasium's `gym.make`."""

from __future__ import annotations

import gymnasium as gym

from hummingbird_rl.envs.disturbances import Regime
from hummingbird_rl.envs.hummingbird_env import HummingbirdEnv
from hummingbird_rl.tasks import TASKS

ENV_IDS = {task: f"Hummingbird-{task.capitalize()}-v0" for task in TASKS}   # e.g. "Hummingbird-Hover-v0"


def register_envs() -> None:
    """Register one Gymnasium id per task (safe to call several times)."""
    for task, env_id in ENV_IDS.items():
        if env_id not in gym.registry:
            gym.register(env_id, entry_point="hummingbird_rl.envs.hummingbird_env:HummingbirdEnv",
                         kwargs={"task": task})


def make_env(task: str, regime: str | Regime = "normal", seed: int | None = None, **kwargs) -> HummingbirdEnv:
    """Build an environment. With a seed, the first reset and the action space are seeded too."""
    env = HummingbirdEnv(task=task, regime=regime, **kwargs)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env
