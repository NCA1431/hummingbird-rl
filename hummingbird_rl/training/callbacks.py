"""Extra training logs: how episodes end, how far from the target, how much energy.

Stable-Baselines3 already logs the mean return, the episode length and the
success rate (from the "is_success" key). For the analysis we also want to
know *why* episodes fail (crash, loss of control, collision...), because a
falling success rate means very different things depending on the cause.
"""

from __future__ import annotations

from collections import Counter, deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from hummingbird_rl.evaluation.rollout import run_episodes, summarise

REASONS = ("success", "time_limit", "crash", "out_of_bounds", "loss_of_control", "collision")


class EpisodeStatsCallback(BaseCallback):
    """Running statistics over the last `window` finished episodes, logged under "episode/"."""

    def __init__(self, window: int = 100):
        super().__init__()
        self.reasons: deque[str] = deque(maxlen=window)
        self.distances: deque[float] = deque(maxlen=window)
        self.energies: deque[float] = deque(maxlen=window)

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if not done:
                continue
            self.reasons.append(info.get("reason") or "unknown")
            if "distance" in info:
                self.distances.append(info["distance"])
            self.energies.append(info.get("energy", np.nan))
            self._record()
        return True

    def _record(self) -> None:
        counts = Counter(self.reasons)
        n = len(self.reasons)
        for r in REASONS:
            self.logger.record(f"episode/end_{r}", counts[r] / n)
        if self.distances:
            self.logger.record("episode/final_distance_cm", 100.0 * float(np.mean(self.distances)))
        self.logger.record("episode/energy_J", float(np.nanmean(self.energies)))


class ValidationCallback(BaseCallback):
    """Every `every` steps, run the current policy on a fixed set of validation episodes.

    We use our own callback rather than SB3's EvalCallback because we want the
    *same* episodes at every evaluation (fixed seeds), so that the learning
    curve measures the policy and not the luck of the draw. The best policy
    (by success rate, then return) is saved as best_model.zip.
    """

    def __init__(self, env, every: int, n_episodes: int, save_dir, seed0: int):
        super().__init__()
        self.env, self.every, self.save_dir = env, every, save_dir
        self.seeds = range(seed0, seed0 + n_episodes)
        self.best = (-1.0, -np.inf)
        self.next_eval = every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_eval:
            self.next_eval += self.every
            self.evaluate()
        return True

    def evaluate(self) -> dict:
        stats = summarise(run_episodes(self.env, self.model, self.seeds, deterministic=True))
        stats.update({f"end_{r}": stats.get(f"end_{r}", 0.0) for r in REASONS})   # same keys every time
        for key, value in stats.items():
            self.logger.record(f"val/{key}", value)
        score = (stats["success_rate"], stats["mean_return"])
        if score > self.best:
            self.best = score
            self.model.save(self.save_dir / "best_model")
        self.logger.dump(self.num_timesteps)
        return stats
