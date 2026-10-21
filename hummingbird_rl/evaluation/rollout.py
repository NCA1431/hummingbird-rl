"""Running episodes with any controller, and summarising them.

"Policy" here means anything with a Stable-Baselines3-style `predict`: a
trained PPO/SAC/TD3 model or one of our classical controllers. Every
episode is started from an explicit seed, so the same list of seeds gives
exactly the same scenes, conditions and initial states to every controller.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from hummingbird_rl.controllers.base import Controller
from hummingbird_rl.envs.hummingbird_env import HummingbirdEnv

VALIDATION_SEED = 1_000_000     # validation episodes use seeds 1,000,000 + i
TEST_SEED = 2_000_000           # test episodes, used only for the final numbers


def run_episode(env: HummingbirdEnv, policy, seed: int, deterministic: bool = True,
                options: dict | None = None, trace: bool = False) -> dict:
    """Fly one episode. With trace=True the result also holds the flight itself, one row per
    decision (time, position, velocity, angles, distance to the current target), plus the
    starting state, the scene and the conditions; the behaviour and failure analyses need them."""
    obs, _ = env.reset(seed=seed, options=options)
    if isinstance(policy, Controller):
        policy.reset()
    start = env.state.copy()
    rows = []
    total, steps = 0.0, 0
    while True:
        action, _ = policy.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        steps += 1
        if trace:
            s = env.state
            rows.append((s.time, *s.pos, *s.vel, *s.euler, info.get("distance", np.nan)))
        if terminated or truncated:
            break
    result = {"seed": seed, "return": total, "length": steps, "success": bool(info["is_success"]),
              "reason": info["reason"], "distance": info.get("distance", np.nan),
              "energy": info["energy"], "waypoints": info.get("waypoints_reached", np.nan)}
    if trace:
        a = np.array(rows)
        result["trace"] = {"t": a[:, 0], "pos": a[:, 1:4], "vel": a[:, 4:7],
                           "yaw": a[:, 7], "pitch": a[:, 8], "roll": a[:, 9], "distance": a[:, 10]}
        result.update(start=start, scene=env.scene, conditions=env.conditions)
    return result


def run_episodes(env: HummingbirdEnv, policy, seeds, deterministic: bool = True, trace: bool = False) -> list[dict]:
    return [run_episode(env, policy, int(s), deterministic, trace=trace) for s in seeds]


def summarise(results: list[dict]) -> dict:
    """Averages over episodes: success rate, return, and how the episodes ended."""
    n = len(results)
    counts = Counter(r["reason"] for r in results)
    out = {
        "success_rate": sum(r["success"] for r in results) / n,
        "mean_return": float(np.mean([r["return"] for r in results])),
        "mean_length": float(np.mean([r["length"] for r in results])),
        "final_distance_cm": 100.0 * float(np.nanmean([r["distance"] for r in results])),
        "energy_J": float(np.mean([r["energy"] for r in results])),
    }
    out.update({f"end_{k}": v / n for k, v in counts.items()})
    return out
