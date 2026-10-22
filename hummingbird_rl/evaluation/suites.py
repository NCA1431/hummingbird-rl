"""The evaluation suites: what we measure on a controller once it is trained.

    test       200 fresh episodes (test seeds) in normal conditions: the headline numbers
    grid       robustness map: mean wind (0-40% of the weight) × lost force of the left wing (0-40%)
    mass       success against the mass of the bird, also beyond the training range (±25%)
    failure    one wing suddenly loses 40% of its force: does the bird survive, how fast does it recover?
               (not on go-to-point: its episodes end before the failure happens)
    behaviour  go-to-point only: depending on where the target is, does the bird turn or back up?

Every suite works the same way for RL agents and classical controllers, and
uses its own block of seeds, so all controllers face exactly the same episodes.
Each returns one row per episode (or per grid cell) and a short summary.
"""

from __future__ import annotations

from functools import partial

import numpy as np

from hummingbird_rl.envs.disturbances import Conditions, grid_conditions, normal
from hummingbird_rl.envs.hummingbird_env import HummingbirdEnv
from hummingbird_rl.evaluation.rollout import TEST_SEED, run_episodes, summarise
from hummingbird_rl.evaluation.stats import mean_interval, wilson_interval
from hummingbird_rl.sim.params import BirdParams
from hummingbird_rl.sim.rotations import world_to_heading
from hummingbird_rl.tasks.base import HOLD_RADIUS, HOLD_STEPS

GRID_WINDS = (0.0, 0.1, 0.2, 0.3, 0.4)        # mean wind, fraction of the weight (training went up to 0.3)
GRID_DAMAGES = (0.0, 0.1, 0.2, 0.3, 0.4)      # force lost by the left wing (training went up to 0.4)
MASSES = (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4)   # training range: 0.75-1.25

# The tasks on which each suite means something. A go-to-point episode ends with the hold,
# about 1 s after the start, so the failure (drawn between 1.5 and 3 s) would never happen.
ALL_TASKS = ("hover", "goto", "obstacles", "navigation")
SUITE_TASKS = {"test": ALL_TASKS, "grid": ALL_TASKS, "mass": ALL_TASKS,
               "failure": ("hover", "obstacles", "navigation"), "behaviour": ("goto",)}

# Separate seed blocks, so that no suite reuses the episodes of another.
SEEDS = {"test": TEST_SEED, "grid": TEST_SEED + 100_000, "mass": TEST_SEED + 200_000,
         "failure": TEST_SEED + 300_000, "behaviour": TEST_SEED + 400_000}


def _rate(rows: list[dict]) -> dict:
    """Success rate with its 95% interval."""
    k = sum(r["success"] for r in rows)
    low, high = wilson_interval(k, len(rows))
    return {"success_rate": k / len(rows), "success_low": low, "success_high": high}


def _plain(rows: list[dict]) -> list[dict]:
    """Episode rows without the heavy objects (trace, scene...), ready for a CSV file."""
    return [{k: v for k, v in r.items() if k not in ("trace", "start", "scene", "conditions")} for r in rows]


def evaluate_test(policy, task: str, env_kwargs: dict, n: int = 200) -> tuple[list[dict], dict]:
    env = HummingbirdEnv(task, "normal", **env_kwargs)
    rows = run_episodes(env, policy, range(SEEDS["test"], SEEDS["test"] + n))
    summary = {**summarise(rows), **_rate(rows)}
    summary["return_low"], summary["return_high"] = mean_interval([r["return"] for r in rows])[1:]
    return _plain(rows), summary


def evaluate_grid(policy, task: str, env_kwargs: dict, n: int = 20) -> tuple[list[dict], dict]:
    """One row per (wind, damage) cell. The "robust area" is the mean success over the whole grid.

    We also want to know HOW the failures happen. A failure "by drift" is an episode
    that reached the time limit without succeeding: the bird was still flying, just
    not close enough to (or not still enough at) its target. The other failures are
    crashes, losses of control and exits from the arena. A controller without integral
    action is expected to fail by drift under a steady wind: it settles off the target.
    """
    cells = []
    for i, wind in enumerate(GRID_WINDS):
        for j, damage in enumerate(GRID_DAMAGES):
            regime = partial(grid_conditions, wind_fraction=wind, damage=damage)
            env = HummingbirdEnv(task, regime, **env_kwargs)
            seed0 = SEEDS["grid"] + 1000 * (i * len(GRID_DAMAGES) + j)
            rows = run_episodes(env, policy, range(seed0, seed0 + n))
            failed = [r for r in rows if not r["success"]]
            cells.append({"wind": wind, "damage": damage, **_rate(rows),
                          "mean_return": float(np.mean([r["return"] for r in rows])),
                          "mean_distance_cm": 100.0 * float(np.nanmean([r["distance"] for r in rows])),
                          "failed": len(failed), "failed_by_drift": sum(r["reason"] == "time_limit" for r in failed)})
    inside = [c["success_rate"] for c in cells if c["wind"] <= 0.3]      # the part covered by training
    failed = sum(c["failed"] for c in cells)
    return cells, {"robust_area": float(np.mean([c["success_rate"] for c in cells])),
                   "robust_area_trained_range": float(np.mean(inside)),
                   "failures_by_drift": sum(c["failed_by_drift"] for c in cells) / failed if failed else float("nan")}


def _with_mass(rng: np.random.Generator, p: BirdParams, mass_factor: float) -> Conditions:
    c = normal(rng, p)
    c.mass_factor = mass_factor
    return c


def evaluate_mass(policy, task: str, env_kwargs: dict, n: int = 20) -> tuple[list[dict], dict]:
    cells = []
    for i, m in enumerate(MASSES):
        env = HummingbirdEnv(task, partial(_with_mass, mass_factor=m), **env_kwargs)
        seed0 = SEEDS["mass"] + 1000 * i
        rows = run_episodes(env, policy, range(seed0, seed0 + n))
        cells.append({"mass_factor": m, **_rate(rows)})
    return cells, {"mean_success": float(np.mean([c["success_rate"] for c in cells]))}


def recovery_time(result: dict) -> float:
    """Seconds from the failure until the bird is back within the hold radius for good.

    This is a settling time: we look for the LAST moment after the failure when
    the bird was farther than 3 cm from its target, and the recovery ends just
    after it. 0 if the bird never left the 3 cm ball; NaN if it is still outside,
    or back for less than a full hold (15 decisions), when the episode ends.
    (A first version counted the first 15 decisions inside after the failure,
    which a bird drifting away slowly passes before it has even left: 0 s.)
    """
    t, d = result["trace"]["t"], result["trace"]["distance"]
    after = t >= result["conditions"].failure_time
    outside = np.flatnonzero(after & (d >= HOLD_RADIUS))
    if len(outside) == 0:
        return 0.0
    last = outside[-1]
    if last >= len(t) - HOLD_STEPS:
        return float("nan")
    return float(t[last + 1] - result["conditions"].failure_time)


def evaluate_failure(policy, task: str, env_kwargs: dict, n: int = 100) -> tuple[list[dict], dict]:
    if task not in SUITE_TASKS["failure"]:
        raise ValueError(f"the failure suite is not defined for {task!r}")
    env = HummingbirdEnv(task, "failure", **env_kwargs)
    results = run_episodes(env, policy, range(SEEDS["failure"], SEEDS["failure"] + n), trace=True)
    rows = []
    for r in results:
        t, d = r["trace"]["t"], r["trace"]["distance"]
        reached = t[-1] >= r["conditions"].failure_time          # the episode lasted until the failure
        after = t >= r["conditions"].failure_time
        rows.append({**_plain([r])[0], "failure_time": r["conditions"].failure_time, "reached_failure": reached,
                     "survived": reached and r["reason"] in ("time_limit", "success"),
                     "max_distance_after_cm": 100.0 * float(d[after].max()) if after.any() else np.nan,
                     "recovery_time_s": recovery_time(r) if reached else np.nan})
    hit = [r for r in rows if r["reached_failure"]]
    k = sum(r["survived"] for r in hit)
    low, high = wilson_interval(k, len(hit))
    times = [r["recovery_time_s"] for r in hit if np.isfinite(r["recovery_time_s"])]
    return rows, {**_rate(rows), "episodes_reaching_failure": len(hit),
                  "survival_rate": k / len(hit) if hit else float("nan"), "survival_low": low, "survival_high": high,
                  "recovered_fraction": len(times) / len(hit) if hit else float("nan"),
                  "median_recovery_s": float(np.median(times)) if times else float("nan"),
                  "median_max_deviation_cm": float(np.median([r["max_distance_after_cm"] for r in hit]))
                  if hit else float("nan")}


def evaluate_behaviour(policy, task: str, env_kwargs: dict, n: int = 200) -> tuple[list[dict], dict]:
    """Go-to-point: for each episode, where the target was at the start and how the bird went there.

    bearing    angle of the target seen from the bird at the start: 0° straight ahead,
               +90° on its left, ±180° behind
    turned     the heading changed by more than 45° at some point of the flight
    net_turn   final heading minus initial heading, between -180° and 180° (same signs as the bearing)
    faced      the bird ended up facing the direction the target was in: |net_turn - bearing| < 45°
    forward    fraction of the moving time spent flying head-first
    backed     most of the moving time was spent flying backwards (forward < 0.5)

    "turned" alone can be misleading: a bird whose heading wanders by more than 45°
    counts as turning even if it never faces its target. "faced" and "forward" say
    whether the turn served to fly head-first towards the target.
    """
    if task not in SUITE_TASKS["behaviour"]:
        raise ValueError("the behaviour suite is defined for the go-to-point task")
    env = HummingbirdEnv(task, "normal", **env_kwargs)
    results = run_episodes(env, policy, range(SEEDS["behaviour"], SEEDS["behaviour"] + n), trace=True)
    rows = []
    for r in results:
        yaw0 = r["start"].euler[0]
        rel = world_to_heading(r["scene"].targets[0] - r["start"].pos, yaw0)
        bearing = bearing_deg(rel)
        tr = r["trace"]
        heading_change = float(np.degrees(np.ptp(np.unwrap(np.r_[yaw0, tr["yaw"]]))))
        net_turn = float(np.degrees(_wrap(tr["yaw"][-1] - yaw0)))
        v_fwd = np.array([world_to_heading(v, y)[0] for v, y in zip(tr["vel"], tr["yaw"])])
        moving = np.linalg.norm(tr["vel"], axis=1) > 0.1
        forward = float(np.mean(v_fwd[moving] > 0)) if moving.any() else float("nan")
        rows.append({**_plain([r])[0], "bearing_deg": bearing, "target_distance": float(np.linalg.norm(rel)),
                     "heading_change_deg": heading_change, "net_turn_deg": net_turn,
                     "turned": heading_change > 45.0,
                     "faced": abs(np.degrees(_wrap(np.radians(net_turn - bearing)))) < 45.0,
                     "forward": forward, "backed": bool(moving.any() and forward < 0.5),
                     "time_s": float(tr["t"][-1])})
    behind = [r for r in rows if abs(r["bearing_deg"]) > 90.0]
    ahead = [r for r in rows if abs(r["bearing_deg"]) <= 90.0]
    return rows, {**_rate(rows), "p_turn_target_behind": _mean(behind, "turned"),
                  "p_faced_target_behind": _mean(behind, "faced"),
                  "forward_target_behind": float(np.nanmean([r["forward"] for r in behind])) if behind else np.nan,
                  "p_backed_target_behind": _mean(behind, "backed"), "p_turn_target_ahead": _mean(ahead, "turned"),
                  "mean_time_s": _mean(rows, "time_s")}


def _wrap(angle: float) -> float:
    """An angle brought back to [-π, π)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def bearing_deg(rel_heading: np.ndarray) -> float:
    """Direction of a point given in the heading frame: 0° ahead, +90° on the left (where yaw > 0
    turns the nose), ±180° behind. The left is -z, since z points along the right wing."""
    return float(np.degrees(np.arctan2(-rel_heading[2], rel_heading[0])))


def _mean(rows: list[dict], key: str) -> float:
    return float(np.mean([r[key] for r in rows])) if rows else float("nan")


SUITES = {"test": evaluate_test, "grid": evaluate_grid, "mass": evaluate_mass,
          "failure": evaluate_failure, "behaviour": evaluate_behaviour}
