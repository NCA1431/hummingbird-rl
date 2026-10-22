"""Recorded episodes for the Watch mode of the web viewer.

The viewer replays episodes, it does not simulate them, so what it shows is
exactly what was measured: the episodes are flown by the same loop and on the
same seeds as the evaluation suites (a test checks that the recorded result
equals the evaluated one).

One JSON file per episode. For every decision (60 per second) we keep what
the viewer needs to draw the bird and its surroundings, one list per quantity:

    t         s
    pos       m, world frame
    quat      orientation (w, x, y, z), body -> world
    vel       m/s, world frame (trail colour, HUD)
    action    the 5 actions; the viewer decodes them into wing amplitude,
              frequency and tilts to animate the wings
    thrust    N, total wing force (downwash particles)
    wind      N, current wind force (wind particles)
    target    index of the current target (it changes in navigation)
    hold      progress of the hold at the target, 0 to 1
    distance  m, to the current target

Frame 0 is the start, before any decision, and frame k the state after k
decisions; action[k] is the action that led to frame k (frame 0: the hover
action the controller is assumed to start from). Numbers are rounded to
0.1 mm and similar: every visitor downloads these files, and nobody can see
a tenth of a millimetre on screen.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hummingbird_rl.controllers.base import Controller
from hummingbird_rl.envs.hummingbird_env import HummingbirdEnv
from hummingbird_rl.sim.aero import decode_action, wing_thrusts

FORMAT = "hummingbird-episode/1"

# Decimals kept for each recorded quantity.
DECIMALS = {"t": 4, "pos": 4, "quat": 5, "vel": 4, "action": 4, "thrust": 6, "wind": 6, "hold": 3, "distance": 4}


def _frame(env: HummingbirdEnv, action: np.ndarray) -> dict:
    s = env.state
    index = getattr(env.task, "index", 0)        # only navigation moves from one target to the next
    # Thrust is recomputed from the action without its small actuator noise (the
    # environment does not expose the noisy one). It only drives the drawing.
    f_l, f_r = wing_thrusts(decode_action(action, env.p), env.p, env.dyn.damage)
    return {"t": s.time, "pos": s.pos, "quat": s.quat, "vel": s.vel, "action": action, "thrust": f_l + f_r,
            "wind": env.gusts.state, "target": int(index), "hold": env.task.hold_progress(),
            "distance": float(np.linalg.norm(s.pos - env.scene.targets[index]))}


def record_episode(env: HummingbirdEnv, policy, seed: int, options: dict | None = None,
                   deterministic: bool = True) -> dict:
    """Fly one episode as `evaluation.rollout.run_episode` does, and keep every frame."""
    obs, _ = env.reset(seed=seed, options=options)
    if isinstance(policy, Controller):
        policy.reset()
    frames = [_frame(env, env.prev_action)]
    total, steps = 0.0, 0
    while True:
        action, _ = policy.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        steps += 1
        frames.append(_frame(env, env.prev_action))      # prev_action is now the (clipped) action just applied
        if terminated or truncated:
            break

    columns = {}
    for key in frames[0]:
        values = [f[key] for f in frames]
        if key == "target":
            columns[key] = values
        else:
            columns[key] = np.round(np.asarray(values, dtype=float), DECIMALS[key]).tolist()
    return {
        "format": FORMAT,
        "task": env.task.name,
        "seed": int(seed),
        "dt": env.p.dt * env.p.action_repeat,
        "scene": env.scene.to_dict(),
        "conditions": env.conditions.to_dict(),
        "result": {"success": bool(info["is_success"]), "reason": info["reason"], "return": float(total),
                   "length": steps, "energy": float(info["energy"]),
                   "waypoints": info.get("waypoints_reached")},
        "frames": columns,
    }


def save_json(data: dict, path: Path) -> None:
    """Compact JSON (no spaces): these files are downloaded by the browser."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":")))


def build_index(folder: Path) -> list[dict]:
    """One line per episode file below `folder`, for the viewer's menus.

    Rebuilt from the files themselves each time, so the index can never list
    an episode that does not exist or miss one that does.
    """
    entries = []
    for path in sorted(folder.rglob("*.json")):
        if path.name == "index.json":
            continue
        ep = json.loads(path.read_text())
        entries.append({"file": path.relative_to(folder).as_posix(), "policy": ep.get("policy"),
                        "task": ep["task"], "suite": ep.get("suite"), "seed": ep["seed"],
                        "success": ep["result"]["success"], "reason": ep["result"]["reason"],
                        "length": ep["result"]["length"]})
    return entries
