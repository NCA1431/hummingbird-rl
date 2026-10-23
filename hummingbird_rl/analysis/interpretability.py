"""E6: what PPO's critic and policy look like near hover, next to the LQR's exact answers.

The LQR gives us something rare in reinforcement learning: a ground truth. For
the linearised bird, xᵀPx is the exact cost-to-go of the LQR from the state x,
the solution of the Bellman equation of that problem. PPO's critic V(s) is also
an estimate of a value, the discounted sum of rewards from s when PPO flies. The
two are not the same quantity (different costs, different policies, rewards
normalised during training), so we do not compare their numbers. We compare
their shape: which states each one considers better than which.

Two measurements, both in the vertical plane through the hover point (forward
offset Δx, vertical offset Δy), the bird otherwise at rest (level, no velocity,
previous action = hover), with no wind and no noise:

  value landscape   V_PPO(s) and −xᵀPx on a fine grid. Spearman's rank
                    correlation tells whether they order the states the same
                    way, close to hover (±5 cm, where the linear model holds)
                    and further out (±15 cm). It only looks at the order, so a
                    cone-shaped critic and a bowl-shaped quadratic with the same
                    contour lines still correlate perfectly: what is tested is
                    the shape of the contours.
  vector field      from each point of a coarser grid, where the bird is 0.1 s
                    later under PID, LQR and PPO: the closed-loop "flow" each
                    controller imposes. Its direction is compared with the LQR's
                    (mean cosine of the angle between the arrows).

Simplifications we accept: one plane only (the lateral offset Δz is left at 0),
the bird always starts at rest, and the deterministic policy is used.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from hummingbird_rl.controllers.base import Controller
from hummingbird_rl.controllers.lqr import LQRController, lqr_state
from hummingbird_rl.envs.hummingbird_env import HummingbirdEnv
from hummingbird_rl.envs.observation import build_observation
from hummingbird_rl.sim import DEFAULT_PARAMS, BirdParams, BirdState, hover_action
from hummingbird_rl.world.scene import START, Scene

HALF = 0.15          # m, half-width of the grids
NEAR = 0.05          # m, "close to hover", where the linearisation is good


def offsets(n: int, half: float = HALF) -> np.ndarray:
    return np.linspace(-half, half, n)


def offset_state(dx: float, dy: float) -> BirdState:
    """The bird at rest, level, Δx ahead of and Δy above the hover point (facing +x)."""
    return BirdState.from_euler(pos=START + np.array([dx, dy, 0.0]))


def observation(state: BirdState, p: BirdParams = DEFAULT_PARAMS) -> np.ndarray:
    """What the hover task would show the controller in this state, without sensor noise."""
    return build_observation(state, [START], [], hover_action(p), 0.0, p.r_bird)


def critic_values(model, obs: np.ndarray) -> np.ndarray:
    """PPO's value network on a batch of observations (in the reward units it was trained on,
    i.e. after Stable-Baselines3's reward normalisation)."""
    import torch

    with torch.no_grad():
        v = model.policy.predict_values(torch.as_tensor(np.asarray(obs), dtype=torch.float32))
    return v.numpy().ravel()


def value_landscape(model, lqr: LQRController, n: int = 61, half: float = HALF) -> dict:
    """V_PPO and −V_LQR on an n x n grid; arrays indexed [iy, ix] (rows = height)."""
    xs = offsets(n, half)
    states = [offset_state(x, y) for y in xs for x in xs]
    out = {"xs": xs, "lqr": np.array([-lqr.cost_to_go(lqr_state(s, START)) for s in states]).reshape(n, n)}
    if model is not None:
        out["ppo"] = critic_values(model, np.stack([observation(s) for s in states])).reshape(n, n)
    return out


def normalise(v: np.ndarray) -> np.ndarray:
    """Rescaled to [0, 1], 1 for the best state: values of different units become comparable in shape."""
    return (v - v.min()) / (v.max() - v.min())


def within(xs: np.ndarray, radius: float) -> np.ndarray:
    """Mask of the grid cells inside the square |Δx|, |Δy| <= radius."""
    X, Y = np.meshgrid(xs, xs)
    return (np.abs(X) <= radius + 1e-9) & (np.abs(Y) <= radius + 1e-9)


def spearman(a: np.ndarray, b: np.ndarray, xs: np.ndarray, radius: float) -> float:
    m = within(xs, radius)
    return float(stats.spearmanr(a[m], b[m])[0])


def axis_values(v: np.ndarray, xs: np.ndarray, at: float = 0.10) -> dict:
    """The normalised value 10 cm ahead, behind, above and below: is being off in one direction
    judged worse than in another? (The LQR's quadratic cost is symmetric by construction.)"""
    v = normalise(v)
    i = lambda d: int(np.argmin(np.abs(xs - d)))   # noqa: E731
    c = i(0.0)
    return {"ahead": float(v[c, i(at)]), "behind": float(v[c, i(-at)]),
            "above": float(v[i(at), c]), "below": float(v[i(-at), c])}


# --- Closed loop -----------------------------------------------------------------------------

def act(policy, obs: np.ndarray) -> np.ndarray:
    """PID, LQR and SB3 agents all answer predict(obs, deterministic=True)."""
    return np.asarray(policy.predict(obs, deterministic=True)[0])


def fly(policy, start: BirdState, n_steps: int, env: HummingbirdEnv | None = None) -> np.ndarray:
    """Positions (n_steps + 1, 3) of the bird flown by `policy` from `start`, in calm air
    (no wind, no noise). Stops early if the episode ends (a crash, for instance)."""
    env = env or HummingbirdEnv("hover", "calm")
    if isinstance(policy, Controller):
        policy.reset()
    obs, _ = env.reset(seed=0, options={"state": start, "scene": Scene(start=START.copy(), targets=[START.copy()])})
    path = [env.state.pos.copy()]
    for _ in range(n_steps):
        obs, _, terminated, truncated, _ = env.step(act(policy, obs))
        path.append(env.state.pos.copy())
        if terminated or truncated:
            break
    return np.array(path)


def vector_field(policy, n: int = 21, half: float = HALF, duration: float = 0.1,
                 p: BirdParams = DEFAULT_PARAMS) -> tuple[np.ndarray, np.ndarray]:
    """Where the bird is `duration` later, from rest at each point of an n x n grid:
    the displacement (Δx, Δy) in m, indexed [iy, ix]."""
    steps = max(1, round(duration / (p.dt * p.action_repeat)))
    xs = offsets(n, half)
    env = HummingbirdEnv("hover", "calm")
    D = np.zeros((n, n, 2))
    for iy, y in enumerate(xs):
        for ix, x in enumerate(xs):
            path = fly(policy, offset_state(x, y), steps, env)
            D[iy, ix] = path[-1, :2] - path[0, :2]
    return xs, D


def alignment(D1: np.ndarray, D2: np.ndarray, xs: np.ndarray, radius: float) -> float:
    """Mean cosine of the angle between two vector fields, over the cells within `radius`
    (1: same directions everywhere, 0: perpendicular on average). The hover point itself,
    where both barely move, is left out."""
    m = within(xs, radius)
    a, b = D1[m], D2[m]
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    ok = (na > 1e-6) & (nb > 1e-6)
    return float(np.mean(np.sum(a[ok] * b[ok], axis=1) / (na[ok] * nb[ok])))


def path_gap(paths_a: list[np.ndarray], paths_b: list[np.ndarray]) -> float:
    """Mean distance (m) between two controllers' positions at the same instant, from the same
    starts: how differently they fly, beyond the first 0.1 s that the arrows show."""
    gaps = []
    for a, b in zip(paths_a, paths_b):
        n = min(len(a), len(b))
        gaps.append(np.mean(np.linalg.norm(a[:n] - b[:n], axis=1)))
    return float(np.mean(gaps))


def trajectory_starts(radius: float = 0.13) -> list[tuple[float, float]]:
    """Five starts around the hover point: above, and the four diagonals."""
    angles = np.radians([90, 45, 135, 225, 315])
    return [(radius * np.cos(a), radius * np.sin(a)) for a in angles]
