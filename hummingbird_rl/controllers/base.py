"""What all classical controllers share: reading the observation, and the predict interface.

The classical controllers see exactly what the RL agents see, the 48-number
observation with its sensor noise, and nothing else (no true state, no
knowledge of the wind or the damage). They do know the nominal model of the
bird, which is what model-based control means. This keeps the comparison
with RL fair.

`predict` has the same signature as in Stable-Baselines3, so a single
evaluation loop can drive a PPO agent or a PID controller without knowing
which one it has.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from hummingbird_rl.envs import observation as ob


@dataclass
class ObsView:
    """The observation turned back into physical units (heading frame, SI)."""
    altitude: float
    vel: np.ndarray                 # m/s, (forward, up, right)
    roll: float
    pitch: float
    rates: np.ndarray               # rad/s, body rates (roll, yaw, pitch)
    targets: list[np.ndarray]       # m, positions relative to the bird, current target first
    obstacles: list[tuple[np.ndarray, float]]   # (centre relative to the bird in m, radius in m)
    hold: float

    @staticmethod
    def parse(obs: np.ndarray) -> "ObsView":
        obs = np.asarray(obs, dtype=float)
        t = obs[ob.TARGETS].reshape(ob.N_TARGETS, 4)
        o = obs[ob.OBSTACLES].reshape(ob.N_OBSTACLES, 5)
        return ObsView(
            altitude=obs[ob.ALTITUDE],
            vel=obs[ob.VELOCITY].copy(),
            roll=obs[ob.ATTITUDE][0],
            pitch=obs[ob.ATTITUDE][1],
            rates=obs[ob.RATES] * ob.RATE_SCALE,
            targets=[row[:3] * ob.POS_SCALE for row in t if row[3] > 0.5],
            obstacles=[(row[:3] * ob.POS_SCALE, row[3] * ob.RADIUS_SCALE) for row in o if row[4] > 0.5],
            hold=obs[ob.HOLD],
        )


class Controller:
    name = "controller"

    def reset(self) -> None:
        """Forget everything from the previous episode (integrators, filters)."""

    def act(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, obs, state=None, episode_start=None, deterministic: bool = True):
        """SB3-style call. A batch of observations (vectorised envs) gets one controller copy per env."""
        obs = np.asarray(obs)
        if obs.ndim == 1:
            return self.act(obs), None
        if not hasattr(self, "_copies") or len(self._copies) != len(obs):
            self._copies = [copy.deepcopy(self) for _ in range(len(obs))]
        actions = []
        for i, c in enumerate(self._copies):
            if episode_start is not None and episode_start[i]:
                c.reset()
            actions.append(c.act(obs[i]))
        return np.array(actions), None
