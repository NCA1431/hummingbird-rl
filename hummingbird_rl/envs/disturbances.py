"""The physical conditions of an episode: wind, mass, wing damage, sudden failure, noise.

A "regime" is a recipe for drawing these conditions at the start of every
episode:

    calm        no wind and no noise at all; for debugging and tests
    normal      a light breeze (mean up to 5% of the weight, gusts of 2%),
                the nominal bird, small sensor and actuator noise
    randomised  domain randomisation: mean wind up to 30% of the weight,
                gusts up to 10%, mass ±25%, and in 70% of the episodes one
                wing has lost up to 40% of its force
    failure     normal conditions, except that one wing suddenly loses 40% of
                its force somewhere between 1.5 and 3 s into the episode

Wind is given as a force in % of the bird's weight because that is what
makes it hard to fly in: 30% means the bird must tilt its thrust by about
17 degrees just to stay in place. The mean wind is horizontal and blows
from a random direction.

For the robustness grid of the evaluation, `grid_conditions` fixes the wind
strength and the damage exactly instead of drawing them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from hummingbird_rl.sim.params import BirdParams

ACTION_NOISE = 0.02      # std of the noise added to each action, in action units
OBS_NOISE = 0.01         # std of the sensor noise, in observation units


@dataclass
class Conditions:
    mean_wind: np.ndarray = field(default_factory=lambda: np.zeros(3))   # N
    gust_sigma: float = 0.0                 # N, size of the gust fluctuations
    mass_factor: float = 1.0
    damage: tuple[float, float] = (0.0, 0.0)  # fraction of force lost by the (left, right) wing
    failure_time: float | None = None       # s, when the sudden failure happens (None: never)
    failure_wing: int = 0                   # 0 left, 1 right
    failure_amount: float = 0.0             # fraction of force lost at failure_time
    action_noise: float = 0.0
    obs_noise: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mean_wind"] = self.mean_wind.tolist()
        return d


def _horizontal(rng: np.random.Generator, magnitude: float) -> np.ndarray:
    """A horizontal vector of the given length pointing in a random direction."""
    angle = rng.uniform(-np.pi, np.pi)
    return magnitude * np.array([np.cos(angle), 0.0, np.sin(angle)])


def _random_wing(rng: np.random.Generator, amount: float) -> tuple[float, float]:
    return (amount, 0.0) if rng.uniform() < 0.5 else (0.0, amount)


def calm(rng: np.random.Generator, p: BirdParams) -> Conditions:
    return Conditions()


def normal(rng: np.random.Generator, p: BirdParams) -> Conditions:
    return Conditions(
        mean_wind=_horizontal(rng, rng.uniform(0.0, 0.05) * p.weight),
        gust_sigma=0.02 * p.weight,
        action_noise=ACTION_NOISE,
        obs_noise=OBS_NOISE,
    )


def randomised(rng: np.random.Generator, p: BirdParams) -> Conditions:
    damage = (0.0, 0.0) if rng.uniform() < 0.3 else _random_wing(rng, rng.uniform(0.0, 0.4))
    return Conditions(
        mean_wind=_horizontal(rng, rng.uniform(0.0, 0.30) * p.weight),
        gust_sigma=rng.uniform(0.0, 0.10) * p.weight,
        mass_factor=rng.uniform(0.75, 1.25),
        damage=damage,
        action_noise=ACTION_NOISE,
        obs_noise=OBS_NOISE,
    )


def failure(rng: np.random.Generator, p: BirdParams) -> Conditions:
    c = normal(rng, p)
    c.failure_time = rng.uniform(1.5, 3.0)
    c.failure_wing = int(rng.integers(0, 2))
    c.failure_amount = 0.4
    return c


def grid_conditions(rng: np.random.Generator, p: BirdParams, wind_fraction: float, damage: float) -> Conditions:
    """One cell of the robustness grid: fixed wind strength and damage on the left wing.

    Gusts grow with the mean wind (a third of it, never below the normal 2%),
    since a strong wind is rarely a steady one.
    """
    return Conditions(
        mean_wind=_horizontal(rng, wind_fraction * p.weight),
        gust_sigma=max(0.02, wind_fraction / 3.0) * p.weight,
        damage=(damage, 0.0),
        action_noise=ACTION_NOISE,
        obs_noise=OBS_NOISE,
    )


Regime = Callable[[np.random.Generator, BirdParams], Conditions]

REGIMES: dict[str, Regime] = {
    "calm": calm,
    "normal": normal,
    "randomised": randomised,
    "failure": failure,
}


def sample_conditions(regime: str | Regime, rng: np.random.Generator, p: BirdParams) -> Conditions:
    """Draw the conditions for one episode. `regime` is a name from REGIMES or any function of (rng, params)."""
    if callable(regime):
        return regime(rng, p)
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}, expected one of {sorted(REGIMES)}")
    return REGIMES[regime](rng, p)
