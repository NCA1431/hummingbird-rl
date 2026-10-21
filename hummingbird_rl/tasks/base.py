"""What all tasks have in common: start state, reward pieces, the "hold" rule.

A task turns the bird's state into a reward and decides when the episode is
won. The environment calls `evaluate` once per decision (60 times a second).
Crashes, leaving the arena and collisions are detected by the environment,
which then asks the task for `failure_reward`.

Reward design. The per-step reward is bounded and mostly positive near the
goal. If it were mostly negative, the quickest way to stop losing points
would be to crash on purpose, a classic failure of RL reward design. Every
task also pays a small price for jerky actions and for energy, so that
smooth, efficient flight is preferred when everything else is equal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hummingbird_rl.sim.dynamics import BirdState
from hummingbird_rl.sim.params import BirdParams
from hummingbird_rl.world.scene import Scene

# "Holding" a target means staying within 3 cm of it, slower than 10 cm/s, for
# 15 decisions in a row (a quarter of a second). Requiring a hold, rather than
# just touching the target, stops the agent from flying straight through it.
HOLD_RADIUS = 0.03      # m
HOLD_SPEED = 0.10       # m/s
HOLD_STEPS = 15

FAILURE_REWARD = -10.0
SMOOTHNESS_WEIGHT = 0.02
ENERGY_WEIGHT = 0.01


@dataclass
class Outcome:
    reward: float
    terminated: bool = False     # the episode ends because the task is won
    success: bool = False
    info: dict = field(default_factory=dict)


class Task:
    name = "base"
    duration = 5.0               # s
    has_obstacles = False

    def __init__(self, params: BirdParams, failure_reward: float = FAILURE_REWARD):
        self.p = params
        self.failure_value = failure_reward
        self.control_dt = params.dt * params.action_repeat          # 1/60 s
        self.n_steps = int(round(self.duration / self.control_dt))  # e.g. 5 s -> 300 decisions
        self.scene: Scene | None = None

    # --- episode start -----------------------------------------------------
    def initial_state(self, scene: Scene, rng: np.random.Generator) -> BirdState:
        """Default start: close to the start point with a small random push, facing any direction."""
        return BirdState.from_euler(
            pos=scene.start + rng.uniform(-0.02, 0.02, 3),
            vel=rng.uniform(-0.05, 0.05, 3),
            yaw=rng.uniform(-np.pi, np.pi),
            pitch=np.radians(rng.uniform(-5, 5)),
            roll=np.radians(rng.uniform(-5, 5)),
            omega=rng.uniform(-0.5, 0.5, 3),
        )

    def reset(self, scene: Scene, state: BirdState) -> None:
        self.scene = scene
        self.step_count = 0
        self.hold_count = 0

    # --- during the episode ------------------------------------------------
    def obs_targets(self) -> list[np.ndarray]:
        """Up to two targets the controller should see, the current one first."""
        return self.scene.targets[:1]

    def hold_progress(self) -> float:
        """Fraction of the required hold already done (0 to 1), shown to the controller."""
        return self.hold_count / HOLD_STEPS

    def update_hold(self, state: BirdState, target: np.ndarray) -> bool:
        """Count consecutive steps spent holding the target; True once the hold is complete."""
        holding = (np.linalg.norm(state.pos - target) < HOLD_RADIUS
                   and np.linalg.norm(state.vel) < HOLD_SPEED)
        self.hold_count = self.hold_count + 1 if holding else 0
        return self.hold_count >= HOLD_STEPS

    def common_terms(self, action: np.ndarray, prev_action: np.ndarray, power: float) -> float:
        """Small penalties shared by every task: jerky actions and energy (relative to hovering)."""
        jerk = float(np.sum((np.asarray(action) - np.asarray(prev_action)) ** 2))
        return -SMOOTHNESS_WEIGHT * jerk - ENERGY_WEIGHT * power / self.p.hover_power()

    def evaluate(self, state: BirdState, action: np.ndarray, prev_action: np.ndarray, power: float) -> Outcome:
        raise NotImplementedError

    def failure_reward(self) -> float:
        return self.failure_value

    def failure_info(self, state: BirdState) -> dict:
        """What we still want to log when the episode ends in a failure."""
        return {"distance": float(np.linalg.norm(state.pos - self.obs_targets()[0]))}
