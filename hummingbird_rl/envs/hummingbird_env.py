"""The Gymnasium environment: one episode of one task under one regime of disturbances.

At every decision (60 per second) the controller sends 5 numbers in [-1, 1];
the environment adds actuator noise, simulates 4 physics steps of 1/240 s
with the current gusts, and returns the next observation, the reward and
whether the episode is over.

An episode can end in three ways, which Gymnasium keeps apart:
  - terminated, because of a failure (crash, out of the arena, tilted past
    80 degrees, obstacle hit): reward -10;
  - terminated, because the task was won (a target held);
  - truncated, because time is up. This distinction matters for learning:
    after a truncation the value of the next state is not zero, the episode
    was simply cut, so the algorithms should bootstrap from it.

The RL agents and the classical controllers use exactly the same interface,
which is what makes the comparison fair.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hummingbird_rl.envs.disturbances import Conditions, Regime, sample_conditions
from hummingbird_rl.envs.observation import CLIP, OBS_SIZE, build_observation
from hummingbird_rl.sim import (
    DEFAULT_PARAMS,
    BirdDynamics,
    BirdParams,
    BirdState,
    GustProcess,
    check_limits,
    hover_action,
)
from hummingbird_rl.tasks import TASKS
from hummingbird_rl.world.collision import collides
from hummingbird_rl.world.sampling import sample_scene
from hummingbird_rl.world.scene import Scene


class HummingbirdEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, task: str = "hover", regime: str | Regime = "normal", params: BirdParams | None = None,
                 failure_reward: float | None = None, n_obstacles: tuple[int, int] | None = None):
        """Two options for experiments: `failure_reward` replaces the task's penalty for a failure
        (-10), and `n_obstacles` (min, max) the number of spheres in the tasks with obstacles."""
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}, expected one of {sorted(TASKS)}")
        self.p = params or DEFAULT_PARAMS
        task_kwargs = {} if failure_reward is None else {"failure_reward": failure_reward}
        self.task = TASKS[task](self.p, **task_kwargs)
        self.scene_options = {} if n_obstacles is None else {"n_obstacles": tuple(n_obstacles)}
        self.regime = regime
        self.dyn = BirdDynamics(self.p)
        self.hover = hover_action(self.p)
        self.observation_space = spaces.Box(-CLIP, CLIP, (OBS_SIZE,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (5,), np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Start a new episode.

        `options` can fix the "scene", the "conditions" or the initial "state"
        instead of drawing them; we use this to replay exactly the same
        situation with different controllers, and in the tests.
        """
        super().reset(seed=seed)
        rng = self.np_random
        options = options or {}
        self.scene: Scene = options.get("scene") or sample_scene(self.task.name, rng, self.p.r_bird,
                                                                 **self.scene_options)
        self.conditions: Conditions = options.get("conditions") or sample_conditions(self.regime, rng, self.p)
        self.state: BirdState = (options["state"].copy() if "state" in options
                                 else self.task.initial_state(self.scene, rng))

        c = self.conditions
        self.dyn.set_conditions(c.mass_factor, c.damage)
        self.gusts = GustProcess(c.mean_wind, c.gust_sigma, rng=rng)
        self.failed = False
        self.task.reset(self.scene, self.state)
        # The controller does not know the bird's mass, so it starts from the
        # nominal hover action, as if it had been hovering just before.
        self.prev_action = self.hover.copy()
        self.energy = 0.0
        return self._observe(), {"conditions": c.to_dict()}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        c = self.conditions
        applied = action
        if c.action_noise > 0.0:
            applied = np.clip(action + c.action_noise * self.np_random.standard_normal(5), -1.0, 1.0)

        # The action is held for 4 physics steps. Limits and collisions are
        # checked after every one of them, so a crash cannot slip through
        # between two decisions.
        reason, power = None, 0.0
        for _ in range(self.p.action_repeat):
            self._apply_failure()
            self.state, sim_info = self.dyn.step(self.state, applied, self.gusts.step(self.p.dt))
            power += sim_info.power / self.p.action_repeat
            self.energy += sim_info.power * self.p.dt
            reason = check_limits(self.state)
            hit = self.task.has_obstacles and collides(self.state.pos, self.scene.obstacles, self.p.r_bird)
            if reason is None and hit:
                reason = "collision"
            if reason is not None:
                break

        if reason is not None:
            reward, terminated, success = self.task.failure_reward(), True, False
            extra = self.task.failure_info(self.state)
        else:
            out = self.task.evaluate(self.state, action, self.prev_action, power)
            reward, terminated, success, extra = out.reward, out.terminated, out.success, out.info
            if terminated:
                reason = "success"
        truncated = not terminated and self.task.step_count >= self.task.n_steps
        if truncated:
            reason = "time_limit"

        self.prev_action = action
        # "is_success" is the key Stable-Baselines3 looks for to log success rates.
        info = {"is_success": success, "reason": reason, "energy": self.energy, "time": self.state.time, **extra}
        return self._observe(), float(reward), terminated, truncated, info

    def _apply_failure(self) -> None:
        """Make one wing lose part of its force once the scheduled failure time is reached."""
        c = self.conditions
        if c.failure_time is None or self.failed or self.state.time < c.failure_time:
            return
        damage = list(self.dyn.damage)
        damage[c.failure_wing] = max(damage[c.failure_wing], c.failure_amount)
        self.dyn.set_conditions(c.mass_factor, (damage[0], damage[1]))
        self.failed = True

    def _observe(self) -> np.ndarray:
        obstacles = self.scene.obstacles if self.task.has_obstacles else []
        return build_observation(self.state, self.task.obs_targets(), obstacles, self.prev_action,
                                 self.task.hold_progress(), self.p.r_bird, self.conditions.obs_noise, self.np_random)
