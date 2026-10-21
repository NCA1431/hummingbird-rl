"""Task 4, navigation: pass three waypoints in order through a cluttered space, then stop.

The first two waypoints only need to be passed (within 4 cm); the last one
must be held, as in go-to-point. The controller sees the current waypoint
and the next one, so it can prepare its turn. A small time penalty pushes it
to finish quickly rather than dawdle.
"""

from __future__ import annotations

import numpy as np

from hummingbird_rl.sim.dynamics import BirdState
from hummingbird_rl.tasks.base import Outcome, Task
from hummingbird_rl.tasks.goto import PROGRESS_WEIGHT, SUCCESS_BONUS
from hummingbird_rl.world.scene import Scene

PASS_RADIUS = 0.04       # m, for the intermediate waypoints
WAYPOINT_BONUS = 20.0
TIME_PENALTY = 0.01      # per decision


class NavigationTask(Task):
    name = "navigation"
    duration = 15.0
    has_obstacles = True

    def reset(self, scene: Scene, state: BirdState) -> None:
        super().reset(scene, state)
        self.index = 0                          # which waypoint we are heading for
        self.prev_distance = float(np.linalg.norm(state.pos - scene.targets[0]))

    @property
    def is_last(self) -> bool:
        return self.index == len(self.scene.targets) - 1

    def obs_targets(self) -> list[np.ndarray]:
        return self.scene.targets[self.index:self.index + 2]

    def hold_progress(self) -> float:
        return super().hold_progress() if self.is_last else 0.0

    def failure_info(self, state: BirdState) -> dict:
        return {**super().failure_info(state), "waypoints_reached": self.index}

    def evaluate(self, state, action, prev_action, power) -> Outcome:
        self.step_count += 1
        target = self.scene.targets[self.index]
        d = float(np.linalg.norm(state.pos - target))
        reward = (PROGRESS_WEIGHT * (self.prev_distance - d)
                  - TIME_PENALTY
                  + self.common_terms(action, prev_action, power))
        self.prev_distance = d
        info = {"distance": d, "waypoints_reached": self.index}

        if not self.is_last and d < PASS_RADIUS:
            # Switch to the next waypoint. The progress reference restarts from the
            # new distance, otherwise the jump in distance would look like a huge
            # step backwards.
            self.index += 1
            self.prev_distance = float(np.linalg.norm(state.pos - self.scene.targets[self.index]))
            info["waypoints_reached"] = self.index
            return Outcome(reward + WAYPOINT_BONUS, info=info)

        if self.is_last and self.update_hold(state, target):
            info["waypoints_reached"] = self.index + 1
            return Outcome(reward + WAYPOINT_BONUS + SUCCESS_BONUS, terminated=True, success=True, info=info)
        return Outcome(reward, info=info)
