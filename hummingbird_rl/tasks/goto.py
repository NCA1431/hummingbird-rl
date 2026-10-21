"""Task 2, go-to-point: fly to a point 10-50 cm away, in any direction, and stop there.

The bird starts facing a random direction, so about half the targets are
behind it. It can then either back up or turn around, and which one it picks
is one of the behaviours we measure.

Reward: points for every centimetre of progress, a bonus that grows close to
the target, and +50 once the target is held (see base.py).
"""

from __future__ import annotations

import numpy as np

from hummingbird_rl.sim.dynamics import BirdState
from hummingbird_rl.tasks.base import Outcome, Task
from hummingbird_rl.world.scene import Scene

PROGRESS_WEIGHT = 100.0      # reward per metre of progress (1 point per centimetre)
SUCCESS_BONUS = 50.0


class GotoTask(Task):
    name = "goto"
    duration = 5.0

    def reset(self, scene: Scene, state: BirdState) -> None:
        super().reset(scene, state)
        self.prev_distance = float(np.linalg.norm(state.pos - self.target))

    @property
    def target(self) -> np.ndarray:
        return self.scene.targets[0]

    def evaluate(self, state, action, prev_action, power) -> Outcome:
        self.step_count += 1
        d = float(np.linalg.norm(state.pos - self.target))
        # Rewarding progress (distance before minus distance now) rather than the
        # distance itself gives a clear signal from the first step, however far
        # the target is; summed over an episode it only depends on where the bird
        # started and where it ended.
        reward = (PROGRESS_WEIGHT * (self.prev_distance - d)
                  + 0.5 * np.exp(-d / 0.03)
                  + self.common_terms(action, prev_action, power))
        self.prev_distance = d

        if self.update_hold(state, self.target):
            return Outcome(reward + SUCCESS_BONUS, terminated=True, success=True, info={"distance": d})
        return Outcome(reward, info={"distance": d})
