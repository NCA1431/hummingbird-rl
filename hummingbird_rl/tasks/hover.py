"""Task 1, hover & recover: get back to the start point after a push and stay there.

Each episode begins as if a gust had just hit the bird: a few centimetres off
target, drifting, tilted and spinning. Because hovering is unstable, doing
nothing is not an option; the controller has to catch the bird and then hold
it still for the rest of the 5 seconds, in gusty air.
"""

from __future__ import annotations

import numpy as np

from hummingbird_rl.sim.dynamics import BirdState
from hummingbird_rl.tasks.base import Outcome, Task
from hummingbird_rl.world.scene import Scene

SUCCESS_RADIUS = 0.03        # m, average distance allowed over the last 2 s
SUCCESS_WINDOW = 2.0         # s


class HoverTask(Task):
    name = "hover"
    duration = 5.0

    def initial_state(self, scene: Scene, rng: np.random.Generator) -> BirdState:
        """A much larger push than the other tasks: this task is about recovering from it."""
        return BirdState.from_euler(
            pos=scene.start + rng.uniform(-0.10, 0.10, 3),
            vel=rng.uniform(-0.3, 0.3, 3),
            yaw=rng.uniform(-np.pi, np.pi),
            pitch=np.radians(rng.uniform(-15, 15)),
            roll=np.radians(rng.uniform(-15, 15)),
            omega=rng.uniform(-2.0, 2.0, 3),
        )

    def reset(self, scene: Scene, state: BirdState) -> None:
        super().reset(scene, state)
        self.distances: list[float] = []

    def evaluate(self, state, action, prev_action, power) -> Outcome:
        self.step_count += 1
        d = float(np.linalg.norm(state.pos - self.scene.targets[0]))
        self.distances.append(d)
        _, pitch, roll = state.euler

        # Close to the target is good (1 at the target, about 0.37 at 5 cm); a
        # tilted or spinning bird is penalised so that "still" means still.
        reward = (np.exp(-d / 0.05)
                  - 0.05 * (roll ** 2 + pitch ** 2)
                  - 0.01 * state.omega[1] ** 2
                  + self.common_terms(action, prev_action, power))

        success = False
        if self.step_count >= self.n_steps:      # judged only at the end of the episode
            window = int(round(SUCCESS_WINDOW / self.control_dt))
            success = float(np.mean(self.distances[-window:])) < SUCCESS_RADIUS
        return Outcome(reward, terminated=False, success=success, info={"distance": d})
