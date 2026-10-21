"""What the controller sees: 48 numbers about the bird, its targets and the nearby obstacles.

Everything is relative to the bird and expressed in its heading frame, i.e.
the world turned by the bird's yaw, so the first axis is always "forward".
With this choice a target 30 cm ahead looks the same whichever way the bird
faces, and the policy does not have to learn the same skill once per
direction. The absolute horizontal position is not given at all; only the
altitude matters, because of the ground and the ceiling.

Each quantity is divided by a typical size so that values are of order 1,
which neural networks train on much better. Empty slots (no second target,
fewer than 5 obstacles in range) are zeros with their "valid" flag at 0.

    index   content                                        scale
    0       altitude                                       1 m
    1-3     velocity (forward, up, right)                  1 m/s
    4-5     roll, pitch                                    1 rad
    6-8     body rates (roll, yaw, pitch)                  10 rad/s
    9-13    previous action                                -
    14-17   current target: dx, dy, dz, valid              0.5 m
    18-21   next target (navigation only), same layout     0.5 m
    22-46   5 nearest obstacles: dx, dy, dz, radius, valid 0.5 m, 0.1 m
    47      progress of the hold at the target (0 to 1)    -

Sensors are not perfect: Gaussian noise is added to the measured entries
(state, targets, obstacles), but not to the previous action, the flags or
the hold progress, which the controller knows exactly.
"""

from __future__ import annotations

import numpy as np

from hummingbird_rl.sim.dynamics import BirdState
from hummingbird_rl.sim.rotations import world_to_heading
from hummingbird_rl.world.collision import nearest_obstacles
from hummingbird_rl.world.scene import Obstacle

OBS_SIZE = 48
N_TARGETS = 2
N_OBSTACLES = 5
OBSTACLE_RANGE = 1.0     # m, obstacles with a larger clearance are not seen
POS_SCALE = 0.5          # m
RADIUS_SCALE = 0.1       # m
RATE_SCALE = 10.0        # rad/s
CLIP = 5.0               # every entry is clipped to [-5, 5]

# Where each block lives, so that tests and analysis code can refer to it by name.
ALTITUDE = 0
VELOCITY = slice(1, 4)
ATTITUDE = slice(4, 6)
RATES = slice(6, 9)
PREV_ACTION = slice(9, 14)
TARGETS = slice(14, 22)
OBSTACLES = slice(22, 47)
HOLD = 47


def build_observation(state: BirdState, targets: list[np.ndarray], obstacles: list[Obstacle],
                      prev_action: np.ndarray, hold_progress: float, r_bird: float,
                      noise: float = 0.0, rng: np.random.Generator | None = None) -> np.ndarray:
    yaw, pitch, roll = state.euler
    obs = np.zeros(OBS_SIZE)
    measured = np.zeros(OBS_SIZE, dtype=bool)     # entries that come from a (noisy) sensor

    obs[ALTITUDE] = state.pos[1]
    obs[VELOCITY] = world_to_heading(state.vel, yaw)
    obs[ATTITUDE] = roll, pitch
    obs[RATES] = state.omega / RATE_SCALE
    measured[:RATES.stop] = True
    obs[PREV_ACTION] = prev_action

    for i, target in enumerate(targets[:N_TARGETS]):
        j = TARGETS.start + 4 * i
        obs[j:j + 3] = world_to_heading(target - state.pos, yaw) / POS_SCALE
        obs[j + 3] = 1.0
        measured[j:j + 3] = True

    for i, o in enumerate(nearest_obstacles(state.pos, obstacles, r_bird, N_OBSTACLES, OBSTACLE_RANGE)):
        j = OBSTACLES.start + 5 * i
        obs[j:j + 3] = world_to_heading(o.center - state.pos, yaw) / POS_SCALE
        obs[j + 3] = o.radius / RADIUS_SCALE
        obs[j + 4] = 1.0
        measured[j:j + 4] = True

    obs[HOLD] = hold_progress

    if noise > 0.0:
        obs[measured] += noise * rng.standard_normal(int(measured.sum()))
    return np.clip(obs, -CLIP, CLIP).astype(np.float32)
