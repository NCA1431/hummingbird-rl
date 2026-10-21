"""What an episode takes place in: a start point, targets to reach and obstacles.

A scene is plain data. The tasks decide what the targets mean (a point to
hover at, a goal, a list of waypoints) and the samplers in sampling.py decide
where they go. Scenes can be saved to JSON so that the web viewer can replay
exactly the same situation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

START = np.array([0.0, 0.5, 0.0])   # m, every episode starts around this point, 50 cm above the ground


@dataclass
class Obstacle:
    """A sphere the bird must not touch."""
    center: np.ndarray      # m, world frame
    radius: float           # m


@dataclass
class Scene:
    start: np.ndarray = field(default_factory=lambda: START.copy())
    targets: list[np.ndarray] = field(default_factory=list)      # in the order they must be reached
    obstacles: list[Obstacle] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "start": self.start.tolist(),
            "targets": [t.tolist() for t in self.targets],
            "obstacles": [{"center": o.center.tolist(), "radius": o.radius} for o in self.obstacles],
        }

    @staticmethod
    def from_dict(d: dict) -> "Scene":
        return Scene(
            start=np.array(d["start"], dtype=float),
            targets=[np.array(t, dtype=float) for t in d["targets"]],
            obstacles=[Obstacle(np.array(o["center"], dtype=float), float(o["radius"])) for o in d["obstacles"]],
        )
