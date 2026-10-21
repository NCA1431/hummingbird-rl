"""Potential field: how the classical controllers steer around obstacles.

Each visible obstacle closer than `avoid_range` (surface to surface) pushes
the bird away, more and more strongly as the gap closes. This is the
gradient of the classical repulsive potential ½ k (1/gap − 1/range)².

A pure push gets stuck when an obstacle sits right on the line to the goal:
attraction and repulsion cancel. We add a sideways component, horizontal and
on the side of the goal, so that the bird slides around the obstacle. It
helps, but it is no guarantee: local minima remain possible, which is the
known weakness of potential fields and one reason to try RL here.

Both PID and LQR use the same field, so they differ only in how they fly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hummingbird_rl.controllers.base import ObsView

UP = np.array([0.0, 1.0, 0.0])


@dataclass(frozen=True)
class FieldGains:
    avoid_range: float = 0.15   # m, obstacles closer than this repel the bird
    k_avoid: float = 0.05       # strength of the repulsion
    k_slide: float = 0.7        # sideways part, as a fraction of the repulsion
    a_max: float = 6.0          # m/s^2, cap on the total push


def repulsion(o: ObsView, r_bird: float, g: FieldGains) -> np.ndarray:
    """Push away from the obstacles that are too close, as an acceleration (m/s², heading frame)."""
    total = np.zeros(3)
    goal = o.targets[0] if o.targets else np.zeros(3)
    for centre, radius in o.obstacles:
        dist = np.linalg.norm(centre)
        gap = dist - radius - r_bird
        if gap >= g.avoid_range or dist < 1e-6:
            continue
        away = -centre / dist
        gap = max(gap, 0.005)
        push = g.k_avoid * (1.0 / gap - 1.0 / g.avoid_range) / gap ** 2
        side = np.cross(UP, away)              # horizontal and perpendicular to "away"
        if np.dot(side, goal) < 0.0:           # slide on the side of the goal
            side = -side
        total += push * (away + g.k_slide * side)
    norm = np.linalg.norm(total)
    if norm > g.a_max:
        total *= g.a_max / norm
    return total
