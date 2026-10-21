"""Geometry between the bird and the obstacles.

The bird is a sphere of radius r_bird and every obstacle is a sphere too, so
all we need are distances between points. "Clearance" is the gap between the
two surfaces: positive means free space, zero or negative means contact.
"""

from __future__ import annotations

import numpy as np

from hummingbird_rl.world.scene import Obstacle


def clearance(pos: np.ndarray, obstacle: Obstacle, r_bird: float) -> float:
    """Gap between the bird's surface and the obstacle's surface (m)."""
    d = pos - obstacle.center
    return float(np.sqrt(d @ d)) - obstacle.radius - r_bird     # faster than np.linalg.norm for one vector


def collides(pos: np.ndarray, obstacles: list[Obstacle], r_bird: float) -> bool:
    return any(clearance(pos, o, r_bird) < 0.0 for o in obstacles)


def min_clearance(pos: np.ndarray, obstacles: list[Obstacle], r_bird: float) -> float:
    """Smallest gap to any obstacle (infinite if there are none)."""
    return min((clearance(pos, o, r_bird) for o in obstacles), default=float("inf"))


def nearest_obstacles(pos: np.ndarray, obstacles: list[Obstacle], r_bird: float,
                      k: int = 5, max_range: float = 1.0) -> list[Obstacle]:
    """The k obstacles with the smallest clearance, ignoring those farther than max_range.

    This is what the bird "sees": a limited number of nearby obstacles, closest first.
    """
    close = [(clearance(pos, o, r_bird), i) for i, o in enumerate(obstacles)]
    close = sorted(c for c in close if c[0] <= max_range)
    return [obstacles[i] for _, i in close[:k]]


def segment_point_distance(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> float:
    """Distance from point p to the segment [a, b]; used to check that obstacles block a path."""
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(a + t * ab - p))
