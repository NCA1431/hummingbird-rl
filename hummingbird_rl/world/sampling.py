"""Random scenes for the four tasks.

Every episode gets a new scene, so the agent cannot memorise a trajectory and
has to learn a general skill. All randomness comes from the generator passed
in, which makes a scene fully reproducible from its seed (we rely on this for
the fixed validation and test sets).

Scenes with obstacles are checked before use: at least one obstacle must
block the straight path, otherwise the task is just go-to-point, and a coarse
grid search must find a way through, otherwise the task is impossible.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import ndimage

from hummingbird_rl.world.collision import segment_point_distance
from hummingbird_rl.world.scene import START, Obstacle, Scene

ALTITUDE_RANGE = (0.15, 0.95)   # m, where targets may be placed
MAX_TRIES = 500


def _unit_vector(rng: np.random.Generator) -> np.ndarray:
    """Direction drawn uniformly on the sphere (normalised Gaussian vector)."""
    v = rng.standard_normal(3)
    return v / np.linalg.norm(v)


def _point_in_ball(rng: np.random.Generator, radius: float) -> np.ndarray:
    """Point drawn uniformly inside a ball (the cube root keeps the density uniform)."""
    return _unit_vector(rng) * radius * rng.uniform() ** (1.0 / 3.0)


def _far_enough(center: np.ndarray, radius: float, points: list[np.ndarray], gap: float) -> bool:
    """True if the obstacle's surface stays at least `gap` away from every point."""
    return all(np.linalg.norm(center - p) - radius >= gap for p in points)


def path_exists(points: list[np.ndarray], obstacles: list[Obstacle], r_bird: float, cell: float = 0.05) -> bool:
    """Coarse check that the bird can fly through the points in order without touching anything.

    We cut the space around the points into 5 cm cubes, mark a cube as blocked
    if its centre is inside an obstacle inflated by the bird's radius, and ask
    whether all the points lie in the same connected region of free cubes.
    """
    pts = np.array(points)
    lo = pts.min(axis=0) - 0.3          # 30 cm of room around the course...
    hi = pts.max(axis=0) + 0.3
    lo[1], hi[1] = 0.05, 1.15           # ...between just above the ground and just below the ceiling
    axes = [np.arange(lo[i], hi[i] + cell, cell) for i in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    free = np.ones(gx.shape, dtype=bool)
    for o in obstacles:
        d2 = (gx - o.center[0]) ** 2 + (gy - o.center[1]) ** 2 + (gz - o.center[2]) ** 2
        free &= d2 > (o.radius + r_bird) ** 2
    labels, _ = ndimage.label(free)        # connected regions of free space (face neighbours)
    ids = set()
    for p in pts:
        idx = tuple(int(round((p[i] - lo[i]) / cell)) for i in range(3))
        if labels[idx] == 0:
            return False                   # a target lies inside an obstacle
        ids.add(labels[idx])
    return len(ids) == 1


# ------------------------------------------------------------------- the tasks

def sample_hover(rng: np.random.Generator, r_bird: float = 0.02) -> Scene:
    """Hover & recover: the target is the start point itself."""
    return Scene(start=START.copy(), targets=[START.copy()])


def sample_goto(rng: np.random.Generator, r_bird: float = 0.02) -> Scene:
    """Go-to-point: a target 10-50 cm away in any direction (including behind and below)."""
    for _ in range(MAX_TRIES):
        target = START + _unit_vector(rng) * rng.uniform(0.10, 0.50)
        if ALTITUDE_RANGE[0] <= target[1] <= ALTITUDE_RANGE[1]:
            return Scene(start=START.copy(), targets=[target])
    raise RuntimeError("could not sample a go-to-point scene")


def sample_obstacles(rng: np.random.Generator, r_bird: float = 0.02) -> Scene:
    """One goal 50-80 cm away with 3-6 spheres placed around the straight path."""
    for _ in range(MAX_TRIES):
        dist = rng.uniform(0.50, 0.80)
        altitude = rng.uniform(0.30, 0.80)
        dy = altitude - START[1]
        horizontal = np.sqrt(dist ** 2 - dy ** 2)
        bearing = rng.uniform(-np.pi, np.pi)
        goal = START + np.array([horizontal * np.cos(bearing), dy, horizontal * np.sin(bearing)])

        obstacles = []
        for _ in range(rng.integers(3, 7)):
            for _ in range(50):
                radius = rng.uniform(0.04, 0.10)
                center = START + rng.uniform(0.2, 0.8) * (goal - START) + _point_in_ball(rng, 0.12)
                if _far_enough(center, radius, [START, goal], gap=0.10):
                    obstacles.append(Obstacle(center, radius))
                    break

        blocking = any(segment_point_distance(START, goal, o.center) < o.radius + r_bird for o in obstacles)
        if obstacles and blocking and path_exists([START, goal], obstacles, r_bird):
            return Scene(start=START.copy(), targets=[goal], obstacles=obstacles)
    raise RuntimeError("could not sample an obstacles scene")


def sample_navigation(rng: np.random.Generator, r_bird: float = 0.02) -> Scene:
    """Three waypoints in a fixed order, with 6-10 spheres scattered along the course."""
    for _ in range(MAX_TRIES):
        waypoints, current = [], START.copy()
        for _ in range(3):
            for _ in range(100):
                bearing = rng.uniform(-np.pi, np.pi)
                step = rng.uniform(0.25, 0.50)
                altitude = rng.uniform(0.20, 0.90)
                dy = altitude - current[1]
                if abs(dy) >= step:
                    continue
                h = np.sqrt(step ** 2 - dy ** 2)
                nxt = current + np.array([h * np.cos(bearing), dy, h * np.sin(bearing)])
                if np.hypot(nxt[0], nxt[2]) <= 1.0:          # stay well inside the arena
                    waypoints.append(nxt)
                    current = nxt
                    break
        if len(waypoints) < 3:
            continue

        course = [START] + waypoints
        obstacles = []
        for _ in range(rng.integers(6, 11)):
            for _ in range(50):
                leg = rng.integers(0, 3)
                a, b = course[leg], course[leg + 1]
                radius = rng.uniform(0.04, 0.10)
                center = a + rng.uniform(0.2, 0.8) * (b - a) + _point_in_ball(rng, 0.15)
                if 0.05 <= center[1] <= 1.15 and _far_enough(center, radius, course, gap=0.10):
                    obstacles.append(Obstacle(center, radius))
                    break

        if path_exists(course, obstacles, r_bird):
            return Scene(start=START.copy(), targets=waypoints, obstacles=obstacles)
    raise RuntimeError("could not sample a navigation scene")


SAMPLERS: dict[str, Callable[[np.random.Generator, float], Scene]] = {
    "hover": sample_hover,
    "goto": sample_goto,
    "obstacles": sample_obstacles,
    "navigation": sample_navigation,
}


def sample_scene(task: str, rng: np.random.Generator, r_bird: float = 0.02) -> Scene:
    return SAMPLERS[task](rng, r_bird)
