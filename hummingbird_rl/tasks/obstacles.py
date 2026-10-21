"""Task 3, obstacles: reach a goal 50-80 cm away that is hidden behind a cluster of spheres.

Same rules as go-to-point (progress reward, hold at the goal), with two
differences: at least one obstacle blocks the straight line, and touching any
obstacle ends the episode with the failure penalty (the environment checks
this). The episode is a little longer, 6 s, to leave time for a detour.
"""

from __future__ import annotations

from hummingbird_rl.tasks.goto import GotoTask


class ObstaclesTask(GotoTask):
    name = "obstacles"
    duration = 6.0
    has_obstacles = True
