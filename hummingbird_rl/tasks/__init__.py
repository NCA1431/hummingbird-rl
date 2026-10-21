"""The four tasks: hover & recover, go-to-point, obstacles, navigation."""

from hummingbird_rl.tasks.base import Outcome, Task
from hummingbird_rl.tasks.goto import GotoTask
from hummingbird_rl.tasks.hover import HoverTask
from hummingbird_rl.tasks.navigation import NavigationTask
from hummingbird_rl.tasks.obstacles import ObstaclesTask

TASKS = {
    "hover": HoverTask,
    "goto": GotoTask,
    "obstacles": ObstaclesTask,
    "navigation": NavigationTask,
}

__all__ = ["Task", "Outcome", "HoverTask", "GotoTask", "ObstaclesTask", "NavigationTask", "TASKS"]
