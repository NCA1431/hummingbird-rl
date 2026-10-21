"""The Gymnasium environment that connects the simulator, the tasks and the controllers."""

from hummingbird_rl.envs.disturbances import REGIMES, Conditions, grid_conditions, sample_conditions
from hummingbird_rl.envs.hummingbird_env import HummingbirdEnv
from hummingbird_rl.envs.registry import ENV_IDS, make_env, register_envs

register_envs()

__all__ = ["HummingbirdEnv", "make_env", "register_envs", "ENV_IDS",
           "Conditions", "REGIMES", "grid_conditions", "sample_conditions"]
