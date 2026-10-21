"""Classical baselines: cascaded PID and LQR, both with a potential field for obstacles."""

from hummingbird_rl.controllers.allocation import allocate
from hummingbird_rl.controllers.base import Controller, ObsView
from hummingbird_rl.controllers.lqr import LQRController, LQRWeights
from hummingbird_rl.controllers.pid import PIDController, PIDGains
from hummingbird_rl.controllers.potential_field import FieldGains, repulsion

CONTROLLERS = {"pid": PIDController, "lqr": LQRController}

__all__ = ["Controller", "ObsView", "allocate", "PIDController", "PIDGains", "LQRController", "LQRWeights",
           "FieldGains", "repulsion", "CONTROLLERS"]
