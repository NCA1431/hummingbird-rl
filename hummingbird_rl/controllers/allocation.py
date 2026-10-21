"""From "what force and torques do I want" to the 5 wing actions (drone people call this the mixer).

The classical controllers reason in physical terms: a total thrust and three
torques. This module inverts the nominal wing model of sim/aero.py:

    F_left + F_right = T                     total thrust
    (F_left - F_right) r_wing = tau_roll     roll
    T · pitch_lever · sin(tilt) = tau_pitch  pitch
    k_yaw · T · r_wing · sin(yaw_tilt) = tau_yaw

with F = k (f A)^2 for each wing. The frequency stays at f0 unless one wing
would need more than its maximum amplitude; only then is it raised (up to
1.2 f0). When a request is impossible, each quantity is clipped on its own,
which is crude but predictable.
"""

from __future__ import annotations

from math import asin, sqrt

import numpy as np

from hummingbird_rl.sim.params import BirdParams


def allocate(thrust: float, torque: np.ndarray, p: BirdParams) -> np.ndarray:
    """Actions in [-1, 1]^5 for a desired total thrust (N) and body torques [roll, yaw, pitch] (N·m)."""
    k = p.thrust_gain
    diff = torque[0] / p.r_wing
    f_left = max(0.5 * (thrust + diff), 0.0)
    f_right = max(0.5 * (thrust - diff), 0.0)

    # Frequency: f0 if both wings can deliver their force at f0, otherwise just enough for the stronger one.
    f_needed = sqrt(max(f_left, f_right) / k) / p.amp_max
    freq = float(np.clip(f_needed, p.f0, p.f0 * (1.0 + p.freq_span)))
    amp_left = min(sqrt(f_left / k) / freq, p.amp_max)
    amp_right = min(sqrt(f_right / k) / freq, p.amp_max)

    t_real = k * freq ** 2 * (amp_left ** 2 + amp_right ** 2)       # thrust actually produced
    t_safe = max(t_real, 1e-6)
    tilt = asin(float(np.clip(torque[2] / (t_safe * p.pitch_lever), -np.sin(p.tilt_max), np.sin(p.tilt_max))))
    yaw_tilt = asin(float(np.clip(torque[1] / (p.k_yaw * t_safe * p.r_wing),
                                  -np.sin(p.yaw_tilt_max), np.sin(p.yaw_tilt_max))))

    return np.clip(np.array([
        2.0 * amp_left / p.amp_max - 1.0,
        2.0 * amp_right / p.amp_max - 1.0,
        (freq / p.f0 - 1.0) / p.freq_span,
        tilt / p.tilt_max,
        yaw_tilt / p.yaw_tilt_max,
    ]), -1.0, 1.0)
