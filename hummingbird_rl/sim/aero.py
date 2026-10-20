"""Wing aerodynamics: from the 5 actions to forces and torques on the bird.

We do not simulate each individual wingbeat. At 40 beats per second the force
of a wing oscillates far faster than the bird can react, so we use its
average over a beat (a "quasi-steady", wingbeat-averaged model). This is the
usual level of detail for flight-control studies, and it lets the physics run
at 240 Hz. The drawback is that the wing's position within a beat plays no
role in the dynamics; it only matters for drawing the bird.

Every function here is pure (no hidden state), which keeps them easy to test
and to port line by line to the web version.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sin

import numpy as np

from hummingbird_rl.sim.params import BirdParams


@dataclass(frozen=True)
class WingCommand:
    """Physical wing settings decoded from the 5 actions."""
    amp_left: float     # rad, half-stroke amplitude of the left wing
    amp_right: float    # rad, same for the right wing
    freq: float         # Hz, wingbeat frequency (shared by both wings)
    tilt: float         # rad, stroke-plane tilt (creates the pitch torque)
    yaw_tilt: float     # rad, opposite tilt of the two stroke planes (creates the yaw torque)


def decode_action(action: np.ndarray, p: BirdParams) -> WingCommand:
    """Map an action in [-1, 1]^5 to wing settings. Out-of-range actions are clipped.

    a1, a2 : amplitude of each wing, from 0 (wing still) to amp_max
    a3     : frequency, from 0.8 f0 to 1.2 f0
    a4     : stroke-plane tilt, from -tilt_max to +tilt_max
    a5     : differential tilt, from -yaw_tilt_max to +yaw_tilt_max
    """
    a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
    return WingCommand(
        amp_left=0.5 * (a[0] + 1.0) * p.amp_max,
        amp_right=0.5 * (a[1] + 1.0) * p.amp_max,
        freq=p.f0 * (1.0 + p.freq_span * a[2]),
        tilt=a[3] * p.tilt_max,
        yaw_tilt=a[4] * p.yaw_tilt_max,
    )


def hover_action(p: BirdParams, mass_factor: float = 1.0) -> np.ndarray:
    """Action that holds a level, still bird in place: equal wings at f0, no tilt.

    Useful as a baseline and as a starting point for the classical controllers.
    It does NOT keep the bird in place after a disturbance: hovering is unstable.
    """
    a_amp = 2.0 * p.hover_amplitude(mass_factor) / p.amp_max - 1.0
    return np.array([a_amp, a_amp, 0.0, 0.0, 0.0])


def wing_speeds(cmd: WingCommand, p: BirdParams) -> tuple[float, float]:
    """Speed of each wing through the air at mid-wing: U = 2π f A r_wing.

    If the stroke angle is A sin(2π f t), this is its peak speed. The real
    average force depends on the mean of U² over a beat, which is half the
    peak value; that factor 1/2 is absorbed into the effective lift coefficient.
    """
    k = 2.0 * pi * cmd.freq * p.r_wing
    return k * cmd.amp_left, k * cmd.amp_right


def wing_thrusts(cmd: WingCommand, p: BirdParams, damage: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float]:
    """Force of each wing, F = ½ ρ C_L (S/2) U² (1 - D).

    Standard lift formula with half the wing area per wing. D is the damage of
    that wing, from 0 (intact) to 1 (useless): a damaged wing simply loses that
    fraction of its force.
    """
    u_l, u_r = wing_speeds(cmd, p)
    q = 0.5 * p.rho * p.c_lift * (p.wing_area / 2.0)
    return q * u_l * u_l * (1.0 - damage[0]), q * u_r * u_r * (1.0 - damage[1])


def control_torques(cmd: WingCommand, f_left: float, f_right: float, p: BirdParams) -> np.ndarray:
    """Steering torques in the body frame, as [roll, yaw, pitch] (N·m).

    roll  : the two wings push at ±r_wing on either side of the body, so a
            stronger left wing lifts the left side (right wing goes down);
    yaw   : opposite stroke-plane tilts turn the bird; positive turns left;
    pitch : the stroke-plane tilt moves the total force ahead of or behind the
            centre of mass; positive raises the nose.
    """
    thrust = f_left + f_right
    tau_roll = (f_left - f_right) * p.r_wing
    tau_yaw = p.k_yaw * thrust * p.r_wing * sin(cmd.yaw_tilt)
    tau_pitch = thrust * p.pitch_lever * sin(cmd.tilt)
    return np.array([tau_roll, tau_yaw, tau_pitch])


def instability_torques(v_body: np.ndarray, p: BirdParams) -> np.ndarray:
    """Speed-dependent torques that make hovering unstable, as [roll, yaw, pitch].

    Moving forward (+x_b) raises the nose, which tilts the thrust backwards;
    moving right (+z_b) rolls the bird left, which tilts the thrust to the
    left. Each reaction opposes the motion but arrives late because the body
    takes time to rotate, so the bird overshoots and the swings grow.
    """
    return np.array([-p.k_u_roll * v_body[2], 0.0, p.k_u_pitch * v_body[0]])


def drag_coefficients(v_body: np.ndarray, p: BirdParams) -> np.ndarray:
    """Drag coefficient along each body axis: [forward/backward, vertical, sideways]."""
    c_x = p.c_fwd if v_body[0] >= 0.0 else p.c_back
    return np.array([c_x, p.c_vert, p.c_side])


def drag_force_body(v_body: np.ndarray, p: BirdParams) -> np.ndarray:
    """Linear drag in the body frame: each component of the velocity is opposed by its own coefficient."""
    return -drag_coefficients(v_body, p) * v_body


def aero_power(cmd: WingCommand, f_left: float, f_right: float, p: BirdParams) -> float:
    """Mechanical power spent by the wings, Σ force × wing speed (W). Used to measure energy."""
    u_l, u_r = wing_speeds(cmd, p)
    return f_left * u_l + f_right * u_r


@dataclass(frozen=True)
class Wrench:
    """Everything the wings, air and gravity do to the bird during one physics step."""
    force_world: np.ndarray    # N, total force in the world frame (gravity included)
    torque_body: np.ndarray    # N·m, total torque in the body frame [roll, yaw, pitch]
    thrust: float              # N, total wing force
    f_left: float              # N
    f_right: float             # N
    power: float               # W


def compute_wrench(
    rot: np.ndarray,
    v_world: np.ndarray,
    omega_body: np.ndarray,
    cmd: WingCommand,
    p: BirdParams,
    damping: np.ndarray,
    damage: tuple[float, float],
    wind_force: np.ndarray,
    mass: float,
) -> Wrench:
    """Add up all forces (world frame) and torques (body frame) acting on the bird.

    rot is the body → world rotation matrix, so rot.T turns a world vector into
    body coordinates and rot[:, 1] is the direction in which the wings push.
    """
    f_l, f_r = wing_thrusts(cmd, p, damage)
    thrust = f_l + f_r
    v_body = rot.T @ v_world

    f_thrust = thrust * rot[:, 1]                        # wings push out of the bird's back
    f_drag = rot @ drag_force_body(v_body, p)            # drag is defined in the body frame
    f_gravity = np.array([0.0, -mass * p.g, 0.0])
    force = f_thrust + f_drag + f_gravity + wind_force

    torque = (control_torques(cmd, f_l, f_r, p)
              + instability_torques(v_body, p)
              - damping * omega_body)                    # damping opposes any rotation
    return Wrench(force, torque, thrust, f_l, f_r, aero_power(cmd, f_l, f_r, p))
