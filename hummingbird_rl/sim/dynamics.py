"""Rigid-body dynamics: moving the bird forward by one physics step.

The bird is a single rigid body. Each step we
  1. turn the action into wing settings and compute every force and torque
     (aero.py),
  2. update the velocities: Newton's law m dv/dt = F for the translation and
     Euler's equation I dω/dt = τ - ω × (I ω) for the rotation,
  3. update position and orientation using the NEW velocities.

Step 3 is the "semi-implicit" (or symplectic) Euler scheme. It costs the same
as the plain Euler method but behaves much better for oscillating systems,
which matters here because hovering is an oscillation. It is still first
order: halving dt halves the error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import radians

import numpy as np

from hummingbird_rl.sim import aero
from hummingbird_rl.sim.params import BirdParams
from hummingbird_rl.sim.rotations import IDENTITY, euler_from_quat, integrate_quat, quat_from_euler, quat_to_matrix


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product of two 3-vectors. np.cross handles arrays of any shape and,
    for a single pair, was a fifth of the whole simulation time."""
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


@dataclass
class BirdState:
    """Everything needed to describe the bird at one instant."""
    pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.5, 0.0]))   # m, world frame
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))                  # m/s, world frame
    quat: np.ndarray = field(default_factory=lambda: IDENTITY.copy())             # orientation (body → world)
    omega: np.ndarray = field(default_factory=lambda: np.zeros(3))                # rad/s, body frame [roll, yaw, pitch]
    time: float = 0.0                                                             # s since the start of the episode
    phase: float = 0.0                                                            # rad, wingbeat phase (drawing only)

    def copy(self) -> "BirdState":
        return BirdState(self.pos.copy(), self.vel.copy(), self.quat.copy(), self.omega.copy(), self.time, self.phase)

    @property
    def euler(self) -> tuple[float, float, float]:
        """(yaw, pitch, roll) in radians."""
        return euler_from_quat(self.quat)

    @staticmethod
    def from_euler(pos, vel=(0.0, 0.0, 0.0), yaw=0.0, pitch=0.0, roll=0.0, omega=(0.0, 0.0, 0.0)) -> "BirdState":
        """Convenience constructor from angles rather than a quaternion."""
        return BirdState(np.array(pos, dtype=float), np.array(vel, dtype=float),
                         quat_from_euler(yaw, pitch, roll), np.array(omega, dtype=float))


@dataclass
class StepInfo:
    """By-products of a step, used for rewards, metrics and the air visualisation."""
    thrust: float           # N
    f_left: float           # N
    f_right: float          # N
    power: float            # W
    wind_force: np.ndarray  # N


class BirdDynamics:
    """Simulates one bird under given physical conditions.

    The conditions are fixed for an episode: mass_factor scales the mass and
    the inertia (1.2 = a 20% heavier bird), and damage = (left, right) is the
    fraction of force each wing has lost.
    """

    def __init__(self, params: BirdParams | None = None, mass_factor: float = 1.0,
                 damage: tuple[float, float] = (0.0, 0.0)):
        self.p = params or BirdParams()
        self.set_conditions(mass_factor, damage)

    def set_conditions(self, mass_factor: float = 1.0, damage: tuple[float, float] = (0.0, 0.0)) -> None:
        self.mass_factor = float(mass_factor)
        self.mass = self.p.mass * self.mass_factor
        # A heavier bird is also harder to rotate; we scale the inertia with the mass.
        self.inertia = self.p.inertia * self.mass_factor
        self.damping = self.inertia / self.p.tau_omega
        self.damage = (float(damage[0]), float(damage[1]))

    def step(self, state: BirdState, action: np.ndarray, wind_force: np.ndarray | None = None,
             dt: float | None = None) -> tuple[BirdState, StepInfo]:
        """Return the state after one step of length dt (default: params.dt).

        The action is applied as given; noise, if any, is added by the environment.
        """
        p = self.p
        dt = p.dt if dt is None else dt
        wind = np.zeros(3) if wind_force is None else np.asarray(wind_force, dtype=float)
        cmd = aero.decode_action(action, p)
        rot = quat_to_matrix(state.quat)

        w = aero.compute_wrench(rot, state.vel, state.omega, cmd, p, self.damping, self.damage, wind, self.mass)

        # Euler's equation for a rigid body in its own frame. The ω × (I ω) term
        # couples the axes when the bird spins about several of them at once.
        i_omega = self.inertia * state.omega
        omega_dot = (w.torque_body - _cross(state.omega, i_omega)) / self.inertia
        acc = w.force_world / self.mass

        new = state.copy()
        new.omega = state.omega + omega_dot * dt          # velocities first...
        new.vel = state.vel + acc * dt
        new.pos = state.pos + new.vel * dt                # ...then positions, with the new velocities
        new.quat = integrate_quat(state.quat, new.omega, dt)
        new.time = state.time + dt
        new.phase = (state.phase + 2.0 * np.pi * cmd.freq * dt) % (2.0 * np.pi)
        return new, StepInfo(w.thrust, w.f_left, w.f_right, w.power, wind)


# --- limits of the arena ---------------------------------------------------------
MAX_TILT = radians(80.0)    # beyond this roll or pitch we consider the bird out of control
GROUND_HEIGHT = 0.02        # m, below this the bird has crashed
ARENA_RADIUS = 1.5          # m, horizontal distance from the arena centre
CEILING = 1.2               # m


def check_limits(state: BirdState) -> str | None:
    """Return why the episode must stop ("crash", "out_of_bounds", "loss_of_control"), or None.

    Obstacle collisions are checked separately, by the tasks that have obstacles.
    """
    if state.pos[1] < GROUND_HEIGHT:
        return "crash"
    if np.hypot(state.pos[0], state.pos[2]) > ARENA_RADIUS or state.pos[1] > CEILING:
        return "out_of_bounds"
    _, pitch, roll = state.euler
    if abs(pitch) > MAX_TILT or abs(roll) > MAX_TILT:
        return "loss_of_control"
    return None
