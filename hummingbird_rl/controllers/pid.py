"""Cascaded PID controller with a potential field for obstacles: the classical baseline.

This is the standard structure of quadrotor autopilots, applied to our bird.
Three loops, each one slower than the one inside it:

    position error  --(P, saturated)-->  desired velocity
    velocity error  --(P + I)-------->  desired acceleration  (+ obstacle repulsion)
    desired accel.  --(geometry)----->  desired thrust, roll and pitch
    attitude error  --(PID)---------->  torques  --(allocation.py)-->  5 actions

The geometry step is the key idea: a hovering bird can only push along its
back, so to accelerate forward it must tilt nose down, and to accelerate
right it must roll right. The tilt is limited to 25 degrees.

Yaw is not controlled, only damped (desired yaw rate = 0): the controller
flies in any direction by tilting, so a target behind is reached by backing
up. Choosing to turn instead is left to the RL agents, and comparing the two
is one of our experiments.

The two integral terms are what let the controller cope with conditions it
does not know about. In the velocity loop, the integral cancels a steady
wind or a heavier bird; it only runs within 15 cm of the target, so that it
does not wind up during long flights. In the attitude loop, it cancels a
steady torque, mainly the one from a damaged wing: roll inertia is so small
that without it a 40% weaker wing would leave the bird tilted by more than a
radian before the PD reacted enough.

Obstacles: the potential field of potential_field.py adds a push away from
the obstacles that are too close to the desired acceleration.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import atan2, cos, radians

import numpy as np

from hummingbird_rl.controllers.allocation import allocate
from hummingbird_rl.controllers.base import Controller, ObsView
from hummingbird_rl.controllers.potential_field import FieldGains, repulsion
from hummingbird_rl.sim.params import DEFAULT_PARAMS, BirdParams


@dataclass(frozen=True)
class PIDGains:
    k_pos: float = 5.0          # 1/s, position error -> desired velocity
    v_max: float = 0.6          # m/s, cap on the desired velocity
    k_vel: float = 10.0         # 1/s, velocity error -> desired acceleration
    k_int: float = 6.0          # 1/s^2, integral of the velocity-loop error
    int_radius: float = 0.15    # m, that integral only runs this close to the target
    a_int_max: float = 3.5      # m/s^2, anti-windup limit (covers a 30% wind or a 25% heavier bird)
    tilt_max: float = radians(25.0)
    w_att: tuple[float, float] = (30.0, 20.0)   # rad/s, natural frequency of the (roll, pitch) loops
    zeta_att: float = 0.8       # damping ratio of the attitude loops
    c_att_int: float = 0.25     # integral gain as a fraction of w^3 (stable below 2 zeta = 1.6)
    tau_int_max: tuple[float, float] = (3e-4, 3e-5)   # N·m, anti-windup for (roll, pitch)
    k_yaw_rate: float = 8.0     # 1/s, yaw-rate damping
    field: FieldGains = FieldGains()


class PIDController(Controller):
    name = "pid"

    def __init__(self, params: BirdParams | None = None, gains: PIDGains | None = None, **changes):
        self.p = params or DEFAULT_PARAMS
        self.g = replace(gains or PIDGains(), **changes)
        self.dt = self.p.dt * self.p.action_repeat
        self.reset()

    def reset(self) -> None:
        self.a_int = np.zeros(3)          # m/s^2, velocity-loop integral
        self.tau_int = np.zeros(2)        # N·m, attitude-loop integral (roll, pitch)

    def act(self, obs: np.ndarray) -> np.ndarray:
        o = ObsView.parse(obs)
        g, p = self.g, self.p
        error = o.targets[0] if o.targets else np.zeros(3)     # target relative to the bird

        # Position loop -> desired velocity, capped so that far targets are approached at v_max.
        v_des = g.k_pos * error
        speed = np.linalg.norm(v_des)
        if speed > g.v_max:
            v_des *= g.v_max / speed

        # Velocity loop -> desired acceleration. The integral sees the velocity error;
        # near the target that is mostly k_pos * position error, so it removes steady offsets.
        v_err = v_des - o.vel
        if np.linalg.norm(error) < g.int_radius:
            self.a_int = np.clip(self.a_int + g.k_int * v_err * self.dt, -g.a_int_max, g.a_int_max)
        a_des = g.k_vel * v_err + self.a_int + repulsion(o, p.r_bird, g.field)

        # Geometry: tilt the thrust towards the desired acceleration.
        lift = max(p.g + a_des[1], 0.2 * p.g)                  # never ask for less than 20% of the weight
        pitch_des = float(np.clip(-atan2(a_des[0], lift), -g.tilt_max, g.tilt_max))
        roll_des = float(np.clip(atan2(a_des[2], lift), -g.tilt_max, g.tilt_max))
        thrust = p.mass * lift / (cos(pitch_des) * cos(roll_des))

        # Attitude loop: a PD on each angle written as a desired angular acceleration
        # (second-order response with natural frequency w_att), times the inertia,
        # plus the integral term in torque units.
        w = np.array(g.w_att)
        kp, kd, ki = w ** 2, 2.0 * g.zeta_att * w, g.c_att_int * w ** 3
        att_err = np.array([roll_des - o.roll, pitch_des - o.pitch])
        inertia_rp = p.inertia[[0, 2]]
        self.tau_int = np.clip(self.tau_int + ki * inertia_rp * att_err * self.dt,
                               -np.array(g.tau_int_max), np.array(g.tau_int_max))
        alpha = np.array([
            kp[0] * att_err[0] - kd[0] * o.rates[0],
            -g.k_yaw_rate * o.rates[1],
            kp[1] * att_err[1] - kd[1] * o.rates[2],
        ])
        torque = p.inertia * alpha + np.array([self.tau_int[0], 0.0, self.tau_int[1]])
        return allocate(thrust, torque, p)
