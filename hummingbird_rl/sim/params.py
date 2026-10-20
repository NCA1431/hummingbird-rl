"""Physical parameters of the simulated hummingbird.

This file is the single place where every number describing the bird lives:
its size and mass, its wings, the air around it, how it steers, how it
resists rotation, how the air slows it down, and the simulation time step.

Two kinds of values appear below:
  - primary parameters, the numbers we choose (mass, wingspan, frequency...);
  - derived quantities, computed from the primary ones (weight, inertia, the
    wing amplitude needed to hover...). They are computed rather than typed
    in, so changing one primary value, say the mass, keeps everything else
    consistent.

Units are SI throughout (kilograms, metres, seconds, newtons) and angles are
in radians.

Body frame: three axes attached to the bird, which move and turn with it
(drawing: docs/figures/body_axes.png).
  - x_b points from the tail towards the beak (forward);
  - y_b points out of the bird's back (up when the bird is level); this is
    the direction in which the wings push;
  - z_b points along the right wing.

        side view                             top view

          y_b  out of the back                  x_b  beak
           ^   (where the wings push)            ^
           |                                     |
    tail ==O==> x_b  beak             left wing --O--> z_b  right wing

    z_b points towards you                y_b points towards you

Rotations about these axes:
    roll  = about x_b, tilting the wings    (positive: right wing goes down)
    yaw   = about y_b, turning the nose     (positive: nose turns left)
    pitch = about z_b, nose up or down      (positive: nose goes up)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import pi, radians, sqrt

import numpy as np


@dataclass(frozen=True)  # read-only: a simulation cannot change its own physics by accident
class BirdParams:
    # --- body -----------------------------------------------------------------
    mass: float = 0.004                 # kg, wings included (a ruby-throated hummingbird is 3-4 g)
    span: float = 0.10                  # m, wingtip to wingtip
    wing_area: float = 5.0e-4           # m², both wings; only its product with c_lift matters
    body_semi_axes: tuple[float, float, float] = (0.04, 0.01, 0.01)  # m, body as an 8 x 2 x 2 cm ellipsoid
    wing_mass: float = 1.0e-4           # kg, one wing (already counted in `mass`)
    wing_root: float = 0.005            # m, from the body's centre line to where the wing starts

    # --- air ------------------------------------------------------------------
    rho: float = 1.225                  # kg/m³, sea level
    g: float = 9.81                     # m/s²

    # --- wings and steering ---------------------------------------------------
    # The controller sends 5 actions in [-1, 1]; these values turn them into
    # physical wing settings.
    f0: float = 40.0                    # Hz, base wingbeat frequency
    freq_span: float = 0.2              # f = f0 * (1 + freq_span * a3), so 32-48 Hz. Narrow on
                                        # purpose: hummingbirds mostly change the size of the stroke.
    amp_max: float = radians(80.0)      # largest half-stroke amplitude (a full sweep of 160°)

    # Effective lift coefficient. Real wings reach about 1-2; ours is higher
    # because the whole wing is reduced to a single point at mid-wing. Chosen
    # so that hovering at f0 takes 80% of amp_max, which leaves a 20% margin.
    c_lift: float = 2.60

    # Pitch: tilting the stroke plane moves the wing force slightly ahead of or
    # behind the centre of mass. Torque = thrust * pitch_lever * sin(tilt).
    tilt_max: float = radians(20.0)
    pitch_lever: float = 0.005          # m

    # Yaw: tilting the two stroke planes in opposite directions.
    # Torque = k_yaw * thrust * r_wing * sin(yaw_tilt). k_yaw is a starting guess.
    yaw_tilt_max: float = radians(20.0)
    k_yaw: float = 0.5

    # --- rotation -------------------------------------------------------------
    # Flapping wings resist body rotation: left alone, a spin decays as
    # exp(-t / tau_omega).
    tau_omega: float = 0.4              # s

    # Hovering flyers are unstable. Drifting forward pitches the nose up, which
    # brakes the drift but overshoots, and each swing is larger than the last.
    # We model this with a torque proportional to speed:
    #   pitch torque = +k_u_pitch * forward speed,  roll torque = -k_u_roll * sideways speed.
    # These values make an uncontrolled swing double about every 0.75 s. Roll
    # gets a smaller value because the body is about four times easier to roll
    # than to pitch. Set both to zero for a stable bird.
    k_u_pitch: float = 1.91e-5          # N·m per m/s
    k_u_roll: float = 1.15e-5           # N·m per m/s

    # --- drag -----------------------------------------------------------------
    # Linear drag in the body frame (force = coefficient * speed). Head-first is
    # cheapest, which gives the agent a reason to turn rather than always back
    # up. Top speeds at a 30° tilt, where the horizontal push is W tan 30° ≈ 0.023 N:
    c_fwd: float = 0.0075               # N·s/m, head-first -> about 3 m/s
    c_back: float = 0.015               # N·s/m, tail-first -> about 1.5 m/s
    c_side: float = 0.023               # N·s/m, sideways   -> about 1 m/s
    c_vert: float = 0.020               # N·s/m, up or down

    # --- simulation -----------------------------------------------------------
    r_bird: float = 0.02                # m, sphere used for collision checks
    dt: float = 1.0 / 240.0             # s, physics step
    action_repeat: int = 4              # physics steps per decision: the controller acts at 60 Hz

    # --- derived quantities ---------------------------------------------------
    @property
    def r_wing(self) -> float:
        """Mid-wing radius used to compute the wing speed (a quarter of the span)."""
        return self.span / 4.0

    @property
    def weight(self) -> float:
        """Weight in newtons (about 0.039 N)."""
        return self.mass * self.g

    @property
    def inertia(self) -> np.ndarray:
        """Moments of inertia [roll, yaw, pitch] in kg·m².

        Body: a solid ellipsoid, I_x = m (b² + c²) / 5 and likewise for y and z.
        Wings: thin rods along z_b from z0 to z0 + L; each adds
        m_w (z0² + z0 L + L²/3) to the roll and yaw axes. Pitching only turns
        the rods about their own length, which they barely resist.
        Result ≈ [3.4, 14.8, 12.9] × 10⁻⁷, so rolling is about 4x easier.
        """
        a, b, c = self.body_semi_axes
        m_body = self.mass - 2.0 * self.wing_mass
        length = self.span / 2.0 - self.wing_root
        z0 = self.wing_root
        i_wing = self.wing_mass * (z0 * z0 + z0 * length + length * length / 3.0)
        ixx = m_body * (b * b + c * c) / 5.0 + 2.0 * i_wing     # roll
        iyy = m_body * (a * a + c * c) / 5.0 + 2.0 * i_wing     # yaw
        izz = m_body * (a * a + b * b) / 5.0                    # pitch
        return np.array([ixx, iyy, izz])

    @property
    def damping(self) -> np.ndarray:
        """Damping coefficients D = I / tau_omega, so every axis decays with the same time constant."""
        return self.inertia / self.tau_omega

    @property
    def thrust_gain(self) -> float:
        """Constant k such that one undamaged wing pushes with k * (f * A)².

        The wing speed at mid-wing is U = 2π f A r_wing and the force is
        ½ ρ C_L (S/2) U²; grouping the constants gives k.
        """
        return 0.5 * self.rho * self.c_lift * (self.wing_area / 2.0) * (2.0 * pi * self.r_wing) ** 2

    def hover_amplitude(self, mass_factor: float = 1.0) -> float:
        """Amplitude at which both wings at f0 carry the weight: 2 k (f0 A)² = W.

        About 64°, i.e. 80% of amp_max. mass_factor asks the same question
        for a heavier or lighter bird (1.2 = 20% heavier).
        """
        w = mass_factor * self.weight
        return sqrt(w / (2.0 * self.thrust_gain)) / self.f0

    def max_thrust(self) -> float:
        """Both wings at amp_max and 1.2 f0: (1.2 / 0.8)² ≈ 2.25 times the weight."""
        f = self.f0 * (1.0 + self.freq_span)
        return 2.0 * self.thrust_gain * (f * self.amp_max) ** 2

    def hover_power(self) -> float:
        """Aerodynamic power while hovering, force × wing speed summed over both wings (≈ 0.27 W).

        Used to express energy use relative to hovering.
        """
        a_h = self.hover_amplitude()
        u = 2.0 * pi * self.f0 * a_h * self.r_wing
        return 2.0 * (self.weight / 2.0) * u

    def with_changes(self, **kwargs) -> "BirdParams":
        """Copy with some values changed, e.g. with_changes(k_u_pitch=0.0, k_u_roll=0.0)."""
        return replace(self, **kwargs)


DEFAULT_PARAMS = BirdParams()
