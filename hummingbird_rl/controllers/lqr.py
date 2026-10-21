"""LQR: the optimal linear controller around hover, our second classical baseline.

Near hover the bird behaves almost like a linear system, x_{k+1} ≈ A x_k + B u_k,
where x is the deviation from hovering still at the target and u the deviation
from the hover action. For a linear system, the feedback that minimises

    J = Σ_k  x_kᵀ Q x_k + u_kᵀ R u_k

is a constant gain, u = −K x (the Linear-Quadratic Regulator). K comes from the
discrete algebraic Riccati equation

    P = Aᵀ P A − Aᵀ P B (R + Bᵀ P B)⁻¹ Bᵀ P A + Q,        K = (R + Bᵀ P B)⁻¹ Bᵀ P A,

and xᵀ P x is the exact cost-to-go from x: the "value function" of this
problem. We will compare it with the value network learnt by PPO.

We do not derive A and B by hand. We linearise the simulator itself, by
central finite differences on the 60 Hz map (four physics steps with the
action held), around hover with the nominal bird and no wind or noise. Like
the PID, the controller knows the nominal model and sees only the observation.

State (11 numbers, heading frame): position error (3), velocity (3), roll,
pitch, body rates (3). Yaw is not in the state, since the observation hides
it and no task depends on the heading; through the body rates the LQR only
damps the yaw rate, like the PID. We add the integral of the position error
(3 more states, the "LQI" variant), otherwise a steady wind leaves a constant
offset of a few centimetres, which is enough to fail the hover task.

Choices and their limits:
- Far targets: the position error is saturated at 15 cm, so the bird always
  believes it is at most 15 cm from its target and stays near the region
  where the linear model holds.
- The drag coefficient jumps between forward and backward flight; at zero
  speed the finite difference takes the average of the two.
- Weights: Bryson's rule (1 / acceptable deviation²) as a start, then tuned
  by hand on validation episodes, as for the PID.
- No fast disturbance estimate: unlike the PID, whose attitude integral
  absorbs the torque of a damaged wing, the LQR trusts its model. Integrating
  the attitude error as well conflicts with the position integral (under
  wind the bird must stay tilted), so we leave it out and accept that a
  badly damaged wing can defeat it.
- Obstacles: the potential field of potential_field.py moves the target away
  from close obstacles (a "carrot"), and the LQR tracks the moved target.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import solve_discrete_are

from hummingbird_rl.controllers.base import Controller, ObsView
from hummingbird_rl.controllers.potential_field import FieldGains, repulsion
from hummingbird_rl.sim import BirdDynamics, BirdState, hover_action
from hummingbird_rl.sim.params import DEFAULT_PARAMS, BirdParams
from hummingbird_rl.sim.rotations import world_to_heading
from hummingbird_rl.world.scene import START

N_STATE, N_ACTION = 11, 5


def lqr_state(state: BirdState, target: np.ndarray) -> np.ndarray:
    """The 11-number LQR state of a bird, relative to its target (heading frame)."""
    yaw, pitch, roll = state.euler
    return np.concatenate([world_to_heading(state.pos - target, yaw), world_to_heading(state.vel, yaw),
                           [roll, pitch], state.omega])


def linearise(p: BirdParams, eps: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
    """A (11×11) and B (11×5) of the 60 Hz map around hover, by central finite differences."""
    dyn = BirdDynamics(p)
    u0 = hover_action(p)

    def step(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        s = BirdState.from_euler(pos=START + x[0:3], vel=x[3:6], pitch=x[7], roll=x[6], omega=x[8:11])
        for _ in range(p.action_repeat):
            s, _ = dyn.step(s, u)
        return lqr_state(s, START)

    A = np.zeros((N_STATE, N_STATE))
    B = np.zeros((N_STATE, N_ACTION))
    for i in range(N_STATE):
        d = np.zeros(N_STATE)
        d[i] = eps
        A[:, i] = (step(d, u0) - step(-d, u0)) / (2 * eps)
    for j in range(N_ACTION):
        d = np.zeros(N_ACTION)
        d[j] = eps
        B[:, j] = (step(np.zeros(N_STATE), u0 + d) - step(np.zeros(N_STATE), u0 - d)) / (2 * eps)
    return A, B


@dataclass(frozen=True)
class LQRWeights:
    """Acceptable deviations (Bryson's rule: Q_ii = 1 / x_max², R_jj = 1 / u_max²), tuned by hand."""
    pos: float = 0.02           # m
    vel: float = 0.2            # m/s
    tilt: float = 0.3           # rad, roll and pitch
    rate: float = 3.0           # rad/s
    action: float = 0.5         # action units, the same for the 5 actions
    integral: float = 2000.0    # weight of the integrated position error (1/(m²·s²))
    saturation: float = 0.15    # m, cap on the position error
    integral_radius: float = 0.15   # m, the integral only runs this close to the target
    integral_max: float = 0.5   # m·s, anti-windup
    carrot_gain: float = 50.0   # 1/s², turns the field's push (m/s²) into a target shift (m)
    field: FieldGains = FieldGains()


class LQRController(Controller):
    name = "lqr"

    def __init__(self, params: BirdParams | None = None, weights: LQRWeights | None = None, **changes):
        self.p = params or DEFAULT_PARAMS
        self.w = replace(weights or LQRWeights(), **changes)
        self.dt = self.p.dt * self.p.action_repeat
        self.u0 = hover_action(self.p)
        self.A, self.B = linearise(self.p)

        # Augmented system with the integrated position error z: z_{k+1} = z_k + dt · position error.
        n = N_STATE + 3
        A = np.zeros((n, n))
        A[:N_STATE, :N_STATE] = self.A
        A[N_STATE:, :3] = self.dt * np.eye(3)
        A[N_STATE:, N_STATE:] = np.eye(3)
        B = np.vstack([self.B, np.zeros((3, N_ACTION))])
        w = self.w
        x_max = np.r_[[w.pos] * 3, [w.vel] * 3, [w.tilt] * 2, [w.rate] * 3]
        self.Q = np.diag(np.r_[1.0 / x_max ** 2, [w.integral] * 3])
        self.R = np.eye(N_ACTION) / w.action ** 2
        self.P = solve_discrete_are(A, B, self.Q, self.R)
        self.K = np.linalg.solve(self.R + B.T @ self.P @ B, B.T @ self.P @ A)
        self.closed_loop = A - B @ self.K
        self.reset()

    def reset(self) -> None:
        self.z = np.zeros(3)

    def act(self, obs: np.ndarray) -> np.ndarray:
        o = ObsView.parse(obs)
        w = self.w
        error = -o.targets[0] if o.targets else np.zeros(3)      # bird minus target
        if np.linalg.norm(error) < w.integral_radius:
            self.z = np.clip(self.z + self.dt * error, -w.integral_max, w.integral_max)
        # Carrot: the field's push moves the (already capped) target by at most 15 cm, so
        # that near an obstacle the repulsion competes on equal terms with the attraction.
        shift = repulsion(o, self.p.r_bird, w.field) / w.carrot_gain
        x_pos = _cap(error, w.saturation) - _cap(shift, w.saturation)
        x = np.r_[x_pos, o.vel, o.roll, o.pitch, o.rates, self.z]
        return np.clip(self.u0 - self.K @ x, -1.0, 1.0)

    def cost_to_go(self, x: np.ndarray) -> float:
        """xᵀ P x for an 11-number state, with no accumulated integral: the LQR's exact value (up to sign)."""
        P = self.P[:N_STATE, :N_STATE]
        return float(x @ P @ x)


def _cap(v: np.ndarray, length: float) -> np.ndarray:
    """v, shortened to `length` if it is longer."""
    n = np.linalg.norm(v)
    return v * (length / n) if n > length else v
