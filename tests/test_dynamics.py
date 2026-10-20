"""Tests for dynamics.py: known analytic cases, and the behaviour we tuned the bird to have."""

from math import exp, log, radians

import numpy as np
import pytest

from hummingbird_rl.sim import BirdDynamics, BirdState, check_limits, hover_action

DT = 1.0 / 240.0


def run(dyn, state, action, seconds, dt=DT, record=None):
    out = []
    for _ in range(int(round(seconds / dt))):
        state, _ = dyn.step(state, action, dt=dt)
        if record is not None:
            out.append(record(state))
    return state, np.array(out)


def growth_rate(signal, dt):
    """Exponential growth rate (1/s) from successive positive peaks of an oscillation."""
    s = signal
    peaks = [i for i in range(1, len(s) - 1) if s[i] > s[i - 1] and s[i] >= s[i + 1] and s[i] > 0]
    assert len(peaks) >= 3, "no clear oscillation"
    t = np.array(peaks) * dt
    a = np.log(signal[peaks])
    return np.polyfit(t, a, 1)[0]


def _free_fall(params, dt, t=0.5):
    dyn = BirdDynamics(params)
    zero = np.array([-1.0, -1.0, 0.0, 0.0, 0.0])     # wings stopped: no thrust
    state, _ = run(dyn, BirdState.from_euler((0, 1.0, 0)), zero, t, dt=dt)
    m, g, c = params.mass, params.g, params.c_vert
    y_exact = 1.0 - (m * g / c) * t + (m * m * g / (c * c)) * (1 - exp(-c * t / m))
    v_exact = -(m * g / c) * (1 - exp(-c * t / m))
    return state, y_exact, v_exact


def test_free_fall_with_linear_drag_matches_analytic(params):
    """Semi-implicit Euler is first order: position error ≈ ½ g dt t (≈ 1 cm after 0.5 s at 240 Hz)."""
    state, y_exact, v_exact = _free_fall(params, 1 / 240)
    assert abs(state.pos[1] - y_exact) < 0.5 * params.g * (1 / 240) * 0.5 * 1.1
    assert state.vel[1] == pytest.approx(v_exact, rel=5e-3)


def test_first_order_convergence(params):
    """Dividing the time step by 4 divides the position error by about 4."""
    s1, y_exact, _ = _free_fall(params, 1 / 240)
    s4, _, _ = _free_fall(params, 1 / 960)
    e1, e4 = abs(s1.pos[1] - y_exact), abs(s4.pos[1] - y_exact)
    assert 3.0 < e1 / e4 < 5.0


@pytest.mark.parametrize("unstable", [False, True])
def test_hover_equilibrium_holds(params, stable_params, unstable):
    p = params if unstable else stable_params
    dyn = BirdDynamics(p)
    start = BirdState.from_euler((0, 0.5, 0))
    end, _ = run(dyn, start.copy(), hover_action(p), 5.0)
    assert np.linalg.norm(end.pos - start.pos) < 1e-3


@pytest.mark.parametrize("axis, angle_index", [(0, 1), (2, 2)])   # forward push → pitch, sideways push → roll
def test_hover_instability_doubling_time(params, axis, angle_index):
    """Calibration target: an uncontrolled push grows with a doubling time of 0.5–1 s."""
    dyn = BirdDynamics(params)
    vel = [0.0, 0.0, 0.0]
    vel[axis] = 0.05
    state = BirdState.from_euler((0, 0.5, 0), vel=vel)
    _, angle = run(dyn, state, hover_action(params), 3.0, record=lambda s: s.euler[angle_index])
    sig = np.abs(angle)  # use the envelope of |angle|
    rate = growth_rate(sig, DT)
    doubling = log(2) / rate
    assert 0.5 <= doubling <= 1.0, f"doubling time {doubling:.2f} s"


def test_without_instability_a_push_dies_out(stable_params):
    dyn = BirdDynamics(stable_params)
    state = BirdState.from_euler((0, 0.5, 0), vel=(0.05, 0, 0))
    end, _ = run(dyn, state, hover_action(stable_params), 4.0)
    assert np.linalg.norm(end.vel) < 0.01


def test_angular_damping_time_constant(params):
    dyn = BirdDynamics(params)
    state = BirdState.from_euler((0, 0.5, 0), omega=(0.0, 1.0, 0.0))   # yaw rate only
    end, _ = run(dyn, state, hover_action(params), params.tau_omega)
    assert end.omega[1] == pytest.approx(exp(-1), rel=0.03)


def test_mass_factor(params):
    heavy = BirdDynamics(params, mass_factor=1.2)
    start = BirdState.from_euler((0, 0.5, 0))
    sinks, _ = run(heavy, start.copy(), hover_action(params), 1.0)
    holds, _ = run(heavy, start.copy(), hover_action(params, 1.2), 1.0)
    assert sinks.pos[1] < 0.5 - 0.01
    assert abs(holds.pos[1] - 0.5) < 1e-3


def test_wing_damage_rolls_towards_damaged_wing(params):
    dyn = BirdDynamics(params, damage=(0.4, 0.0))
    # Uncontrolled, the bird spins quickly: look at the first 30 ms only.
    end, _ = run(dyn, BirdState.from_euler((0, 0.5, 0)), hover_action(params), 0.03)
    assert -radians(60) < end.euler[2] < -radians(2)      # negative roll = left wing down


def test_determinism(params):
    dyn = BirdDynamics(params)
    rng = np.random.default_rng(3)
    actions = rng.uniform(-1, 1, size=(200, 5))
    def go():
        s = BirdState.from_euler((0, 0.5, 0), yaw=0.3)
        for a in actions:
            s, _ = dyn.step(s, a)
        return s
    a, b = go(), go()
    assert np.array_equal(a.pos, b.pos) and np.array_equal(a.quat, b.quat)


def test_limits():
    assert check_limits(BirdState.from_euler((0, 0.5, 0))) is None
    assert check_limits(BirdState.from_euler((0, 0.01, 0))) == "crash"
    assert check_limits(BirdState.from_euler((1.6, 0.5, 0))) == "out_of_bounds"
    assert check_limits(BirdState.from_euler((0, 1.3, 0))) == "out_of_bounds"
    assert check_limits(BirdState.from_euler((0, 0.5, 0), roll=radians(85))) == "loss_of_control"
    assert check_limits(BirdState.from_euler((0, 0.5, 0), pitch=radians(-85))) == "loss_of_control"
