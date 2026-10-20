"""Tests for aero.py: forces, torque signs, drag and power."""

from math import radians, tan

import numpy as np
import pytest

from hummingbird_rl.sim import aero
from hummingbird_rl.sim.aero import decode_action, hover_action


def _thrust(action, p, damage=(0.0, 0.0)):
    return sum(aero.wing_thrusts(decode_action(action, p), p, damage))


def test_hover_action_gives_thrust_equal_to_weight(params):
    assert _thrust(hover_action(params), params) == pytest.approx(params.weight, rel=1e-9)
    assert _thrust(hover_action(params, 1.2), params) == pytest.approx(1.2 * params.weight, rel=1e-9)


def test_action_ranges(params):
    lo, hi = decode_action(-np.ones(5), params), decode_action(np.ones(5), params)
    assert lo.amp_left == 0 and hi.amp_left == pytest.approx(params.amp_max)
    assert lo.freq == pytest.approx(0.8 * params.f0) and hi.freq == pytest.approx(1.2 * params.f0)
    assert hi.tilt == pytest.approx(radians(20)) and lo.yaw_tilt == pytest.approx(-radians(20))
    assert decode_action(5 * np.ones(5), params) == hi  # out-of-range actions are clipped


def test_damage_reduces_thrust_proportionally(params):
    cmd = decode_action(hover_action(params), params)
    fl0, fr0 = aero.wing_thrusts(cmd, params)
    fl, fr = aero.wing_thrusts(cmd, params, damage=(0.4, 0.0))
    assert fl == pytest.approx(0.6 * fl0) and fr == pytest.approx(fr0)


def test_torque_signs(params):
    a = hover_action(params)
    # stronger left wing → positive roll torque (right wing down)
    left = a.copy()
    left[0] += 0.1
    cmd = decode_action(left, params)
    fl, fr = aero.wing_thrusts(cmd, params)
    assert aero.control_torques(cmd, fl, fr, params)[0] > 0
    # positive a4 → nose-up pitch torque; positive a5 → left yaw torque
    for idx, axis in [(3, 2), (4, 1)]:
        b = a.copy()
        b[idx] = 0.5
        cmd = decode_action(b, params)
        fl, fr = aero.wing_thrusts(cmd, params)
        tau = aero.control_torques(cmd, fl, fr, params)
        assert tau[axis] > 0 and np.count_nonzero(np.abs(tau) > 1e-15) == 1


def test_zero_amplitude_gives_no_thrust_and_no_power(params):
    cmd = decode_action(np.array([-1.0, -1.0, 0.0, 0.0, 0.0]), params)
    fl, fr = aero.wing_thrusts(cmd, params)
    assert fl == 0 and fr == 0 and aero.aero_power(cmd, fl, fr, params) == 0


def test_hover_power_is_positive_and_consistent(params):
    cmd = decode_action(hover_action(params), params)
    fl, fr = aero.wing_thrusts(cmd, params)
    assert aero.aero_power(cmd, fl, fr, params) == pytest.approx(params.hover_power(), rel=1e-9)
    assert 0.05 < params.hover_power() < 1.0  # order of magnitude of a hovering hummingbird (W)


def test_instability_torques_tilt_thrust_against_motion(params):
    tau = aero.instability_torques(np.array([0.2, 0.0, 0.0]), params)   # moving forward
    assert tau[2] > 0 and tau[0] == 0                                       # nose up
    tau = aero.instability_torques(np.array([0.0, 0.0, 0.2]), params)   # moving right
    assert tau[0] < 0 and tau[2] == 0                                       # roll left
    assert np.all(aero.instability_torques(np.zeros(3), params) == 0)


def test_drag_opposes_motion_and_depends_on_direction(params):
    for v in [np.array([1.0, 0, 0]), np.array([-1.0, 0, 0]), np.array([0, 0, 1.0]), np.array([0, -1.0, 0])]:
        assert np.dot(aero.drag_force_body(v, params), v) < 0
    fwd = -aero.drag_force_body(np.array([1.0, 0, 0]), params)[0]
    back = aero.drag_force_body(np.array([-1.0, 0, 0]), params)[0]
    side = -aero.drag_force_body(np.array([0, 0, 1.0]), params)[2]
    assert fwd < back < side


@pytest.mark.parametrize("axis, speed_sign, target", [(0, 1, 3.0), (0, -1, 1.5), (2, 1, 1.0)])
def test_terminal_speeds_at_30_degrees(params, axis, speed_sign, target):
    """Drag balances the horizontal thrust of a 30° tilt (W tan 30°) at the target speed."""
    v = np.zeros(3)
    v[axis] = speed_sign * target
    drag = abs(aero.drag_force_body(v, params)[axis])
    assert drag == pytest.approx(params.weight * tan(radians(30)), rel=0.05)
