"""Tests for rotations.py: round trips, sign conventions and the heading frame."""

from math import radians

import numpy as np

from hummingbird_rl.sim.rotations import (
    euler_from_quat,
    heading_to_world,
    integrate_quat,
    quat_from_euler,
    quat_to_matrix,
    world_to_heading,
)


def test_matrix_is_orthonormal(rng):
    for _ in range(50):
        q = quat_from_euler(*rng.uniform(-1.4, 1.4, size=3))
        r = quat_to_matrix(q)
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(r), 1.0)


def test_euler_round_trip(rng):
    for _ in range(200):
        yaw = rng.uniform(-np.pi, np.pi)
        pitch = rng.uniform(-radians(85), radians(85))
        roll = rng.uniform(-np.pi * 0.95, np.pi * 0.95)
        y2, p2, r2 = euler_from_quat(quat_from_euler(yaw, pitch, roll))
        assert np.allclose([y2, p2, r2], [yaw, pitch, roll], atol=1e-9)


def test_sign_conventions():
    # positive pitch = nose up: the forward axis gains a +y component
    fwd = quat_to_matrix(quat_from_euler(0.0, 0.3, 0.0))[:, 0]
    assert fwd[1] > 0
    # positive roll = right wing down: the up axis tilts towards +z (the right)
    up = quat_to_matrix(quat_from_euler(0.0, 0.0, 0.3))[:, 1]
    assert up[2] > 0
    # positive yaw = nose turns left (towards −z)
    fwd = quat_to_matrix(quat_from_euler(0.3, 0.0, 0.0))[:, 0]
    assert fwd[2] < 0


def test_quaternion_stays_unit_after_long_integration(rng):
    q = quat_from_euler(0.1, 0.2, 0.3)
    omega = np.array([3.0, -2.0, 5.0])
    for _ in range(24000):  # 100 s at 240 Hz
        q = integrate_quat(q, omega, 1.0 / 240.0)
    assert abs(np.linalg.norm(q) - 1.0) < 1e-12


def test_constant_rate_rotation_matches_analytic():
    # rotating about the body yaw axis at 1 rad/s for 0.5 s gives yaw = 0.5
    q = quat_from_euler(0.0, 0.0, 0.0)
    for _ in range(120):
        q = integrate_quat(q, np.array([0.0, 1.0, 0.0]), 1.0 / 240.0)
    assert np.isclose(euler_from_quat(q)[0], 0.5, atol=1e-9)


def test_heading_frame(rng):
    for _ in range(50):
        yaw = rng.uniform(-np.pi, np.pi)
        pitch = rng.uniform(-1.0, 1.0)
        fwd = quat_to_matrix(quat_from_euler(yaw, pitch, 0.0))[:, 0]
        h = world_to_heading(fwd, yaw)
        assert abs(h[2]) < 1e-12 and h[0] > 0          # straight ahead in the heading frame
        v = rng.normal(size=3)
        assert np.allclose(heading_to_world(world_to_heading(v, yaw), yaw), v)
        assert np.isclose(world_to_heading(v, yaw)[1], v[1])  # vertical unchanged
