"""Orientation maths: quaternions, Euler angles and the heading frame.

Inside the simulator the bird's orientation is a unit quaternion
q = (w, x, y, z). Quaternions have no singular orientation and are cheap to
integrate, which is why physics engines use them. For the controller we
convert to Euler angles (yaw, pitch, roll), which are easier to read and to
reason about.

Euler convention: R = R_y(yaw) · R_z(pitch) · R_x(roll), where R maps a vector
written in the body frame to the world frame. In words: start level, turn
the nose by `yaw` about the vertical, raise it by `pitch`, then bank by `roll`
about the bird's own long axis. Signs (see params.py for the axes):
    yaw   > 0 : nose turns left (towards -z)
    pitch > 0 : nose goes up
    roll  > 0 : right wing goes down

Heading frame: the world frame turned by the bird's yaw only. In it, "x" is
straight ahead of the bird in the horizontal plane and "y" is still vertical.
The observation uses it so that what the controller sees does not depend on
which way the bird happens to face.
"""

from __future__ import annotations

from math import asin, atan2, cos, sin, sqrt

import numpy as np

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])   # no rotation


def quat_mul(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Hamilton product q ⊗ r: the rotation r followed by q (both in their own frames)."""
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = r
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Rescale to unit length. Rounding errors slowly break |q| = 1, so we renormalise every step."""
    return q / sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation by `angle` (rad) about `axis`: q = (cos(angle/2), sin(angle/2) · unit axis)."""
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-15 or angle == 0.0:
        return IDENTITY.copy()
    s = sin(angle / 2.0) / n
    return np.array([cos(angle / 2.0), axis[0] * s, axis[1] * s, axis[2] * s])


def quat_from_euler(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Quaternion for the yaw → pitch → roll sequence above: q_y(yaw) ⊗ q_z(pitch) ⊗ q_x(roll)."""
    qy = np.array([cos(yaw / 2), 0.0, sin(yaw / 2), 0.0])
    qz = np.array([cos(pitch / 2), 0.0, 0.0, sin(pitch / 2)])
    qx = np.array([cos(roll / 2), sin(roll / 2), 0.0, 0.0])
    return quat_mul(quat_mul(qy, qz), qx)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix R (body → world). Column i of R is body axis i expressed in the world frame,
    so R[:, 0] is the beak direction and R[:, 1] the direction the wings push."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def euler_from_matrix(r: np.ndarray) -> tuple[float, float, float]:
    """Recover (yaw, pitch, roll) from R = R_y(yaw) R_z(pitch) R_x(roll).

    Multiplying the three matrices out gives, among other entries,
        R[1,0] = sin(pitch),   R[1,1] = cos(pitch) cos(roll),   R[1,2] = -cos(pitch) sin(roll),
        R[0,0] = cos(yaw) cos(pitch),   R[2,0] = -sin(yaw) cos(pitch),
    which we invert below. This breaks down only at pitch = ±90°, and the
    environment ends an episode long before that (at 80°).
    """
    pitch = asin(max(-1.0, min(1.0, r[1, 0])))   # clamp guards against rounding just above 1
    roll = atan2(-r[1, 2], r[1, 1])
    yaw = atan2(-r[2, 0], r[0, 0])
    return yaw, pitch, roll


def euler_from_quat(q: np.ndarray) -> tuple[float, float, float]:
    return euler_from_matrix(quat_to_matrix(q))


def integrate_quat(q: np.ndarray, omega_body: np.ndarray, dt: float) -> np.ndarray:
    """Advance the orientation by dt, given the angular velocity in the body frame.

    Over a short step the bird turns by |ω| dt about the axis ω, so we apply
    that small rotation on the right (right = in the body frame):
    q ← q ⊗ rotation(ω, |ω| dt). This is exact for a constant ω, unlike the
    usual first-order update q + ½ q ⊗ (0, ω) dt.
    """
    angle = float(np.linalg.norm(omega_body)) * dt
    if angle < 1e-15:
        return q
    dq = quat_from_axis_angle(omega_body, angle)
    return quat_normalize(quat_mul(q, dq))


def rot_y(angle: float) -> np.ndarray:
    """Rotation matrix about the vertical axis y."""
    c, s = cos(angle), sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def world_to_heading(vec_world: np.ndarray, yaw: float) -> np.ndarray:
    """Express a world vector in the heading frame (undo the yaw): R_y(-yaw) · u."""
    return rot_y(-yaw) @ np.asarray(vec_world, dtype=float)


def heading_to_world(vec_heading: np.ndarray, yaw: float) -> np.ndarray:
    """Inverse of world_to_heading."""
    return rot_y(yaw) @ np.asarray(vec_heading, dtype=float)
