/**
 * Orientation maths: quaternions, Euler angles and the heading frame.
 *
 * Port of hummingbird_rl/sim/rotations.py (conventions explained there).
 * Quaternions are (w, x, y, z), as in Python; three.js stores them as
 * (x, y, z, w), so the conversion happens only when drawing. The world frame
 * is the same as three.js's: x forward, y up, z to the right.
 *
 * Every expression keeps the order of the Python one: floating-point
 * addition is not associative, and the parity tests compare to 1e-9.
 */

import type { Vec3 } from "./params";

export type Quat = [number, number, number, number];
/** 3x3 matrix stored by rows: m[i][j] is row i, column j. */
export type Mat3 = [Vec3, Vec3, Vec3];

export const IDENTITY: Readonly<Quat> = [1.0, 0.0, 0.0, 0.0];

export function norm3(v: Vec3): number {
  return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

/** m v */
export function matVec(m: Mat3, v: Vec3): Vec3 {
  return [
    m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
    m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
    m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
  ];
}

/** mᵀ v: turns a world vector into body coordinates when m is body -> world. */
export function matTVec(m: Mat3, v: Vec3): Vec3 {
  return [
    m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
    m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
    m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
  ];
}

/** Hamilton product q ⊗ r. */
export function quatMul(q: Readonly<Quat>, r: Readonly<Quat>): Quat {
  const [w1, x1, y1, z1] = q;
  const [w2, x2, y2, z2] = r;
  return [
    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
  ];
}

export function quatNormalize(q: Readonly<Quat>): Quat {
  const n = Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

export function quatFromAxisAngle(axis: Vec3, angle: number): Quat {
  const n = norm3(axis);
  if (n < 1e-15 || angle === 0.0) return [...IDENTITY] as Quat;
  const s = Math.sin(angle / 2.0) / n;
  return [Math.cos(angle / 2.0), axis[0] * s, axis[1] * s, axis[2] * s];
}

/** q_y(yaw) ⊗ q_z(pitch) ⊗ q_x(roll). */
export function quatFromEuler(yaw: number, pitch: number, roll: number): Quat {
  const qy: Quat = [Math.cos(yaw / 2), 0.0, Math.sin(yaw / 2), 0.0];
  const qz: Quat = [Math.cos(pitch / 2), 0.0, 0.0, Math.sin(pitch / 2)];
  const qx: Quat = [Math.cos(roll / 2), Math.sin(roll / 2), 0.0, 0.0];
  return quatMul(quatMul(qy, qz), qx);
}

/** Rotation matrix body -> world; column 0 is the beak direction, column 1 the thrust direction. */
export function quatToMatrix(q: Readonly<Quat>): Mat3 {
  const [w, x, y, z] = q;
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
  ];
}

/** (yaw, pitch, roll) from R = R_y(yaw) R_z(pitch) R_x(roll); valid away from pitch = ±90°. */
export function eulerFromMatrix(r: Mat3): Vec3 {
  const pitch = Math.asin(Math.max(-1.0, Math.min(1.0, r[1][0])));
  const roll = Math.atan2(-r[1][2], r[1][1]);
  const yaw = Math.atan2(-r[2][0], r[0][0]);
  return [yaw, pitch, roll];
}

export function eulerFromQuat(q: Readonly<Quat>): Vec3 {
  return eulerFromMatrix(quatToMatrix(q));
}

/** Turn by |ω| dt about ω, applied in the body frame (exact for a constant ω). */
export function integrateQuat(q: Readonly<Quat>, omegaBody: Vec3, dt: number): Quat {
  const angle = norm3(omegaBody) * dt;
  if (angle < 1e-15) return [...q] as Quat;
  const dq = quatFromAxisAngle(omegaBody, angle);
  return quatNormalize(quatMul(q, dq));
}

export function rotY(angle: number): Mat3 {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return [
    [c, 0.0, s],
    [0.0, 1.0, 0.0],
    [-s, 0.0, c],
  ];
}

/** A world vector in the heading frame (the yaw undone): R_y(-yaw) u. */
export function worldToHeading(vecWorld: Vec3, yaw: number): Vec3 {
  return matVec(rotY(-yaw), vecWorld);
}

export function headingToWorld(vecHeading: Vec3, yaw: number): Vec3 {
  return matVec(rotY(yaw), vecHeading);
}
