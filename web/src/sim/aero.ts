/**
 * Wing aerodynamics: from the 5 actions to forces and torques on the bird.
 *
 * Port of hummingbird_rl/sim/aero.py: wingbeat-averaged ("quasi-steady")
 * forces, so the position of the wing within a beat plays no role in the
 * physics, only in the drawing. All functions are pure, as in Python.
 */

import {
  type BirdParams,
  type Vec3,
  hoverAmplitude,
  rWing,
} from "./params";
import { type Mat3, matTVec, matVec } from "./rotations";

export interface WingCommand {
  ampLeft: number; // rad, half-stroke amplitude of the left wing
  ampRight: number; // rad
  freq: number; // Hz, shared by both wings
  tilt: number; // rad, stroke-plane tilt (pitch torque)
  yawTilt: number; // rad, opposite tilt of the two stroke planes (yaw torque)
}

const clip1 = (x: number) => Math.min(Math.max(x, -1.0), 1.0);

/** Actions in [-1, 1]^5 to wing settings; out-of-range actions are clipped. */
export function decodeAction(action: ArrayLike<number>, p: BirdParams): WingCommand {
  const a = [0, 1, 2, 3, 4].map((i) => clip1(action[i]));
  return {
    ampLeft: 0.5 * (a[0] + 1.0) * p.amp_max,
    ampRight: 0.5 * (a[1] + 1.0) * p.amp_max,
    freq: p.f0 * (1.0 + p.freq_span * a[2]),
    tilt: a[3] * p.tilt_max,
    yawTilt: a[4] * p.yaw_tilt_max,
  };
}

/** Equal wings at f0, no tilt: holds a still, level bird (but hovering is unstable). */
export function hoverAction(p: BirdParams, massFactor = 1.0): number[] {
  const aAmp = (2.0 * hoverAmplitude(p, massFactor)) / p.amp_max - 1.0;
  return [aAmp, aAmp, 0.0, 0.0, 0.0];
}

/** Mid-wing speed of each wing, U = 2π f A r_wing. */
export function wingSpeeds(cmd: WingCommand, p: BirdParams): [number, number] {
  const k = 2.0 * Math.PI * cmd.freq * rWing(p);
  return [k * cmd.ampLeft, k * cmd.ampRight];
}

/** Force of each wing, ½ ρ C_L (S/2) U² (1 - D). */
export function wingThrusts(
  cmd: WingCommand,
  p: BirdParams,
  damage: [number, number] = [0.0, 0.0],
): [number, number] {
  const [uL, uR] = wingSpeeds(cmd, p);
  const q = 0.5 * p.rho * p.c_lift * (p.wing_area / 2.0);
  return [q * uL * uL * (1.0 - damage[0]), q * uR * uR * (1.0 - damage[1])];
}

/** Steering torques in the body frame, [roll, yaw, pitch]. */
export function controlTorques(cmd: WingCommand, fLeft: number, fRight: number, p: BirdParams): Vec3 {
  const thrust = fLeft + fRight;
  const tauRoll = (fLeft - fRight) * rWing(p);
  const tauYaw = p.k_yaw * thrust * rWing(p) * Math.sin(cmd.yawTilt);
  const tauPitch = thrust * p.pitch_lever * Math.sin(cmd.tilt);
  return [tauRoll, tauYaw, tauPitch];
}

/** Speed-dependent torques that make hovering unstable, [roll, yaw, pitch]. */
export function instabilityTorques(vBody: Vec3, p: BirdParams): Vec3 {
  return [-p.k_u_roll * vBody[2], 0.0, p.k_u_pitch * vBody[0]];
}

export function dragCoefficients(vBody: Vec3, p: BirdParams): Vec3 {
  const cX = vBody[0] >= 0.0 ? p.c_fwd : p.c_back;
  return [cX, p.c_vert, p.c_side];
}

export function dragForceBody(vBody: Vec3, p: BirdParams): Vec3 {
  const c = dragCoefficients(vBody, p);
  return [-c[0] * vBody[0], -c[1] * vBody[1], -c[2] * vBody[2]];
}

/** Mechanical power of the wings, Σ force × wing speed (W). */
export function aeroPower(cmd: WingCommand, fLeft: number, fRight: number, p: BirdParams): number {
  const [uL, uR] = wingSpeeds(cmd, p);
  return fLeft * uL + fRight * uR;
}

export interface Wrench {
  forceWorld: Vec3; // N, gravity included
  torqueBody: Vec3; // N·m, [roll, yaw, pitch]
  thrust: number;
  fLeft: number;
  fRight: number;
  power: number;
}

/** All forces (world frame) and torques (body frame) acting on the bird during one step. */
export function computeWrench(
  rot: Mat3,
  vWorld: Vec3,
  omegaBody: Vec3,
  cmd: WingCommand,
  p: BirdParams,
  damp: Vec3,
  damage: [number, number],
  windForce: Vec3,
  mass: number,
): Wrench {
  const [fL, fR] = wingThrusts(cmd, p, damage);
  const thrust = fL + fR;
  const vBody = matTVec(rot, vWorld);

  const fThrust: Vec3 = [thrust * rot[0][1], thrust * rot[1][1], thrust * rot[2][1]]; // out of the back
  const fDrag = matVec(rot, dragForceBody(vBody, p));
  const fGravity: Vec3 = [0.0, -mass * p.g, 0.0];
  const force: Vec3 = [0, 1, 2].map(
    (i) => fThrust[i] + fDrag[i] + fGravity[i] + windForce[i],
  ) as Vec3;

  const ctrl = controlTorques(cmd, fL, fR, p);
  const inst = instabilityTorques(vBody, p);
  const torque: Vec3 = [0, 1, 2].map(
    (i) => ctrl[i] + inst[i] - damp[i] * omegaBody[i],
  ) as Vec3;
  return { forceWorld: force, torqueBody: torque, thrust, fLeft: fL, fRight: fR, power: aeroPower(cmd, fL, fR, p) };
}
