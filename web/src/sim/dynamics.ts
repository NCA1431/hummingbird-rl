/**
 * Rigid-body dynamics: moving the bird forward by one physics step.
 *
 * Port of hummingbird_rl/sim/dynamics.py: Newton for the translation, Euler's
 * equation I dω/dt = τ - ω × (I ω) for the rotation, and the semi-implicit
 * Euler scheme (velocities first, then positions with the new velocities).
 */

import { type WingCommand, computeWrench, decodeAction } from "./aero";
import { type BirdParams, DEFAULT_PARAMS, type Vec3, inertia, radians } from "./params";
import { type Quat, IDENTITY, eulerFromQuat, integrateQuat, quatFromEuler, quatToMatrix } from "./rotations";

function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

export interface BirdState {
  pos: Vec3; // m, world frame
  vel: Vec3; // m/s, world frame
  quat: Quat; // orientation, body -> world
  omega: Vec3; // rad/s, body frame [roll, yaw, pitch]
  time: number; // s
  phase: number; // rad, wingbeat phase (drawing only)
}

export function stateFromEuler(
  pos: Vec3,
  vel: Vec3 = [0, 0, 0],
  yaw = 0.0,
  pitch = 0.0,
  roll = 0.0,
  omega: Vec3 = [0, 0, 0],
): BirdState {
  return { pos: [...pos], vel: [...vel], quat: quatFromEuler(yaw, pitch, roll), omega: [...omega], time: 0.0, phase: 0.0 };
}

export function defaultState(): BirdState {
  return { pos: [0.0, 0.5, 0.0], vel: [0, 0, 0], quat: [...IDENTITY] as Quat, omega: [0, 0, 0], time: 0.0, phase: 0.0 };
}

/** (yaw, pitch, roll) in radians. */
export function eulerOf(state: BirdState): Vec3 {
  return eulerFromQuat(state.quat);
}

export interface StepInfo {
  thrust: number; // N
  fLeft: number;
  fRight: number;
  power: number; // W
  windForce: Vec3; // N
  cmd: WingCommand; // the decoded wing settings, for drawing the wings
}

export class BirdDynamics {
  readonly p: BirdParams;
  massFactor = 1.0;
  mass = 0.0;
  inertia: Vec3 = [0, 0, 0];
  damping: Vec3 = [0, 0, 0];
  damage: [number, number] = [0.0, 0.0];

  constructor(params: BirdParams = DEFAULT_PARAMS, massFactor = 1.0, damage: [number, number] = [0.0, 0.0]) {
    this.p = params;
    this.setConditions(massFactor, damage);
  }

  /** A heavier bird is uniformly denser: the inertia scales with the mass. */
  setConditions(massFactor = 1.0, damage: [number, number] = [0.0, 0.0]): void {
    this.massFactor = massFactor;
    this.mass = this.p.mass * massFactor;
    const i = inertia(this.p);
    this.inertia = [i[0] * massFactor, i[1] * massFactor, i[2] * massFactor];
    this.damping = [
      this.inertia[0] / this.p.tau_omega,
      this.inertia[1] / this.p.tau_omega,
      this.inertia[2] / this.p.tau_omega,
    ];
    this.damage = [damage[0], damage[1]];
  }

  step(state: BirdState, action: ArrayLike<number>, windForce: Vec3 = [0, 0, 0], dt?: number): [BirdState, StepInfo] {
    const p = this.p;
    const h = dt ?? p.dt;
    const cmd = decodeAction(action, p);
    const rot = quatToMatrix(state.quat);
    const w = computeWrench(rot, state.vel, state.omega, cmd, p, this.damping, this.damage, windForce, this.mass);

    const io: Vec3 = [0, 1, 2].map((i) => this.inertia[i] * state.omega[i]) as Vec3;
    const gyro = cross(state.omega, io);
    const omegaDot = [0, 1, 2].map((i) => (w.torqueBody[i] - gyro[i]) / this.inertia[i]);
    const acc = [0, 1, 2].map((i) => w.forceWorld[i] / this.mass);

    const omega = [0, 1, 2].map((i) => state.omega[i] + omegaDot[i] * h) as Vec3; // velocities first...
    const vel = [0, 1, 2].map((i) => state.vel[i] + acc[i] * h) as Vec3;
    const pos = [0, 1, 2].map((i) => state.pos[i] + vel[i] * h) as Vec3; // ...then positions
    const next: BirdState = {
      pos,
      vel,
      quat: integrateQuat(state.quat, omega, h),
      omega,
      time: state.time + h,
      phase: (state.phase + 2.0 * Math.PI * cmd.freq * h) % (2.0 * Math.PI),
    };
    return [next, { thrust: w.thrust, fLeft: w.fLeft, fRight: w.fRight, power: w.power, windForce, cmd }];
  }
}

// --- limits of the arena -----------------------------------------------------------------
export const MAX_TILT = radians(80.0);
export const GROUND_HEIGHT = 0.02; // m
export const ARENA_RADIUS = 1.5; // m
export const CEILING = 1.2; // m

export type LimitReason = "crash" | "out_of_bounds" | "loss_of_control";

/** Why the flight must stop, or null. Obstacle collisions are checked separately. */
export function checkLimits(state: BirdState): LimitReason | null {
  if (state.pos[1] < GROUND_HEIGHT) return "crash";
  if (Math.hypot(state.pos[0], state.pos[2]) > ARENA_RADIUS || state.pos[1] > CEILING) return "out_of_bounds";
  const [, pitch, roll] = eulerOf(state);
  if (Math.abs(pitch) > MAX_TILT || Math.abs(roll) > MAX_TILT) return "loss_of_control";
  return null;
}
