/**
 * Cascaded PID with a potential field: the classical baseline, in the browser.
 *
 * Port of hummingbird_rl/controllers/pid.py, where each loop is explained:
 * position -> desired velocity (P, capped at 0.6 m/s), velocity -> desired
 * acceleration (P + I near the target, plus the obstacle push), acceleration
 * -> thrust and tilt (a hovering bird can only push along its back), tilt ->
 * torques (PID), torques -> actions (allocation). Yaw is only damped, so a
 * target behind is reached by backing up. Same gains as in Python; the parity
 * test flies whole episodes with both and compares every action.
 */

import { type BirdParams, DEFAULT_PARAMS, type Vec3, inertia, radians } from "../sim/params";
import { allocate } from "./allocation";
import { type Controller, clip, norm, parseObs } from "./controller";
import { repulsion } from "./potentialField";

export const PID_GAINS = {
  kPos: 5.0, // 1/s, position error -> desired velocity
  vMax: 0.6, // m/s, cap on the desired velocity
  kVel: 10.0, // 1/s, velocity error -> desired acceleration
  kInt: 6.0, // 1/s², integral of the velocity-loop error
  intRadius: 0.15, // m, that integral only runs this close to the target
  aIntMax: 3.5, // m/s², anti-windup limit
  tiltMax: radians(25.0),
  wAtt: [30.0, 20.0], // rad/s, natural frequency of the (roll, pitch) loops
  zetaAtt: 0.8, // damping ratio of the attitude loops
  cAttInt: 0.25, // integral gain as a fraction of w³
  tauIntMax: [3e-4, 3e-5], // N·m, anti-windup for (roll, pitch)
  kYawRate: 8.0, // 1/s, yaw-rate damping
};

export class PIDController implements Controller {
  readonly name = "pid";
  private readonly p: BirdParams;
  private readonly dt: number;
  private aInt: Vec3 = [0, 0, 0]; // m/s², velocity-loop integral
  private tauInt: [number, number] = [0, 0]; // N·m, attitude-loop integral (roll, pitch)

  constructor(params: BirdParams = DEFAULT_PARAMS) {
    this.p = params;
    this.dt = params.dt * params.action_repeat;
  }

  reset(): void {
    this.aInt = [0, 0, 0];
    this.tauInt = [0, 0];
  }

  act(obs: ArrayLike<number>): number[] {
    const o = parseObs(obs);
    const g = PID_GAINS;
    const p = this.p;
    const error: Vec3 = o.targets.length ? o.targets[0] : [0, 0, 0]; // target relative to the bird

    // Position loop -> desired velocity, capped.
    const vDes = error.map((e) => g.kPos * e) as Vec3;
    const speed = norm(vDes);
    if (speed > g.vMax) for (let i = 0; i < 3; i++) vDes[i] *= g.vMax / speed;

    // Velocity loop -> desired acceleration; the integral only runs near the target.
    const vErr = [0, 1, 2].map((i) => vDes[i] - o.vel[i]) as Vec3;
    if (norm(error) < g.intRadius) {
      this.aInt = [0, 1, 2].map((i) => clip(this.aInt[i] + g.kInt * vErr[i] * this.dt, -g.aIntMax, g.aIntMax)) as Vec3;
    }
    const push = repulsion(o, p.r_bird);
    const aDes = [0, 1, 2].map((i) => g.kVel * vErr[i] + this.aInt[i] + push[i]);

    // Geometry: tilt the thrust towards the desired acceleration.
    const lift = Math.max(p.g + aDes[1], 0.2 * p.g);
    const pitchDes = clip(-Math.atan2(aDes[0], lift), -g.tiltMax, g.tiltMax);
    const rollDes = clip(Math.atan2(aDes[2], lift), -g.tiltMax, g.tiltMax);
    const thrust = (p.mass * lift) / (Math.cos(pitchDes) * Math.cos(rollDes));

    // Attitude loop: PD written as a desired angular acceleration, times the inertia, plus the integral.
    const w = g.wAtt;
    const kp = w.map((x) => x ** 2);
    const kd = w.map((x) => 2.0 * g.zetaAtt * x);
    const ki = w.map((x) => g.cAttInt * x ** 3);
    const attErr = [rollDes - o.roll, pitchDes - o.pitch];
    const I = inertia(p);
    const inertiaRP = [I[0], I[2]];
    this.tauInt = [0, 1].map((i) =>
      clip(this.tauInt[i] + ki[i] * inertiaRP[i] * attErr[i] * this.dt, -g.tauIntMax[i], g.tauIntMax[i]),
    ) as [number, number];
    const alpha = [
      kp[0] * attErr[0] - kd[0] * o.rates[0],
      -g.kYawRate * o.rates[1],
      kp[1] * attErr[1] - kd[1] * o.rates[2],
    ];
    const torque: Vec3 = [
      I[0] * alpha[0] + this.tauInt[0],
      I[1] * alpha[1] + 0.0,
      I[2] * alpha[2] + this.tauInt[1],
    ];
    return allocate(thrust, torque, p);
  }
}
