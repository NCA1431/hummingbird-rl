/**
 * What every controller of the live mode shares: the interface, and reading the observation.
 *
 * Port of hummingbird_rl/controllers/base.py. The classical controllers see
 * exactly what the networks see, the 48-number observation, and turn it back
 * into physical units before using it.
 */

import * as ob from "../sim/observation";
import type { Vec3 } from "../sim/params";

export interface Controller {
  readonly name: string;
  /** Forget the previous episode (integrators). */
  reset(): void;
  /** The 5 actions for one observation. */
  act(obs: ArrayLike<number>): number[];
}

/** The observation in physical units (heading frame, SI). */
export interface ObsView {
  altitude: number;
  vel: Vec3; // m/s, (forward, up, right)
  roll: number;
  pitch: number;
  rates: Vec3; // rad/s, body rates (roll, yaw, pitch)
  targets: Vec3[]; // m, relative to the bird, current target first
  obstacles: { centre: Vec3; radius: number }[]; // centre relative to the bird (m), radius (m)
  hold: number;
}

export function parseObs(obs: ArrayLike<number>): ObsView {
  const v = (i: number, scale = 1.0): Vec3 => [obs[i] * scale, obs[i + 1] * scale, obs[i + 2] * scale];
  const targets: Vec3[] = [];
  for (let i = 0; i < ob.N_TARGETS; i++) {
    const j = ob.TARGETS + 4 * i;
    if (obs[j + 3] > 0.5) targets.push(v(j, ob.POS_SCALE));
  }
  const obstacles: { centre: Vec3; radius: number }[] = [];
  for (let i = 0; i < ob.N_OBSTACLES; i++) {
    const j = ob.OBSTACLES + 5 * i;
    if (obs[j + 4] > 0.5) obstacles.push({ centre: v(j, ob.POS_SCALE), radius: obs[j + 3] * ob.RADIUS_SCALE });
  }
  return {
    altitude: obs[ob.ALTITUDE],
    vel: v(ob.VELOCITY),
    roll: obs[ob.ATTITUDE],
    pitch: obs[ob.ATTITUDE + 1],
    rates: v(ob.RATES, ob.RATE_SCALE),
    targets,
    obstacles,
    hold: obs[ob.HOLD],
  };
}

export const clip = (x: number, lo: number, hi: number) => Math.min(Math.max(x, lo), hi);
export const norm = (v: ArrayLike<number>) => Math.hypot(...Array.from(v));
