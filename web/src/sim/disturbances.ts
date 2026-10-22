/**
 * The physical conditions of an episode: wind, mass, wing damage, sudden failure, noise.
 *
 * Port of hummingbird_rl/envs/disturbances.py, same regimes and same ranges:
 *   calm        no wind, no noise
 *   normal      breeze up to 5% of the weight, gusts of 2%, small sensor and actuator noise
 *   randomised  wind up to 30%, gusts up to 10%, mass ±25%, one wing down by up to 40% (70% of episodes)
 *   failure     normal, then one wing suddenly loses 40% between 1.5 and 3 s
 * Field names stay in Python's snake_case: the same objects are written in
 * the episode files and read back by the viewer.
 */

import { type BirdParams, type Vec3, weight } from "./params";
import type { Rng } from "./random";

export const ACTION_NOISE = 0.02; // std of the noise added to each action
export const OBS_NOISE = 0.01; // std of the sensor noise, in observation units

export interface Conditions {
  mean_wind: Vec3; // N
  gust_sigma: number; // N
  mass_factor: number;
  damage: [number, number]; // fraction of force lost by the (left, right) wing
  failure_time: number | null; // s, when the sudden failure happens (null: never)
  failure_wing: number; // 0 left, 1 right
  failure_amount: number; // fraction of force lost at failure_time
  action_noise: number;
  obs_noise: number;
}

export const REGIMES = ["normal", "randomised", "failure", "calm"] as const;
export type RegimeName = (typeof REGIMES)[number];

export function calmConditions(): Conditions {
  return {
    mean_wind: [0, 0, 0],
    gust_sigma: 0.0,
    mass_factor: 1.0,
    damage: [0.0, 0.0],
    failure_time: null,
    failure_wing: 0,
    failure_amount: 0.0,
    action_noise: 0.0,
    obs_noise: 0.0,
  };
}

/** A horizontal vector of the given length pointing in a random direction. */
function horizontal(rng: Rng, magnitude: number): Vec3 {
  const angle = rng.between(-Math.PI, Math.PI);
  return [magnitude * Math.cos(angle), 0.0, magnitude * Math.sin(angle)];
}

function randomWing(rng: Rng, amount: number): [number, number] {
  return rng.uniform() < 0.5 ? [amount, 0.0] : [0.0, amount];
}

export function sampleConditions(regime: RegimeName, rng: Rng, p: BirdParams): Conditions {
  const w = weight(p);
  const c = calmConditions();
  if (regime === "calm") return c;
  if (regime === "randomised") {
    const damage: [number, number] = rng.uniform() < 0.3 ? [0.0, 0.0] : randomWing(rng, rng.between(0.0, 0.4));
    return {
      ...c,
      mean_wind: horizontal(rng, rng.between(0.0, 0.3) * w),
      gust_sigma: rng.between(0.0, 0.1) * w,
      mass_factor: rng.between(0.75, 1.25),
      damage,
      action_noise: ACTION_NOISE,
      obs_noise: OBS_NOISE,
    };
  }
  const normal: Conditions = {
    ...c,
    mean_wind: horizontal(rng, rng.between(0.0, 0.05) * w),
    gust_sigma: 0.02 * w,
    action_noise: ACTION_NOISE,
    obs_noise: OBS_NOISE,
  };
  if (regime === "failure") {
    normal.failure_time = rng.between(1.5, 3.0);
    normal.failure_wing = rng.integer(0, 2);
    normal.failure_amount = 0.4;
  }
  return normal;
}
