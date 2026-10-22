/**
 * What the controller sees: 48 numbers about the bird, its targets and the nearby obstacles.
 *
 * Port of hummingbird_rl/envs/observation.py, where the layout is explained
 * (heading frame, scales, valid flags). The result is rounded to float32 like
 * the Python one, since the networks were trained on float32 inputs.
 */

import { type BirdState, eulerOf } from "./dynamics";
import type { Vec3 } from "./params";
import type { Rng } from "./random";
import { worldToHeading } from "./rotations";
import { type Obstacle, nearestObstacles } from "./world";

export const OBS_SIZE = 48;
export const N_TARGETS = 2;
export const N_OBSTACLES = 5;
export const OBSTACLE_RANGE = 1.0; // m
export const POS_SCALE = 0.5; // m
export const RADIUS_SCALE = 0.1; // m
export const RATE_SCALE = 10.0; // rad/s
export const CLIP = 5.0;

// Where each block lives (start index, and length where it is a block).
export const ALTITUDE = 0;
export const VELOCITY = 1; // 3 values
export const ATTITUDE = 4; // roll, pitch
export const RATES = 6; // 3 values
export const PREV_ACTION = 9; // 5 values
export const TARGETS = 14; // 2 x (dx, dy, dz, valid)
export const OBSTACLES = 22; // 5 x (dx, dy, dz, radius, valid)
export const HOLD = 47;

export function buildObservation(
  state: BirdState,
  targets: Vec3[],
  obstacles: Obstacle[],
  prevAction: ArrayLike<number>,
  holdProgress: number,
  rBird: number,
  noise = 0.0,
  rng?: Rng,
): Float32Array {
  const [yaw, pitch, roll] = eulerOf(state);
  const obs = new Float64Array(OBS_SIZE);
  const measured = new Array<boolean>(OBS_SIZE).fill(false); // entries that come from a (noisy) sensor

  obs[ALTITUDE] = state.pos[1];
  obs.set(worldToHeading(state.vel, yaw), VELOCITY);
  obs[ATTITUDE] = roll;
  obs[ATTITUDE + 1] = pitch;
  for (let i = 0; i < 3; i++) obs[RATES + i] = state.omega[i] / RATE_SCALE;
  for (let i = 0; i < RATES + 3; i++) measured[i] = true;
  for (let i = 0; i < 5; i++) obs[PREV_ACTION + i] = prevAction[i];

  targets.slice(0, N_TARGETS).forEach((target, i) => {
    const j = TARGETS + 4 * i;
    const rel = worldToHeading([target[0] - state.pos[0], target[1] - state.pos[1], target[2] - state.pos[2]], yaw);
    for (let k = 0; k < 3; k++) {
      obs[j + k] = rel[k] / POS_SCALE;
      measured[j + k] = true;
    }
    obs[j + 3] = 1.0;
  });

  nearestObstacles(state.pos, obstacles, rBird, N_OBSTACLES, OBSTACLE_RANGE).forEach((o, i) => {
    const j = OBSTACLES + 5 * i;
    const c = o.center;
    const rel = worldToHeading([c[0] - state.pos[0], c[1] - state.pos[1], c[2] - state.pos[2]], yaw);
    for (let k = 0; k < 3; k++) obs[j + k] = rel[k] / POS_SCALE;
    obs[j + 3] = o.radius / RADIUS_SCALE;
    obs[j + 4] = 1.0;
    for (let k = 0; k < 4; k++) measured[j + k] = true;
  });

  obs[HOLD] = holdProgress;

  if (noise > 0.0) {
    if (!rng) throw new Error("sensor noise needs a random generator");
    for (let i = 0; i < OBS_SIZE; i++) if (measured[i]) obs[i] += noise * rng.normal();
  }
  return Float32Array.from(obs, (v) => Math.min(Math.max(v, -CLIP), CLIP));
}
