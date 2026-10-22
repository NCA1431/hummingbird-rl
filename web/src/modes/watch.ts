/**
 * Watch mode: replaying the episodes recorded by `scripts/export_web.py episodes`.
 *
 * Nothing is simulated here: an episode file holds the state of the bird after
 * every decision (60 per second), and we only interpolate between two
 * consecutive frames, so what is shown is exactly what was measured. The wing
 * phase is the one thing we integrate ourselves, from the recorded frequency,
 * because the recording does not keep it.
 */

import type { Vec3 } from "../sim/params";
import type { Quat } from "../sim/rotations";

export interface Episode {
  format: string;
  task: string;
  seed: number;
  policy?: string;
  suite?: string;
  dt: number; // s between two frames (1/60)
  scene: { start: Vec3; targets: Vec3[]; obstacles: { center: Vec3; radius: number }[] };
  conditions: {
    mean_wind: Vec3;
    gust_sigma: number;
    mass_factor: number;
    damage: [number, number];
    failure_time: number | null;
    failure_wing: number;
    failure_amount: number;
  };
  result: { success: boolean; reason: string; return: number; length: number; energy: number };
  frames: {
    t: number[];
    pos: Vec3[];
    quat: Quat[];
    vel: Vec3[];
    action: number[][];
    thrust: number[];
    wind: Vec3[];
    target: number[];
    hold: number[];
    distance: number[];
  };
}

export interface IndexEntry {
  file: string;
  policy: string;
  task: string;
  suite: string;
  seed: number;
  success: boolean;
  reason: string;
  length: number;
}

/** The bird's state at time t (s), between two recorded frames. */
export interface Sample {
  index: number; // the frame just before t
  pos: Vec3;
  quat: Quat;
  vel: Vec3;
  action: number[]; // the action applied between this frame and the next
  thrust: number;
  wind: Vec3;
  target: number;
  hold: number;
  distance: number;
}

const lerp = (a: number, b: number, u: number) => a + (b - a) * u;
const lerp3 = (a: Vec3, b: Vec3, u: number): Vec3 => [lerp(a[0], b[0], u), lerp(a[1], b[1], u), lerp(a[2], b[2], u)];

/** Normalised linear interpolation of two quaternions: close enough to slerp for 1/60 s apart. */
function nlerp(a: Quat, b: Quat, u: number): Quat {
  const sign = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3] < 0 ? -1 : 1; // shortest way round
  const q = [0, 1, 2, 3].map((i) => lerp(a[i], sign * b[i], u)) as Quat;
  const n = Math.hypot(q[0], q[1], q[2], q[3]);
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

export function duration(ep: Episode): number {
  return ep.frames.t[ep.frames.t.length - 1];
}

export function sampleEpisode(ep: Episode, t: number): Sample {
  const f = ep.frames;
  const last = f.t.length - 1;
  const x = Math.min(Math.max(t / ep.dt, 0), last);
  const k = Math.min(Math.floor(x), Math.max(last - 1, 0));
  const u = last === 0 ? 0 : x - k;
  const next = Math.min(k + 1, last);
  return {
    index: k,
    pos: lerp3(f.pos[k], f.pos[next], u),
    quat: nlerp(f.quat[k], f.quat[next], u),
    vel: lerp3(f.vel[k], f.vel[next], u),
    action: f.action[next],
    thrust: lerp(f.thrust[k], f.thrust[next], u),
    wind: lerp3(f.wind[k], f.wind[next], u),
    target: f.target[u < 0.5 ? k : next],
    hold: lerp(f.hold[k], f.hold[next], u),
    distance: lerp(f.distance[k], f.distance[next], u),
  };
}

/** Wing damage at time t: the episode's damage, plus the sudden failure once it has happened. */
export function damageAt(ep: Episode, t: number): [number, number] {
  const c = ep.conditions;
  const d: [number, number] = [c.damage[0], c.damage[1]];
  if (c.failure_time !== null && t >= c.failure_time) {
    d[c.failure_wing] = Math.max(d[c.failure_wing], c.failure_amount);
  }
  return d;
}

/** Simulated time, advanced by real time × speed; it stops at the end of the episode. */
export class Playback {
  time = 0.0;
  speed = 1.0;
  playing = true;

  constructor(public episode: Episode) {}

  /** Advances by `realDt` seconds of real time; returns the simulated time elapsed. */
  advance(realDt: number): number {
    if (!this.playing) return 0.0;
    const end = duration(this.episode);
    const before = this.time;
    this.time = Math.min(this.time + realDt * this.speed, end);
    if (this.time >= end) this.playing = false;
    return this.time - before;
  }

  seek(t: number): void {
    this.time = Math.min(Math.max(t, 0), duration(this.episode));
  }

  restart(): void {
    this.time = 0.0;
    this.playing = true;
  }
}
