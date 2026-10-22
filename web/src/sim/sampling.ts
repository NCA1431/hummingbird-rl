/**
 * Random scenes for the four tasks, in the browser.
 *
 * Port of hummingbird_rl/world/sampling.py: same distances, altitudes and
 * obstacle sizes, and the same two checks for the scenes with obstacles (at
 * least one obstacle blocks the straight path, and a coarse search finds a
 * way through). The random draws are the browser's own (random.ts), so a
 * scene here is a new one, never one of the Python test scenes.
 */

import type { Vec3 } from "./params";
import type { Rng } from "./random";
import { type Obstacle, type Scene, START } from "./world";

const ALTITUDE_RANGE: [number, number] = [0.15, 0.95]; // m, where go-to-point targets may be placed
const MAX_TRIES = 500;

const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scale = (a: Vec3, k: number): Vec3 => [a[0] * k, a[1] * k, a[2] * k];
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len = (a: Vec3) => Math.sqrt(dot(a, a));

/** Direction drawn uniformly on the sphere (normalised Gaussian vector). */
function unitVector(rng: Rng): Vec3 {
  const v: Vec3 = [rng.normal(), rng.normal(), rng.normal()];
  return scale(v, 1.0 / len(v));
}

/** Point drawn uniformly inside a ball (the cube root keeps the density uniform). */
function pointInBall(rng: Rng, radius: number): Vec3 {
  return scale(unitVector(rng), radius * rng.uniform() ** (1.0 / 3.0));
}

/** True if the obstacle's surface stays at least `gap` away from every point. */
function farEnough(center: Vec3, radius: number, points: Vec3[], gap: number): boolean {
  return points.every((p) => len(sub(center, p)) - radius >= gap);
}

/** Distance from point p to the segment [a, b]. */
export function segmentPointDistance(a: Vec3, b: Vec3, p: Vec3): number {
  const ab = sub(b, a);
  const t = Math.min(Math.max(dot(sub(p, a), ab) / Math.max(dot(ab, ab), 1e-12), 0.0), 1.0);
  return len(sub(add(a, scale(ab, t)), p));
}

/**
 * Coarse check that the bird can fly through the points in order without touching anything.
 *
 * As in Python: 5 cm cubes around the course, a cube is blocked if its centre
 * is inside an obstacle inflated by the bird's radius, and all the points must
 * lie in the same connected region of free cubes (face neighbours). We find
 * that region with a breadth-first search from the first point.
 */
export function pathExists(points: Vec3[], obstacles: Obstacle[], rBird: number, cell = 0.05): boolean {
  const lo: Vec3 = [0, 1, 2].map((i) => Math.min(...points.map((p) => p[i])) - 0.3) as Vec3;
  const hi: Vec3 = [0, 1, 2].map((i) => Math.max(...points.map((p) => p[i])) + 0.3) as Vec3;
  lo[1] = 0.05;
  hi[1] = 1.15;
  const n = [0, 1, 2].map((i) => Math.ceil((hi[i] + cell - lo[i]) / cell)); // like np.arange(lo, hi + cell, cell)
  const index = (i: number, j: number, k: number) => (i * n[1] + j) * n[2] + k;
  const free = new Uint8Array(n[0] * n[1] * n[2]);
  for (let i = 0; i < n[0]; i++)
    for (let j = 0; j < n[1]; j++)
      for (let k = 0; k < n[2]; k++) {
        const c: Vec3 = [lo[0] + i * cell, lo[1] + j * cell, lo[2] + k * cell];
        free[index(i, j, k)] = obstacles.every((o) => {
          const d = sub(c, o.center);
          return dot(d, d) > (o.radius + rBird) ** 2;
        })
          ? 1
          : 0;
      }

  const cellOf = (p: Vec3) => [0, 1, 2].map((i) => Math.round((p[i] - lo[i]) / cell));
  const starts = points.map(cellOf);
  if (starts.some(([i, j, k]) => !free[index(i, j, k)])) return false; // a point lies inside an obstacle

  const seen = new Uint8Array(free.length);
  const queue = [starts[0]];
  seen[index(starts[0][0], starts[0][1], starts[0][2])] = 1;
  const steps = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
  while (queue.length) {
    const [i, j, k] = queue.pop()!;
    for (const [di, dj, dk] of steps) {
      const a = i + di;
      const b = j + dj;
      const c = k + dk;
      if (a < 0 || b < 0 || c < 0 || a >= n[0] || b >= n[1] || c >= n[2]) continue;
      const id = index(a, b, c);
      if (free[id] && !seen[id]) {
        seen[id] = 1;
        queue.push([a, b, c]);
      }
    }
  }
  return starts.every(([i, j, k]) => seen[index(i, j, k)] === 1);
}

// ------------------------------------------------------------------- the tasks

function sampleGoto(rng: Rng): Scene {
  for (let t = 0; t < MAX_TRIES; t++) {
    const target = add(START, scale(unitVector(rng), rng.between(0.1, 0.5)));
    if (target[1] >= ALTITUDE_RANGE[0] && target[1] <= ALTITUDE_RANGE[1]) {
      return { start: [...START], targets: [target], obstacles: [] };
    }
  }
  throw new Error("could not sample a go-to-point scene");
}

function sampleObstacles(rng: Rng, rBird: number, nObstacles: [number, number]): Scene {
  for (let t = 0; t < MAX_TRIES; t++) {
    const dist = rng.between(0.5, 0.8);
    const altitude = rng.between(0.3, 0.8);
    const dy = altitude - START[1];
    const h = Math.sqrt(dist ** 2 - dy ** 2);
    const bearing = rng.between(-Math.PI, Math.PI);
    const goal = add(START, [h * Math.cos(bearing), dy, h * Math.sin(bearing)]);

    const obstacles: Obstacle[] = [];
    const count = rng.integer(nObstacles[0], nObstacles[1] + 1);
    for (let n = 0; n < count; n++) {
      for (let a = 0; a < 50; a++) {
        const radius = rng.between(0.04, 0.1);
        const center = add(add(START, scale(sub(goal, START), rng.between(0.2, 0.8))), pointInBall(rng, 0.12));
        if (farEnough(center, radius, [START, goal], 0.1)) {
          obstacles.push({ center, radius });
          break;
        }
      }
    }
    const blocking = obstacles.some((o) => segmentPointDistance(START, goal, o.center) < o.radius + rBird);
    if (obstacles.length && blocking && pathExists([START, goal], obstacles, rBird)) {
      return { start: [...START], targets: [goal], obstacles };
    }
  }
  throw new Error("could not sample an obstacles scene");
}

function sampleNavigation(rng: Rng, rBird: number, nObstacles: [number, number]): Scene {
  for (let t = 0; t < MAX_TRIES; t++) {
    const waypoints: Vec3[] = [];
    let current: Vec3 = [...START];
    for (let w = 0; w < 3; w++) {
      for (let a = 0; a < 100; a++) {
        const bearing = rng.between(-Math.PI, Math.PI);
        const step = rng.between(0.25, 0.5);
        const altitude = rng.between(0.2, 0.9);
        const dy = altitude - current[1];
        if (Math.abs(dy) >= step) continue;
        const h = Math.sqrt(step ** 2 - dy ** 2);
        const next = add(current, [h * Math.cos(bearing), dy, h * Math.sin(bearing)]);
        if (Math.hypot(next[0], next[2]) <= 1.0) {
          // stay well inside the arena
          waypoints.push(next);
          current = next;
          break;
        }
      }
    }
    if (waypoints.length < 3) continue;

    const course = [START, ...waypoints];
    const obstacles: Obstacle[] = [];
    const count = rng.integer(nObstacles[0], nObstacles[1] + 1);
    for (let n = 0; n < count; n++) {
      for (let a = 0; a < 50; a++) {
        const leg = rng.integer(0, 3);
        const [p, q] = [course[leg], course[leg + 1]];
        const radius = rng.between(0.04, 0.1);
        const center = add(add(p, scale(sub(q, p), rng.between(0.2, 0.8))), pointInBall(rng, 0.15));
        if (center[1] >= 0.05 && center[1] <= 1.15 && farEnough(center, radius, course, 0.1)) {
          obstacles.push({ center, radius });
          break;
        }
      }
    }
    if (pathExists(course, obstacles, rBird)) return { start: [...START], targets: waypoints, obstacles };
  }
  throw new Error("could not sample a navigation scene");
}

export function sampleScene(task: string, rng: Rng, rBird = 0.02, nObstacles: [number, number] = [3, 6]): Scene {
  switch (task) {
    case "hover":
      return { start: [...START], targets: [[...START]], obstacles: [] };
    case "goto":
      return sampleGoto(rng);
    case "obstacles":
      return sampleObstacles(rng, rBird, nObstacles);
    case "navigation":
      return sampleNavigation(rng, rBird, nObstacles);
    default:
      throw new Error(`unknown task ${task}`);
  }
}
