/**
 * Scenes and the geometry between the bird and the obstacles.
 *
 * Port of the parts of hummingbird_rl/world/scene.py and collision.py that
 * the browser needs. The bird and the obstacles are spheres, so everything
 * reduces to distances between points.
 */

import type { Vec3 } from "./params";

export const START: Vec3 = [0.0, 0.5, 0.0]; // m, every episode starts around this point

export interface Obstacle {
  center: Vec3; // m, world frame
  radius: number; // m
}

export interface Scene {
  start: Vec3;
  targets: Vec3[]; // in the order they must be reached
  obstacles: Obstacle[];
}

/** Gap between the bird's surface and the obstacle's surface (m). */
export function clearance(pos: Vec3, o: Obstacle, rBird: number): number {
  const d: Vec3 = [pos[0] - o.center[0], pos[1] - o.center[1], pos[2] - o.center[2]];
  return Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) - o.radius - rBird;
}

export function collides(pos: Vec3, obstacles: Obstacle[], rBird: number): boolean {
  return obstacles.some((o) => clearance(pos, o, rBird) < 0.0);
}

/** The k obstacles with the smallest clearance, ignoring those beyond maxRange; closest first. */
export function nearestObstacles(pos: Vec3, obstacles: Obstacle[], rBird: number, k = 5, maxRange = 1.0): Obstacle[] {
  return obstacles
    .map((o, i) => ({ c: clearance(pos, o, rBird), i }))
    .filter((x) => x.c <= maxRange)
    .sort((a, b) => a.c - b.c || a.i - b.i) // ties broken by index, as Python's tuple sort does
    .slice(0, k)
    .map((x) => obstacles[x.i]);
}
