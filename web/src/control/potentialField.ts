/**
 * Potential field: how the PID and the LQR steer around obstacles.
 *
 * Port of hummingbird_rl/controllers/potential_field.py: each obstacle closer
 * than 15 cm pushes the bird away (gradient of ½ k (1/gap − 1/range)²), with a
 * horizontal sideways part on the side of the goal so that the bird slides
 * round instead of stopping dead. Local minima remain possible.
 */

import type { Vec3 } from "../sim/params";
import { type ObsView, norm } from "./controller";

export const FIELD = {
  avoidRange: 0.15, // m, obstacles closer than this repel the bird
  kAvoid: 0.05, // strength of the repulsion
  kSlide: 0.7, // sideways part, as a fraction of the repulsion
  aMax: 6.0, // m/s², cap on the total push
};

/** Push away from the obstacles that are too close, as an acceleration (m/s², heading frame). */
export function repulsion(o: ObsView, rBird: number, g = FIELD): Vec3 {
  const total: Vec3 = [0, 0, 0];
  const goal: Vec3 = o.targets.length ? o.targets[0] : [0, 0, 0];
  for (const { centre, radius } of o.obstacles) {
    const dist = norm(centre);
    let gap = dist - radius - rBird;
    if (gap >= g.avoidRange || dist < 1e-6) continue;
    const away: Vec3 = [-centre[0] / dist, -centre[1] / dist, -centre[2] / dist];
    gap = Math.max(gap, 0.005);
    const push = (g.kAvoid * (1.0 / gap - 1.0 / g.avoidRange)) / gap ** 2;
    // up × away: horizontal and perpendicular to "away"; flipped to the side of the goal.
    let side: Vec3 = [away[2], 0.0, -away[0]];
    if (side[0] * goal[0] + side[1] * goal[1] + side[2] * goal[2] < 0.0) side = [-side[0], -side[1], -side[2]];
    for (let i = 0; i < 3; i++) total[i] += push * (away[i] + g.kSlide * side[i]);
  }
  const n = norm(total);
  if (n > g.aMax) for (let i = 0; i < 3; i++) total[i] *= g.aMax / n;
  return total;
}
