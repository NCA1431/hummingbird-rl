/**
 * LQR with integral action: the optimal linear controller around hover, in the browser.
 *
 * Port of the control law of hummingbird_rl/controllers/lqr.py. The gain K is
 * not computed here: Python linearises the simulator, solves the Riccati
 * equation and writes K to lqr_gains.json (`scripts/export_web.py golden`).
 * What remains is u = u0 − K x, with x the 11-number state read from the
 * observation (position error capped at 15 cm, minus the obstacle "carrot")
 * and the 3 integrals of the position error.
 */

import { type BirdParams, DEFAULT_PARAMS, type Vec3 } from "../sim/params";
import { type Controller, clip, norm, parseObs } from "./controller";
import gains from "./lqr_gains.json";
import { repulsion } from "./potentialField";

/** v, shortened to `length` if it is longer. */
function cap(v: Vec3, length: number): Vec3 {
  const n = norm(v);
  return n > length ? (v.map((x) => x * (length / n)) as Vec3) : v;
}

export class LQRController implements Controller {
  readonly name = "lqr";
  private readonly p: BirdParams;
  private z: Vec3 = [0, 0, 0]; // m·s, integral of the position error

  constructor(params: BirdParams = DEFAULT_PARAMS) {
    this.p = params;
  }

  reset(): void {
    this.z = [0, 0, 0];
  }

  act(obs: ArrayLike<number>): number[] {
    const o = parseObs(obs);
    const g = gains;
    const error: Vec3 = o.targets.length ? (o.targets[0].map((x) => -x) as Vec3) : [0, 0, 0]; // bird minus target
    if (norm(error) < g.integral_radius) {
      this.z = this.z.map((z, i) => clip(z + g.dt * error[i], -g.integral_max, g.integral_max)) as Vec3;
    }
    // The carrot: the field's push moves the (capped) target by at most 15 cm.
    const shift = repulsion(o, this.p.r_bird).map((a) => a / g.carrot_gain) as Vec3;
    const errCapped = cap(error, g.saturation);
    const shiftCapped = cap(shift, g.saturation);
    const x = [
      ...[0, 1, 2].map((i) => errCapped[i] - shiftCapped[i]),
      ...o.vel,
      o.roll,
      o.pitch,
      ...o.rates,
      ...this.z,
    ];
    return g.u0.map((u, j) => {
      let kx = 0.0;
      for (let i = 0; i < x.length; i++) kx += g.K[j][i] * x[i];
      return clip(u - kx, -1.0, 1.0);
    });
  }
}
