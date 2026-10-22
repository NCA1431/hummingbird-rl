/**
 * Wind gusts: an Ornstein-Uhlenbeck process around a mean wind force.
 *
 * Port of hummingbird_rl/sim/wind.py. The random draws differ from NumPy's
 * (see random.ts), so only the statistics can be compared; a test checks
 * that each horizontal component has standard deviation sigma.
 */

import type { Vec3 } from "./params";
import { Rng } from "./random";

export class GustProcess {
  readonly meanForce: Vec3; // N, mean wind (horizontal)
  readonly sigma: number; // N, size of the horizontal fluctuations
  readonly tau: number; // s, correlation time of a gust
  state: Vec3; // N, current wind force
  private readonly sig: Vec3;
  private readonly rng: Rng;

  constructor(meanForce: Vec3, sigma = 0.0, rng: Rng = new Rng(), tau = 0.5, verticalRatio = 0.3) {
    this.meanForce = [...meanForce] as Vec3;
    this.sigma = sigma;
    this.tau = tau;
    this.rng = rng;
    this.sig = [sigma, verticalRatio * sigma, sigma];
    // Start from the steady-state distribution, so gusts are fully developed at once.
    this.state = [0, 1, 2].map((i) => this.meanForce[i] + this.sig[i] * rng.normal()) as Vec3;
  }

  static calm(): GustProcess {
    return new GustProcess([0.0, 0.0, 0.0], 0.0);
  }

  /** Exact OU update over dt: w ← w̄ + (w - w̄) e^(-dt/τ) + σ sqrt(1 - e^(-2dt/τ)) · noise. */
  step(dt: number): Vec3 {
    if (this.tau <= 0.0) return this.state;
    const decay = Math.exp(-dt / this.tau);
    const noise = Math.sqrt(1.0 - decay * decay);
    this.state = [0, 1, 2].map(
      (i) =>
        this.meanForce[i] + (this.state[i] - this.meanForce[i]) * decay + this.sig[i] * noise * this.rng.normal(),
    ) as Vec3;
    return this.state;
  }
}
