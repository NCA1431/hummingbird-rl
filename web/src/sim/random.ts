/**
 * A small seeded random generator for the browser (gusts, sensor and actuator noise).
 *
 * JavaScript's Math.random cannot be seeded, and we want a sandbox run to be
 * repeatable. It does not reproduce NumPy's draws: the parity tests never
 * need the same random numbers, because they feed recorded inputs.
 */

export class Rng {
  private state: number;
  private spare: number | null = null;

  constructor(seed = 1) {
    this.state = seed >>> 0;
  }

  /** Uniform in [0, 1): the "mulberry32" generator, 32 bits of state, good enough for noise. */
  uniform(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Standard normal, by the Box-Muller transform (which gives two values per draw; we keep the second). */
  normal(): number {
    if (this.spare !== null) {
      const s = this.spare;
      this.spare = null;
      return s;
    }
    const u1 = 1.0 - this.uniform(); // in (0, 1], so the log is finite
    const u2 = this.uniform();
    const r = Math.sqrt(-2.0 * Math.log(u1));
    this.spare = r * Math.sin(2.0 * Math.PI * u2);
    return r * Math.cos(2.0 * Math.PI * u2);
  }
}
