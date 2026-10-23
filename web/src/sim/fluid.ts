/**
 * The air around the followed bird: a 3D smoke simulation by Jos Stam's "Stable Fluids"
 * (1999), ported from our C++ fluid simulation (fluid_grid.h) and scaled down to a hummingbird.
 *
 * The method, as in the C++ version: the air is a velocity field on a regular grid, advanced
 * at each step by
 *   1. forces: the birds push the air (below), and the air relaxes towards the wind;
 *   2. vorticity confinement (Fedkiw, Stam & Jensen 2001): a coarse grid smooths vortices out,
 *      so we give back to each one a little of the spin it loses;
 *   3. pressure projection: we solve a Poisson equation for the pressure and subtract its
 *      gradient, so that the air stays incompressible (what flows into a cell flows out);
 *   4. semi-Lagrangian advection: each cell takes the value found one step upstream (we follow
 *      the velocity backwards and interpolate), which is stable for any time step;
 * The C++ code also projects once before the advection; we project once, after it, and start
 * each pressure solve from the previous pressure instead of zero (it changes little from one
 * step to the next), which converges as well for half the cost: the browser must do all this
 * between two frames.
 * The smoke is a passive dye carried the same way: it only shows where the air goes.
 *
 * What changes from the C++ version:
 *   - SI units: metres, seconds, m/s, and pressures in pascals; the grid is a box of 40 cm
 *     (32³ cells of 1.25 cm) that follows the bird, the bird two thirds of the way up, so that
 *     its jet below is in view. When the box moves, the fields are shifted cell by cell, as
 *     the C++ window does, and the new cells are filled with the wind;
 *   - the bird as a source of momentum. The simulator gives its thrust T; the wings push the air
 *     the other way with the same force (Newton's third law), spread over the two sectors they
 *     sweep, each wing in proportion to its own force. The force pulses twice per wingbeat
 *     (downstroke and upstroke both lift, as in real hummingbirds), so in slow motion the jet
 *     comes in puffs that roll up into vortex rings by themselves;
 *   - open sides: the box is a window on the air, not a closed tank, so values are copied
 *     across its sides instead of reflected, except at the ground, which the jet hits and
 *     spreads along. Obstacles are solid cells.
 * It stays a one-way coupling: the air does not act back on the simulated bird, whose flight is
 * the one the controllers flew. And at 1.25 cm per cell the wings themselves are far below the
 * resolution: we see the jet and its large eddies, not the flow around a feather.
 */

import type { Vec3 } from "./params";

export const N = 40; // cells per side, inside
export const SIZE = 0.4; // m, side of the box
export const H = SIZE / N; // m, cell size
const G = N + 2; // with one layer of ghost cells on each side
const CELLS = G * G * G;
const RHO = 1.225; // kg/m³
const ITER = 10; // Gauss-Seidel sweeps per pressure solve (warm-started from the last one)
const VORT_EPS = 3.0; // vorticity confinement strength (Fedkiw et al.'s epsilon)
const WIND_TAU = 0.25; // s: the air in the box relaxes towards the wind in about this time
const DYE_TAU = 1.2; // s: lifetime of the smoke
const DYE_RATE = 24.0; // smoke released per second in the cells the wings push
const SHELL_RATE = 4.0; // smoke released per second on a shell around the bird (twice as much on top)
const BIRD_HEIGHT = 0.68; // where the followed bird sits in the box, as a fraction of its height

export const IX = (i: number, j: number, k: number): number => i + G * (j + G * k);

/** One bird, as the fluid needs it. */
export interface FluidBird {
  pos: Vec3;
  up: Vec3; // thrust direction (body up axis), unit
  fwd: Vec3; // body forward axis (towards the beak), unit
  right: Vec3; // body right axis (along the right wing), unit
  thrust: number; // N, total
  ampLeft: number; // rad, half-stroke amplitudes
  ampRight: number;
  damage: [number, number]; // force lost by each wing, 0..1
  phase: number; // rad, wingbeat phase
  freq: number; // Hz
  span: number; // m, wingtip to wingtip
  root: number; // m, wing root distance from the body axis
}

/**
 * How the air is shown: the smoke (alone, or coloured by the speed, the pressure or the spin of
 * the air it is in), tracer particles in a laser sheet, or the surfaces wrapping the vortices.
 */
export type FluidView = "smoke" | "sheet" | "tubes" | "speed" | "pressure" | "spin";

/**
 * The adaptive colour scales, as in the C++ code: each follows the largest value seen in the
 * smoke, smoothed over a few steps, so the palette uses its whole range.
 */
export interface FluidScales {
  speed: number; // m/s
  pressure: number; // Pa
  vortices: number; // 1/s, spin (vorticity magnitude)
  q: number; // 1/s², Q-criterion
}

export class Fluid {
  readonly u = new Float64Array(CELLS);
  readonly v = new Float64Array(CELLS);
  readonly w = new Float64Array(CELLS);
  readonly dye = new Float64Array(CELLS);
  readonly p = new Float64Array(CELLS); // pressure potential of the last projection, m²/s
  private readonly u0 = new Float64Array(CELLS);
  private readonly v0 = new Float64Array(CELLS);
  private readonly w0 = new Float64Array(CELLS);
  private readonly d0 = new Float64Array(CELLS);
  private readonly div = new Float64Array(CELLS);
  private readonly cx = new Float64Array(CELLS);
  private readonly cy = new Float64Array(CELLS);
  private readonly cz = new Float64Array(CELLS);
  private readonly mag = new Float64Array(CELLS);
  readonly solid = new Uint8Array(CELLS);
  /** World position of the lower corner of the box (cell 1, 1, 1 starts there). */
  readonly origin: Vec3 = [0, 0, 0];
  readonly scales: FluidScales = { speed: 0.5, pressure: 0.05, vortices: 20, q: 100 };
  /** Q-criterion of each cell (1/s²): positive where rotation beats shear, i.e. inside a vortex. */
  readonly q = new Float64Array(CELLS);
  private lastDt = 1 / 60;
  private started = false;

  /** World position of the centre of cell (i, j, k). */
  cellCentre(i: number, j: number, k: number): Vec3 {
    return [this.origin[0] + (i - 0.5) * H, this.origin[1] + (j - 0.5) * H, this.origin[2] + (k - 0.5) * H];
  }

  /** Is the ground the bottom of the box? Then it is a wall, not an open side. */
  get onGround(): boolean {
    return this.origin[1] <= 1e-9;
  }

  /** Empties the box around `center`, filled with the wind's air speed. */
  reset(center: Vec3, wind: Vec3): void {
    this.u.fill(wind[0]);
    this.v.fill(wind[1]);
    this.w.fill(wind[2]);
    this.dye.fill(0);
    this.p.fill(0);
    const o = Fluid.originFor(center);
    this.origin[0] = o[0];
    this.origin[1] = o[1];
    this.origin[2] = o[2];
    this.started = true;
  }

  /** Forgets the air: the next `follow` starts again from still air around the bird. */
  stop(): void {
    this.started = false;
  }

  /** Where the box should start for a bird at `center`: the bird two thirds up, the box above the ground. */
  static originFor(center: Vec3): Vec3 {
    return [center[0] - SIZE / 2, Math.max(0, center[1] - BIRD_HEIGHT * SIZE), center[2] - SIZE / 2];
  }

  /**
   * Moves the box with the bird, by whole cells and only once it is two cells off (so the smoke
   * does not shimmer at every frame). The fields move with it; the new cells get the wind.
   */
  follow(center: Vec3, wind: Vec3): void {
    if (!this.started) {
      this.reset(center, wind);
      return;
    }
    const want = Fluid.originFor(center);
    const shift = [0, 1, 2].map((a) => {
      const cells = Math.round((want[a] - this.origin[a]) / H);
      return Math.abs(cells) >= 2 ? cells : 0;
    });
    // The box never goes below the ground: its origin is 0 at most.
    if (this.origin[1] + shift[1] * H < 0) shift[1] = -Math.round(this.origin[1] / H);
    if (shift.every((s) => s === 0)) return;
    if (shift.some((s) => Math.abs(s) >= N)) {
      this.reset(center, wind); // a jump (a new flight, a seek): start again
      return;
    }
    this.shift(this.u, shift, wind[0]);
    this.shift(this.v, shift, wind[1]);
    this.shift(this.w, shift, wind[2]);
    this.shift(this.dye, shift, 0);
    this.shift(this.p, shift, 0);
    for (let a = 0; a < 3; a++) this.origin[a] += shift[a] * H;
    if (this.origin[1] < 1e-9) this.origin[1] = 0;
  }

  /** f moved by `s` cells: cell (i, j, k) takes the value of cell (i + s), `fill` where there is none. */
  private shift(f: Float64Array, s: number[], fill: number): void {
    const t = this.d0;
    t.fill(fill);
    for (let k = 0; k < G; k++) {
      const sk = k + s[2];
      if (sk < 0 || sk >= G) continue;
      for (let j = 0; j < G; j++) {
        const sj = j + s[1];
        if (sj < 0 || sj >= G) continue;
        for (let i = 0; i < G; i++) {
          const si = i + s[0];
          if (si < 0 || si >= G) continue;
          t[IX(i, j, k)] = f[IX(si, sj, sk)];
        }
      }
    }
    f.set(t);
  }

  /** Marks the cells inside the obstacles as solid (to do after every move of the box). */
  setObstacles(obstacles: { center: Vec3; radius: number }[]): void {
    this.solid.fill(0);
    for (const o of obstacles) {
      const r2 = o.radius * o.radius;
      const lo = (a: number) => Math.max(1, Math.floor((o.center[a] - o.radius - this.origin[a]) / H) + 1);
      const hi = (a: number) => Math.min(N, Math.floor((o.center[a] + o.radius - this.origin[a]) / H) + 1);
      for (let k = lo(2); k <= hi(2); k++)
        for (let j = lo(1); j <= hi(1); j++)
          for (let i = lo(0); i <= hi(0); i++) {
            const c = this.cellCentre(i, j, k);
            const d2 = (c[0] - o.center[0]) ** 2 + (c[1] - o.center[1]) ** 2 + (c[2] - o.center[2]) ** 2;
            if (d2 <= r2) this.solid[IX(i, j, k)] = 1;
          }
    }
  }

  /**
   * The push of one bird's wings on the air during `dt`, and the smoke released around it.
   * Returns the momentum given to the air along the jet (kg·m/s), which the tests compare with T dt.
   *
   * Each wing's share of the thrust goes as the square of its amplitude times what is left of it
   * after damage (the frequency is shared). How it is spread depends on the step:
   *   - a step shorter than a quarter of a wingbeat (slow motion) resolves the beat: the force
   *     acts where the wing is now, a strip from root to tip at its stroke angle -A cos(phase)
   *     (the drawn wings use the same), as strong as the wing is fast: sin² of the phase, twice
   *     per beat since both strokes lift, and more towards the tip, where the wing moves fastest
   *     (the force on a slice of wing goes as its speed squared, so as r²). The wing also drags
   *     the air along its stroke, forward then back, which averages out over a beat;
   *   - a longer step would only alias the beat: the mean force is spread over the two sectors
   *     the wings sweep, straight down.
   */
  addBird(b: FluidBird, dt: number): number {
    const tip = b.span / 2;
    const reach = tip + 2 * H;
    const lo = (a: number) => Math.max(1, Math.floor((b.pos[a] - reach - this.origin[a]) / H) + 1);
    const hi = (a: number) => Math.min(N, Math.floor((b.pos[a] + reach - this.origin[a]) / H) + 1);
    const fl = b.ampLeft * b.ampLeft * (1 - b.damage[0]);
    const fr = b.ampRight * b.ampRight * (1 - b.damage[1]);
    const total = fl + fr;
    if (b.thrust <= 0 || total <= 0) return 0;
    const resolved = b.freq * dt <= 0.25;
    const pulse = resolved ? 2 * Math.sin(b.phase) ** 2 : 1;
    const theta = [-b.ampLeft * Math.cos(b.phase), -b.ampRight * Math.cos(b.phase)]; // stroke angles, + = forward
    const sweep = Math.sign(Math.sin(b.phase)); // dtheta/dt goes as sin(phase): forward in the downstroke

    // First pass: each cell's weight for each wing, in a layer of the stroke plane one cell thick.
    const cells: number[] = [];
    const weights: number[] = [];
    const sides: number[] = [];
    let wl = 0;
    let wr = 0;
    for (let k = lo(2); k <= hi(2); k++)
      for (let j = lo(1); j <= hi(1); j++)
        for (let i = lo(0); i <= hi(0); i++) {
          const n = IX(i, j, k);
          if (this.solid[n]) continue;
          const c = this.cellCentre(i, j, k);
          const r: Vec3 = [c[0] - b.pos[0], c[1] - b.pos[1], c[2] - b.pos[2]];
          const along = r[0] * b.up[0] + r[1] * b.up[1] + r[2] * b.up[2];
          const thick = 1 - Math.abs(along) / H;
          if (thick <= 0) continue;
          const x = r[0] * b.fwd[0] + r[1] * b.fwd[1] + r[2] * b.fwd[2];
          const z = r[0] * b.right[0] + r[1] * b.right[1] + r[2] * b.right[2];
          const rad = Math.sqrt(x * x + z * z);
          if (rad < b.root || rad > tip + 0.5 * H) continue;
          const side = z >= 0 ? 1 : 0;
          const edge = Math.min(1, (tip + 0.5 * H - rad) / H); // soft tip
          let wgt: number;
          if (resolved) {
            // Distance from the cell to the wing's line, in the stroke plane.
            const th = theta[side];
            const dirX = Math.sin(th);
            const dirZ = (side ? 1 : -1) * Math.cos(th);
            const proj = x * dirX + z * dirZ;
            if (proj <= 0) continue;
            const perp = Math.abs(x * dirZ - z * dirX);
            const near = 1 - perp / (1.2 * H); // the chord, about 1.5 cm: a cell or so
            if (near <= 0) continue;
            wgt = thick * near * edge * (rad / tip) ** 2;
          } else {
            const amp = side ? b.ampRight : b.ampLeft;
            if (Math.atan2(Math.abs(x), Math.abs(z)) > amp) continue; // inside the sector, ±amplitude
            wgt = thick * edge;
          }
          cells.push(n);
          weights.push(wgt);
          sides.push(side);
          if (side) wr += wgt;
          else wl += wgt;
        }

    // Second pass: each wing's force spread over its cells. A force F on a volume V of air of
    // density rho changes its velocity by F dt / (rho V); the jet goes against the thrust.
    let momentum = 0;
    const vol = H * H * H;
    const DRAG = 0.4; // stroke-wise push on the air, per unit of lift (hovering wings: lift/drag about 2.5)
    for (let c = 0; c < cells.length; c++) {
      const n = cells[c];
      const side = sides[c];
      const wSide = side ? wr : wl;
      if (wSide <= 0) continue;
      const force = b.thrust * pulse * ((side ? fr : fl) / total) * (weights[c] / wSide); // N on this cell
      const dv = (force * dt) / (RHO * vol);
      this.u[n] -= dv * b.up[0];
      this.v[n] -= dv * b.up[1];
      this.w[n] -= dv * b.up[2];
      if (resolved) {
        // Along the stroke: the direction of increasing stroke angle, cos(th) fwd - s sin(th) right.
        const th = theta[side];
        const s = side ? 1 : -1;
        const h = DRAG * dv * sweep;
        for (let a = 0; a < 3; a++) {
          const t = Math.cos(th) * b.fwd[a] - s * Math.sin(th) * b.right[a];
          if (a === 0) this.u[n] += h * t;
          else if (a === 1) this.v[n] += h * t;
          else this.w[n] += h * t;
        }
      }
      momentum += force * dt;
      this.dye[n] = Math.min(this.dye[n] + DYE_RATE * dt * weights[c] * (resolved ? 3 : 1), 1.5);
    }

    // Smoke also comes off a thin shell around the bird, above it as well: the air the wings suck
    // in from above carries it down, so the whole flow through the wings is seen, not only below.
    const shell = 0.035;
    for (let k = lo(2); k <= hi(2); k++)
      for (let j = lo(1); j <= hi(1); j++)
        for (let i = lo(0); i <= hi(0); i++) {
          const n = IX(i, j, k);
          if (this.solid[n]) continue;
          const c = this.cellCentre(i, j, k);
          const d = Math.sqrt((c[0] - b.pos[0]) ** 2 + (c[1] - b.pos[1]) ** 2 + (c[2] - b.pos[2]) ** 2);
          const e = (d - shell) / H;
          if (e * e > 4) continue;
          // More above the bird, where the air comes from: up to twice as much at the top.
          const upward = ((c[0] - b.pos[0]) * b.up[0] + (c[1] - b.pos[1]) * b.up[1] + (c[2] - b.pos[2]) * b.up[2]) / (d + 1e-9);
          const top = 1 + Math.max(upward, 0);
          this.dye[n] = Math.min(this.dye[n] + SHELL_RATE * top * dt * Math.exp(-e * e), 1.5);
        }
    return momentum;
  }

  /** The air relaxes towards the wind (as in the C++ code: v += alpha (V_wind - v)). */
  relaxToWind(wind: Vec3, dt: number): void {
    const a = 1 - Math.exp(-dt / WIND_TAU);
    for (let n = 0; n < CELLS; n++) {
      if (this.solid[n]) continue;
      this.u[n] += a * (wind[0] - this.u[n]);
      this.v[n] += a * (wind[1] - this.v[n]);
      this.w[n] += a * (wind[2] - this.w[n]);
    }
  }

  /** One step of the solver, after the forces of this step were added. */
  step(dt: number): void {
    if (dt <= 0) return;
    this.vorticityConfinement(dt);
    this.u0.set(this.u);
    this.v0.set(this.v);
    this.w0.set(this.w);
    this.advect(1, this.u, this.u0, dt);
    this.advect(2, this.v, this.v0, dt);
    this.advect(3, this.w, this.w0, dt);
    // The pressure potential is P dt / rho: rescaled to this step's dt, the last one is a good start.
    if (this.lastDt !== dt) {
      const r = dt / this.lastDt;
      for (let n = 0; n < CELLS; n++) this.p[n] *= r;
    }
    this.project(this.u, this.v, this.w, this.p, this.div);
    this.lastDt = dt;
    this.computeQ();
    this.d0.set(this.dye);
    this.advect(0, this.dye, this.d0, dt);
    const keep = Math.exp(-dt / DYE_TAU);
    for (let n = 0; n < CELLS; n++) this.dye[n] = this.solid[n] ? 0 : this.dye[n] * keep;
    this.measure();
    this.scrubNaN();
  }

  /** Pressure of cell n in pascals: the projection's potential p is P dt / rho. */
  pressure(n: number): number {
    return (RHO * this.p[n]) / this.lastDt;
  }

  speed(n: number): number {
    const { u, v, w } = this;
    return Math.sqrt(u[n] * u[n] + v[n] * v[n] + w[n] * w[n]);
  }

  /** |curl u| of cell n, 1/s (computed during the last step). */
  vorticity(n: number): number {
    return this.mag[n];
  }

  /**
   * Boundary values. The sides of the box are open: a ghost cell copies its neighbour. Solid
   * cells hold still air. The ground, when it is the bottom of the box, is a wall: the vertical
   * velocity is reflected (no air through it), the horizontal ones slide along it.
   */
  private bnd(bc: number, x: Float64Array): void {
    const floor = this.onGround;
    for (let k = 1; k <= N; k++)
      for (let j = 1; j <= N; j++) {
        x[IX(0, j, k)] = x[IX(1, j, k)];
        x[IX(N + 1, j, k)] = x[IX(N, j, k)];
        x[IX(j, k, 0)] = x[IX(j, k, 1)];
        x[IX(j, k, N + 1)] = x[IX(j, k, N)];
        x[IX(j, N + 1, k)] = x[IX(j, N, k)];
        x[IX(j, 0, k)] = floor && bc === 2 ? -x[IX(j, 1, k)] : x[IX(j, 1, k)];
      }
    if (bc !== 0) {
      for (let n = 0; n < CELLS; n++) if (this.solid[n]) x[n] = 0;
    }
  }

  /** Gauss-Seidel sweeps for x = (x0 + a * sum of the six neighbours) / c. */
  private linSolve(bc: number, x: Float64Array, x0: Float64Array, a: number, c: number): void {
    const inv = 1 / c;
    const sx = 1;
    const sy = G;
    const sz = G * G;
    for (let it = 0; it < ITER; it++) {
      for (let k = 1; k <= N; k++)
        for (let j = 1; j <= N; j++) {
          let n = IX(1, j, k);
          for (let i = 1; i <= N; i++, n++) {
            x[n] = (x0[n] + a * (x[n - sx] + x[n + sx] + x[n - sy] + x[n + sy] + x[n - sz] + x[n + sz])) * inv;
          }
        }
      this.bnd(bc, x);
    }
  }

  /** Makes (u, v, w) divergence-free; p receives the pressure potential (m²/s). */
  private project(u: Float64Array, v: Float64Array, w: Float64Array, p: Float64Array, div: Float64Array): void {
    const sy = G;
    const sz = G * G;
    for (let k = 1; k <= N; k++)
      for (let j = 1; j <= N; j++) {
        let n = IX(1, j, k);
        for (let i = 1; i <= N; i++, n++) {
          div[n] = -0.5 * H * (u[n + 1] - u[n - 1] + v[n + sy] - v[n - sy] + w[n + sz] - w[n - sz]);
        }
      }
    this.bnd(0, div);
    this.bnd(0, p);
    this.linSolve(0, p, div, 1, 6);
    for (let k = 1; k <= N; k++)
      for (let j = 1; j <= N; j++) {
        let n = IX(1, j, k);
        for (let i = 1; i <= N; i++, n++) {
          u[n] -= (0.5 * (p[n + 1] - p[n - 1])) / H;
          v[n] -= (0.5 * (p[n + sy] - p[n - sy])) / H;
          w[n] -= (0.5 * (p[n + sz] - p[n - sz])) / H;
        }
      }
    this.bnd(1, u);
    this.bnd(2, v);
    this.bnd(3, w);
  }

  /** d = d0 taken one step upstream, with trilinear interpolation (semi-Lagrangian advection). */
  private advect(bc: number, d: Float64Array, d0: Float64Array, dt: number): void {
    const dt0 = dt / H; // cells travelled per (m/s)
    const u = bc === 0 ? this.u : this.u0;
    const v = bc === 0 ? this.v : this.v0;
    const w = bc === 0 ? this.w : this.w0;
    const GG = G * G;
    const hi = N + 0.5;
    for (let k = 1; k <= N; k++)
      for (let j = 1; j <= N; j++) {
        let n = IX(1, j, k);
        for (let i = 1; i <= N; i++, n++) {
          let x = i - dt0 * u[n];
          let y = j - dt0 * v[n];
          let z = k - dt0 * w[n];
          x = x < 0.5 ? 0.5 : x > hi ? hi : x;
          y = y < 0.5 ? 0.5 : y > hi ? hi : y;
          z = z < 0.5 ? 0.5 : z > hi ? hi : z;
          const ia = x | 0; // x > 0, so truncation is the floor
          const ja = y | 0;
          const ka = z | 0;
          const s1 = x - ia;
          const t1 = y - ja;
          const q1 = z - ka;
          const s0 = 1 - s1;
          const t0 = 1 - t1;
          const q0 = 1 - q1;
          const a = ia + G * (ja + G * ka);
          d[n] =
            q0 * (s0 * (t0 * d0[a] + t1 * d0[a + G]) + s1 * (t0 * d0[a + 1] + t1 * d0[a + 1 + G])) +
            q1 * (s0 * (t0 * d0[a + GG] + t1 * d0[a + G + GG]) + s1 * (t0 * d0[a + 1 + GG] + t1 * d0[a + 1 + G + GG]));
        }
      }
    this.bnd(bc, d);
  }

  /** Vorticity confinement: a force eps h (N x omega), N pointing towards stronger spin. */
  private vorticityConfinement(dt: number): void {
    const { u, v, w, cx, cy, cz, mag } = this;
    const sy = G;
    const sz = G * G;
    const inv2h = 1 / (2 * H);
    for (let k = 1; k <= N; k++)
      for (let j = 1; j <= N; j++) {
        let n = IX(1, j, k);
        for (let i = 1; i <= N; i++, n++) {
          cx[n] = (w[n + sy] - w[n - sy] - (v[n + sz] - v[n - sz])) * inv2h;
          cy[n] = (u[n + sz] - u[n - sz] - (w[n + 1] - w[n - 1])) * inv2h;
          cz[n] = (v[n + 1] - v[n - 1] - (u[n + sy] - u[n - sy])) * inv2h;
          mag[n] = Math.sqrt(cx[n] * cx[n] + cy[n] * cy[n] + cz[n] * cz[n]); // (Math.hypot is several times slower)
        }
      }
    const f = VORT_EPS * H * dt;
    for (let k = 2; k < N; k++)
      for (let j = 2; j < N; j++) {
        let n = IX(2, j, k);
        for (let i = 2; i < N; i++, n++) {
          if (this.solid[n]) continue;
          let nx = 0.5 * (mag[n + 1] - mag[n - 1]);
          let ny = 0.5 * (mag[n + sy] - mag[n - sy]);
          let nz = 0.5 * (mag[n + sz] - mag[n - sz]);
          const len = Math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9;
          nx /= len;
          ny /= len;
          nz /= len;
          u[n] += f * (ny * cz[n] - nz * cy[n]);
          v[n] += f * (nz * cx[n] - nx * cz[n]);
          w[n] += f * (nx * cy[n] - ny * cx[n]);
        }
      }
  }

  /**
   * The Q-criterion (Hunt et al. 1988), Q = (|Omega|² - |S|²) / 2 from the rotation and strain
   * parts of the velocity gradient: positive where the air spins more than it is sheared. Its
   * positive regions are the vortex cores; the surfaces drawn by the "vortex tubes" view are Q = const.
   */
  private computeQ(): void {
    const { u, v, w, q } = this;
    const sy = G;
    const sz = G * G;
    const k2 = 1 / (2 * H);
    let qmax = 0;
    q.fill(0);
    for (let k = 2; k < N; k++)
      for (let j = 2; j < N; j++) {
        let n = IX(2, j, k);
        for (let i = 2; i < N; i++, n++) {
          if (this.solid[n]) continue;
          const ux = (u[n + 1] - u[n - 1]) * k2;
          const uy = (u[n + sy] - u[n - sy]) * k2;
          const uz = (u[n + sz] - u[n - sz]) * k2;
          const vx = (v[n + 1] - v[n - 1]) * k2;
          const vy = (v[n + sy] - v[n - sy]) * k2;
          const vz = (v[n + sz] - v[n - sz]) * k2;
          const wx = (w[n + 1] - w[n - 1]) * k2;
          const wy = (w[n + sy] - w[n - sy]) * k2;
          const wz = (w[n + sz] - w[n - sz]) * k2;
          const sxy = 0.5 * (uy + vx);
          const sxz = 0.5 * (uz + wx);
          const syz = 0.5 * (vz + wy);
          const oxy = 0.5 * (uy - vx);
          const oxz = 0.5 * (uz - wx);
          const oyz = 0.5 * (vz - wy);
          const s2 = ux * ux + vy * vy + wz * wz + 2 * (sxy * sxy + sxz * sxz + syz * syz);
          const o2 = 2 * (oxy * oxy + oxz * oxz + oyz * oyz);
          q[n] = 0.5 * (o2 - s2);
          if (q[n] > qmax) qmax = q[n];
        }
      }
    this.scales.q = Math.max(0.9 * this.scales.q + 0.1 * qmax, 1);
  }

  /** The air velocity at a world point (m/s), interpolated; the wind outside the box. */
  velocityAt(p: Vec3, out: Vec3): Vec3 {
    const x = (p[0] - this.origin[0]) / H + 0.5;
    const y = (p[1] - this.origin[1]) / H + 0.5;
    const z = (p[2] - this.origin[2]) / H + 0.5;
    if (x < 0.5 || y < 0.5 || z < 0.5 || x > N + 0.5 || y > N + 0.5 || z > N + 0.5) {
      out[0] = out[1] = out[2] = Number.NaN;
      return out;
    }
    const ia = x | 0;
    const ja = y | 0;
    const ka = z | 0;
    const s1 = x - ia;
    const t1 = y - ja;
    const q1 = z - ka;
    const a = ia + G * (ja + G * ka);
    const GG = G * G;
    const lerp = (f: Float64Array) =>
      (1 - q1) * ((1 - s1) * ((1 - t1) * f[a] + t1 * f[a + G]) + s1 * ((1 - t1) * f[a + 1] + t1 * f[a + 1 + G])) +
      q1 * ((1 - s1) * ((1 - t1) * f[a + GG] + t1 * f[a + G + GG]) + s1 * ((1 - t1) * f[a + 1 + GG] + t1 * f[a + 1 + G + GG]));
    out[0] = lerp(this.u);
    out[1] = lerp(this.v);
    out[2] = lerp(this.w);
    return out;
  }

  /** Follows the largest speed, pressure and spin where there is smoke (smoothed, as in the C++ code). */
  private measure(): void {
    let sp = 0;
    let pr = 0;
    let vo = 0;
    for (let k = 1; k <= N; k++)
      for (let j = 1; j <= N; j++) {
        let n = IX(1, j, k);
        for (let i = 1; i <= N; i++, n++) {
          if (this.dye[n] < 0.03) continue;
          sp = Math.max(sp, this.speed(n));
          pr = Math.max(pr, Math.abs(this.pressure(n)));
          vo = Math.max(vo, this.mag[n]);
        }
      }
    const s = this.scales;
    s.speed = Math.max(0.9 * s.speed + 0.1 * sp, 0.05);
    s.pressure = Math.max(0.9 * s.pressure + 0.1 * pr, 1e-3);
    s.vortices = Math.max(0.9 * s.vortices + 0.1 * vo, 1);
  }

  /** A NaN would spread to the whole box through the advection: start the air again instead. */
  private scrubNaN(): void {
    let acc = 0;
    for (let n = 0; n < CELLS; n++) acc += this.u[n] + this.v[n] + this.w[n];
    if (Number.isFinite(acc)) return;
    this.u.fill(0);
    this.v.fill(0);
    this.w.fill(0);
    this.p.fill(0);
  }

  /** Mean divergence magnitude (1/s) inside the box, for the tests. */
  meanDivergence(): number {
    let s = 0;
    for (let k = 2; k < N; k++)
      for (let j = 2; j < N; j++)
        for (let i = 2; i < N; i++) {
          const n = IX(i, j, k);
          s += Math.abs((this.u[n + 1] - this.u[n - 1] + this.v[n + G] - this.v[n - G] + this.w[n + G * G] - this.w[n - G * G]) / (2 * H));
        }
    return s / (N - 2) ** 3;
  }

  /** Runs one projection on the current velocity (for the tests). */
  projectNow(): void {
    this.project(this.u, this.v, this.w, this.p, this.div);
  }
}
