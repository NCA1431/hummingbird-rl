/**
 * What the page draws of the air: the colours of the smoke, and tracer particles in a laser sheet.
 *
 * Kept apart from the drawing itself (view/air.ts) because it runs in a worker, next to the
 * solver: the browser's main thread only receives the finished pictures.
 *
 * The palettes are those of the fluid simulation this solver comes from: Viridis for speeds,
 * RdBu for signed quantities (blue below zero, white at zero, red above) and Inferno for
 * magnitudes like the spin.
 */

import { type Fluid, type FluidView, H, IX, N, SIZE } from "./fluid";
import type { Vec3 } from "./params";

/** N·s/m: drag coefficient that turns the wind's push into an air speed (the tail-first value, mid-range of the four). */
export const C_AIR = 0.015;

/** The air speed (m/s) that gives the push `force` (N) through a linear drag `c`: the simulator's wind is a force. */
export function windAirSpeed(force: Vec3, c = C_AIR): Vec3 {
  return [force[0] / c, force[1] / c, force[2] / c];
}

// --- Palettes ---------------------------------------------------------------------------

const clamp01 = (t: number) => (t < 0 ? 0 : t > 1 ? 1 : t);
/** Lifts the dark low end of Inferno, which would hide faint smoke. */
export const gammaUp = (t: number, g = 0.55) => clamp01(t) ** g;

/** Viridis (Matplotlib), the published polynomial fit. */
export function viridis(t: number): Vec3 {
  t = clamp01(t);
  const r = 0.2777273272 + t * (0.1050930431 + t * (-0.3308618287 + t * (-4.634230499 + t * (6.2282699363 + t * (4.7763849977 + t * -5.4354558559)))));
  const g = 0.0054073445 + t * (1.4046135299 + t * (0.2148475595 + t * (-5.7991009734 + t * (14.1799333668 + t * (-13.7451453777 + t * 4.6458526122)))));
  const b = 0.3340998053 + t * (1.3845901626 + t * (0.095095163 + t * (-19.3324409563 + t * (56.6905526007 + t * (-65.3530326334 + t * 26.3124352496)))));
  return [clamp01(r), clamp01(g), clamp01(b)];
}

function stops(table: number[][], t: number): Vec3 {
  const u = t * (table.length - 1);
  const i = Math.min(Math.max(Math.floor(u), 0), table.length - 2);
  const f = u - i;
  return [0, 1, 2].map((c) => table[i][c] + f * (table[i + 1][c] - table[i][c])) as Vec3;
}

const RDBU = [
  [0.019, 0.188, 0.38], [0.129, 0.4, 0.674], [0.263, 0.576, 0.765], [0.573, 0.773, 0.871], [0.82, 0.898, 0.941],
  [0.969, 0.969, 0.969], [0.992, 0.859, 0.78], [0.957, 0.647, 0.51], [0.839, 0.376, 0.302], [0.698, 0.094, 0.169],
  [0.404, 0.0, 0.122],
];
const INFERNO = [
  [0.0, 0.0, 0.016], [0.106, 0.047, 0.255], [0.29, 0.047, 0.42], [0.471, 0.11, 0.428], [0.647, 0.173, 0.376],
  [0.812, 0.267, 0.275], [0.929, 0.412, 0.145], [0.984, 0.608, 0.024], [0.988, 1.0, 0.643],
];

/** RdBu (ColorBrewer), for a signed value in [-1, 1]: blue below, white at 0, red above. */
export const rdbu = (t: number): Vec3 => stops(RDBU, (Math.min(Math.max(t, -1), 1) + 1) / 2);
/** Inferno (Matplotlib), for a magnitude in [0, 1]. */
export const inferno = (t: number): Vec3 => stops(INFERNO, clamp01(t));

/** The CSS gradient of a view's palette, for the key ("" for the views without one). */
export function paletteCss(view: FluidView): string {
  const f =
    view === "speed" || view === "tubes"
      ? viridis
      : view === "pressure"
        ? (t: number) => rdbu(2 * t - 1)
        : view === "spin"
          ? (t: number) => inferno(gammaUp(t))
          : null;
  if (!f) return "";
  const css = (c: Vec3) => `rgb(${c.map((x) => Math.round(255 * x)).join(",")})`;
  return `linear-gradient(to right, ${Array.from({ length: 9 }, (_, i) => css(f(i / 8))).join(", ")})`;
}

// --- The 3D texture ---------------------------------------------------------------------

/**
 * The air into an RGBA texture of N³ texels (one per cell). For the smoke views the opacity is
 * the smoke's, the colour says what the view shows; for the vortex tubes the opacity channel
 * holds Q (scaled to its current maximum), which the shader turns into surfaces, and the colour
 * is the air's speed there.
 */
export function paint(f: Fluid, view: FluidView, center: Vec3, out: Uint8Array): void {
  const s = f.scales;
  let o = 0;
  for (let k = 1; k <= N; k++)
    for (let j = 1; j <= N; j++)
      for (let i = 1; i <= N; i++, o += 4) {
        const n = IX(i, j, k);
        if (view === "tubes") {
          const q = f.q[n] / s.q;
          if (q <= 0) {
            out[o + 3] = 0;
            continue;
          }
          const c = viridis(f.speed(n) / s.speed);
          out[o] = 255 * c[0];
          out[o + 1] = 255 * c[1];
          out[o + 2] = 255 * c[2];
          out[o + 3] = 255 * Math.min(q, 1);
          continue;
        }
        const d = f.dye[n];
        if (d < 0.004 || view === "sheet") {
          out[o + 3] = 0;
          continue;
        }
        let c: Vec3;
        if (view === "speed") c = viridis(f.speed(n) / s.speed);
        else if (view === "pressure") c = rdbu(f.pressure(n) / s.pressure);
        else if (view === "spin") c = inferno(gammaUp(f.vorticity(n) / s.vortices));
        else {
          // Tinted by direction from the followed bird, so that the smoke of each side can be
          // told apart; darker than pastels, for a light sky.
          const cc = f.cellCentre(i, j, k);
          const dx = cc[0] - center[0];
          const dy = cc[1] - center[1];
          const dz = cc[2] - center[2];
          const l = Math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6;
          c = [0.42 + 0.3 * (dx / l), 0.4 + 0.25 * (dy / l), 0.6 + 0.3 * (dz / l)];
        }
        out[o] = 255 * c[0];
        out[o + 1] = 255 * c[1];
        out[o + 2] = 255 * c[2];
        out[o + 3] = 255 * Math.min(d, 1);
      }
}

// --- Tracers in a laser sheet -----------------------------------------------------------

/**
 * The horizontal direction the laser sheet faces, from the camera's orientation alone: where
 * the camera looks, or, when it looks straight down (the Top camera), the top of the screen.
 * We first took the direction from the camera to the bird; seen from above, that direction is
 * only a few millimetres long and turns with every small move of the bird, so the sheet (a
 * line from up there) spun around, faster at higher speeds. `forward` and `up`: the camera's
 * viewing and up directions, in world coordinates.
 */
export function sheetFacing(forward: Vec3, up: Vec3): Vec3 {
  // Looking down, the top of the screen points the way the camera faces; looking up, the other
  // way. Adding it to the view direction gives the facing at any tilt, never a zero vector.
  const s = forward[1] < 0 ? 1 : -1;
  const x = forward[0] + s * up[0];
  const z = forward[2] + s * up[2];
  const l = Math.hypot(x, z);
  return l < 1e-9 ? [0, 0, 1] : [x / l, 0, z / l];
}

export const TRACERS = 12000;
const SHEET_HALF = 0.006; // m, half the thickness of the sheet
const LIFE = 1.5; // s of simulated time, then a tracer is seeded again
const STREAK = 0.015; // s: each tracer is drawn as its path over the last 15 ms (a camera's exposure)

/**
 * Tracer particles in a thin vertical sheet through the bird, facing the camera: what an
 * experiment shows when it seeds the air with fog and lights one slice of it with a laser.
 * Each tracer is carried by the simulated air (second-order Runge-Kutta in the interpolated
 * velocity) and drawn as a short streak along its motion; tracers that leave the sheet, the box
 * or their life are seeded again at random in the sheet, so its density stays even.
 */
export class Tracers {
  readonly pos = new Float64Array(TRACERS * 3);
  private readonly vel = new Float64Array(TRACERS * 3);
  private readonly age = new Float64Array(TRACERS);
  readonly center: Vec3 = [0, 0, 0];
  readonly normal: Vec3 = [0, 0, 1];
  private seeded = false;

  /**
   * The sheet goes through `center`, facing `eye` (its normal is the horizontal direction from
   * the eye). It follows both smoothly, as if the laser and the camera followed the bird in a lab:
   * when the bird moves across the sheet the tracers are carried across with it, and when the
   * camera turns around the bird they turn with the sheet, about the vertical through the bird.
   * Only their motion within the sheet is the air's. A sudden turn (a new camera) reseeds it.
   */
  aim(center: Vec3, eye: Vec3, f: Fluid): void {
    let nx = center[0] - eye[0];
    let nz = center[2] - eye[2];
    const l = Math.hypot(nx, nz) || 1;
    nx /= l;
    nz /= l;
    const [ox, , oz] = this.normal;
    const jump = nx * ox + nz * oz < Math.cos((60 * Math.PI) / 180);
    if (this.seeded && !jump) {
      // Across: the bird's move along the old normal, given to every tracer.
      const across = (center[0] - this.center[0]) * ox + (center[2] - this.center[2]) * oz;
      // The turn from the old normal to the new one, as a unit complex number in the (x, z) plane.
      const re = nx * ox + nz * oz;
      const im = nz * ox - nx * oz;
      for (let p = 0; p < TRACERS; p++) {
        const k = 3 * p;
        const dx = this.pos[k] + across * ox - center[0];
        const dz = this.pos[k + 2] + across * oz - center[2];
        this.pos[k] = center[0] + dx * re - dz * im;
        this.pos[k + 2] = center[2] + dx * im + dz * re;
      }
    }
    this.normal[0] = nx;
    this.normal[1] = 0;
    this.normal[2] = nz;
    this.center[0] = center[0];
    this.center[1] = center[1];
    this.center[2] = center[2];
    if (!this.seeded || jump) {
      for (let p = 0; p < TRACERS; p++) this.seed(p, f);
      this.seeded = true;
    }
  }

  reset(): void {
    this.seeded = false;
  }

  /** A tracer at a random place of the sheet, within the height of the box. */
  private seed(p: number, f: Fluid): void {
    const [nx, , nz] = this.normal;
    // In-plane axes: horizontal (normal x up) and vertical.
    const ex = -nz;
    const ez = nx;
    const a = (Math.random() - 0.5) * SIZE;
    const b = (Math.random() - 0.5) * SIZE;
    const c = (Math.random() * 2 - 1) * SHEET_HALF;
    const k = 3 * p;
    this.pos[k] = this.center[0] + a * ex + c * nx;
    this.pos[k + 1] = f.origin[1] + SIZE / 2 + b;
    this.pos[k + 2] = this.center[2] + a * ez + c * nz;
    this.vel[k] = this.vel[k + 1] = this.vel[k + 2] = 0;
    this.age[p] = Math.random() * LIFE;
  }

  update(f: Fluid, dt: number): void {
    const v1: Vec3 = [0, 0, 0];
    const v2: Vec3 = [0, 0, 0];
    const mid: Vec3 = [0, 0, 0];
    const [nx, , nz] = this.normal;
    for (let p = 0; p < TRACERS; p++) {
      const k = 3 * p;
      const x: Vec3 = [this.pos[k], this.pos[k + 1], this.pos[k + 2]];
      f.velocityAt(x, v1);
      if (Number.isNaN(v1[0])) {
        this.seed(p, f);
        continue;
      }
      // Midpoint rule: the velocity half a step ahead gives the step.
      mid[0] = x[0] + 0.5 * dt * v1[0];
      mid[1] = x[1] + 0.5 * dt * v1[1];
      mid[2] = x[2] + 0.5 * dt * v1[2];
      f.velocityAt(mid, v2);
      if (Number.isNaN(v2[0])) {
        this.seed(p, f);
        continue;
      }
      this.pos[k] += dt * v2[0];
      this.pos[k + 1] += dt * v2[1];
      this.pos[k + 2] += dt * v2[2];
      this.vel[k] = v2[0];
      this.vel[k + 1] = v2[1];
      this.vel[k + 2] = v2[2];
      this.age[p] += dt;
      const off = (this.pos[k] - this.center[0]) * nx + (this.pos[k + 2] - this.center[2]) * nz;
      if (Math.abs(off) > 2 * SHEET_HALF || this.age[p] > LIFE || this.pos[k + 1] < f.origin[1] || f.solid[cellOf(f, this.pos, k)]) {
        this.seed(p, f);
      }
    }
  }

  /**
   * Streaks for the drawing: two points per tracer (where it is, and where it was STREAK ago),
   * and their colours, bright green-white and brighter the faster it goes, fading to the tail.
   */
  write(positions: Float32Array, colors: Float32Array, speedScale: number): void {
    for (let p = 0; p < TRACERS; p++) {
      const k = 3 * p;
      const vx = this.vel[k];
      const vy = this.vel[k + 1];
      const vz = this.vel[k + 2];
      const sp = Math.sqrt(vx * vx + vy * vy + vz * vz);
      // A still tracer is still drawn, as a dot 1.5 mm long.
      const len = Math.max(sp * STREAK, 0.0015);
      const ux = sp > 1e-6 ? vx / sp : 0;
      const uy = sp > 1e-6 ? vy / sp : 1;
      const uz = sp > 1e-6 ? vz / sp : 0;
      const o = 6 * p;
      positions[o] = this.pos[k];
      positions[o + 1] = this.pos[k + 1];
      positions[o + 2] = this.pos[k + 2];
      positions[o + 3] = this.pos[k] - ux * len;
      positions[o + 4] = this.pos[k + 1] - uy * len;
      positions[o + 5] = this.pos[k + 2] - uz * len;
      const b = 0.45 + 0.55 * Math.min(sp / speedScale, 1);
      const fade = Math.min(this.age[p] / 0.1, 1, (LIFE - this.age[p]) / 0.2); // no popping in or out
      const c = 8 * p;
      colors[c] = 0.55 * b;
      colors[c + 1] = b;
      colors[c + 2] = 0.6 * b;
      colors[c + 3] = 0.9 * fade;
      colors[c + 4] = 0.55 * b;
      colors[c + 5] = b;
      colors[c + 6] = 0.6 * b;
      colors[c + 7] = 0;
    }
  }
}

/** The cell index of the point stored at pos[k..k+2]. */
function cellOf(f: Fluid, pos: Float64Array, k: number): number {
  const i = Math.min(Math.max(Math.floor((pos[k] - f.origin[0]) / H) + 1, 0), N + 1);
  const j = Math.min(Math.max(Math.floor((pos[k + 1] - f.origin[1]) / H) + 1, 0), N + 1);
  const l = Math.min(Math.max(Math.floor((pos[k + 2] - f.origin[2]) / H) + 1, 0), N + 1);
  return IX(i, j, l);
}
