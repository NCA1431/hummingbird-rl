/**
 * The smoke simulation: the solver's invariants, the bird's push on the air, and the palettes.
 */

import { describe, expect, it } from "vitest";

import { Fluid, type FluidBird, H, IX, N } from "../src/sim/fluid";
import { DEFAULT_PARAMS, type Vec3, hoverAmplitude, weight } from "../src/sim/params";
import { C_AIR, TRACERS, Tracers, inferno, rdbu, viridis, windAirSpeed } from "../src/sim/fluidView";

const p = DEFAULT_PARAMS;
const RHO = 1.225;
const at: Vec3 = [0, 0.5, 0];

function hoverBird(changes: Partial<FluidBird> = {}): FluidBird {
  const a = hoverAmplitude(p);
  return {
    pos: [...at], up: [0, 1, 0], fwd: [1, 0, 0], right: [0, 0, 1], thrust: weight(p), ampLeft: a, ampRight: a,
    damage: [0, 0], phase: 0, freq: 40, span: p.span, root: p.wing_root, ...changes,
  };
}

/** Total momentum of the air in the box (kg·m/s), relative to `wind`. */
function momentum(f: Fluid, wind: Vec3 = [0, 0, 0]): Vec3 {
  const m: Vec3 = [0, 0, 0];
  for (let k = 1; k <= N; k++)
    for (let j = 1; j <= N; j++)
      for (let i = 1; i <= N; i++) {
        const n = IX(i, j, k);
        m[0] += RHO * H ** 3 * (f.u[n] - wind[0]);
        m[1] += RHO * H ** 3 * (f.v[n] - wind[1]);
        m[2] += RHO * H ** 3 * (f.w[n] - wind[2]);
      }
  return m;
}

describe("solver", () => {
  it("the projection makes the air incompressible", () => {
    const f = new Fluid();
    f.reset(at, [0, 0, 0]);
    // A field that converges into sheets every 8 cells (the size of the jet): strongly divergent.
    for (let k = 0; k < N + 2; k++)
      for (let j = 0; j < N + 2; j++)
        for (let i = 0; i < N + 2; i++) {
          f.u[IX(i, j, k)] = Math.sin((2 * Math.PI * i) / 8) * Math.cos((2 * Math.PI * j) / 16);
          f.v[IX(i, j, k)] = 0.5 * Math.sin((2 * Math.PI * k) / 8);
        }
    const before = f.meanDivergence();
    for (let r = 0; r < 4; r++) f.projectNow(); // warm-started sweeps add up, as from step to step
    expect(f.meanDivergence()).toBeLessThan(0.1 * before);
  });

  it("a uniform wind crosses the box unchanged", () => {
    const f = new Fluid();
    const wind: Vec3 = [0.6, 0, -0.3];
    f.reset(at, wind);
    for (let s = 0; s < 10; s++) {
      f.relaxToWind(wind, 1 / 60);
      f.step(1 / 60);
    }
    const n = IX(N / 2, N / 2, N / 2);
    expect(f.u[n]).toBeCloseTo(0.6, 9);
    expect(f.w[n]).toBeCloseTo(-0.3, 9);
  });

  it("the box follows the bird by whole cells, and the new air is the wind", () => {
    const f = new Fluid();
    const wind: Vec3 = [0.2, 0, 0];
    f.reset(at, wind);
    const n = IX(10, 10, 10);
    f.dye[n] = 1;
    f.follow([at[0] + 3 * H, at[1], at[2]], wind); // three cells to the right
    expect(f.dye[IX(7, 10, 10)]).toBe(1);
    expect(f.u[IX(N, 10, 10)]).toBe(0.2);
    f.follow([at[0] + 3.4 * H, at[1], at[2]], wind); // less than two cells: it stays
    expect(f.dye[IX(7, 10, 10)]).toBe(1);
  });
});

describe("the bird's push", () => {
  it("gives the air exactly the thrust times the time, downwards (Newton's third law)", () => {
    const f = new Fluid();
    f.reset(at, [0, 0, 0]);
    const dt = 1 / 60; // longer than a quarter of a beat: the mean force, no pulse
    const given = f.addBird(hoverBird(), dt);
    expect(given).toBeCloseTo(weight(p) * dt, 12);
    const m = momentum(f);
    expect(m[1]).toBeCloseTo(-weight(p) * dt, 9);
    expect(Math.abs(m[0]) + Math.abs(m[2])).toBeLessThan(1e-12);
  });

  it("a damaged wing pushes less air than the other", () => {
    const f = new Fluid();
    f.reset(at, [0, 0, 0]);
    f.addBird(hoverBird({ damage: [0.5, 0] }), 1 / 60);
    let left = 0;
    let right = 0;
    for (let k = 1; k <= N; k++)
      for (let j = 1; j <= N; j++)
        for (let i = 1; i <= N; i++) {
          const z = f.cellCentre(i, j, k)[2];
          if (z < 0) left += f.v[IX(i, j, k)];
          else right += f.v[IX(i, j, k)];
        }
    expect(left / right).toBeCloseTo(0.5, 1);
  });

  it("a hovering bird blows a jet of about 2 to 3 m/s below it, as momentum theory says", () => {
    const f = new Fluid();
    f.reset(at, [0, 0, 0]);
    const dt = 1 / 240;
    const b = hoverBird();
    for (let s = 0; s < 72; s++) {
      b.phase = (b.phase + 2 * Math.PI * b.freq * dt) % (2 * Math.PI);
      f.addBird(b, dt);
      f.step(dt);
    }
    // Mean vertical speed in a small column 8 cm below the bird, and 8 cm above it.
    const mean = (dy: number) => {
      let s = 0;
      let c = 0;
      for (let k = 1; k <= N; k++)
        for (let j = 1; j <= N; j++)
          for (let i = 1; i <= N; i++) {
            const q = f.cellCentre(i, j, k);
            if (Math.hypot(q[0], q[2]) < 0.02 && Math.abs(q[1] - (at[1] + dy)) < H) {
              s += f.v[IX(i, j, k)];
              c += 1;
            }
          }
      return s / c;
    };
    expect(-mean(-0.08)).toBeGreaterThan(1.0); // momentum theory: 1.7 m/s at the wings, 3.4 far below
    expect(-mean(-0.08)).toBeLessThan(4.0);
    expect(Math.abs(mean(0.08))).toBeLessThan(0.5 * -mean(-0.08)); // the air above is drawn in, more slowly
  });
});

describe("laser sheet", () => {
  it("tracers ride the air: in a uniform wind they all move with it, and stay in their sheet", () => {
    const f = new Fluid();
    const wind: Vec3 = [0, 0, 0.5]; // along the sheet (whose normal will be x)
    f.reset(at, wind);
    const t = new Tracers();
    t.aim(at, [at[0] - 1, at[1], at[2]], f); // camera on the -x side: normal +x
    expect(t.normal).toEqual([1, 0, 0]);
    const before = Float64Array.from(t.pos);
    t.update(f, 0.01);
    let moved = 0;
    for (let p = 0; p < TRACERS; p++) {
      const dz = t.pos[3 * p + 2] - before[3 * p + 2];
      if (Math.abs(dz - 0.005) < 1e-9 && t.pos[3 * p] === before[3 * p]) moved += 1;
    }
    expect(moved / TRACERS).toBeGreaterThan(0.6); // the others left the box or their life: seeded again
  });
});

describe("laser sheet, turning", () => {
  it("turns with the camera: each tracer keeps its place in the sheet, nothing is reseeded", () => {
    const f = new Fluid();
    f.reset(at, [0, 0, 0]);
    const t = new Tracers();
    t.aim(at, [at[0] - 1, at[1], at[2]], f);
    const inSheet = (p: number) => {
      const dx = t.pos[3 * p] - at[0];
      const dz = t.pos[3 * p + 2] - at[2];
      return [dx * t.normal[0] + dz * t.normal[2], -dx * t.normal[2] + dz * t.normal[0], t.pos[3 * p + 1]];
    };
    const before = Array.from({ length: 50 }, (_, p) => inSheet(p));
    const a = (20 * Math.PI) / 180; // the camera turns by 20° around the bird
    t.aim(at, [at[0] - Math.cos(a), at[1], at[2] - Math.sin(a)], f);
    before.forEach((b, p) => inSheet(p).forEach((x, i) => expect(x).toBeCloseTo(b[i], 12)));
  });
});

describe("colours and wind", () => {
  it("the palettes of the C++ simulation", () => {
    viridis(0).forEach((c, i) => expect(c).toBeCloseTo([0.278, 0.005, 0.334][i], 2));
    viridis(1).forEach((c, i) => expect(c).toBeCloseTo([0.993, 0.906, 0.144][i], 1));
    expect(rdbu(0)).toEqual([0.969, 0.969, 0.969]); // white at zero
    expect(inferno(1)).toEqual([0.988, 1.0, 0.643]);
  });

  it("a push of 40% of the weight is about 1 m/s of air", () => {
    const v = windAirSpeed([0.4 * weight(p), 0, 0]);
    expect(v[0]).toBeCloseTo((0.4 * weight(p)) / C_AIR, 12);
    expect(v[0]).toBeGreaterThan(0.9);
    expect(v[0]).toBeLessThan(1.2);
  });
});
