/**
 * The random parts of the browser simulator: they cannot match NumPy draw for
 * draw, so we check their statistics instead.
 */

import { describe, expect, it } from "vitest";

import { Rng } from "../src/sim/random";
import { GustProcess } from "../src/sim/wind";

function meanStd(values: number[]): [number, number] {
  const m = values.reduce((a, b) => a + b, 0) / values.length;
  const v = values.reduce((a, b) => a + (b - m) * (b - m), 0) / (values.length - 1);
  return [m, Math.sqrt(v)];
}

describe("random generator", () => {
  it("is repeatable and gives standard normal draws", () => {
    expect(new Rng(7).uniform()).toBe(new Rng(7).uniform());
    const rng = new Rng(1);
    const [m, s] = meanStd(Array.from({ length: 20000 }, () => rng.normal()));
    expect(Math.abs(m)).toBeLessThan(0.03);
    expect(Math.abs(s - 1)).toBeLessThan(0.03);
  });
});

describe("gusts", () => {
  it("fluctuate around the mean with standard deviation sigma (0.3 sigma vertically)", () => {
    const sigma = 0.01;
    const gusts = new GustProcess([0.005, 0, -0.002], sigma, new Rng(3));
    const xs: number[] = [];
    const ys: number[] = [];
    // One sample every 0.5 s (the correlation time), so the samples are nearly independent.
    for (let k = 0; k < 4000 * 120; k++) {
      const w = gusts.step(1 / 240);
      if (k % 120 === 0) {
        xs.push(w[0]);
        ys.push(w[1]);
      }
    }
    const [mx, sx] = meanStd(xs);
    const [my, sy] = meanStd(ys);
    expect(Math.abs(mx - 0.005)).toBeLessThan(0.1 * sigma);
    expect(Math.abs(sx / sigma - 1)).toBeLessThan(0.1);
    expect(Math.abs(my)).toBeLessThan(0.1 * sigma);
    expect(Math.abs(sy / (0.3 * sigma) - 1)).toBeLessThan(0.1);
  });

  it("stay exactly at the mean without fluctuations", () => {
    const gusts = new GustProcess([0.01, 0, 0], 0.0);
    for (let k = 0; k < 100; k++) gusts.step(1 / 240);
    expect(gusts.state).toEqual([0.01, 0, 0]);
  });
});
