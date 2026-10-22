/** Watch mode and wing drawing: interpolation between recorded frames, damage over time, wing angles. */

import { describe, expect, it } from "vitest";

import { type Episode, Playback, damageAt, duration, sampleEpisode } from "../src/modes/watch";
import { sampleCurve, wingAngles } from "../src/view/wingKinematics";

/** A made-up episode: 3 frames 1/60 s apart, moving 6 cm along x, target switching at frame 2. */
function episode(failure: number | null = null): Episode {
  const dt = 1 / 60;
  return {
    format: "hummingbird-episode/1", task: "navigation", seed: 0, dt,
    scene: { start: [0, 0.5, 0], targets: [[0.1, 0.5, 0], [0.2, 0.5, 0]], obstacles: [] },
    conditions: { mean_wind: [0, 0, 0], gust_sigma: 0, mass_factor: 1, damage: [0.1, 0],
      failure_time: failure, failure_wing: 1, failure_amount: 0.4 },
    result: { success: false, reason: "time_limit", return: 0, length: 2, energy: 0 },
    frames: {
      t: [0, dt, 2 * dt], pos: [[0, 0.5, 0], [0.03, 0.5, 0], [0.06, 0.5, 0]],
      quat: [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], vel: [[0, 0, 0], [1.8, 0, 0], [1.8, 0, 0]],
      action: [[0.6, 0.6, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0]], thrust: [0.04, 0.05, 0.06],
      wind: [[0, 0, 0], [0, 0, 0], [0, 0, 0]], target: [0, 0, 1], hold: [0, 0, 0], distance: [0.1, 0.07, 0.04],
    },
  };
}

describe("watch mode", () => {
  it("interpolates between frames and uses the action that leads to the next one", () => {
    const ep = episode();
    const s = sampleEpisode(ep, 0.5 / 60);
    expect(s.index).toBe(0);
    expect(s.pos[0]).toBeCloseTo(0.015, 12);
    expect(s.thrust).toBeCloseTo(0.045, 12);
    expect(s.action).toEqual([1, 1, 0, 0, 0]);
  });

  it("stays on the first and last frames outside the episode", () => {
    const ep = episode();
    expect(sampleEpisode(ep, -1).pos).toEqual([0, 0.5, 0]);
    expect(sampleEpisode(ep, 10).pos[0]).toBeCloseTo(0.06, 12);
    expect(sampleEpisode(ep, 10).target).toBe(1);
  });

  it("adds the sudden failure to the damage once its time has come", () => {
    const ep = episode(1 / 60);
    expect(damageAt(ep, 0)).toEqual([0.1, 0]);
    expect(damageAt(ep, 1.5 / 60)).toEqual([0.1, 0.4]);
  });

  it("plays at the chosen speed and stops at the end", () => {
    const pb = new Playback(episode());
    pb.speed = 0.5;
    expect(pb.advance(1 / 60)).toBeCloseTo(0.5 / 60, 12);
    pb.advance(10);
    expect(pb.time).toBe(duration(pb.episode));
    expect(pb.playing).toBe(false);
  });
});

describe("wing angles", () => {
  it("repeat every beat and start the downstroke with the wing at the back", () => {
    const a = wingAngles(0.3, 1.0);
    const b = wingAngles(0.3 + 2 * Math.PI, 1.0);
    expect(b.stroke).toBeCloseTo(a.stroke, 12);
    expect(b.pitch).toBeCloseTo(a.pitch, 12);
    expect(wingAngles(0, 1.0).stroke).toBe(-1.0);
  });

  it("interpolate the measured curves linearly and cyclically", () => {
    const curve = Array.from({ length: 64 }, (_, i) => i);
    expect(sampleCurve(curve, (2 * Math.PI * 10.5) / 64)).toBeCloseTo(10.5, 12);
    expect(sampleCurve(curve, (2 * Math.PI * 63.5) / 64)).toBeCloseTo(31.5, 12); // halfway from 63 back to 0
  });
});
