/**
 * Live mode: the scenes and conditions drawn in the browser, and the episodes flown there.
 *
 * The random draws differ from NumPy's, so here we check the rules rather than
 * numbers: every scene respects the ranges of hummingbird_rl/world/sampling.py,
 * a seed always gives the same flight, and two controllers given the same seed
 * meet exactly the same situation.
 */

import { describe, expect, it } from "vitest";

import { LQRController } from "../src/control/lqr";
import { PIDController } from "../src/control/pid";
import { simulateEpisode } from "../src/modes/live";
import { REGIMES, sampleConditions } from "../src/sim/disturbances";
import { DEFAULT_PARAMS, type Vec3, weight } from "../src/sim/params";
import { Rng } from "../src/sim/random";
import { pathExists, sampleScene, segmentPointDistance } from "../src/sim/sampling";
import { START } from "../src/sim/world";

const p = DEFAULT_PARAMS;
const dist = (a: Vec3, b: Vec3) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const seeds = Array.from({ length: 30 }, (_, i) => 1000 + i);

describe("scenes", () => {
  it("go-to-point: a target 10-50 cm away, between 15 and 95 cm high", () => {
    for (const s of seeds) {
      const t = sampleScene("goto", new Rng(s)).targets[0];
      expect(dist(t, START)).toBeGreaterThanOrEqual(0.1);
      expect(dist(t, START)).toBeLessThanOrEqual(0.5);
      expect(t[1]).toBeGreaterThanOrEqual(0.15);
      expect(t[1]).toBeLessThanOrEqual(0.95);
    }
  });

  it("obstacles: a goal 50-80 cm away, 1-6 spheres, the straight path blocked, a way through", () => {
    for (const s of seeds) {
      const scene = sampleScene("obstacles", new Rng(s));
      const goal = scene.targets[0];
      expect(dist(goal, START)).toBeGreaterThanOrEqual(0.5 - 1e-9);
      expect(dist(goal, START)).toBeLessThanOrEqual(0.8 + 1e-9);
      expect(scene.obstacles.length).toBeGreaterThanOrEqual(1);
      expect(scene.obstacles.length).toBeLessThanOrEqual(6);
      expect(scene.obstacles.some((o) => segmentPointDistance(START, goal, o.center) < o.radius + p.r_bird)).toBe(true);
      for (const o of scene.obstacles) {
        expect(dist(o.center, START) - o.radius).toBeGreaterThanOrEqual(0.1 - 1e-9);
        expect(dist(o.center, goal) - o.radius).toBeGreaterThanOrEqual(0.1 - 1e-9);
      }
      expect(pathExists([START, goal], scene.obstacles, p.r_bird)).toBe(true);
    }
  });

  it("navigation: 3 waypoints 25-50 cm apart, inside 1 m of the centre, a way through", () => {
    for (const s of seeds) {
      const scene = sampleScene("navigation", new Rng(s));
      expect(scene.targets).toHaveLength(3);
      const course = [START, ...scene.targets];
      for (let i = 1; i < course.length; i++) {
        expect(dist(course[i], course[i - 1])).toBeGreaterThanOrEqual(0.25 - 1e-9);
        expect(dist(course[i], course[i - 1])).toBeLessThanOrEqual(0.5 + 1e-9);
        expect(Math.hypot(course[i][0], course[i][2])).toBeLessThanOrEqual(1.0);
      }
      expect(pathExists(course, scene.obstacles, p.r_bird)).toBe(true);
    }
  });

  it("the path search sees a blocked target and a wall", () => {
    const goal: Vec3 = [0.6, 0.5, 0.0];
    expect(pathExists([START, goal], [], p.r_bird)).toBe(true);
    expect(pathExists([START, goal], [{ center: goal, radius: 0.1 }], p.r_bird)).toBe(false); // goal inside
    // A sphere of 1 m around the goal, but not the start: no way in.
    expect(pathExists([START, goal], [{ center: [1.1, 0.5, 0.0], radius: 0.45 }], p.r_bird)).toBe(true);
    const shell = Array.from({ length: 200 }, (_, i) => {
      const phi = Math.acos(1 - (2 * (i + 0.5)) / 200);
      const theta = Math.PI * (1 + Math.sqrt(5)) * i;
      const c: Vec3 = [goal[0] + 0.2 * Math.sin(phi) * Math.cos(theta), goal[1] + 0.2 * Math.cos(phi), goal[2] + 0.2 * Math.sin(phi) * Math.sin(theta)];
      return { center: c, radius: 0.06 };
    });
    expect(pathExists([START, goal], shell, p.r_bird)).toBe(false); // goal enclosed in a shell of spheres
  });
});

describe("conditions", () => {
  it("stay within the ranges of each regime", () => {
    const w = weight(p);
    for (const s of seeds) {
      const rng = new Rng(s);
      for (const regime of REGIMES) {
        const c = sampleConditions(regime, rng, p);
        const wind = Math.hypot(...c.mean_wind) / w;
        expect(c.mean_wind[1]).toBe(0.0);
        expect(wind).toBeLessThanOrEqual(regime === "randomised" ? 0.3 : 0.05 + 1e-12);
        if (regime !== "randomised") expect(c.mass_factor).toBe(1.0);
        if (regime === "failure") {
          expect(c.failure_time).toBeGreaterThanOrEqual(1.5);
          expect(c.failure_time).toBeLessThanOrEqual(3.0);
          expect(c.failure_amount).toBe(0.4);
        } else expect(c.failure_time).toBeNull();
        if (regime === "calm") expect(c.action_noise + c.obs_noise + c.gust_sigma + wind).toBe(0.0);
      }
    }
  });
});

describe("live episodes", () => {
  it("a seed always gives the same flight", () => {
    const a = simulateEpisode("goto", "normal", new PIDController(), 7);
    const b = simulateEpisode("goto", "normal", new PIDController(), 7);
    expect(b).toEqual(a);
  });

  it("two controllers with the same seed meet the same scene, conditions and start", () => {
    for (const task of ["hover", "obstacles", "navigation"]) {
      const a = simulateEpisode(task, "randomised", new PIDController(), 11);
      const b = simulateEpisode(task, "randomised", new LQRController(), 11);
      expect(b.scene).toEqual(a.scene);
      expect(b.conditions).toEqual(a.conditions);
      expect(b.frames.pos[0]).toEqual(a.frames.pos[0]);
      expect(b.frames.quat[0]).toEqual(a.frames.quat[0]);
    }
  });

  it("two controllers with the same seed meet the same gusts, decision after decision", () => {
    const a = simulateEpisode("obstacles", "randomised", new PIDController(), 12);
    const b = simulateEpisode("obstacles", "randomised", new LQRController(), 12);
    const n = Math.min(a.result.length, b.result.length);
    expect(n).toBeGreaterThan(30);
    expect(b.frames.wind.slice(0, n)).toEqual(a.frames.wind.slice(0, n));
  });

  it("after a loss of control, goes on for a while but does not count it", () => {
    let checked = 0;
    for (const s of seeds.slice(0, 10)) {
      const ep = simulateEpisode("hover", "failure", new LQRController(), s);
      if (ep.result.reason !== "loss_of_control") continue;
      checked += 1;
      expect(ep.endFrame).toBe(ep.result.length);
      const extra = ep.frames.t.length - 1 - ep.result.length;
      expect(extra).toBeGreaterThan(0);
      expect(ep.frames.t.length - 1).toBeLessThanOrEqual(300); // never beyond the 5 s of a hover episode
      for (const values of Object.values(ep.frames)) expect(values).toHaveLength(ep.frames.t.length);
    }
    expect(checked).toBeGreaterThanOrEqual(8);
    const won = simulateEpisode("goto", "normal", new PIDController(), 5);
    expect(won.endFrame).toBeUndefined(); // nothing is added after a success
    expect(won.frames.t).toHaveLength(won.result.length + 1);
  });

  it("has the format of a recorded episode: one frame per decision plus the start", () => {
    const ep = simulateEpisode("navigation", "normal", new PIDController(), 3);
    expect(ep.endFrame).toBeUndefined();
    const n = ep.result.length + 1;
    for (const values of Object.values(ep.frames)) expect(values).toHaveLength(n);
    expect(ep.frames.t[0]).toBe(0.0);
    // A failure stops the 4 physics steps of the last decision early, so the last frame can come a little before.
    expect(ep.frames.t[n - 1]).toBeLessThanOrEqual(ep.result.length / 60 + 1e-9);
    expect(ep.frames.t[n - 1]).toBeGreaterThan((ep.result.length - 1) / 60);
    expect(ep.frames.target.every((v, i, a) => i === 0 || v >= a[i - 1])).toBe(true); // targets only move on
  });

  it("the PID flies as well as in Python (100% on hover and go-to-point in normal conditions)", () => {
    for (const task of ["hover", "goto"]) {
      const wins = seeds.slice(0, 20).filter((s) => simulateEpisode(task, "normal", new PIDController(), s).result.success);
      expect(wins.length, task).toBeGreaterThanOrEqual(19);
    }
  });

  it("the LQR loses control after a sudden wing failure, as in Python (4% survival)", () => {
    const lost = seeds
      .slice(0, 10)
      .filter((s) => simulateEpisode("hover", "failure", new LQRController(), s).result.reason === "loss_of_control");
    expect(lost.length).toBeGreaterThanOrEqual(8);
  });
});
