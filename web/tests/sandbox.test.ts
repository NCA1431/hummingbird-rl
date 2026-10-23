/**
 * Sandbox: the conditions set by hand, the placement tools, and episodes flown in a given scene.
 */

import { describe, expect, it } from "vitest";

import { PIDController } from "../src/control/pid";
import { simulateEpisode } from "../src/modes/live";
import {
  ARENA_LIMIT,
  DEFAULT_SETTINGS,
  MAX_WAYPOINTS,
  type SandboxSettings,
  clearScene,
  erase,
  inside,
  place,
  sandboxConditions,
  sceneProblems,
  settingsFrom,
  toolsFor,
  waitingEpisode,
} from "../src/modes/sandbox";
import { duration, endTime, sampleEpisode } from "../src/modes/watch";
import { ACTION_NOISE, OBS_NOISE } from "../src/sim/disturbances";
import { DEFAULT_PARAMS, type Vec3, weight } from "../src/sim/params";
import { START, type Scene } from "../src/sim/world";

const p = DEFAULT_PARAMS;
const settings = (changes: Partial<SandboxSettings> = {}): SandboxSettings => ({ ...structuredClone(DEFAULT_SETTINGS), ...changes });
const empty = (): Scene => ({ start: [...START], targets: [], obstacles: [] });
const at0: Vec3 = [0.3, 0.4, 0.2];

describe("conditions", () => {
  it("turn the sliders into the same conditions the regimes draw", () => {
    const c = sandboxConditions(settings({ wind: 0.2, windDir: 90, gusts: 0.05, mass: 1.2, damage: [0.3, 0] }), p);
    const w = weight(p);
    expect(c.mean_wind[0]).toBeCloseTo(0, 12);
    expect(c.mean_wind[2]).toBeCloseTo(0.2 * w, 12); // 90°: towards +z
    expect(c.mean_wind[1]).toBe(0);
    expect(c.gust_sigma).toBeCloseTo(0.05 * w, 12);
    expect(c.mass_factor).toBe(1.2);
    expect(c.damage).toEqual([0.3, 0]);
    expect(c.failure_time).toBeNull();
    expect([c.action_noise, c.obs_noise]).toEqual([ACTION_NOISE, OBS_NOISE]);
    expect(sandboxConditions(settings({ noise: false }), p).obs_noise).toBe(0);
  });

  it("a sudden failure needs a wing", () => {
    const c = sandboxConditions(settings({ failureWing: 1, failureTime: 1.5, failureAmount: 0.5 }), p);
    expect([c.failure_time, c.failure_wing, c.failure_amount]).toEqual([1.5, 1, 0.5]);
  });

  it("an episode's conditions come back unchanged through the sliders' settings", () => {
    const s = settings({ wind: 0.17, windDir: 233, gusts: 0.04, mass: 0.85, damage: [0, 0.25], failureWing: 0, failureTime: 2.3 });
    const back = sandboxConditions(settingsFrom(sandboxConditions(s, p), p), p);
    const c = sandboxConditions(s, p);
    back.mean_wind.forEach((v, i) => expect(v).toBeCloseTo(c.mean_wind[i], 12));
    expect({ ...back, mean_wind: 0 }).toEqual({ ...c, mean_wind: 0 });
  });
});

describe("placing", () => {
  const at: Vec3 = [0.4, 0.6, -0.2];

  it("go-to-point and obstacles have one target, which the click moves", () => {
    for (const task of ["goto", "obstacles"]) {
      let scene = place(empty(), task, "target", at, 0.05);
      scene = place(scene, task, "target", [0.1, 0.3, 0.1], 0.05);
      expect(scene.targets).toEqual([[0.1, 0.3, 0.1]]);
    }
  });

  it("navigation builds its course click after click, up to a limit", () => {
    let scene = empty();
    for (let i = 0; i < MAX_WAYPOINTS + 2; i++) scene = place(scene, "navigation", "target", [0.1 * i, 0.5, 0], 0.05);
    expect(scene.targets).toHaveLength(MAX_WAYPOINTS);
    expect(scene.targets[2]).toEqual([0.2, 0.5, 0]);
    // The Target tool is then greyed out, and comes back once a waypoint is erased.
    expect(toolsFor("navigation", scene).target).toBe(false);
    expect(toolsFor("navigation", erase(scene, "target", 0)).target).toBe(true);
    expect(toolsFor("goto", scene).target).toBe(true); // go-to-point: the click moves its one target
  });

  it("obstacles only where the task has them, and nothing outside the arena", () => {
    expect(place(empty(), "goto", "obstacle", at, 0.05).obstacles).toHaveLength(0);
    expect(place(empty(), "hover", "target", at, 0.05).targets).toHaveLength(0);
    const far = place(empty(), "obstacles", "obstacle", [3, 0.5, 4], 0.05).obstacles[0].center;
    expect(Math.hypot(far[0], far[2])).toBeCloseTo(ARENA_LIMIT, 12);
  });

  it("clearing keeps the target of go-to-point and obstacles, not a navigation course", () => {
    const scene: Scene = { start: [...START], targets: [at], obstacles: [{ center: at, radius: 0.05 }] };
    expect(clearScene(scene, "obstacles")).toEqual({ start: START, targets: [at], obstacles: [] });
    expect(clearScene(scene, "navigation").targets).toEqual([]);
  });

  it("erasing removes one target or one obstacle, and leaves the rest in order", () => {
    const scene: Scene = {
      start: [...START],
      targets: [[0.1, 0.5, 0], [0.2, 0.5, 0], [0.3, 0.5, 0]],
      obstacles: [{ center: at, radius: 0.05 }, { center: [0, 0.4, 0.3], radius: 0.06 }],
    };
    const a = erase(scene, "target", 1);
    expect(a.targets).toEqual([[0.1, 0.5, 0], [0.3, 0.5, 0]]);
    expect(a.obstacles).toEqual(scene.obstacles);
    const b = erase(scene, "obstacle", 0);
    expect(b.obstacles).toEqual([{ center: [0, 0.4, 0.3], radius: 0.06 }]);
    expect(scene.targets).toHaveLength(3); // the scene given is not changed
  });

  it("the preview is drawn where the click would place", () => {
    expect(inside([0.2, 0.5, 0.1])).toEqual([0.2, 0.5, 0.1]);
    const edge = inside([3, 0.5, 4]);
    expect(Math.hypot(edge[0], edge[2])).toBeCloseTo(ARENA_LIMIT, 12);
    expect(place(empty(), "goto", "target", [3, 0.5, 4], 0.05).targets[0]).toEqual(edge);
  });

  it("says what would spoil the flight", () => {
    expect(sceneProblems(empty(), "navigation", p.r_bird)).toHaveLength(1);
    const blocked: Scene = { start: [...START], targets: [at], obstacles: [{ center: [...START], radius: 0.05 }] };
    expect(sceneProblems(blocked, "obstacles", p.r_bird)[0]).toMatch(/start/);
    expect(sceneProblems({ ...blocked, obstacles: [] }, "obstacles", p.r_bird)).toEqual([]);
  });
});

describe("flights in a given scene", () => {
  it("a scene that cannot be flown yet is still shown, the bird waiting at the start", () => {
    const scene: Scene = { start: [...START], targets: [], obstacles: [{ center: at0, radius: 0.05 }] };
    const ep = waitingEpisode("navigation", scene, sandboxConditions(settings(), p));
    expect(ep.scene).toEqual(scene);
    expect([duration(ep), endTime(ep)]).toEqual([0, 0]);
    expect(ep.result.reason).toBe("no_course");
    expect(sampleEpisode(ep, 0.5).pos).toEqual(START);
  });

  it("use the scene and the conditions given, and stay repeatable", () => {
    const scene: Scene = { start: [...START], targets: [[0.3, 0.6, 0.2]], obstacles: [{ center: [0.15, 0.55, 0.1], radius: 0.04 }] };
    const conditions = sandboxConditions(settings({ wind: 0.1, windDir: 45 }), p);
    const a = simulateEpisode("obstacles", "normal", new PIDController(), 5, { scene, conditions }, "sandbox");
    const b = simulateEpisode("obstacles", "normal", new PIDController(), 5, { scene, conditions }, "sandbox");
    expect(a.scene).toEqual(scene);
    expect(a.conditions).toEqual(conditions);
    expect(a.suite).toBe("sandbox");
    expect(b).toEqual(a);
  });

  it("a navigation course built by hand is flown waypoint after waypoint", () => {
    const scene: Scene = { start: [...START], targets: [[0.2, 0.5, 0], [0.2, 0.5, 0.2], [0, 0.6, 0.2]], obstacles: [] };
    const conditions = sandboxConditions(settings({ wind: 0, gusts: 0, noise: false }), p);
    const ep = simulateEpisode("navigation", "normal", new PIDController(), 1, { scene, conditions }, "sandbox");
    expect(ep.result.success).toBe(true);
    expect(Math.max(...ep.frames.target)).toBe(2);
  });
});
