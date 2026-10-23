/**
 * Sandbox mode: the visitor builds the situation, the controllers fly it.
 *
 * Everything the live mode draws at random can be set by hand here: where the
 * target, the waypoints and the obstacles are, the wind, the bird's mass, the
 * damage to each wing, a sudden failure, the noise. The episode itself is
 * flown exactly as in the live mode (same environment, same tasks), from the
 * fixed start point; only the start push and the noise still come from the
 * seed. This module holds the logic, without any page element, so that it
 * can be tested; the controls are in ui/sandboxPanel.ts.
 */

import { ACTION_NOISE, type Conditions, OBS_NOISE } from "../sim/disturbances";
import { type BirdParams, type Vec3, weight } from "../sim/params";
import { type Obstacle, type Scene, START, clearance } from "../sim/world";
import type { Episode } from "./watch";

export interface SandboxSettings {
  wind: number; // mean wind, fraction of the normal bird's weight
  windDir: number; // degrees in the horizontal plane, 0 = towards +x, 90 = towards +z
  gusts: number; // size of the gusts, fraction of the weight
  mass: number; // mass factor
  damage: [number, number]; // force lost by the (left, right) wing from the start, 0 to 1
  failureWing: -1 | 0 | 1; // -1: no sudden failure; 0 left, 1 right
  failureTime: number; // s
  failureAmount: number; // force lost at that moment, 0 to 1
  noise: boolean; // sensor and actuator noise, as in normal conditions
}

export const DEFAULT_SETTINGS: Readonly<SandboxSettings> = Object.freeze({
  wind: 0.05,
  windDir: 0,
  gusts: 0.02,
  mass: 1.0,
  damage: [0, 0] as [number, number],
  failureWing: -1 as const,
  failureTime: 2.0,
  failureAmount: 0.4,
  noise: true,
});

export const MAX_WAYPOINTS = 6;
export const MAX_BIRDS = 4;
export const ARENA_LIMIT = 1.3; // m from the centre: placements stay inside the 1.5 m arena

/** The episode's conditions, in the same form as those drawn at random. */
export function sandboxConditions(s: SandboxSettings, p: BirdParams): Conditions {
  const w = weight(p);
  const a = (s.windDir * Math.PI) / 180;
  return {
    mean_wind: [s.wind * w * Math.cos(a), 0, s.wind * w * Math.sin(a)],
    gust_sigma: s.gusts * w,
    mass_factor: s.mass,
    damage: [s.damage[0], s.damage[1]],
    failure_time: s.failureWing < 0 ? null : s.failureTime,
    failure_wing: Math.max(s.failureWing, 0),
    failure_amount: s.failureWing < 0 ? 0 : s.failureAmount,
    action_noise: s.noise ? ACTION_NOISE : 0,
    obs_noise: s.noise ? OBS_NOISE : 0,
  };
}

export type Tool = "target" | "obstacle" | "erase" | null;

/** Which tools make sense for a task: hover's target is the start point, only two tasks have obstacles. */
export function toolsFor(task: string): { target: boolean; obstacle: boolean; erase: boolean } {
  return { target: task !== "hover", obstacle: task === "obstacles" || task === "navigation", erase: task !== "hover" };
}

/** Keeps a point inside the arena, horizontally (also where the preview of the next placement is drawn). */
export function inside(point: Vec3): Vec3 {
  const r = Math.hypot(point[0], point[2]);
  const k = r > ARENA_LIMIT ? ARENA_LIMIT / r : 1;
  return [point[0] * k, point[1], point[2] * k];
}

/**
 * The scene after one click with a tool. Go-to-point and obstacles have one target, which
 * the click moves; navigation adds a waypoint at the end of the course.
 */
export function place(scene: Scene, task: string, tool: Tool, point: Vec3, radius: number): Scene {
  const p = inside(point);
  const next: Scene = { start: [...scene.start], targets: scene.targets.map((t) => [...t] as Vec3), obstacles: [...scene.obstacles] };
  const tools = toolsFor(task);
  if (tool === "target" && tools.target) {
    if (task === "navigation") {
      if (next.targets.length < MAX_WAYPOINTS) next.targets.push(p);
    } else next.targets = [p];
  } else if (tool === "obstacle" && tools.obstacle) {
    next.obstacles.push({ center: p, radius });
  }
  return next;
}

/** The scene without one of its targets or obstacles (the Erase tool). */
export function erase(scene: Scene, kind: "target" | "obstacle", index: number): Scene {
  return {
    start: [...scene.start],
    targets: kind === "target" ? scene.targets.filter((_, i) => i !== index) : scene.targets,
    obstacles: kind === "obstacle" ? scene.obstacles.filter((_, i) => i !== index) : scene.obstacles,
  };
}

/**
 * An "episode" of a single frame, the bird waiting at the start: what is shown while the
 * scene cannot be flown (a navigation course without waypoints), so that every edit is seen.
 */
export function waitingEpisode(task: string, scene: Scene, conditions: Conditions): Episode {
  const f = (x: number[]) => [x];
  return {
    format: "hummingbird-episode/1",
    task,
    seed: 0,
    policy: "",
    suite: "sandbox",
    dt: 1 / 60,
    scene,
    conditions,
    result: { success: false, reason: "no_course", return: 0, length: 0, energy: 0 },
    frames: {
      t: [0],
      pos: [[...scene.start] as Vec3],
      quat: [[1, 0, 0, 0]],
      vel: [[0, 0, 0]],
      action: f([0.6, 0.6, 0, 0, 0]),
      thrust: [0],
      wind: [[...conditions.mean_wind] as Vec3],
      target: [0],
      hold: [0],
      distance: [0],
    },
  };
}

/** Removes the obstacles, and the waypoints in navigation (the course is then built click by click). */
export function clearScene(scene: Scene, task: string): Scene {
  return { start: [...scene.start], targets: task === "navigation" ? [] : scene.targets, obstacles: [] };
}

/** What would make the flight meaningless, in words; empty if the scene is fine. */
export function sceneProblems(scene: Scene, task: string, rBird: number): string[] {
  const problems: string[] = [];
  if (task === "hover") return problems;
  if (scene.targets.length === 0) problems.push("place at least one waypoint (Target tool)");
  const blocked = (q: Vec3, o: Obstacle) => clearance(q, o, rBird) < 0;
  if (scene.obstacles.some((o) => blocked(START, o))) problems.push("an obstacle covers the start: the flight ends at once");
  if (scene.obstacles.some((o) => scene.targets.some((t) => blocked(t, o)))) {
    problems.push("an obstacle covers a target: it cannot be reached");
  }
  return problems;
}

/**
 * Sandbox settings that reproduce an episode's conditions exactly, to edit a live flight (the
 * sliders show them rounded to their steps until they are moved).
 */
export function settingsFrom(c: Episode["conditions"], p: BirdParams): SandboxSettings {
  const w = weight(p);
  return {
    wind: Math.hypot(c.mean_wind[0], c.mean_wind[2]) / w,
    windDir: (((Math.atan2(c.mean_wind[2], c.mean_wind[0]) * 180) / Math.PI) + 360) % 360,
    gusts: c.gust_sigma / w,
    mass: c.mass_factor,
    damage: [c.damage[0], c.damage[1]],
    failureWing: c.failure_time === null ? -1 : (c.failure_wing as 0 | 1),
    failureTime: c.failure_time ?? DEFAULT_SETTINGS.failureTime,
    failureAmount: c.failure_time === null ? DEFAULT_SETTINGS.failureAmount : c.failure_amount,
    noise: (c.obs_noise ?? 0) > 0,
  };
}
