/**
 * Parity tests for the live mode: networks, controllers, tasks and the environment loop.
 *
 * Golden files written by `scripts/export_web.py golden`:
 *   policies.json  three small networks (PPO, SAC, TD3 layouts) and their actions;
 *   episodes.json  one episode per task flown by the PID and by the LQR, without
 *                  noise (steady wind, mass, damage, a sudden failure), with every
 *                  action, reward and position.
 * The networks run in float32, so they are compared to 1e-5. Everything else is
 * float64 in both languages, in the same order of operations, except one step:
 * the observation is rounded to float32. A difference of 1e-15 before that
 * rounding can land on either side of a float32 value, a change of 1e-8 in one
 * observation entry, which the controllers' gains turn into up to 1e-7 in an
 * action (seen near the end of the navigation episode, close to an obstacle).
 * Hence 1e-6 for the actions. The positions, which integrate the actions,
 * still agree to 1e-9: the flights are the same.
 */

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import type { Controller } from "../src/control/controller";
import { LQRController } from "../src/control/lqr";
import { MlpPolicy, type PolicyFile } from "../src/control/mlp";
import { PIDController } from "../src/control/pid";
import type { Conditions } from "../src/sim/disturbances";
import { HummingbirdEnv } from "../src/sim/env";
import type { Vec3 } from "../src/sim/params";
import type { Quat } from "../src/sim/rotations";
import type { Scene } from "../src/sim/world";

function load<T>(name: string): T {
  return JSON.parse(readFileSync(new URL(`./golden/${name}.json`, import.meta.url), "utf8")) as T;
}

interface PolicyCase {
  policy: PolicyFile;
  observations: number[][];
  expected: number[][];
}

interface EpisodeCase {
  task: string;
  controller: string;
  scene: Scene;
  conditions: Conditions;
  start: { pos: Vec3; vel: Vec3; quat: Quat; omega: Vec3 };
  actions: number[][];
  rewards: number[];
  pos: Vec3[];
  result: { reason: string; success: boolean; length: number; energy: number };
}

describe("networks", () => {
  const cases = load<{ cases: PolicyCase[] }>("policies").cases;

  it.each(cases.map((c) => [c.policy.algo, c] as const))("%s gives the same actions as PyTorch", (_, c) => {
    const net = new MlpPolicy(c.policy);
    let worst = 0.0;
    c.observations.forEach((obs, i) => {
      net.act(obs).forEach((a, j) => (worst = Math.max(worst, Math.abs(a - c.expected[i][j]))));
    });
    expect(worst).toBeLessThan(1e-5);
  });

  it("the networks really saturate (otherwise the test would not check clipping and tanh)", () => {
    const all = cases.flatMap((c) => c.expected.flat());
    expect(all.filter((a) => Math.abs(a) > 0.99).length).toBeGreaterThan(5);
  });
});

describe("whole episodes: tasks, environment and classical controllers", () => {
  const cases = load<{ cases: EpisodeCase[] }>("episodes").cases;
  const close = (a: number, b: number, tol: number) => Math.abs(a - b) <= tol * Math.max(1.0, Math.abs(b));

  it.each(cases.map((c) => [`${c.task} ${c.controller}`, c] as const))("%s", (_, c) => {
    const env = new HummingbirdEnv(c.task, "calm");
    let obs = env.reset(0, {
      scene: c.scene,
      conditions: c.conditions,
      state: { ...c.start, time: 0.0, phase: 0.0 },
    });
    const controller: Controller = c.controller === "pid" ? new PIDController() : new LQRController();
    controller.reset();
    let k = 0;
    let worstAction = 0.0;
    let worstPos = 0.0;
    for (;;) {
      const action = controller.act(obs);
      action.forEach((a, j) => (worstAction = Math.max(worstAction, Math.abs(a - c.actions[k][j]))));
      const r = env.step(action);
      expect(close(r.reward, c.rewards[k], 1e-8), `reward at step ${k}`).toBe(true);
      env.state.pos.forEach((x, j) => (worstPos = Math.max(worstPos, Math.abs(x - c.pos[k][j]))));
      obs = r.obs;
      k += 1;
      if (r.terminated || r.truncated) {
        expect(r.reason).toBe(c.result.reason);
        expect(r.success).toBe(c.result.success);
        break;
      }
      expect(k).toBeLessThan(c.result.length); // the TypeScript episode must not run longer
    }
    expect(k).toBe(c.result.length);
    expect(worstAction).toBeLessThan(1e-6);
    expect(worstPos).toBeLessThan(1e-9);
    expect(close(env.energy, c.result.energy, 1e-8)).toBe(true);
  });

  it("the golden episodes cover every way an episode ends that matters here", () => {
    const reasons = new Set(cases.map((c) => c.result.reason));
    expect(reasons).toEqual(new Set(["success", "time_limit", "loss_of_control"]));
  });
});
