/**
 * Play mode: birds flown in real time after a target that moves, among obstacles that move.
 *
 * What we check is what makes the mode honest: the controllers get the same
 * observation as in training (with the target hidden when asked, and the
 * obstacles hidden from networks that never saw any), the flight never ends by
 * itself, a fall is followed by a restart at the start point, and all the birds
 * feel the same wind.
 */

import { describe, expect, it } from "vitest";

import type { Controller } from "../src/control/controller";
import { LQRController } from "../src/control/lqr";
import { PIDController } from "../src/control/pid";
import { PlaySession, RESPAWN_DELAY, defaultPlayScene } from "../src/modes/play";
import { calmConditions } from "../src/sim/disturbances";
import { hoverAction } from "../src/sim/aero";
import * as ob from "../src/sim/observation";
import { DEFAULT_PARAMS, type Vec3, weight } from "../src/sim/params";

const p = DEFAULT_PARAMS;
const dist = (a: Vec3, b: Vec3) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

/** A PID that also keeps every observation it was given. */
class Spy implements Controller {
  readonly name: string;
  readonly seen: Float32Array[] = [];
  private readonly pid = new PIDController(p);
  constructor(name = "spy") {
    this.name = name;
  }
  reset(): void {
    this.pid.reset();
  }
  act(obs: ArrayLike<number>): number[] {
    this.seen.push(Float32Array.from(obs));
    return this.pid.act(obs);
  }
}

const pid = (id = 0) => ({ id, controller: new PIDController(p), seesObstacles: true });
const fly = (s: PlaySession, seconds: number) => {
  for (let t = 0; t < seconds; t += 0.25) s.advance(0.25);
};

describe("play: flying after a moving target", () => {
  it("the PID reaches the target and holds it, and the flight does not end", () => {
    const scene = defaultPlayScene();
    scene.obstacles = [];
    const s = new PlaySession(scene, calmConditions(), [pid()], 1);
    fly(s, 4);
    const b = s.birds[0];
    expect(dist(b.state.pos, scene.targets[0])).toBeLessThan(0.03);
    expect(b.holdProgress).toBe(1); // full, and it stays full: no success ends the flight
    fly(s, 20);
    expect(b.down).toBeNull();
    expect(s.time).toBeGreaterThan(23.9);
    expect(b.holdProgress).toBe(1);
  });

  it("follows the target when it is moved, and the hold starts again", () => {
    const scene = defaultPlayScene();
    scene.obstacles = [];
    const s = new PlaySession(scene, calmConditions(), [pid()], 2);
    fly(s, 4);
    const next: Vec3 = [-0.2, 0.7, 0.25];
    s.moveTarget(next);
    expect(s.birds[0].holdProgress).toBe(0);
    fly(s, 5);
    expect(dist(s.birds[0].state.pos, next)).toBeLessThan(0.03);
  });

  it("runs whole decisions only, and catches up at most 30 per call", () => {
    const s = new PlaySession(defaultPlayScene(), calmConditions(), [pid()], 3);
    expect(s.advance(0.5 * s.controlDt)).toBe(0);
    expect(s.advance(0.5 * s.controlDt)).toBe(1);
    expect(s.advance(10.0)).toBe(30); // a long pause (a hidden tab) is dropped, not owed
    expect(s.advance(0.0)).toBe(0);
  });
});

describe("play: what the controllers see", () => {
  it("a hidden target leaves its slot empty, and the PID stays where it is", () => {
    const scene = defaultPlayScene();
    scene.obstacles = [];
    const spy = new Spy();
    const s = new PlaySession(scene, calmConditions(), [{ id: 0, controller: spy, seesObstacles: true }], 4);
    s.seesTarget = false;
    const start = [...s.birds[0].state.pos] as Vec3;
    fly(s, 3);
    for (const obs of spy.seen) {
      for (let k = 0; k < 8; k++) expect(obs[ob.TARGETS + k]).toBe(0); // both target slots, flags at 0
      expect(obs[ob.HOLD]).toBe(0);
    }
    expect(dist(s.birds[0].state.pos, start)).toBeLessThan(0.05); // the target is 30 cm away
    s.seesTarget = true;
    fly(s, 4);
    expect(dist(s.birds[0].state.pos, scene.targets[0])).toBeLessThan(0.03);
  });

  it("a network trained without obstacles does not see them; the others do", () => {
    const blind = new Spy("ppo_goto_normal_s1");
    const seeing = new Spy("ppo_obstacles_normal_s1");
    const s = new PlaySession(
      defaultPlayScene(),
      calmConditions(),
      [
        { id: 0, controller: blind, seesObstacles: false },
        { id: 1, controller: seeing, seesObstacles: true },
      ],
      5,
    );
    s.advance(s.controlDt);
    const flags = (obs: Float32Array) => Array.from({ length: ob.N_OBSTACLES }, (_, i) => obs[ob.OBSTACLES + 5 * i + 4]);
    expect(flags(blind.seen[0]).every((f) => f === 0)).toBe(true);
    expect(flags(seeing.seen[0])[0]).toBe(1);
  });
});

describe("play: falls and restarts", () => {
  it("an obstacle pushed onto a bird makes it fall; it starts again from the start point", () => {
    const scene = defaultPlayScene();
    const s = new PlaySession(scene, calmConditions(), [pid()], 6);
    fly(s, 3);
    const b = s.birds[0];
    s.moveObstacle(0, [...b.state.pos] as Vec3);
    s.advance(s.controlDt);
    expect(b.down?.reason).toBe("collision");
    expect(b.falls).toBe(1);
    const lying = [...b.state.pos] as Vec3;
    s.advance(0.5 * RESPAWN_DELAY);
    expect(b.state.pos).toEqual(lying); // still, until the delay is over
    s.moveObstacle(0, [0.6, 0.3, -0.6]); // out of the way before it comes back
    fly(s, 0.75 * RESPAWN_DELAY);
    expect(b.down).toBeNull();
    // A new path, from the start point, not a line from where it fell.
    expect(b.trail.length).toBeLessThan(20);
    expect(dist(b.trail[0], scene.start)).toBeLessThan(0.05);
  });

  it("a bird that is not given lift falls to the ground and comes back", () => {
    const off: Controller = { name: "off", reset() {}, act: () => [-1, -1, -1, 0, 0] };
    const s = new PlaySession(defaultPlayScene(), calmConditions(), [{ id: 0, controller: off, seesObstacles: true }], 7);
    fly(s, 2);
    expect(s.birds[0].falls).toBeGreaterThanOrEqual(1);
  });
});

describe("play: the birds and the conditions", () => {
  it("every bird feels the same wind; changing a controller keeps the bird where it is", () => {
    const c = calmConditions();
    c.mean_wind = [0.2 * weight(p), 0, 0];
    c.gust_sigma = 0.05 * weight(p);
    const scene = defaultPlayScene();
    scene.obstacles = [];
    // Two identical controllers, the same start push: identical flights only if the gusts are shared.
    const s = new PlaySession(scene, c, [pid(0), pid(1)], 8);
    s.birds[1].state = structuredClone(s.birds[0].state);
    s.birds[1].now = structuredClone(s.birds[0].now);
    fly(s, 2);
    expect(s.birds[1].state.pos).toEqual(s.birds[0].state.pos);

    const where = [...s.birds[1].state.pos];
    const kept = s.birds[1].pilot.controller;
    s.setPilots([pid(0), pid(1)]); // the same controller again: the bird keeps its own (and its integrators)
    expect(s.birds[1].pilot.controller).toBe(kept);
    const replacement = new LQRController(p);
    s.setPilots([pid(0), { id: 1, controller: replacement, seesObstacles: true }]);
    expect(s.birds[1].state.pos).toEqual(where);
    expect(s.birds[1].pilot.controller).toBe(replacement);
    s.setPilots([{ id: 1, controller: replacement, seesObstacles: true }]); // bird 0 removed
    expect(s.birds).toHaveLength(1);
    expect(s.birds[0].state.pos).toEqual(where);
  });

  it("mass and damage apply at once, and a scheduled sudden failure is dropped", () => {
    const s = new PlaySession(defaultPlayScene(), calmConditions(), [pid()], 9);
    const c = calmConditions();
    c.mass_factor = 1.2;
    c.damage = [0.3, 0];
    c.failure_time = 1.0;
    s.setConditions(c);
    expect(s.birds[0].dyn.massFactor).toBe(1.2);
    expect(s.birds[0].dyn.damage).toEqual([0.3, 0]);
    expect(s.conditions.failure_time).toBeNull();
    expect(s.birds[0].now.action).toEqual(hoverAction(p));
  });

  it("draws a bird between its last two decisions", () => {
    const scene = defaultPlayScene();
    scene.obstacles = [];
    const s = new PlaySession(scene, calmConditions(), [pid()], 10);
    fly(s, 1);
    s.advance(0.5 * s.controlDt);
    const b = s.birds[0];
    const mid = s.sample(0).pos;
    for (let k = 0; k < 3; k++) expect(mid[k]).toBeCloseTo(0.5 * (b.prev.pos[k] + b.now.pos[k]), 9);
  });
});
