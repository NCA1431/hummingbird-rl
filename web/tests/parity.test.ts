/**
 * Parity tests: the TypeScript simulator must reproduce the Python one.
 *
 * The golden files are written by `scripts/export_web.py golden`: recorded
 * inputs (start state, conditions, actions, wind) and the outputs Python
 * computed from them. Both languages use 64-bit floats and we kept the order
 * of every operation, so the only differences left come from the last bit of
 * functions like sin or sqrt; 1e-9 leaves a wide margin above those and
 * catches any real mistake (a wrong sign or coefficient shows up at 1e-3).
 */

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import { hoverAction } from "../src/sim/aero";
import { type BirdState, BirdDynamics, checkLimits } from "../src/sim/dynamics";
import { buildObservation } from "../src/sim/observation";
import {
  DEFAULT_PARAMS,
  type Vec3,
  damping,
  hoverAmplitude,
  hoverPower,
  inertia,
  maxThrust,
  paramsFromPython,
  rWing,
  thrustGain,
  weight,
} from "../src/sim/params";
import { type Quat, eulerFromQuat } from "../src/sim/rotations";
import type { Obstacle } from "../src/sim/world";

const TOL = 1e-9;

function load<T>(name: string): T {
  return JSON.parse(readFileSync(new URL(`./golden/${name}.json`, import.meta.url), "utf8")) as T;
}

interface GoldenCase {
  params: Record<string, unknown>;
  mass_factor: number;
  damage: [number, number];
  start: BirdState;
  actions: number[][];
  wind: Vec3[];
  expected: {
    pos: Vec3[];
    vel: Vec3[];
    quat: Quat[];
    omega: Vec3[];
    time: number[];
    phase: number[];
    euler: Vec3[];
    thrust: number[];
    power: number[];
  };
}

interface GoldenDynamics {
  default_params: Record<string, unknown>;
  derived: Record<string, number | number[]>;
  cases: GoldenCase[];
  limits: { pos: Vec3; quat: Quat; expected: string | null }[];
}

interface GoldenObservation {
  state: { pos: Vec3; vel: Vec3; quat: Quat; omega: Vec3 };
  targets: Vec3[];
  obstacles: Obstacle[];
  prev_action: number[];
  hold: number;
  r_bird: number;
  expected: number[];
}

const dynamics = load<GoldenDynamics>("dynamics");
const observations = load<{ cases: GoldenObservation[] }>("observations").cases;

/** Largest error relative to (1 + |expected|): relative for large values, absolute near zero. */
function worstError(actual: ArrayLike<number>, expected: ArrayLike<number>): number {
  let worst = 0.0;
  for (let i = 0; i < expected.length; i++) {
    worst = Math.max(worst, Math.abs(actual[i] - expected[i]) / (1.0 + Math.abs(expected[i])));
  }
  return worst;
}

/** Angles that may wrap around ±π (the wingbeat phase is taken modulo 2π). */
function angleError(a: number, b: number): number {
  const d = a - b;
  return Math.abs(d - 2.0 * Math.PI * Math.round(d / (2.0 * Math.PI)));
}

describe("parameters", () => {
  it("defaults are exactly Python's", () => {
    for (const [key, value] of Object.entries(dynamics.default_params)) {
      expect((DEFAULT_PARAMS as unknown as Record<string, unknown>)[key], key).toEqual(value);
    }
  });

  it("derived quantities match", () => {
    const d = dynamics.derived;
    const p = DEFAULT_PARAMS;
    const pairs: [string, number[], number[]][] = [
      ["r_wing", [rWing(p)], [d.r_wing as number]],
      ["weight", [weight(p)], [d.weight as number]],
      ["inertia", inertia(p), d.inertia as number[]],
      ["damping", damping(p), d.damping as number[]],
      ["thrust_gain", [thrustGain(p)], [d.thrust_gain as number]],
      ["hover_amplitude", [hoverAmplitude(p)], [d.hover_amplitude as number]],
      ["hover_amplitude_heavy", [hoverAmplitude(p, 1.2)], [d.hover_amplitude_heavy as number]],
      ["max_thrust", [maxThrust(p)], [d.max_thrust as number]],
      ["hover_power", [hoverPower(p)], [d.hover_power as number]],
      ["hover_action", hoverAction(p), d.hover_action as number[]],
    ];
    for (const [name, actual, expected] of pairs) {
      // Relative to the value itself here: inertias are ~1e-7, so (1 + |x|) would hide everything.
      for (let i = 0; i < expected.length; i++) {
        expect(Math.abs(actual[i] - expected[i]) / Math.abs(expected[i] || 1), name).toBeLessThan(1e-12);
      }
    }
  });
});

describe("dynamics", () => {
  it.each(dynamics.cases.map((c, i) => [i, c] as const))("case %i follows Python", (_, c) => {
    const p = paramsFromPython(c.params);
    const dyn = new BirdDynamics(p, c.mass_factor, c.damage);
    let state: BirdState = { ...c.start };
    const errors: Record<string, number> = {};
    const record = (name: string, e: number) => (errors[name] = Math.max(errors[name] ?? 0, e));

    c.actions.forEach((action, k) => {
      let info;
      for (let s = 0; s < p.action_repeat; s++) [state, info] = dyn.step(state, action, c.wind[k]);
      const e = c.expected;
      record("pos", worstError(state.pos, e.pos[k]));
      record("vel", worstError(state.vel, e.vel[k]));
      record("quat", worstError(state.quat, e.quat[k]));
      record("omega", worstError(state.omega, e.omega[k]));
      record("time", worstError([state.time], [e.time[k]]));
      record("phase", angleError(state.phase, e.phase[k]));
      record("euler", worstError(eulerFromQuat(state.quat), e.euler[k]));
      record("thrust", worstError([info!.thrust], [e.thrust[k]]));
      record("power", worstError([info!.power], [e.power[k]]));
    });
    for (const [name, e] of Object.entries(errors)) expect(e, name).toBeLessThan(TOL);
  });

  it("arena limits give the same verdicts", () => {
    for (const l of dynamics.limits) {
      const state: BirdState = { pos: l.pos, vel: [0, 0, 0], quat: l.quat, omega: [0, 0, 0], time: 0, phase: 0 };
      expect(checkLimits(state), JSON.stringify(l.pos)).toBe(l.expected);
    }
  });
});

describe("observation", () => {
  it.each(observations.map((c, i) => [i, c] as const))("case %i matches Python", (_, c) => {
    const state: BirdState = { ...c.state, time: 0, phase: 0 };
    const obs = buildObservation(state, c.targets, c.obstacles, c.prev_action, c.hold, c.r_bird);
    expect(obs.length).toBe(48);
    // Both sides round to float32 at the end; allow one float32 step in case a
    // last-bit difference in float64 falls on a rounding boundary.
    for (let i = 0; i < 48; i++) expect(Math.abs(obs[i] - c.expected[i]), `entry ${i}`).toBeLessThan(1e-6);
  });
});
