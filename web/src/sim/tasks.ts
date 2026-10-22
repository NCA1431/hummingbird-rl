/**
 * The four tasks: start state, reward, holds and waypoints, when an episode is won.
 *
 * Port of hummingbird_rl/tasks/*.py; the reasons for each reward term are
 * explained there. In short:
 *   hover        recover from a push and stay within 3 cm on average over the last 2 s (5 s episode)
 *   goto         fly 10-50 cm and hold the target (within 3 cm, slower than 10 cm/s, for 15 decisions)
 *   obstacles    the same, 50-80 cm away behind spheres; touching one ends the episode (6 s)
 *   navigation   pass two waypoints (within 4 cm), then hold the third (15 s)
 * The environment (env.ts) detects crashes and collisions and asks the task
 * for its failure reward.
 */

import { type BirdState, eulerOf, stateFromEuler } from "./dynamics";
import { type BirdParams, type Vec3, hoverPower, radians } from "./params";
import type { Rng } from "./random";
import type { Scene } from "./world";

export const HOLD_RADIUS = 0.03; // m
export const HOLD_SPEED = 0.1; // m/s
export const HOLD_STEPS = 15;
export const PASS_RADIUS = 0.04; // m, navigation's intermediate waypoints
const FAILURE_REWARD = -10.0;
const SMOOTHNESS_WEIGHT = 0.02;
const ENERGY_WEIGHT = 0.01;
const PROGRESS_WEIGHT = 100.0; // per metre of progress
const SUCCESS_BONUS = 50.0;
const WAYPOINT_BONUS = 20.0;
const TIME_PENALTY = 0.01;
const HOVER_SUCCESS_RADIUS = 0.03; // m, mean distance allowed over the last 2 s
const HOVER_SUCCESS_WINDOW = 2.0; // s

export interface Outcome {
  reward: number;
  terminated: boolean; // the episode ends because the task is won
  success: boolean;
  distance: number;
}

const dist = (a: Vec3, b: Vec3) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

export abstract class Task {
  abstract readonly name: string;
  abstract readonly duration: number; // s
  readonly hasObstacles: boolean = false;
  readonly p: BirdParams;
  readonly controlDt: number;
  scene!: Scene;
  stepCount = 0;
  holdCount = 0;
  index = 0; // current target (only navigation moves on)

  constructor(p: BirdParams) {
    this.p = p;
    this.controlDt = p.dt * p.action_repeat;
  }

  get nSteps(): number {
    return Math.round(this.duration / this.controlDt);
  }

  /** Default start: close to the start point with a small push, facing any direction. */
  initialState(scene: Scene, rng: Rng): BirdState {
    const s = scene.start;
    return stateFromEuler(
      [s[0] + rng.between(-0.02, 0.02), s[1] + rng.between(-0.02, 0.02), s[2] + rng.between(-0.02, 0.02)],
      [rng.between(-0.05, 0.05), rng.between(-0.05, 0.05), rng.between(-0.05, 0.05)],
      rng.between(-Math.PI, Math.PI),
      radians(rng.between(-5, 5)),
      radians(rng.between(-5, 5)),
      [rng.between(-0.5, 0.5), rng.between(-0.5, 0.5), rng.between(-0.5, 0.5)],
    );
  }

  reset(scene: Scene, _state: BirdState): void {
    this.scene = scene;
    this.stepCount = 0;
    this.holdCount = 0;
    this.index = 0;
  }

  /** Up to two targets the controller sees, the current one first. */
  obsTargets(): Vec3[] {
    return this.scene.targets.slice(0, 1);
  }

  holdProgress(): number {
    return this.holdCount / HOLD_STEPS;
  }

  protected updateHold(state: BirdState, target: Vec3): boolean {
    const v = state.vel;
    const holding = dist(state.pos, target) < HOLD_RADIUS && Math.hypot(v[0], v[1], v[2]) < HOLD_SPEED;
    this.holdCount = holding ? this.holdCount + 1 : 0;
    return this.holdCount >= HOLD_STEPS;
  }

  /** Small penalties shared by every task: jerky actions and energy (relative to hovering). */
  protected commonTerms(action: ArrayLike<number>, prevAction: ArrayLike<number>, power: number): number {
    let jerk = 0.0;
    for (let i = 0; i < 5; i++) jerk += (action[i] - prevAction[i]) ** 2;
    return -SMOOTHNESS_WEIGHT * jerk - (ENERGY_WEIGHT * power) / hoverPower(this.p);
  }

  abstract evaluate(state: BirdState, action: ArrayLike<number>, prevAction: ArrayLike<number>, power: number): Outcome;

  failureReward(): number {
    return FAILURE_REWARD;
  }

  /** Distance to the current target. */
  distance(state: BirdState): number {
    return dist(state.pos, this.scene.targets[this.index]);
  }
}

class HoverTask extends Task {
  readonly name = "hover";
  readonly duration = 5.0;
  private distances: number[] = [];

  /** A much larger push than the other tasks: this task is about recovering from it. */
  initialState(scene: Scene, rng: Rng): BirdState {
    const s = scene.start;
    return stateFromEuler(
      [s[0] + rng.between(-0.1, 0.1), s[1] + rng.between(-0.1, 0.1), s[2] + rng.between(-0.1, 0.1)],
      [rng.between(-0.3, 0.3), rng.between(-0.3, 0.3), rng.between(-0.3, 0.3)],
      rng.between(-Math.PI, Math.PI),
      radians(rng.between(-15, 15)),
      radians(rng.between(-15, 15)),
      [rng.between(-2, 2), rng.between(-2, 2), rng.between(-2, 2)],
    );
  }

  reset(scene: Scene, state: BirdState): void {
    super.reset(scene, state);
    this.distances = [];
  }

  evaluate(state: BirdState, action: ArrayLike<number>, prevAction: ArrayLike<number>, power: number): Outcome {
    this.stepCount += 1;
    const d = dist(state.pos, this.scene.targets[0]);
    this.distances.push(d);
    const [, pitch, roll] = eulerOf(state);
    const reward =
      Math.exp(-d / 0.05) -
      0.05 * (roll ** 2 + pitch ** 2) -
      0.01 * state.omega[1] ** 2 +
      this.commonTerms(action, prevAction, power);
    let success = false;
    if (this.stepCount >= this.nSteps) {
      // judged only at the end of the episode
      const window = Math.round(HOVER_SUCCESS_WINDOW / this.controlDt);
      const last = this.distances.slice(-window);
      success = last.reduce((a, b) => a + b, 0) / last.length < HOVER_SUCCESS_RADIUS;
    }
    return { reward, terminated: false, success, distance: d };
  }
}

class GotoTask extends Task {
  readonly name: string = "goto";
  readonly duration: number = 5.0;
  private prevDistance = 0.0;

  reset(scene: Scene, state: BirdState): void {
    super.reset(scene, state);
    this.prevDistance = dist(state.pos, scene.targets[0]);
  }

  evaluate(state: BirdState, action: ArrayLike<number>, prevAction: ArrayLike<number>, power: number): Outcome {
    this.stepCount += 1;
    const target = this.scene.targets[0];
    const d = dist(state.pos, target);
    const reward =
      PROGRESS_WEIGHT * (this.prevDistance - d) + 0.5 * Math.exp(-d / 0.03) + this.commonTerms(action, prevAction, power);
    this.prevDistance = d;
    if (this.updateHold(state, target)) return { reward: reward + SUCCESS_BONUS, terminated: true, success: true, distance: d };
    return { reward, terminated: false, success: false, distance: d };
  }
}

class ObstaclesTask extends GotoTask {
  readonly name = "obstacles";
  readonly duration = 6.0;
  readonly hasObstacles = true;
}

class NavigationTask extends Task {
  readonly name = "navigation";
  readonly duration = 15.0;
  readonly hasObstacles = true;
  private prevDistance = 0.0;

  get isLast(): boolean {
    return this.index === this.scene.targets.length - 1;
  }

  reset(scene: Scene, state: BirdState): void {
    super.reset(scene, state);
    this.prevDistance = dist(state.pos, scene.targets[0]);
  }

  obsTargets(): Vec3[] {
    return this.scene.targets.slice(this.index, this.index + 2);
  }

  holdProgress(): number {
    return this.isLast ? super.holdProgress() : 0.0;
  }

  evaluate(state: BirdState, action: ArrayLike<number>, prevAction: ArrayLike<number>, power: number): Outcome {
    this.stepCount += 1;
    const target = this.scene.targets[this.index];
    const d = dist(state.pos, target);
    const reward = PROGRESS_WEIGHT * (this.prevDistance - d) - TIME_PENALTY + this.commonTerms(action, prevAction, power);
    this.prevDistance = d;
    if (!this.isLast && d < PASS_RADIUS) {
      // Next waypoint; the progress reference restarts from the new distance.
      this.index += 1;
      this.prevDistance = dist(state.pos, this.scene.targets[this.index]);
      return { reward: reward + WAYPOINT_BONUS, terminated: false, success: false, distance: d };
    }
    if (this.isLast && this.updateHold(state, target)) {
      return { reward: reward + WAYPOINT_BONUS + SUCCESS_BONUS, terminated: true, success: true, distance: d };
    }
    return { reward, terminated: false, success: false, distance: d };
  }
}

export const TASKS = ["hover", "goto", "obstacles", "navigation"] as const;

export function makeTask(name: string, p: BirdParams): Task {
  switch (name) {
    case "hover":
      return new HoverTask(p);
    case "goto":
      return new GotoTask(p);
    case "obstacles":
      return new ObstaclesTask(p);
    case "navigation":
      return new NavigationTask(p);
    default:
      throw new Error(`unknown task ${name}`);
  }
}
