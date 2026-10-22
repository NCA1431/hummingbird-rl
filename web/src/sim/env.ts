/**
 * One episode of one task, in the browser: the same loop as the Python environment.
 *
 * Port of hummingbird_rl/envs/hummingbird_env.py. At every decision (60 per
 * second) the controller's 5 actions get the actuator noise, are held for 4
 * physics steps of 1/240 s with the current gusts, the limits and collisions
 * are checked after every physics step, and the task scores the result. An
 * episode ends by a failure (crash, out of the arena, tilted past 80°,
 * obstacle hit), by a success, or at the time limit. So a live flight stops
 * by itself at the arena limits, before the integration could diverge.
 *
 * Randomness comes from four separate generators: the scene, the conditions
 * and start state, the gusts, and the sensor and actuator noise. Two
 * controllers given the same seed therefore meet exactly the same situation,
 * gust for gust, even though they draw a different amount of sensor noise
 * along the way (the number of visible obstacles changes with the path).
 *
 * `stepAfterEnd` is not part of the Python environment: it lets the live mode
 * show what happens after a loss of control, which the episode itself does
 * not count.
 */

import { hoverAction } from "./aero";
import { type Conditions, type RegimeName, sampleConditions } from "./disturbances";
import { ARENA_RADIUS, type BirdState, BirdDynamics, CEILING, GROUND_HEIGHT, checkLimits } from "./dynamics";
import { buildObservation } from "./observation";
import { type BirdParams, DEFAULT_PARAMS } from "./params";
import { Rng } from "./random";
import { sampleScene } from "./sampling";
import { type Task, makeTask } from "./tasks";
import { GustProcess } from "./wind";
import { type Scene, collides } from "./world";

export interface ResetOptions {
  scene?: Scene;
  conditions?: Conditions;
  state?: BirdState;
}

export interface StepResult {
  obs: Float32Array;
  reward: number;
  terminated: boolean;
  truncated: boolean;
  success: boolean;
  reason: string | null; // "success", "time_limit", "crash", "out_of_bounds", "loss_of_control", "collision"
  distance: number;
}

const clip1 = (x: number) => Math.min(Math.max(x, -1.0), 1.0);

export class HummingbirdEnv {
  readonly p: BirdParams;
  readonly task: Task;
  readonly regime: RegimeName;
  readonly dyn: BirdDynamics;
  scene!: Scene;
  conditions!: Conditions;
  state!: BirdState;
  gusts!: GustProcess;
  prevAction: number[] = [];
  energy = 0.0;
  private failed = false;
  private noise = new Rng();

  constructor(task: string, regime: RegimeName = "normal", params: BirdParams = DEFAULT_PARAMS) {
    this.p = params;
    this.task = makeTask(task, params);
    this.regime = regime;
    this.dyn = new BirdDynamics(params);
  }

  reset(seed: number, options: ResetOptions = {}): Float32Array {
    // Four streams from one seed (the constants only make the four seeds different).
    const sceneRng = new Rng(seed);
    const startRng = new Rng((seed ^ 0x5bd1e995) >>> 0);
    const gustRng = new Rng((seed ^ 0x9e3779b9) >>> 0);
    this.noise = new Rng((seed ^ 0x27d4eb2f) >>> 0);
    this.scene = options.scene ?? sampleScene(this.task.name, sceneRng, this.p.r_bird);
    this.conditions = options.conditions ?? sampleConditions(this.regime, startRng, this.p);
    this.state = options.state ? structuredClone(options.state) : this.task.initialState(this.scene, startRng);

    const c = this.conditions;
    this.dyn.setConditions(c.mass_factor, c.damage);
    this.gusts = new GustProcess(c.mean_wind, c.gust_sigma, gustRng);
    this.failed = false;
    this.task.reset(this.scene, this.state);
    // The controller does not know the bird's mass: it starts from the nominal hover action.
    this.prevAction = hoverAction(this.p);
    this.energy = 0.0;
    return this.observe();
  }

  step(action: ArrayLike<number>): StepResult {
    const a = Array.from({ length: 5 }, (_, i) => clip1(action[i]));
    const c = this.conditions;
    const applied = c.action_noise > 0.0 ? a.map((x) => clip1(x + c.action_noise * this.noise.normal())) : a;

    let reason: string | null = null;
    let power = 0.0;
    for (let k = 0; k < this.p.action_repeat; k++) {
      this.applyFailure();
      const [next, info] = this.dyn.step(this.state, applied, this.gusts.step(this.p.dt));
      this.state = next;
      power += info.power / this.p.action_repeat;
      this.energy += info.power * this.p.dt;
      reason = checkLimits(this.state);
      if (reason === null && this.task.hasObstacles && collides(this.state.pos, this.scene.obstacles, this.p.r_bird)) {
        reason = "collision";
      }
      if (reason !== null) break;
    }

    let reward: number;
    let terminated: boolean;
    let success = false;
    let distance: number;
    if (reason !== null) {
      reward = this.task.failureReward();
      terminated = true;
      distance = this.task.distance(this.state);
    } else {
      const out = this.task.evaluate(this.state, a, this.prevAction, power);
      ({ reward, terminated, success, distance } = out);
      if (terminated) reason = "success";
    }
    const truncated = !terminated && this.task.stepCount >= this.task.nSteps;
    if (truncated) reason = "time_limit";
    this.prevAction = a;
    return { obs: this.observe(), reward, terminated, truncated, success, reason, distance };
  }

  /**
   * One more decision after the episode has ended, physics only: no reward, no
   * 80° rule. A bird out of control can spin at more than 100 rad/s, where a
   * 1/240 s step would turn it by half a radian and the integration would
   * diverge, so each physics step is cut into `substeps`. Returns why the flight
   * must now stop for good, or null: the bird reached the ground, left the
   * arena, or spins faster than `maxSpin` (rad/s) even for those small steps.
   */
  stepAfterEnd(action: ArrayLike<number>, substeps = 8, maxSpin = 400.0): { obs: Float32Array; stop: string | null } {
    const a = Array.from({ length: 5 }, (_, i) => clip1(action[i]));
    const c = this.conditions;
    const applied = c.action_noise > 0.0 ? a.map((x) => clip1(x + c.action_noise * this.noise.normal())) : a;
    const h = this.p.dt / substeps;
    let stop: string | null = null;
    for (let k = 0; k < this.p.action_repeat && stop === null; k++) {
      this.applyFailure();
      const wind = this.gusts.step(this.p.dt);
      for (let j = 0; j < substeps; j++) [this.state] = this.dyn.step(this.state, applied, wind, h);
      const [x, y, z] = this.state.pos;
      if (y < GROUND_HEIGHT) stop = "ground";
      else if (Math.hypot(x, z) > ARENA_RADIUS || y > CEILING) stop = "out_of_bounds";
      else if (Math.hypot(...this.state.omega) > maxSpin) stop = "spin";
    }
    this.prevAction = a;
    return { obs: this.observe(), stop };
  }

  /** One wing loses part of its force once the scheduled failure time is reached. */
  private applyFailure(): void {
    const c = this.conditions;
    if (c.failure_time === null || this.failed || this.state.time < c.failure_time) return;
    const damage: [number, number] = [...this.dyn.damage];
    damage[c.failure_wing] = Math.max(damage[c.failure_wing], c.failure_amount);
    this.dyn.setConditions(c.mass_factor, damage);
    this.failed = true;
  }

  private observe(): Float32Array {
    const obstacles = this.task.hasObstacles ? this.scene.obstacles : [];
    return buildObservation(
      this.state,
      this.task.obsTargets(),
      obstacles,
      this.prevAction,
      this.task.holdProgress(),
      this.p.r_bird,
      this.conditions.obs_noise,
      this.noise,
    );
  }
}
