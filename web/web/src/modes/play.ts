/**
 * Play mode: the birds fly in real time, for as long as you like, after a target
 * you move yourself, among obstacles you can move too.
 *
 * The other modes compute a whole episode first and then replay it. Here the
 * scene changes while the birds fly, so we step them as the page runs: at every
 * decision (60 per simulated second) each controller reads the scene as it is
 * now, through the same 48-number observation as in training, and the bird is
 * moved by the same physics (4 steps of 1/240 s, gusts, noise, the arena
 * limits). What changes is the episode around it:
 *   - there is no time limit and no success that ends the flight: holding the
 *     target only fills the "holding" gauge, which stays full;
 *   - a bird that hits the ground, leaves the arena or touches an obstacle
 *     stops where it is, and starts again from the start point a second later
 *     (the controller is reset, as for a new episode). The episodes' 80° rule
 *     is not used: it ends a training episode because a bird tilted that far is
 *     usually lost, but here we let it fight on, and it often recovers (a
 *     target dragged fast makes the birds tilt hard). Only a spin too fast for
 *     the simulation to follow stops it as well, as after the end of a live flight;
 *   - the target can be hidden from the birds: its slot in the observation is
 *     then empty (zeros, "valid" flag 0), which no controller met in training;
 *   - a network trained on a task without obstacles (hover, go-to-point) does
 *     not see them, as in training; it can still hit them. PID and LQR see them.
 * All the birds feel the same wind and gusts, as in the sandbox. This module
 * has no page element, so that the tests can fly it; main.ts draws it.
 */

import type { Controller } from "../control/controller";
import { decodeAction, hoverAction, wingThrusts } from "../sim/aero";
import type { Conditions } from "../sim/disturbances";
import { ARENA_RADIUS, type BirdState, BirdDynamics, CEILING, GROUND_HEIGHT, stateFromEuler } from "../sim/dynamics";
import { buildObservation } from "../sim/observation";
import { type BirdParams, DEFAULT_PARAMS, type Vec3, radians } from "../sim/params";
import { Rng } from "../sim/random";
import type { Quat } from "../sim/rotations";
import { HOLD_RADIUS, HOLD_SPEED, HOLD_STEPS } from "../sim/tasks";
import { GustProcess } from "../sim/wind";
import { type Obstacle, type Scene, START, collides } from "../sim/world";

// s of REAL time a fallen bird stays still (blinking) before starting again, whatever the speed of
// the simulation: at ×1/64 a second of simulated time would be a minute of waiting, at ×4 a blink.
export const RESPAWN_DELAY = 1.0;
export const TRAIL_SECONDS = 4.0; // s of path drawn behind each bird
const MAX_STEPS_PER_CALL = 30; // decisions per frame at most: a hidden tab must not freeze the page on return
// A bird tilted past 80° can spin fast. Above FAST_SPIN a physics step of 1/240 s is cut into
// SUBSTEPS (at 100 rad/s it would turn the bird by half a radian at once and the integration would
// diverge); above MAX_SPIN even those are too coarse, and the bird is stopped. As in env.ts's stepAfterEnd.
const FAST_SPIN = 25.0; // rad/s, far above a flying bird's rates
const SUBSTEPS = 8;
const MAX_SPIN = 400.0; // rad/s

/** Who flies a bird: its controller, and whether it sees the obstacles (as in its training). */
export interface PlayPilot {
  id: number; // the bird's identity in the panel, kept when another bird is removed
  controller: Controller;
  seesObstacles: boolean;
}

/** What is drawn of a bird at one decision. */
export interface PlayFrame {
  pos: Vec3;
  quat: Quat;
  vel: Vec3;
  action: number[]; // the action being applied (for the wings)
  thrust: number; // N
  hold: number; // 0..1
  distance: number; // m, to the target
}

/**
 * The scene a Play session starts with: the target 30 cm ahead of the start, and one obstacle
 * beside the way, on the far side from the camera's first view so that it hides nothing.
 */
export function defaultPlayScene(): Scene {
  return {
    start: [...START] as Vec3,
    targets: [[0.3, 0.5, 0.0]],
    obstacles: [{ center: [0.15, 0.6, -0.2], radius: 0.06 }],
  };
}

const dist = (a: Vec3, b: Vec3) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const clip1 = (x: number) => Math.min(Math.max(x, -1.0), 1.0);

export class PlayBird {
  pilot: PlayPilot;
  state!: BirdState;
  prev!: PlayFrame; // the frame before `now`: drawing interpolates between the two
  now!: PlayFrame;
  /** Why the bird is down, since when (simulated time) and for how long already (real time), or null while it flies. */
  down: { reason: string; at: number; waited: number } | null = null;
  falls = 0; // how many times it has had to start again
  readonly trail: Vec3[] = [];
  readonly dyn: BirdDynamics;
  private prevAction: number[] = [];
  private holdCount = 0;
  private readonly noise: Rng;
  private readonly startRng: Rng;

  constructor(
    pilot: PlayPilot,
    private readonly session: PlaySession,
    seed: number,
  ) {
    this.pilot = pilot;
    this.dyn = new BirdDynamics(session.p);
    this.noise = new Rng(seed);
    this.startRng = new Rng((seed ^ 0x5bd1e995) >>> 0);
    this.applyConditions();
    this.respawn();
  }

  get holdProgress(): number {
    return this.holdCount / HOLD_STEPS;
  }

  applyConditions(): void {
    const c = this.session.conditions;
    this.dyn.setConditions(c.mass_factor, c.damage);
  }

  /** Back at the start point with the small push of a new go-to-point episode, the controller reset. */
  respawn(): void {
    const s = this.session.scene.start;
    const r = this.startRng;
    this.state = stateFromEuler(
      [s[0] + r.between(-0.02, 0.02), s[1] + r.between(-0.02, 0.02), s[2] + r.between(-0.02, 0.02)],
      [r.between(-0.05, 0.05), r.between(-0.05, 0.05), r.between(-0.05, 0.05)],
      r.between(-Math.PI, Math.PI),
      radians(r.between(-5, 5)),
      radians(r.between(-5, 5)),
      [r.between(-0.5, 0.5), r.between(-0.5, 0.5), r.between(-0.5, 0.5)],
    );
    // The controller does not know the bird's mass: it starts from the nominal hover action.
    this.prevAction = hoverAction(this.session.p);
    this.holdCount = 0;
    this.pilot.controller.reset();
    this.down = null;
    this.trail.length = 0;
    this.now = this.frame();
    this.prev = this.now;
  }

  /** A new controller for this bird, which keeps flying from where it is. */
  setPilot(pilot: PlayPilot): void {
    this.pilot = pilot;
    pilot.controller.reset();
  }

  /** The target has moved: any hold so far was a hold somewhere else. */
  resetHold(): void {
    this.holdCount = 0;
  }

  /** What the controller is given now. */
  observe(): Float32Array {
    const se = this.session;
    return buildObservation(
      this.state,
      se.seesTarget ? se.scene.targets.slice(0, 1) : [],
      this.pilot.seesObstacles ? se.scene.obstacles : [],
      this.prevAction,
      se.seesTarget ? this.holdProgress : 0.0,
      se.p.r_bird,
      se.conditions.obs_noise,
      this.noise,
    );
  }

  /** One decision: the controller acts, the physics runs for its 4 steps with the shared `winds`. */
  step(winds: Vec3[]): void {
    const se = this.session;
    this.prev = this.now;
    if (this.down) return; // it lies still until it starts again (PlaySession.advance)
    const out = this.pilot.controller.act(this.observe());
    const a = Array.from({ length: 5 }, (_, i) => clip1(out[i]));
    const c = se.conditions;
    const applied = c.action_noise > 0.0 ? a.map((x) => clip1(x + c.action_noise * this.noise.normal())) : a;
    let reason: string | null = null;
    for (const wind of winds) {
      const spin = Math.hypot(...this.state.omega);
      if (spin <= FAST_SPIN) [this.state] = this.dyn.step(this.state, applied, wind);
      else for (let j = 0; j < SUBSTEPS; j++) [this.state] = this.dyn.step(this.state, applied, wind, se.p.dt / SUBSTEPS);
      reason = this.stopReason();
      if (reason !== null) break;
    }
    this.prevAction = a;
    if (reason !== null) {
      this.down = { reason, at: se.time, waited: 0.0 };
      this.falls += 1;
    }
    // The gauge of the go-to-point task, but it stays full instead of ending the flight.
    const target = se.scene.targets[0];
    const v = this.state.vel;
    const holding = dist(this.state.pos, target) < HOLD_RADIUS && Math.hypot(v[0], v[1], v[2]) < HOLD_SPEED;
    this.holdCount = holding ? Math.min(this.holdCount + 1, HOLD_STEPS) : 0;
    this.now = this.frame();
    this.trail.push([...this.state.pos] as Vec3);
    if (this.trail.length > Math.round(TRAIL_SECONDS / se.controlDt)) this.trail.shift();
  }

  /** Why the flight must stop now, or null: the ground, the arena's limits, an obstacle, a spin too fast. */
  private stopReason(): string | null {
    const [x, y, z] = this.state.pos;
    if (y < GROUND_HEIGHT) return "crash";
    if (Math.hypot(x, z) > ARENA_RADIUS || y > CEILING) return "out_of_bounds";
    if (collides(this.state.pos, this.session.scene.obstacles, this.session.p.r_bird)) return "collision";
    if (Math.hypot(...this.state.omega) > MAX_SPIN) return "spin";
    return null;
  }

  private frame(): PlayFrame {
    const p = this.session.p;
    const [fl, fr] = wingThrusts(decodeAction(this.prevAction, p), p, this.dyn.damage);
    return {
      pos: [...this.state.pos] as Vec3,
      quat: [...this.state.quat] as Quat,
      vel: [...this.state.vel] as Vec3,
      action: [...this.prevAction],
      thrust: fl + fr,
      hold: this.holdProgress,
      distance: dist(this.state.pos, this.session.scene.targets[0]),
    };
  }
}

export class PlaySession {
  readonly p: BirdParams;
  readonly controlDt: number;
  scene: Scene;
  conditions: Conditions;
  seesTarget = true;
  time = 0.0; // s of simulated time
  birds: PlayBird[] = [];
  private gusts: GustProcess;
  private readonly gustRng: Rng;
  private acc = 0.0; // simulated time not yet stepped (less than one decision)
  private nextSeed: number;

  constructor(scene: Scene, conditions: Conditions, pilots: PlayPilot[], seed: number, params: BirdParams = DEFAULT_PARAMS) {
    this.p = params;
    this.controlDt = params.dt * params.action_repeat;
    this.scene = scene;
    this.conditions = playConditions(conditions);
    this.gustRng = new Rng((seed ^ 0x9e3779b9) >>> 0);
    this.gusts = new GustProcess(conditions.mean_wind, conditions.gust_sigma, this.gustRng);
    this.nextSeed = seed;
    this.setPilots(pilots);
  }

  /**
   * The birds, by their panel identity: a bird already flying keeps its state (and its controller,
   * if unchanged: a PID keeps its integrators); a new one starts at the start point.
   */
  setPilots(pilots: PlayPilot[]): void {
    const old = new Map(this.birds.map((b) => [b.pilot.id, b]));
    this.birds = pilots.map((pilot) => {
      const bird = old.get(pilot.id);
      if (!bird) return new PlayBird(pilot, this, (this.nextSeed = (this.nextSeed * 1103515245 + 12345) >>> 0));
      const same = bird.pilot.controller.name === pilot.controller.name && bird.pilot.seesObstacles === pilot.seesObstacles;
      if (!same) bird.setPilot(pilot);
      return bird;
    });
  }

  /** New wind, mass or damage, felt at once. */
  setConditions(c: Conditions): void {
    const w = this.conditions;
    const windChanged = c.mean_wind.some((x, i) => x !== w.mean_wind[i]) || c.gust_sigma !== w.gust_sigma;
    this.conditions = playConditions(c);
    if (windChanged) this.gusts = new GustProcess(c.mean_wind, c.gust_sigma, this.gustRng);
    for (const b of this.birds) b.applyConditions();
  }

  moveTarget(pos: Vec3): void {
    this.scene.targets[0] = [...pos] as Vec3;
    for (const b of this.birds) b.resetHold();
  }

  moveObstacle(index: number, pos: Vec3, radius?: number): void {
    const o = this.scene.obstacles[index];
    this.scene.obstacles[index] = { center: [...pos] as Vec3, radius: radius ?? o.radius };
  }

  addObstacle(o: Obstacle): void {
    this.scene.obstacles.push({ center: [...o.center] as Vec3, radius: o.radius });
  }

  removeObstacle(index: number): void {
    this.scene.obstacles.splice(index, 1);
  }

  /** Every bird back at the start. */
  restartAll(): void {
    for (const b of this.birds) b.respawn();
  }

  /**
   * Runs `simDt` seconds of simulated time, decision by decision; returns how many decisions ran.
   * `realDt`: the real time that went by meanwhile (0 while paused), which the fallen birds wait
   * out before starting again; by default the same as simDt (as at ×1, and in the tests).
   */
  advance(simDt: number, realDt: number = simDt): number {
    if (realDt > 0) {
      for (const b of this.birds) {
        if (!b.down) continue;
        b.down.waited += realDt;
        if (b.down.waited >= RESPAWN_DELAY - 1e-9) b.respawn(); // the times add up a frame at a time
      }
    }
    this.acc += simDt;
    let n = 0;
    while (this.acc >= this.controlDt && n < MAX_STEPS_PER_CALL) {
      // One wind for all the birds, physics step by physics step: they feel the same gusts.
      const winds = Array.from({ length: this.p.action_repeat }, () => [...this.gusts.step(this.p.dt)] as Vec3);
      for (const b of this.birds) b.step(winds);
      this.time += this.controlDt;
      this.acc -= this.controlDt;
      n += 1;
    }
    if (n === MAX_STEPS_PER_CALL) this.acc = 0.0; // what could not be caught up is dropped, not owed
    return n;
  }

  /** How far between the last two decisions the drawing is (0..1). */
  get fraction(): number {
    return Math.min(this.acc / this.controlDt, 1.0);
  }

  /** The wind force now (N), the same for every bird. */
  get wind(): Vec3 {
    return [...this.gusts.state] as Vec3;
  }

  /** Bird `i` as drawn now, between its last two frames. */
  sample(i: number): PlayFrame {
    const b = this.birds[i];
    const u = b.down ? 1.0 : this.fraction;
    const lerp = (x: number, y: number) => x + (y - x) * u;
    const a = b.prev;
    const z = b.now;
    const sign = a.quat.reduce((s, q, k) => s + q * z.quat[k], 0) < 0 ? -1 : 1;
    const q = a.quat.map((x, k) => lerp(x, sign * z.quat[k]));
    const n = Math.hypot(...q);
    return {
      pos: a.pos.map((x, k) => lerp(x, z.pos[k])) as Vec3,
      quat: q.map((x) => x / n) as Quat,
      vel: a.vel.map((x, k) => lerp(x, z.vel[k])) as Vec3,
      action: a.action.map((x, k) => lerp(x, z.action[k])),
      thrust: lerp(a.thrust, z.thrust),
      hold: z.hold,
      distance: dist(a.pos.map((x, k) => lerp(x, z.pos[k])) as Vec3, this.scene.targets[0]),
    };
  }
}

/** The conditions of a session: those set in the panel, without a scheduled sudden failure (no clock to schedule it on). */
function playConditions(c: Conditions): Conditions {
  return { ...c, mean_wind: [...c.mean_wind] as Vec3, damage: [...c.damage] as [number, number], failure_time: null };
}

/** Words for why a bird fell, for the panel. */
export function fallWords(reason: string): string {
  switch (reason) {
    case "collision":
      return "hit an obstacle";
    case "crash":
      return "hit the ground";
    case "out_of_bounds":
      return "left the arena";
    case "spin":
      return "spun out of control";
    case "loss_of_control":
      return "lost control (tilted past 80°)";
    default:
      return reason;
  }
}
