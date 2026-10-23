/**
 * Live mode: a new episode, flown in the browser by the chosen controller.
 *
 * The whole episode is simulated at once (at most 15 s of flight, a few tens
 * of milliseconds of computation) and written in the same format as the
 * recorded ones. The viewer then plays it with the Watch mode's machinery:
 * same timeline, speeds, cameras, trail and HUD. The difference with the
 * recorded episodes is where it comes from: a new random situation drawn in
 * the browser, never one of the Python test scenes.
 *
 * One addition: an episode ends at 80° of tilt ("loss of control"), which says
 * nothing of what the bird would do next. In the live mode we go on, the
 * controller still trying, until the episode's normal time limit or until the
 * bird reaches the ground, leaves the arena or spins too fast to simulate.
 * Sometimes it falls; sometimes the 80° was a brief overshoot and the bird
 * recovers. These frames are marked (`endFrame`): they are shown, not counted
 * in the result. We do not do it after a collision, since the bird would fly
 * through the obstacle (the simulator has no contact forces).
 */

import type { Controller } from "../control/controller";
import { decodeAction, wingThrusts } from "../sim/aero";
import type { RegimeName } from "../sim/disturbances";
import { HummingbirdEnv, type ResetOptions } from "../sim/env";
import type { Vec3 } from "../sim/params";
import type { Quat } from "../sim/rotations";
import type { Episode } from "./watch";

export const LIVE_FORMAT = "hummingbird-episode/1";

/**
 * One episode. `options` can fix the scene and the conditions instead of drawing them (the
 * sandbox does); the start push and the noise always come from `seed`.
 */
export function simulateEpisode(
  task: string,
  regime: RegimeName,
  controller: Controller,
  seed: number,
  options: ResetOptions = {},
  suite = "live",
): Episode {
  const env = new HummingbirdEnv(task, regime);
  let obs = env.reset(seed, options);
  controller.reset();
  const f: Episode["frames"] = { t: [], pos: [], quat: [], vel: [], action: [], thrust: [], wind: [], target: [], hold: [], distance: [] };
  const record = () => {
    const s = env.state;
    const [fl, fr] = wingThrusts(decodeAction(env.prevAction, env.p), env.p, env.dyn.damage);
    f.t.push(s.time);
    f.pos.push([...s.pos] as Vec3);
    f.quat.push([...s.quat] as Quat);
    f.vel.push([...s.vel] as Vec3);
    f.action.push([...env.prevAction]);
    f.thrust.push(fl + fr);
    f.wind.push([...env.gusts.state] as Vec3);
    f.target.push(env.task.index);
    f.hold.push(env.task.holdProgress());
    f.distance.push(env.task.distance(s));
  };

  record();
  let total = 0.0;
  let length = 0;
  for (;;) {
    const r = env.step(controller.act(obs));
    obs = r.obs;
    total += r.reward;
    length += 1;
    record();
    if (r.terminated || r.truncated) {
      const c = env.conditions;
      const ep: Episode = {
        format: LIVE_FORMAT,
        task,
        seed,
        policy: controller.name,
        suite,
        dt: env.p.dt * env.p.action_repeat,
        scene: env.scene,
        conditions: { ...c, damage: [...c.damage] as [number, number] },
        result: { success: r.success, reason: r.reason ?? "unknown", return: total, length, energy: env.energy },
        frames: f,
      };
      if (r.reason === "loss_of_control") {
        ep.endFrame = length;
        for (let k = length; k < env.task.nSteps; k++) {
          const after = env.stepAfterEnd(controller.act(obs));
          obs = after.obs;
          record();
          if (after.stop !== null) break;
        }
      }
      return ep;
    }
  }
}

/** A new random seed for the next live flight. */
export function newSeed(): number {
  return Math.floor(Math.random() * 1_000_000);
}
