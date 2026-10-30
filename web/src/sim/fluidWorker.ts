/**
 * The air's worker: the solver and what is drawn of it, off the page's main thread.
 *
 * One solver step of the 40³ grid takes about 10 ms, a good part of a frame (16 ms at 60 Hz):
 * on the main thread it would make the whole page stutter. Here it runs in parallel with the
 * drawing. The page sends what happened in the simulated flight since the last step (time,
 * birds, wind); the worker steps the air and sends back the finished pictures: the 3D texture
 * of the smoke, and the streaks of the tracers in the laser-sheet view. The buffers travel back
 * and forth (they are transferred, not copied).
 */

import { Fluid, type FluidBird, type FluidScales, type FluidView } from "./fluid";
import { TRACERS, Tracers, paint } from "./fluidView";
import type { Vec3 } from "./params";

export interface StepRequest {
  dt: number; // s of simulated time to advance (0: only repaint, e.g. a new view while paused)
  center: Vec3; // the followed bird
  eye: Vec3; // a point in front of the laser sheet, on the camera's side: the sheet faces it
  wind: Vec3; // m/s, the air speed of the wind
  birds: FluidBird[];
  obstacles: { center: Vec3; radius: number }[] | null; // null: unchanged
  view: FluidView;
  reset: boolean; // forget the air (a new flight, a seek)
  texture: ArrayBuffer; // N³ x 4 bytes, to fill
  lines: ArrayBuffer; // TRACERS x 6 floats
  colors: ArrayBuffer; // TRACERS x 8 floats
}

export interface StepReply {
  texture: ArrayBuffer;
  lines: ArrayBuffer;
  colors: ArrayBuffer;
  origin: Vec3;
  scales: FluidScales;
  sheet: { center: Vec3; normal: Vec3 } | null;
  ms: number; // time spent, for the page's own measure
}

const fluid = new Fluid();
const tracers = new Tracers();
let obstacles: { center: Vec3; radius: number }[] = [];
let solidsAt = "";

type Poster = { postMessage(message: unknown, transfer: Transferable[]): void };

self.onmessage = (e: MessageEvent<StepRequest>) => {
  const r = e.data;
  const t0 = performance.now();
  if (r.reset) {
    fluid.stop();
    tracers.reset();
  }
  if (r.obstacles) {
    obstacles = r.obstacles;
    solidsAt = "";
  }
  fluid.follow(r.center, r.wind);
  const at = fluid.origin.join(",");
  if (at !== solidsAt) {
    fluid.setObstacles(obstacles);
    solidsAt = at;
  }
  if (r.dt > 0) {
    fluid.relaxToWind(r.wind, r.dt);
    for (const b of r.birds) fluid.addBird(b, r.dt);
    fluid.step(r.dt);
  }
  const texture = new Uint8Array(r.texture);
  paint(fluid, r.view, r.center, texture);
  let sheet: StepReply["sheet"] = null;
  if (r.view === "sheet") {
    tracers.aim(r.center, r.eye, fluid);
    if (r.dt > 0) tracers.update(fluid, r.dt);
    tracers.write(new Float32Array(r.lines, 0, TRACERS * 6), new Float32Array(r.colors, 0, TRACERS * 8), fluid.scales.speed);
    sheet = { center: [...tracers.center] as Vec3, normal: [...tracers.normal] as Vec3 };
  }
  const reply: StepReply = {
    texture: r.texture,
    lines: r.lines,
    colors: r.colors,
    origin: [...fluid.origin] as Vec3,
    scales: { ...fluid.scales },
    sheet,
    ms: performance.now() - t0,
  };
  (self as unknown as Poster).postMessage(reply, [r.texture, r.lines, r.colors]);
};
