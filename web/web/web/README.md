# Hummingbird RL — web viewer

Vite + TypeScript + three.js. Everything runs in the browser: no server.

```powershell
npm ci             # once, and after any change to package-lock.json
npm test           # the TypeScript simulator against the Python one, and the viewer's logic
npm run dev        # local server with live reload
npm run build      # static site in dist/ (type check first)
```

- `src/sim/` — line-by-line port of `hummingbird_rl/sim`, of the observation builder, the
  disturbances, the scene samplers, the tasks and the environment loop (`env.ts`).
  Field names of the parameters are the Python ones, so both can be compared side by side.
- `src/control/` — the PID, the LQR (its gain `lqr_gains.json` is computed in Python), the
  potential field, the allocation, and the forward pass of the exported networks (`mlp.ts`).
- `src/modes/watch.ts` — replay of an episode (interpolation between frames); `src/modes/live.ts` —
  a new episode simulated in the browser, in the same format, then replayed the same way. After a
  loss of control, a live flight goes on until the bird reaches the ground or the episode's time
  limit, with a grey trail, a hatched band on the timeline and a note in the HUD: shown, not counted.
- `src/view/` — the stage (camera, lights, arena), the bird, its wing motion from measured
  data (`wingData.json`, Song et al. 2016), targets, obstacles, trail and wind arrow; `air.ts`, the
  "air" menu: the air of `src/sim/fluid.ts` (Stam's Stable Fluids, ported from an earlier C++
  version: 40³ cells of 1 cm around the followed bird, pushed by the simulated thrust where each
  wing is in its stroke, and by the wind), run in a worker (`fluidWorker.ts`, pictures in
  `fluidView.ts`) and shown as smoke (ray-marched 3D texture), a laser sheet of tracer particles,
  vortex tubes (Q-criterion surfaces), or the smoke coloured by speed, pressure or spin.
- `src/ui/panel.ts` — controls (Recorded / Live / Sandbox / Play) and head-up display; `src/ui/words.ts` — the
  names of tasks, controllers (a transfer or a fine-tuning is said as such) and endings; `src/ui/sandboxPanel.ts` and
  `src/modes/sandbox.ts` — the sandbox: targets, numbered waypoints and obstacles placed by clicking on the
  ground (a see-through preview shows where and how high), dragged to move them (Shift + drag: up and down)
  and removed with Erase; wind, mass, wing damage,
  sudden failure and noise set by hand; up to four controllers flying the same situation, each bird in its
  colour, drawn or as the 3D model, the camera following the one chosen. Every change flies again.
- `src/modes/play.ts` — Play: the birds stepped in real time while the visitor drags the target or an
  obstacle (Shift + drag: up and down; F: the object picked follows the mouse, Shift: up and down), with no end: a bird that falls
  starts again from the start point. Each controller gets the training observation, with the target's
  slot empty when "the birds see the target" is off, and the obstacles hidden from networks trained
  without any. The Play section of `src/ui/sandboxPanel.ts`, the dragging in `src/main.ts`.
- `tests/golden/` — inputs and outputs recorded in Python by `scripts/export_web.py golden`
  (simulator, observations, networks, whole PID and LQR episodes); regenerate them, with
  `src/control/lqr_gains.json`, only if the physics, the tasks or the controllers change, then rerun `npm test`.
- `public/episodes/` — recorded episodes and their `index.json`, written by
  `scripts/export_web.py episodes`; `public/policies/` — exported networks and their `index.json`,
  written by `scripts/export_web.py policy` (the live mode offers PID and LQR without them).
- `public/models/hummingbird.glb` and `hummingbird_specgloss.webp` — the 3D hummingbird (CC BY 4.0,
  credits in the page and the main README), shrunk from 24.7 MB to 1.6 MB by `tools/shrink-model.mjs`
  (instructions inside), and drawn by `src/view/feathers.ts`, a port of the earlier Panda3D viewer's shader.

Controls: Space = play/pause, R = restart, N = new live flight; drag to orbit, scroll to zoom.
In Play: drag the target or an obstacle, Shift + drag to move it up and down, F to make the object
picked follow the mouse (Shift held: up and down; Esc stops it), Delete to remove the obstacle picked.
The top panel and the controls fold to their essentials with their corner button (the top one
starts folded on phones).
Cameras: Follow, Side, Top, Behind, Bird's eye (first person at the bird's eye, along its heading,
horizon kept level) and Overview.
