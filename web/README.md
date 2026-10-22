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
  data (`wingData.json`, Song et al. 2016), targets, obstacles, trail and wind arrow.
- `src/ui/panel.ts` — controls (Recorded / Live) and head-up display.
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
Cameras: Follow, Side, Top, Behind, Bird's eye (first person at the bird's eye, along its heading,
horizon kept level) and Overview.
