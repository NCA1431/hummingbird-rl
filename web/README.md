# Hummingbird RL — web viewer

Vite + TypeScript + three.js. Everything runs in the browser: no server.

```powershell
npm ci             # once, and after any change to package-lock.json
npm test           # the TypeScript simulator against the Python one, and the viewer's logic
npm run dev        # local server with live reload
npm run build      # static site in dist/ (type check first)
```

- `src/sim/` — line-by-line port of `hummingbird_rl/sim` and of the observation builder.
  Field names of the parameters are the Python ones, so both can be compared side by side.
- `src/modes/watch.ts` — replay of recorded episodes (interpolation between frames).
- `src/view/` — the stage (camera, lights, arena), the bird, its wing motion from measured
  data (`wingData.json`, Song et al. 2016), targets, obstacles, trail and wind arrow.
- `src/ui/panel.ts` — controls and head-up display.
- `tests/golden/` — inputs and outputs recorded in Python by `scripts/export_web.py golden`;
  regenerate them only if the physics changes, then rerun `npm test`.
- `public/episodes/` — recorded episodes and their `index.json`, written by
  `scripts/export_web.py episodes`; `public/policies/` — exported networks (live mode, later).
- `public/models/hummingbird.glb` and `hummingbird_specgloss.webp` — the 3D hummingbird (CC BY 4.0,
  credits in the page and the main README), shrunk from 24.7 MB to 1.6 MB by `tools/shrink-model.mjs`
  (instructions inside), and drawn by `src/view/feathers.ts`, a port of the earlier Panda3D viewer's shader.

Controls: Space = play/pause, R = restart; drag to orbit, scroll to zoom.
