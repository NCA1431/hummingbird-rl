# Hummingbird RL — web viewer

Vite + TypeScript + three.js. Everything runs in the browser: no server.

```powershell
npm install        # once, and after any change to package.json
npm test           # the TypeScript simulator against the Python one (golden files)
npm run dev        # local server with live reload
npm run build      # static site in dist/ (type check first)
```

- `src/sim/` — line-by-line port of `hummingbird_rl/sim` and of the observation builder.
  Field names of the parameters are the Python ones, so both can be compared side by side.
- `tests/golden/` — inputs and outputs recorded in Python by `scripts/export_web.py golden`;
  regenerate them only if the physics changes, then rerun `npm test`.
- `public/` — files served as they are: recorded episodes and policies, written by
  `scripts/export_web.py episodes` and `scripts/export_web.py policy`.
