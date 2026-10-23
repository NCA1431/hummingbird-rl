# Hummingbird RL

Reinforcement learning for a simulated hummingbird: PPO, SAC and TD3 against PID and LQR
on hover, go-to-point, obstacle and navigation tasks. Work in progress: the simulator, the
environment, the PID and LQR baselines, the training pipeline and the evaluation are done; the
web viewer comes next.

## Setup (Windows, PowerShell, with uv)

```powershell
cd C:\Users\nca18\projects\hummingbird-rl
uv sync                 # creates .venv with Python 3.12, installs PyTorch (CUDA) and everything else, writes uv.lock
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
uv run pytest           # 139 passed
```

`uv.lock` pins every version: commit it.

## Layout

```
hummingbird_rl/
  sim/      the bird: parameters, rotations, aerodynamics, wind gusts, rigid-body dynamics
  world/    scenes (targets and obstacles), collisions, random scene generation
  tasks/    hover & recover, go-to-point, obstacles, navigation: rewards and success rules
  envs/     the Gymnasium environment, the observation, the disturbance regimes
  controllers/  classical baselines: PID and LQR, with a potential field for obstacles; action mixer
  training/     train.py (PPO, SAC, TD3 with Stable-Baselines3), logging callbacks
  evaluation/   episodes on fixed seeds; suites (test, robustness grid, mass, failure, behaviour); intervals
  export/       recorded episodes, policy networks and golden test files for the web viewer
configs/algos/      hyperparameters of each algorithm
configs/experiments/  lists of runs for scripts/launch.py
scripts/    launch.py, evaluate.py, eval_summary.py, export_web.py, runs_summary.py, pid_baseline.py,
            benchmark_device.py, extract_dryad.py
tests/      one test file per module
web/        the web viewer: Vite + TypeScript + three.js, with a TypeScript port of the simulator
data/vanderbilt/  hummingbird wing kinematics (Song et al. 2016, Dryad doi:10.5061/dryad.8ch1b),
                  extracted from the dataset's MATLAB figures by scripts/extract_dryad.py
```

## Training

```powershell
uv run python -m hummingbird_rl.training.train --algo ppo --task hover --steps 1e6 --wandb
uv run python scripts/launch.py configs/experiments/pilots.yaml --parallel 7     # a batch of runs
uv run python scripts/pid_baseline.py                                           # the PID's numbers
```

Runs are saved in `results/runs/<algo>_<task>_<regime>_s<seed>_<date>_<time>/`.

Quick check that everything is wired up:

```powershell
uv run python -c "from hummingbird_rl.envs import make_env; env = make_env('goto', seed=0); print(env.step(env.hover)[1])"
```

## Web viewer

Needs Node.js (LTS). Two modes. **Recorded** replays test episodes exactly as they were measured.
**Live** flies new random episodes in the browser: the simulator, the tasks, the PID, the LQR and the
trained networks are ported to TypeScript and checked against the Python code on recorded inputs and
outputs (`web/tests/golden/`): whole PID and LQR episodes agree to 1e-9 m, the networks to 1e-5.

```powershell
uv run python scripts/export_web.py episodes --policy pid --tasks hover goto obstacles navigation
uv run python scripts/export_web.py episodes --policy ppo_goto_normal_s1      # a trained agent, on its task
uv run python scripts/export_web.py policy --policy ppo_goto_normal_s1        # its network, for the live mode
cd web
npm ci                  # once, and after any change to package-lock.json
npm test                # parity of the TypeScript simulator with the Python one, and the viewer's logic
npm run dev             # local server, open the address it prints
```

The site is published on GitHub Pages by `.github/workflows/deploy-web.yml` at every push that changes `web/`.

## Credits

- 3D hummingbird in the viewer: ["Animated Hovering Flying Hummingbird Loop"](https://sketchfab.com/3d-models/animated-hovering-flying-hummingbird-loop-102884713a2742ce829e368d2a790c45)
  by LasquetiSpice, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes: textures
  resized and converted to WebP, unused animation and keyframes removed (`web/tools/shrink-model.mjs`), rendered
  with our own shader (`web/src/view/feathers.ts`); its root motion is ignored and its wingbeat replayed at the simulated frequency.
- Wing kinematics: Song, Tobalske, Powers, Hedrick & Luo (2016), *R. Soc. Open Sci.* 3:160230,
  Dryad doi:10.5061/dryad.8ch1b.

