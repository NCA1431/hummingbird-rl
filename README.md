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
uv run pytest           # 134 passed
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
scripts/    launch.py, evaluate.py, export_web.py, runs_summary.py, pid_baseline.py, benchmark_device.py
tests/      one test file per module
web/        the web viewer: Vite + TypeScript + three.js, with a TypeScript port of the simulator
data/vanderbilt/  hummingbird wing kinematics (Song et al. 2016, Dryad doi:10.5061/dryad.8ch1b)
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

Needs Node.js (LTS). The simulator is ported to TypeScript and checked against the Python one
on recorded trajectories (`web/tests/golden/`, written by `scripts/export_web.py golden`).

```powershell
cd web
npm install             # once, and after any change to package.json
npm test                # parity of the TypeScript simulator with the Python one
npm run dev             # local server, open the address it prints
```
