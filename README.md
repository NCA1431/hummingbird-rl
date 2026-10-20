# Hummingbird RL

Reinforcement learning for a simulated hummingbird: PPO, SAC and TD3 against PID and LQR
on hover, go-to-point, obstacle and navigation tasks. Work in progress: the simulator, the
environment, the PID baseline and the training pipeline are done; LQR, evaluation and the web
viewer come next.

## Setup (Windows, PowerShell, with uv)

```powershell
cd C:\Users\nca18\projects\hummingbird-rl
uv sync                 # creates .venv with Python 3.12, installs PyTorch (CUDA) and everything else, writes uv.lock
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
uv run pytest           # 104 passed (about 30 s)
```

`uv.lock` pins every version: commit it.

## Layout

```
hummingbird_rl/
  sim/      the bird: parameters, rotations, aerodynamics, wind gusts, rigid-body dynamics
  world/    scenes (targets and obstacles), collisions, random scene generation
  tasks/    hover & recover, go-to-point, obstacles, navigation: rewards and success rules
  envs/     the Gymnasium environment, the observation, the disturbance regimes
  controllers/  PID + potential field (classical baseline), thrust/torque -> action mixer
  training/     train.py (PPO, SAC, TD3 with Stable-Baselines3), logging callbacks
  evaluation/   running episodes on fixed seeds, summaries
configs/algos/      hyperparameters of each algorithm
configs/experiments/  lists of runs for scripts/launch.py
scripts/    launch.py, pid_baseline.py, benchmark_device.py
tests/      one test file per module
data/vanderbilt/  hummingbird wing kinematics (Song et al. 2016, Dryad doi:10.5061/dryad.8ch1b)
```

## Training

```powershell
uv run python -m hummingbird_rl.training.train --algo ppo --task hover --steps 1e6 --wandb
uv run python scripts/launch.py configs/experiments/pilots.yaml --parallel 7     # a batch of runs
uv run python scripts/pid_baseline.py                                           # the PID's numbers
```

Runs are saved in `results/runs/<algo>_<task>_<regime>_s<seed>/`.

Quick check that everything is wired up:

```powershell
uv run python -c "from hummingbird_rl.envs import make_env; env = make_env('goto', seed=0); print(env.step(env.hover)[1])"
```
