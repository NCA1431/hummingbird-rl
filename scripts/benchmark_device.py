"""Does SAC (and TD3) train faster on the CPU or on the GPU of this machine? About 3 minutes.

    uv run python scripts/benchmark_device.py

SAC and TD3 do one gradient step per environment step on small networks, so
the GPU's strength (large matrix products) competes with its overhead
(launching many tiny kernels, copying each batch). Only a measurement on the
actual laptop tells which wins. PPO is not tested: it trains on large
batches only every 4096 steps and is clearly fine on the CPU.
"""

import time

import torch
from stable_baselines3 import SAC, TD3

from hummingbird_rl.envs import HummingbirdEnv

FILL, TIMED = 2_000, 3_000      # random steps to fill the buffer, then timed learning steps


def speed(algo, device: str) -> float:
    model = algo("MlpPolicy", HummingbirdEnv("hover"), device=device, learning_starts=FILL, buffer_size=50_000,
                 policy_kwargs={"net_arch": [256, 256]}, seed=0, verbose=0)
    model.learn(FILL)
    t0 = time.time()
    model.learn(TIMED, reset_num_timesteps=False)
    return TIMED / (time.time() - t0)


if __name__ == "__main__":
    torch.set_num_threads(1)     # as in the parallel training jobs
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for algo in (SAC, TD3):
        results = {d: speed(algo, d) for d in devices}
        line = "   ".join(f"{d}: {v:5.0f} steps/s" for d, v in results.items())
        best = max(results, key=results.get)
        print(f"{algo.__name__:4s} {line}   -> use --device {best}")
