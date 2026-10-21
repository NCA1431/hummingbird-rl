"""Success rate of the PID controller on every task and regime (about 2 minutes).

    uv run python scripts/pid_baseline.py
    uv run python scripts/pid_baseline.py --tasks hover goto --regimes normal --episodes 50

Uses the validation seeds, so the numbers are directly comparable with the
val/ curves of the RL runs in W&B.
"""

import argparse
import time

from hummingbird_rl.controllers import PIDController
from hummingbird_rl.envs import make_env
from hummingbird_rl.evaluation import VALIDATION_SEED, run_episodes, summarise

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["hover", "goto", "obstacles", "navigation"])
    ap.add_argument("--regimes", nargs="+", default=["normal", "randomised", "failure"])
    ap.add_argument("--episodes", type=int, default=50)
    args = ap.parse_args()

    seeds = range(VALIDATION_SEED, VALIDATION_SEED + args.episodes)
    print(f"{'task':11s}{'regime':12s}{'success':>8s}{'return':>9s}{'dist cm':>9s}   how episodes end")
    for task in args.tasks:
        for regime in args.regimes:
            t0 = time.time()
            s = summarise(run_episodes(make_env(task, regime), PIDController(), seeds))
            ends = ", ".join(f"{k[4:]} {v:.0%}" for k, v in s.items() if k.startswith("end_"))
            print(f"{task:11s}{regime:12s}{s['success_rate']:8.0%}{s['mean_return']:9.1f}"
                  f"{s['final_distance_cm']:9.1f}   {ends}   ({time.time() - t0:.0f} s)")
