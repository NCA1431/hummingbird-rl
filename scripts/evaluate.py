"""Evaluate a controller (PID, LQR or a trained agent) on the evaluation suites.

    uv run python scripts/evaluate.py --policy pid --tasks hover goto obstacles navigation
    uv run python scripts/evaluate.py --policy ppo_hover_normal_s0 --suites test grid mass failure
    uv run python scripts/evaluate.py --policy ppo_goto_normal_s0 --suites behaviour --wandb

--policy is "pid", "lqr", a run name (the most recent run with that name is used)
or the path of a model .zip. For a trained agent the final model is evaluated
(--which best for the best validation model), on the task it was trained on
unless --tasks says otherwise.

The suites are described in hummingbird_rl/evaluation/suites.py. Results go to
results/eval/<policy>/<task>/<suite>.csv (one row per episode or grid cell) and
summary.json; with --wandb the summaries and tables are also logged, as a W&B
run in the group "eval".
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hummingbird_rl.controllers import CONTROLLERS  # noqa: E402
from hummingbird_rl.evaluation.suites import SUITE_TASKS, SUITES  # noqa: E402
from hummingbird_rl.training.train import ALGOS, parse_pairs, resolve_model  # noqa: E402


def load_policy(spec: str, which: str):
    """The policy object and a description of it (name, training task, environment options)."""
    if spec in CONTROLLERS:
        return CONTROLLERS[spec](), {"name": spec, "task": None, "env": {}}
    run_dir = resolve_model(spec).parent
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    model = ALGOS[config["algo"]].load(run_dir / ("model.zip" if which == "final" else "best_model.zip"), device="cpu")
    return model, {"name": run_dir.name.rsplit("_", 2)[0], "run": run_dir.name, "which": which,
                   "algo": config["algo"], "task": config["task"], "env": config.get("env") or {}}


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))      # every column, in order of appearance
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--which", choices=["final", "best"], default="final")
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--suites", nargs="+", default=["test"], choices=sorted(SUITES))
    ap.add_argument("--env", nargs="*", default=[], metavar="KEY=VALUE", help="environment options (PID/LQR)")
    ap.add_argument("--episodes", type=int, default=None, help="episodes per suite (per cell for grid and mass)")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--project", default="hummingbird-rl")
    args = ap.parse_args()

    policy, meta = load_policy(args.policy, args.which)
    tasks = args.tasks or ([meta["task"]] if meta["task"] else None)
    if not tasks:
        ap.error("--tasks is required for pid and lqr")
    env_kwargs = {**meta["env"], **parse_pairs(args.env)}

    for task in tasks:
        out = ROOT / "results" / "eval" / meta["name"] / task
        out.mkdir(parents=True, exist_ok=True)
        summary_path = out / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        run = None
        if args.wandb:
            import wandb
            run = wandb.init(project=args.project, group="eval", job_type="eval", name=f"eval_{meta['name']}_{task}",
                             config={**meta, "task": task, "env": env_kwargs, "suites": args.suites})
        for suite in args.suites:
            if task not in SUITE_TASKS[suite]:
                print(f"[{meta['name']} | {task} | {suite}] skipped: this suite is not defined for {task}")
                continue
            t0 = time.time()
            kwargs = {"n": args.episodes} if args.episodes else {}
            rows, result = SUITES[suite](policy, task, env_kwargs, **kwargs)
            write_csv(out / f"{suite}.csv", rows)
            summary[suite] = {**result, "evaluated": time.strftime("%Y-%m-%d %H:%M"), **meta}
            shown = ", ".join(f"{k} {v:.3g}" for k, v in result.items()
                              if isinstance(v, float) and not k.startswith("end_"))
            print(f"[{meta['name']} | {task} | {suite}] {shown}  ({time.time() - t0:.0f} s)")
            if run is not None:
                run.summary.update({f"{suite}/{k}": v for k, v in result.items() if isinstance(v, (int, float))})
                run.log({f"{suite}/table": wandb.Table(columns=list(rows[0]), data=[list(r.values()) for r in rows])})
        summary_path.write_text(json.dumps(summary, indent=2, default=str))
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
