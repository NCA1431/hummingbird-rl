"""Run a list of jobs from a YAML file, a few at a time.

    uv run python scripts/launch.py configs/experiments/pilots.yaml --parallel 7
    uv run python scripts/launch.py configs/experiments/pilots.yaml --dry-run

By default each job is one call to hummingbird_rl.training.train. A file can
name another script instead, with a line `script: scripts/evaluate.py`; this
is how the evaluations of many trained agents are run in parallel too.
A job's output goes to results/logs/<job name>_<date>_<time>.log. Jobs start
in the order of the file, so the longest should come first. A job that fails
does not stop the others.
"""

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hummingbird_rl.training.train import parse_args  # noqa: E402


def run_name(job: list[str]) -> str:
    a = parse_args(job)
    return a.name or f"{a.algo}_{a.task}_{a.regime}_s{a.seed}"


def job_name(job: list[str], script: str | None) -> str:
    """The run name for a training job; for another script, its name and first argument value,
    e.g. evaluate_ppo_hover_normal_s1, plus "_best" for an evaluation of the best model, so that
    two evaluations of the same run never share a log file."""
    if script is None:
        return run_name(job)
    values = [a for a in job if not a.startswith("--")]
    name = f"{Path(script).stem}_{values[0]}" if values else Path(script).stem
    best = any(a == "--which" and b == "best" for a, b in zip(job, job[1:]))
    return name + "_best" if best else name


def command(job: list[str], script: str | None) -> list[str]:
    if script is None:
        return [sys.executable, "-m", "hummingbird_rl.training.train", *job]
    return [sys.executable, str(ROOT / script), *job]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--parallel", type=int, default=7, help="jobs at the same time (one CPU core each)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    spec = yaml.safe_load(Path(args.file).read_text())
    script = spec.get("script")                 # None: the training script
    common = shlex.split(spec.get("common", ""))
    jobs = [shlex.split(j) + common for j in spec["jobs"]]
    logs = ROOT / "results" / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for job in jobs:
            print(job_name(job, script), "<-", " ".join(job))
        return

    pending, running, t_start = list(jobs), [], time.time()
    while pending or running:
        while pending and len(running) < args.parallel:
            job = pending.pop(0)
            name = job_name(job, script)
            log = open(logs / f"{name}_{time.strftime('%Y-%m-%d_%H%M%S')}.log", "w")
            proc = subprocess.Popen(command(job, script), stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            running.append((proc, name, log, time.time()))
            print(f"{time.strftime('%H:%M')}  started  {name}", flush=True)
        for item in list(running):
            proc, name, log, t0 = item
            if proc.poll() is not None:
                log.close()
                running.remove(item)
                status = "done   " if proc.returncode == 0 else "FAILED "
                print(f"{time.strftime('%H:%M')}  {status} {name} ({(time.time() - t0) / 60:.0f} min)", flush=True)
        time.sleep(5)
    print(f"all jobs finished in {(time.time() - t_start) / 3600:.1f} h")


if __name__ == "__main__":
    main()
