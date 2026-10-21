"""Run a list of training jobs from a YAML file, a few at a time.

    uv run python scripts/launch.py configs/experiments/pilots.yaml --parallel 7
    uv run python scripts/launch.py configs/experiments/pilots.yaml --dry-run

Each job is one call to hummingbird_rl.training.train; its output goes to
results/logs/<run name>_<date>_<time>.log. Jobs start in the order of the file, so the
longest should come first. A job that fails does not stop the others.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--parallel", type=int, default=7, help="jobs at the same time (one CPU core each)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    spec = yaml.safe_load(Path(args.file).read_text())
    common = shlex.split(spec.get("common", ""))
    jobs = [shlex.split(j) + common for j in spec["jobs"]]
    logs = ROOT / "results" / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for job in jobs:
            print(run_name(job), "<-", " ".join(job))
        return

    pending, running, t_start = list(jobs), [], time.time()
    while pending or running:
        while pending and len(running) < args.parallel:
            job = pending.pop(0)
            name = run_name(job)
            log = open(logs / f"{name}_{time.strftime('%Y-%m-%d_%H%M%S')}.log", "w")
            proc = subprocess.Popen([sys.executable, "-m", "hummingbird_rl.training.train", *job],
                                    stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
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
