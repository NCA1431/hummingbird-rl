"""Learning curves of every training run, in one small CSV file.

Each run's TensorBoard files (results/runs/<run>/tb/) hold everything SB3 logged
during training: tens of megabytes per run, most of it the rollout statistics.
The figures only need the validation curves (val/*: success rate, return, final
distance, how episodes ended, measured on the same fixed validation episodes at
every evaluation), so we keep those and write them, one line per run, step and
metric, to results/curves.csv: a few megabytes for all the runs, which
scripts/make_figures.py reads.

    uv run python scripts/export_curves.py

A run trained twice under the same name (a restart) appears twice, with its date:
make_figures.py keeps the latest.
"""

import csv
import sys
from pathlib import Path

import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"


def run_curves(run_dir: Path) -> list[dict]:
    """The val/* scalars of one run, as rows (step, metric, value)."""
    rows = []
    for events in sorted((run_dir / "tb").glob("*/events.out.tfevents.*")):
        acc = EventAccumulator(str(events), size_guidance={"scalars": 0})  # 0: keep every point
        acc.Reload()
        for tag in acc.Tags()["scalars"]:
            if tag.startswith("val/"):
                rows += [{"step": e.step, "metric": tag[4:], "value": e.value} for e in acc.Scalars(tag)]
    return rows


def main() -> None:
    out = ROOT / "results" / "curves.csv"
    runs = sorted(d for d in RUNS.iterdir() if (d / "config.yaml").exists())
    if not runs:
        sys.exit(f"no runs in {RUNS}")
    n = 0
    with out.open("w", newline="") as f:
        fields = ["run", "name", "algo", "task", "regime", "seed", "init_from", "env", "step", "metric", "value"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for d in runs:
            config = yaml.safe_load((d / "config.yaml").read_text())
            name = d.name.rsplit("_", 2)[0]  # without the date and time
            meta = {
                "run": d.name, "name": name, "algo": config["algo"], "task": config["task"],
                "regime": config["regime"], "seed": config["seed"], "init_from": config.get("init_from") or "",
                "env": ";".join(f"{k}={v}" for k, v in (config.get("env") or {}).items()),
            }
            rows = run_curves(d)
            for r in rows:
                w.writerow({**meta, **r})
            n += 1
            print(f"{d.name}: {len(rows)} points")
    print(f"wrote {out} ({n} runs, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
