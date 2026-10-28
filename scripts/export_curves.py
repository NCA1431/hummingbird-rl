"""Learning curves of every training run, in one small CSV file.

Each run's TensorBoard files (results/runs/<run>/tb/) hold everything SB3 logged
during training: tens of megabytes per run, most of it the rollout statistics.
The figures only need the validation curves (val/*: success rate, return, final
distance, how episodes ended, measured on the same fixed validation episodes at
every evaluation), so we keep those and write them, one line per run, step and
metric, to results/curves.csv: a few megabytes for all the runs, which
scripts/make_figures.py reads.

    uv run python scripts/export_curves.py
    uv run python scripts/export_curves.py --prefix train/ val/ --only sac_hover td3_hover --out results/offpolicy.csv

The second form exports other scalars too, here SB3's training diagnostics (train/ent_coef is
SAC's temperature, train/critic_loss, ...) of the runs whose name contains "sac_hover" or "td3_hover",
to results/offpolicy.csv.

A run trained twice under the same name (a restart) appears twice, with its date:
make_figures.py keeps the latest.
"""

import argparse
import csv
import sys
from pathlib import Path

import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"


def run_curves(run_dir: Path, prefixes: list[str]) -> list[dict]:
    """The scalars of one run whose tag starts with one of `prefixes`, as rows (step, metric, value).
    Validation tags lose their "val/" (the figures expect "success_rate"); others keep their full name."""
    rows = []
    for events in sorted((run_dir / "tb").glob("*/events.out.tfevents.*")):
        acc = EventAccumulator(str(events), size_guidance={"scalars": 0})  # 0: keep every point
        acc.Reload()
        for tag in acc.Tags()["scalars"]:
            if any(tag.startswith(p) for p in prefixes):
                metric = tag[4:] if tag.startswith("val/") else tag
                rows += [{"step": e.step, "metric": metric, "value": e.value} for e in acc.Scalars(tag)]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", nargs="+", default=["val/"], help="tags to keep (default: the validation curves)")
    ap.add_argument("--only", nargs="*", default=[], help="keep only the runs whose name contains one of these")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "curves.csv")
    args = ap.parse_args()
    out = args.out
    runs = sorted(d for d in RUNS.iterdir() if (d / "config.yaml").exists()
                  and (not args.only or any(o in d.name for o in args.only)))
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
            rows = run_curves(d, args.prefix)
            for r in rows:
                w.writerow({**meta, **r})
            n += 1
            print(f"{d.name}: {len(rows)} points")
    print(f"wrote {out} ({n} runs, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
