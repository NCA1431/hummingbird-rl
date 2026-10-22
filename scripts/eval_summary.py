"""Every evaluation result in one table per task, with the seeds of each experiment grouped.

    uv run python scripts/eval_summary.py
    uv run python scripts/eval_summary.py --tasks hover goto

Reads results/eval/<policy>/<task>/summary.json (written by evaluate.py). For each
task, one line per controller: PID and LQR, then every trained configuration with
its seeds grouped (ppo_hover_normal_s1..s3 -> ppo_hover_normal). Each column is
the mean over the seeds; the success rate also lists every seed, because with 3
seeds the spread says as much as the mean. A column stays empty when that test
was not run for the controller.

Everything is also written, one line per run and task, to results/eval/all_results.csv,
which the figures of the report will read.
"""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "results" / "eval"

# (test, key in its summary, column title, format); "pct" prints a fraction as a percentage.
COLUMNS = [
    ("test", "success_rate", "success", "pct"),
    ("test", "final_distance_cm", "dist cm", "{:.2f}"),
    ("test", "energy_J", "energy J", "{:.2f}"),
    ("test", "end_collision", "collision", "pct"),
    ("test", "end_loss_of_control", "lost ctrl", "pct"),
    ("grid", "robust_area", "robust", "{:.2f}"),
    ("grid", "robust_area_trained_range", "robust<=.3", "{:.2f}"),
    ("mass", "mean_success", "mass", "{:.2f}"),
    ("failure", "survival_rate", "survival", "pct"),
    ("failure", "median_recovery_s", "recov s", "{:.2f}"),
    ("behaviour", "p_turn_target_behind", "turns", "pct"),
    ("behaviour", "p_backed_target_behind", "backs up", "pct"),
]


def group_of(name: str) -> str:
    """ppo_hover_normal_s2 -> ppo_hover_normal; ppo_hover_normal_s2_best -> ppo_hover_normal_best."""
    return re.sub(r"_s\d+(_best)?$", r"\1", name)


def value(summary: dict, suite: str, key: str) -> float | None:
    if suite not in summary:
        return None
    v = summary[suite].get(key)
    if v is None and key.startswith("end_"):
        return 0.0          # summarise() only lists the endings that occurred
    return float(v) if isinstance(v, (int, float)) else None


def fmt(x: float, f: str) -> str:
    if x is None or not np.isfinite(x):
        return "-"
    return f"{100 * x:.0f}%" if f == "pct" else f.format(x)


def load() -> list[dict]:
    """One entry per (run, task): its name, group, task and summary."""
    runs = []
    for path in sorted(EVAL.glob("*/*/summary.json")):
        runs.append({"name": path.parent.parent.name, "group": group_of(path.parent.parent.name),
                     "task": path.parent.name, "summary": json.loads(path.read_text())})
    return runs


def print_task(task: str, runs: list[dict]) -> None:
    groups: dict[str, list[dict]] = {}
    for r in runs:
        groups.setdefault(r["group"], []).append(r)
    used = [c for c in COLUMNS if any(c[0] in r["summary"] for r in runs)]     # drop tests never run here
    order = [g for g in ("pid", "lqr") if g in groups] + sorted(g for g in groups if g not in ("pid", "lqr"))

    rows = []
    for g in order:
        members = sorted(groups[g], key=lambda r: r["name"])
        cells = [g, str(len(members))]
        for suite, key, _, f in used:
            ran = [v for r in members if (v := value(r["summary"], suite, key)) is not None]
            values = [v for v in ran if np.isfinite(v)]      # e.g. no recovery time when nobody survived
            text = (fmt(float(np.mean(values)), f) if values else "-") if ran else ""
            if key == "success_rate" and suite == "test" and len(values) > 1:
                text += " (" + " ".join(f"{100 * v:.0f}" for v in values) + ")"
            cells.append(text)
        rows.append(cells)

    header = ["", "n"] + [title for _, _, title, _ in used]
    widths = [max(len(row[i]) for row in rows + [header]) for i in range(len(header))]
    print(f"\n=== {task} ===")
    print("  ".join(h.ljust(w) if i == 0 else h.rjust(w) for i, (h, w) in enumerate(zip(header, widths))))
    for row in rows:
        print("  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(row, widths))))


def write_csv(runs: list[dict]) -> Path:
    """All numbers of every summary, flattened as "<test>/<key>" columns."""
    rows = []
    for r in runs:
        row = {"name": r["name"], "group": r["group"], "task": r["task"]}
        for suite, result in r["summary"].items():
            for key, v in result.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    row[f"{suite}/{key}"] = v
        rows.append(row)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    path = EVAL / "all_results.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["hover", "goto", "obstacles", "navigation"])
    args = ap.parse_args()
    runs = load()
    if not runs:
        raise SystemExit(f"no summary.json under {EVAL}; run scripts/evaluate.py first")
    for task in args.tasks:
        task_runs = [r for r in runs if r["task"] == task]
        if task_runs:
            print_task(task, task_runs)
    print(f"\nall numbers, one line per run: {write_csv(runs)}")


if __name__ == "__main__":
    main()
