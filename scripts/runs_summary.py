"""One line per training run, read from its local TensorBoard logs (a few seconds).

    uv run python scripts/runs_summary.py                  # every run in results/runs
    uv run python scripts/runs_summary.py --filter sac_    # only runs whose name contains "sac_"

For each run: how far it went, the final and best validation success rate,
when it first reached 90%, the final return and distance, the training speed,
and how the last validation episodes ended. A quick way to compare runs as
text, without opening W&B.
"""

import argparse
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
ENDS = ["success", "time_limit", "crash", "out_of_bounds", "loss_of_control", "collision"]


def load_scalars(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    """Every logged curve of a run, as {tag: [(step, value), ...]}."""
    scalars: dict[str, list[tuple[int, float]]] = {}
    for tb in sorted((run_dir / "tb").glob("*")):
        ea = EventAccumulator(str(tb), size_guidance={"scalars": 0})   # 0 = keep every point
        ea.Reload()
        for tag in ea.Tags()["scalars"]:
            scalars.setdefault(tag, []).extend((e.step, e.value) for e in ea.Scalars(tag))
    return scalars


def last(scalars: dict, tag: str) -> float:
    return scalars[tag][-1][1] if tag in scalars else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", default="", help="only runs whose folder name contains this text")
    args = ap.parse_args()

    runs = sorted(p for p in (ROOT / "results" / "runs").iterdir() if p.is_dir() and args.filter in p.name)
    print(f"{'run':28s}{'steps':>7s}{'final':>7s}{'best':>7s}{'@step':>7s}{'90% at':>8s}"
          f"{'return':>8s}{'dist cm':>8s}{'steps/s':>8s}   last validation ended by")
    for run in runs:
        s = load_scalars(run)
        name = run.name.rsplit("_", 2)[0]          # drop the date and time
        success = s.get("val/success_rate", [])
        if not success:
            print(f"{name:28s}  no validation logged yet")
            continue
        best_step, best = max(success, key=lambda p: (p[1], -p[0]))   # earliest best
        first90 = next((step for step, v in success if v >= 0.9), None)
        ends = ", ".join(f"{e} {last(s, f'val/end_{e}'):.0%}" for e in ENDS if last(s, f"val/end_{e}") > 0)
        print(f"{name:28s}{success[-1][0] / 1e6:6.2f}M{success[-1][1]:7.0%}{best:7.0%}{best_step / 1e6:6.2f}M"
              f"{(f'{first90 / 1e6:.2f}M' if first90 else '-'):>8s}{last(s, 'val/mean_return'):8.1f}"
              f"{last(s, 'val/final_distance_cm'):8.1f}{last(s, 'time/fps'):8.0f}   {ends}")


if __name__ == "__main__":
    main()