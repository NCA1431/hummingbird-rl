"""Every evaluation result, logged to W&B as data, so that the report can draw its own charts.

make_figures.py draws the figures of the PDF as images. In the W&B report we would
rather hand W&B the numbers: a chart that W&B draws from runs can be hovered,
filtered and regrouped by the reader, and every point in it leads back to the run
it came from. So this script logs one W&B run per controller and task, in the
group "results":

  config   who it is: algorithm, algo, task, regime, variant, seed, checkpoint, and
           `controller` ("PPO randomised", "SAC normal (best)", "PID"...), the key to group
           by (W&B writes the key in front of each legend entry, so we gave it the name
           a reader expects: "controller: PPO randomised")
  tags     the experiments it takes part in (E1 ... E5), to filter a run set on
  summary  every score of all_results.csv, a few read off the learning curve (steps
           until validation success first reaches 90%, best and final value) and, for
           SAC and TD3 on hover, the test success of the best checkpoint
  history  the curves, each with its own x axis (W&B's define_metric):
             val/*     validation during training                     x = val/env_steps
             mass/*    success vs the bird's mass                      x = mass/mass_factor
             wind/*    success vs wind, overall and per damage level   x = wind/wind_pct
             damage/*  success vs wing damage, overall and per wind    x = damage/damage_pct
  chart    go-to-point: target bearing vs net turn, one dot per episode (E1b)

and one run in the group "results_tables" with the tables behind all this
(all_results, the wind x damage grid of every run and its mean per controller, the mass
sweeps, the behaviour episodes), for table panels and custom charts, and the PNG
figures of results/figures/ (make_figures.py, interpret.py) for the report's image
panels: the heatmaps, the E1b scatter and E6, which W&B cannot draw itself.

In the report, a run set filtered on group = results, with its panels grouped by
config.controller, shows the mean over seeds and the range of the seeds as a band: the
same convention as the PNG figures.

The PID and the LQR do not learn. So that they can still be drawn as reference
lines on the learning curves, as in the PNG figures, their test scores are logged as
flat val/* curves from 0 to the longest training on that task (config.reference_line
is then true). It is a drawing convention, not a measurement during training.

    uv run python scripts/log_results_wandb.py --dry-run                       # print, send nothing
    uv run python scripts/log_results_wandb.py --only pid ppo_hover_normal_s1  # a few runs, to check
    uv run python scripts/log_results_wandb.py --replace                       # delete the old ones, log all

It uses the W&B login of this computer (uv run wandb login): no key appears here.
"""

import argparse
import logging
import math
import os
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hummingbird_rl.analysis.plots import ALGO_NAME  # noqa: E402

EVAL = ROOT / "results" / "eval"
GROUP, TABLES = "results", "results_tables"

# The run names say who an agent is: <algo>_<task>_<variant>_s<seed>, then _best for
# the best-validation checkpoint (only SAC and TD3 on hover were evaluated both ways).
NAME = re.compile(r"^(ppo|sac|td3)_(hover|goto|obstacles|navigation)_(.+?)_s(\d+)(_best)?$")
VARIANT = {
    "normal": "normal", "randomised": "randomised",
    "finetuned": "fine-tuned +30%", "finetuned_long": "fine-tuned +100%",
    "noclip": "no clipping", "noadvnorm": "no advantage normalisation",
    "lambda0": "GAE λ = 0", "lambda1": "GAE λ = 1", "nohoverinit": "no hover start",
    "fixedalpha": "fixed α = 0.2", "f50": "failure penalty −50", "from_obstacles": "from obstacles",
    "norewnorm": "no reward normalisation", "norewnorm_noadvnorm": "neither normalisation",
    "annealed": "annealed learning rate",
}
ABLATIONS = {("ppo", "hover"): {"normal", "noclip", "noadvnorm", "norewnorm", "norewnorm_noadvnorm", "lambda0",
                                "lambda1", "nohoverinit"},
             ("sac", "hover"): {"normal", "fixedalpha", "annealed"}, ("td3", "hover"): {"normal", "annealed"},
             ("ppo", "navigation"): {"normal", "f50"}}
REFERENCE = ["success_rate", "mean_return", "final_distance_cm"]  # val/* metrics the PID and LQR get a flat line for


def finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def plain(df: pd.DataFrame) -> pd.DataFrame:
    """Missing values as None rather than NaN, which W&B tables do not all display."""
    return df.astype(object).where(df.notna(), None)


# --- Who is who --------------------------------------------------------------------------

def who(name: str, task: str, train: dict | None) -> dict:
    """The config of a result run, from the controller's name and its training config."""
    if name in ("pid", "lqr"):
        return {"policy": name, "algorithm": name.upper(), "algo": name, "kind": "classical", "task": task,
                "trained_on": None, "variant": "classical", "regime": None, "seed": None, "checkpoint": None,
                "controller": name.upper(), "reference_line": True}
    m = NAME.match(name)
    if m is None:
        raise ValueError(f"cannot read the run name {name!r}")
    algo, trained_on, variant, seed, best = m.groups()
    source = train.get("init_from") if train else None
    return {
        "policy": name, "algorithm": ALGO_NAME[algo], "algo": algo, "kind": "rl", "task": task,
        "trained_on": trained_on, "variant": variant,
        # finetuned runs were trained normal first, then randomised: their own config says randomised
        "regime": train["regime"] if train else ("randomised" if "randomised" in variant else "normal"),
        "seed": int(seed), "checkpoint": "best" if best else "last",
        "controller": f"{ALGO_NAME[algo]} {VARIANT.get(variant, variant)}" + (" (best)" if best else ""),
        "train_run": train["run"] if train else None,
        # a Windows path, written on the training computer: only the run it names is useful here
        "init_from": re.split(r"[\\/]", source)[-2] if isinstance(source, str) and source else None,
        "env": train["env"] if train and isinstance(train.get("env"), str) else None,
        "reference_line": False,
    }


def experiments(c: dict) -> list[str]:
    """The experiments a result run takes part in, as tags."""
    algo, task, variant, last = c["algo"], c["task"], c["variant"], c["checkpoint"] != "best"
    if c["kind"] == "classical":  # the references, wherever the figures draw them
        tags = ["E1"] + (["E1b"] if task == "goto" else []) + (["E2", "E3"] if task in ("hover", "goto") else [])
        return tags + (["E5"] if task == "navigation" else []) + ["reference"]
    tags = []
    if variant == "normal" and last:
        tags.append("E1")
    if variant == "normal" and last and task == "goto":
        tags.append("E1b")
    if task in ("hover", "goto") and variant in ("normal", "randomised"):
        tags.append("E2")
    if algo == "ppo" and task in ("hover", "goto") and variant in ("normal", "finetuned", "finetuned_long",
                                                                    "randomised") and last:
        tags.append("E3")
    if variant in ABLATIONS.get((algo, task), set()) and last:
        tags.append("E4")
    if algo == "ppo" and task == "navigation" and variant in ("normal", "from_obstacles"):
        tags.append("E5")
    return tags


# --- What each run holds -----------------------------------------------------------------

def read(eval_dir: Path, name: str, task: str, suite: str) -> pd.DataFrame | None:
    path = eval_dir / name / task / f"{suite}.csv"
    return pd.read_csv(path) if path.exists() else None


def curve_of(curves: pd.DataFrame | None, name: str) -> pd.DataFrame | None:
    """The validation curves of one training run: one row per evaluation step, one column per metric."""
    if curves is None:
        return None
    c = curves[curves["name"] == name]
    if c.empty:
        return None
    c = c[c["run"] == c["run"].max()]  # trained twice under one name: the latest (the run names end with the date)
    return c.pivot_table(index="step", columns="metric", values="value").sort_index()


def curve_scores(curve: pd.DataFrame) -> dict:
    s = curve["success_rate"]
    hit = s.index[s >= 0.9]
    out = {"train/final_val_success": float(s.iloc[-1]), "train/best_val_success": float(s.max()),
           "train/env_steps": int(s.index[-1]), "train/reached_90pct": int(len(hit) > 0)}
    if len(hit):  # left out when never reached, rather than a made-up number of steps
        out["train/steps_to_90pct"] = int(hit[0])
    return out


def sweeps(grid: pd.DataFrame | None, mass: pd.DataFrame | None) -> list[tuple[str, list[dict]]]:
    """The mass sweep and the two cuts through the wind x damage grid, as rows of history."""
    out = []
    if mass is not None:
        out.append(("mass/mass_factor", [
            {"mass/mass_factor": float(r.mass_factor), "mass/success": float(r.success_rate),
             "mass/success_low": float(r.success_low), "mass/success_high": float(r.success_high)}
            for r in mass.itertuples()]))
    if grid is not None:
        g = grid.pivot_table(index="wind", columns="damage", values="success_rate")
        out.append(("wind/wind_pct", [
            {"wind/wind_pct": round(100 * w), "wind/success": float(row.mean()),
             **{f"wind/success_damage_{round(100 * d):02d}": float(v) for d, v in row.items()}}
            for w, row in g.iterrows()]))
        out.append(("damage/damage_pct", [
            {"damage/damage_pct": round(100 * d), "damage/success": float(col.mean()),
             **{f"damage/success_wind_{round(100 * w):02d}": float(v) for w, v in col.items()}}
            for d, col in g.items()]))
    return out


def plan(eval_dir: Path, curves_path: Path, only: list[str] | None) -> tuple[list[dict], dict]:
    """Everything to log, without W&B: one entry per result run, and the tables."""
    results = pd.read_csv(eval_dir / "all_results.csv")
    curves = pd.read_csv(curves_path) if curves_path.exists() else None
    train = {}
    if curves is not None:
        latest = curves.sort_values("run").drop_duplicates("name", keep="last")
        train = {r["name"]: r.to_dict() for _, r in latest.iterrows()}
    rows = {(r["name"], r["task"]): r for _, r in results.iterrows()}

    # The longest training on each task: where the flat reference lines stop.
    longest = {}
    for (name, task) in rows:
        c = curve_of(curves, name)
        if c is not None:
            longest[task] = max(longest.get(task, 0), int(c.index[-1]))

    runs, grids, masses, behaviours = [], [], [], []
    for (name, task), r in rows.items():
        if only and name not in only:
            continue
        base = name.removesuffix("_best")
        c = who(name, task, train.get(base))
        summary = {k: float(v) for k, v in r.items() if k not in ("name", "group", "task") and finite(v)}
        history = []
        curve = curve_of(curves, base) if c["kind"] == "rl" and c["checkpoint"] == "last" else None
        if curve is not None:
            summary.update(curve_scores(curve))
            history.append(("val/env_steps", [{"val/env_steps": int(step), **{f"val/{m}": float(v)
                                                for m, v in row.items() if finite(v)}}
                                              for step, row in curve.iterrows()]))
        if c["reference_line"] and task in longest:
            flat = {f"val/{m}": float(r[f"test/{m}"]) for m in REFERENCE if finite(r.get(f"test/{m}"))}
            history.append(("val/env_steps", [{"val/env_steps": 0, **flat}, {"val/env_steps": longest[task], **flat}]))
        best = rows.get((f"{name}_best", task))
        if best is not None:  # SAC and TD3 on hover: the last model and the best checkpoint, side by side
            summary.update({f"best_checkpoint/{k.replace('/', '_')}": float(best[k])
                            for k in ("test/success_rate", "grid/robust_area") if finite(best.get(k))})
        grid, mass = read(eval_dir, name, task, "grid"), read(eval_dir, name, task, "mass")
        history += sweeps(grid, mass)
        behaviour = read(eval_dir, name, task, "behaviour")
        ident = {"controller": c["controller"], "policy": name, "seed": c["seed"], "task": task}
        if grid is not None:
            grids.append(grid.assign(wind_pct=(100 * grid["wind"]).round().astype(int),
                                     damage_pct=(100 * grid["damage"]).round().astype(int), **ident))
        if mass is not None:
            masses.append(mass.assign(**ident))
        if behaviour is not None:
            behaviour = behaviour[["seed", "bearing_deg", "net_turn_deg", "heading_change_deg", "turned", "faced",
                                   "backed", "success", "time_s"]].rename(columns={"seed": "episode_seed"})
            behaviours.append(behaviour.assign(**ident))
        seed = f" #{c['seed']}" if c["seed"] is not None else ""
        runs.append({"name": f"{task} · {c['controller']}{seed}", "config": c, "tags": experiments(c),
                     "summary": summary, "history": history, "behaviour": behaviour})

    tables = {"all_results": results}
    if grids:
        g = pd.concat(grids, ignore_index=True)
        tables["grid"] = g[["controller", "policy", "seed", "task", "wind_pct", "damage_pct", "success_rate",
                            "success_low", "success_high", "mean_distance_cm", "failed_by_drift"]]
        tables["grid_mean"] = (g.groupby(["controller", "task", "wind_pct", "damage_pct"], as_index=False)
                                .agg(success_rate=("success_rate", "mean"), seeds=("success_rate", "size")))
    if masses:
        tables["mass"] = pd.concat(masses, ignore_index=True)
    if behaviours:
        tables["behaviour"] = pd.concat(behaviours, ignore_index=True)
    return runs, tables


# --- W&B -----------------------------------------------------------------------------------

def replace_old(project: str, entity: str | None) -> None:
    """Delete the runs this script logged before (its two groups only), so that nothing is counted twice."""
    import wandb

    api = wandb.Api()
    path = f"{entity or api.default_entity}/{project}"
    old = [r for g in (GROUP, TABLES) for r in api.runs(path, filters={"group": g})]
    for r in old:
        r.delete()
    print(f"deleted {len(old)} runs from {path} (groups {GROUP}, {TABLES})")


def log(runs: list[dict], tables: dict, project: str, entity: str | None, mode: str, figures: Path) -> None:
    import wandb

    # W&B copies each run's config into its own debug log, a file Windows opens in its old
    # encoding (cp1252), which has no λ, α or −. Writing that line fails and Python prints the
    # whole traceback, although the run itself is sent correctly. We drop such lines quietly.
    logging.raiseExceptions = False

    for k, item in enumerate(runs, 1):
        run = wandb.init(project=project, entity=entity, group=GROUP, job_type="results", name=item["name"],
                         config=item["config"], tags=item["tags"], dir=str(ROOT / "results"), mode=mode)
        points = 0
        for x, rows in item["history"]:
            # W&B puts the last logged value of every metric in the summary. For the validation
            # curve that is the final validation, fine; for a sweep it would be the last point
            # (the heaviest mass, the strongest wind), which a bar chart could pick by mistake.
            sweep = not x.startswith("val/")
            run.define_metric(x, summary="none" if sweep else None)
            run.define_metric(x.split("/")[0] + "/*", step_metric=x, summary="none" if sweep else None)
            for row in rows:
                run.log(row)
            points += len(rows)
        b = item["behaviour"]
        if b is not None:
            # the controller in every row too, so that a chart over several runs can colour by it
            columns = ["bearing_deg", "net_turn_deg", "turned", "success", "time_s"]
            table = wandb.Table(dataframe=plain(b[columns].assign(controller=item["config"]["controller"])))
            title = f"{item['config']['controller']}: net turn vs target bearing"
            scatter = wandb.plot.scatter(table, "bearing_deg", "net_turn_deg", title=title)
            run.log({"behaviour/turn_vs_bearing": scatter})
        run.summary.update(item["summary"])
        run.finish()
        print(f"[{k}/{len(runs)}] {item['name']}: {len(item['summary'])} scores, {points} curve points")

    run = wandb.init(project=project, entity=entity, group=TABLES, job_type="results", name="results tables",
                     dir=str(ROOT / "results"), mode=mode)
    run.log({name: wandb.Table(dataframe=plain(df)) for name, df in tables.items()})
    # The figures W&B cannot draw itself (heatmaps, E1b, E6), for the report's image panels.
    pictures = sorted(figures.glob("*.png")) if figures.is_dir() else []
    if pictures:
        run.log({f"figures/{p.stem}": wandb.Image(str(p)) for p in pictures})
    run.finish()
    print("results tables: " + ", ".join(f"{n} ({len(df)} rows)" for n, df in tables.items()))
    print(f"figures: {len(pictures)} images from {figures}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", type=Path, default=EVAL)
    ap.add_argument("--curves", type=Path, default=ROOT / "results" / "curves.csv")
    ap.add_argument("--only", nargs="*", help="log only these controllers (names as in all_results.csv)")
    ap.add_argument("--project", default="hummingbird-rl")
    ap.add_argument("--entity", default=None, help="W&B user or team (default: the one you are logged in as)")
    ap.add_argument("--replace", action="store_true", help="first delete the runs of the groups results and "
                                                           "results_tables")
    ap.add_argument("--figures", type=Path, default=ROOT / "results" / "figures",
                    help="folder of PNG figures to attach to the tables run (for the report's image panels)")
    ap.add_argument("--offline", action="store_true", help="write the runs locally only (send later: wandb sync)")
    ap.add_argument("--dry-run", action="store_true", help="print what would be logged; nothing goes to W&B")
    args = ap.parse_args()

    runs, tables = plan(args.eval, args.curves, args.only)
    if not runs:
        sys.exit("nothing to log: check --eval and --only")
    if args.dry_run:
        for item in runs:
            points = sum(len(rows) for _, rows in item["history"])
            print(f"{item['name']:<45} tags {','.join(item['tags']):<16} {len(item['summary']):>3} scores, "
                  f"{points:>4} curve points" + (", behaviour chart" if item["behaviour"] is not None else ""))
        print(f"{len(runs)} runs; tables: " + ", ".join(f"{n} ({len(df)} rows)" for n, df in tables.items()))
        return
    os.environ.setdefault("WANDB_SILENT", "true")  # one line per run from us, rather than W&B's banner
    if args.replace and not args.offline:
        replace_old(args.project, args.entity)
    log(runs, tables, args.project, args.entity, "offline" if args.offline else "online", args.figures)


if __name__ == "__main__":
    main()
