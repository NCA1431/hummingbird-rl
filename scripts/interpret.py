"""E6: PPO's critic and policy near hover, compared with the LQR (and the PID).

    uv run python scripts/interpret.py            # PPO hover normal, seeds 1-3, best models
    uv run python scripts/interpret.py --wandb    # and upload the figures and numbers to W&B

For every PPO model: its value landscape next to the LQR's (−xᵀPx), and the
Spearman rank correlation between the two within ±5 cm and ±15 cm of the hover
point. For the model chosen by validation (the highest validation success, then
the highest mean validation success; or --show): the closed-loop vector fields
of PID, LQR and PPO, and five trajectories of 2 s each. The method and its
limits are described in hummingbird_rl/analysis/interpretability.py.

Writes results/figures/e6_value_landscape.png and e6_vector_field.png, and the
numbers behind them to results/e6/ (summary.json, value_grid.csv,
vector_field.csv, trajectories.csv). Needs the trained models in results/runs/.
Takes about a minute.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hummingbird_rl.analysis import interpretability as it  # noqa: E402
from hummingbird_rl.analysis.plots import ALGO, ALGO_LIGHT, BLUES, INK, INK_2, INK_3, save, style  # noqa: E402
from hummingbird_rl.controllers import LQRController, PIDController  # noqa: E402
from hummingbird_rl.training.train import ALGOS, resolve_model  # noqa: E402
from hummingbird_rl.world.scene import START  # noqa: E402

SEEDS = ["ppo_hover_normal_s1", "ppo_hover_normal_s2", "ppo_hover_normal_s3"]
COLOR = {"PID": INK_2, "LQR": INK_3, "PPO": ALGO["ppo"]}


def load(name: str, which: str):
    best = resolve_model(name)                       # the best model of the latest run with this name
    path = best if which == "best" else best.parent / "model.zip"
    return ALGOS["ppo"].load(path, device="cpu"), path


def by_validation(names: list[str], curves: Path) -> str:
    """The run with the highest validation success, ties broken by its mean validation success."""
    if not curves.exists():
        return names[0]
    c = pd.read_csv(curves)
    c = c[(c["metric"] == "success_rate") & c["name"].isin(names)]
    score = {n: (g["value"].max(), g["value"].mean()) for n, g in c.groupby("name")}
    return max(names, key=lambda n: score.get(n, (-1.0, -1.0)))


# --- Figures -------------------------------------------------------------------------------

def heatmap(ax, v, xs, title):
    cm = 100 * xs
    ext = [cm[0], cm[-1], cm[0], cm[-1]]
    image = ax.imshow(it.normalise(v), origin="lower", extent=ext, cmap=BLUES, vmin=0, vmax=1, aspect="equal")
    lines = ax.contour(cm, cm, it.normalise(v), levels=[0.2, 0.4, 0.6, 0.8, 0.9], colors="white", linewidths=0.8)
    ax.clabel(lines, fontsize=6.5, fmt="%.1f")
    near = 100 * it.NEAR
    ax.add_patch(plt.Rectangle((-near, -near), 2 * near, 2 * near, fill=False, edgecolor=INK, linewidth=0.9,
                               linestyle="--"))
    ax.plot(0, 0, marker="+", color=INK, markersize=9, markeredgewidth=1.4)
    ax.set_title(title)
    ax.set_xlabel("forward offset Δx (cm)")
    ax.grid(False)
    return image


def value_figure(L: dict, shown: str, rho: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), layout="constrained")
    xs = L["xs"]
    heatmap(axes[0], L[shown], xs, f"PPO critic V(s), {shown.split('_')[-1].replace('s', 'seed ')}")
    image = heatmap(axes[1], L["lqr"], xs, r"LQR: $-x^{\top}Px$ (exact, linear model)")
    axes[0].set_ylabel("vertical offset Δy (cm)")
    cbar = fig.colorbar(image, ax=axes[:2], shrink=0.9)
    cbar.set_label("value rescaled to [0, 1]  (1 = best state)")

    ax = axes[2]
    a, b = it.normalise(L["lqr"]), it.normalise(L[shown])
    near = it.within(xs, it.NEAR)
    ax.scatter(a[~near], b[~near], s=4, color=ALGO_LIGHT["ppo"], linewidth=0, label="5 to 15 cm from hover")
    ax.scatter(a[near], b[near], s=5, color=ALGO["ppo"], linewidth=0, label="within ±5 cm")
    ax.plot([0, 1], [0, 1], color=INK_3, linewidth=0.9, linestyle=":")
    r = rho[shown]
    others = ", ".join(f"{v['near']:.2f}" for v in rho.values())
    ax.set_title(f"Same ordering? Spearman ρ = {r['near']:.2f} (±5 cm), {r['far']:.2f} (±15 cm)\n"
                 f"all seeds, ±5 cm: {others}", fontsize=9)
    ax.set_xlabel("LQR value (rescaled)")
    ax.set_ylabel("PPO critic (rescaled)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", markerscale=3)
    fig.suptitle("E6 · Value near hover: PPO's critic vs the LQR's exact cost-to-go "
                 "(bird at rest and level, Δz = 0, dashed square = ±5 cm)", x=0.01, ha="left", fontsize=11,
                 fontweight="bold")
    return fig


def field_figure(xs, fields: dict, paths: dict, align: dict, gaps: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True)
    cm = 100 * xs
    X, Y = np.meshgrid(cm, cm)
    spacing = cm[1] - cm[0]
    k = 1.4 * spacing / max(100 * np.linalg.norm(D, axis=2).max() for D in fields.values())   # same scale for all
    for ax, (name, D) in zip(axes, fields.items()):
        ax.quiver(X, Y, 100 * D[..., 0] * k, 100 * D[..., 1] * k, angles="xy", scale_units="xy", scale=1,
                  color=INK_3, width=0.0035, headwidth=3.5, alpha=0.9, zorder=1)
        for path in paths[name]:
            p = 100 * (path[:, :2] - START[:2])
            ax.plot(p[:, 0], p[:, 1], color=COLOR[name], linewidth=1.8, zorder=3)
            ax.plot(p[0, 0], p[0, 1], "o", color="white", markeredgecolor=COLOR[name], markersize=5, zorder=4)
        ax.plot(0, 0, marker="+", color=INK, markersize=10, markeredgewidth=1.5, zorder=5)
        if name == "LQR":
            title = "LQR (reference)\ncorrects height first, then slides across"
        else:
            a = align[name]
            title = (f"{name} · arrows vs LQR: {a['near']:.2f} (±5 cm), {a['far']:.2f} (±15 cm)\n"
                     f"paths {100 * gaps[name + '-LQR']:.1f} cm from LQR's"
                     + (f", {100 * gaps['PPO-PID']:.1f} cm from PID's" if name == "PPO" else ""))
        ax.set_title(title, fontsize=9.5)
        ax.set_xlim(cm[0] - spacing, cm[-1] + spacing)
        ax.set_ylim(cm[0] - spacing, cm[-1] + spacing)
        ax.set_aspect("equal")
        ax.set_xlabel("forward offset Δx (cm)")
    axes[0].set_ylabel("vertical offset Δy (cm)")
    fig.suptitle("E6 · Where each controller takes the bird from rest: in 0.1 s (arrows, same scale in the three "
                 "panels) and in 2 s (lines); calm air", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))   # leave room for the title above the two-line panel titles
    return fig


# --- Main ------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policies", nargs="+", default=SEEDS, help="PPO run names (hover)")
    ap.add_argument("--which", choices=["best", "final"], default="best")
    ap.add_argument("--show", default=None, help="the run whose landscape and vector field are drawn")
    ap.add_argument("--curves", type=Path, default=ROOT / "results" / "curves.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--wandb", action="store_true", help="also upload the figures and the numbers to W&B")
    ap.add_argument("--project", default="hummingbird-rl")
    args = ap.parse_args()
    style()

    lqr, pid = LQRController(), PIDController()
    models = {}
    for name in args.policies:
        try:
            models[name], path = load(name, args.which)
            print(f"{name}: {path}")
        except FileNotFoundError as e:
            print(f"{name}: skipped ({e})")
    if not models:
        sys.exit("no PPO model found in results/runs")
    shown = args.show or by_validation(list(models), args.curves)

    # 1. Value landscapes (61 x 61) and their rank correlation with the LQR's.
    L = it.value_landscape(None, lqr)
    rho, axes = {}, {"LQR": it.axis_values(L["lqr"], L["xs"])}
    for name, model in models.items():
        L[name] = it.value_landscape(model, lqr)["ppo"]
        rho[name] = {"near": it.spearman(L[name], L["lqr"], L["xs"], it.NEAR),
                     "far": it.spearman(L[name], L["lqr"], L["xs"], it.HALF)}
        axes[name] = it.axis_values(L[name], L["xs"])
        print(f"{name}: Spearman rho {rho[name]['near']:.3f} within ±5 cm, {rho[name]['far']:.3f} within ±15 cm")

    # 2. Closed-loop vector fields (21 x 21, 0.1 s) and 2 s trajectories, for the model shown.
    controllers = {"PID": pid, "LQR": lqr, "PPO": models[shown]}
    # In 0.1 s from rest the bird mostly moves vertically (thrust acts at once, a horizontal move
    # needs a tilt first), so we also compare the directions after 0.5 s and the whole 2 s paths.
    fields, later, paths = {}, {}, {}
    for label, policy in controllers.items():
        xs, fields[label] = it.vector_field(policy)
        _, later[label] = it.vector_field(policy, duration=0.5)
        paths[label] = [it.fly(policy, it.offset_state(*s), 120) for s in it.trajectory_starts()]
    align = {label: {"near": it.alignment(fields[label], fields["LQR"], xs, it.NEAR),
                     "far": it.alignment(fields[label], fields["LQR"], xs, it.HALF),
                     "near_0.5s": it.alignment(later[label], later["LQR"], xs, it.NEAR),
                     "far_0.5s": it.alignment(later[label], later["LQR"], xs, it.HALF)} for label in ("PID", "PPO")}
    gaps = {"PID-LQR": it.path_gap(paths["PID"], paths["LQR"]), "PPO-LQR": it.path_gap(paths["PPO"], paths["LQR"]),
            "PPO-PID": it.path_gap(paths["PPO"], paths["PID"])}
    for label, a in align.items():
        print(f"{label} vs LQR, direction of the displacement: after 0.1 s {a['near']:.3f} (±5 cm), "
              f"{a['far']:.3f} (±15 cm); after 0.5 s {a['near_0.5s']:.3f}, {a['far_0.5s']:.3f}")
    print("mean gap between the 2 s paths: " + ", ".join(f"{k} {100 * v:.1f} cm" for k, v in gaps.items()))

    figures = args.out / "figures"
    written = [save(value_figure(L, shown, rho), figures, "e6_value_landscape"),
               save(field_figure(xs, fields, paths, align, gaps), figures, "e6_vector_field")]
    for p in written:
        print(f"wrote {p}")

    # The numbers behind the figures.
    data = args.out / "e6"
    data.mkdir(parents=True, exist_ok=True)
    X, Y = np.meshgrid(100 * L["xs"], 100 * L["xs"])
    pd.DataFrame({"dx_cm": X.ravel(), "dy_cm": Y.ravel(), "lqr": L["lqr"].ravel(),
                  **{n: L[n].ravel() for n in models}}).to_csv(data / "value_grid.csv", index=False)
    Xf, Yf = np.meshgrid(100 * xs, 100 * xs)
    pd.concat([pd.DataFrame({"controller": label, "dx_cm": Xf.ravel(), "dy_cm": Yf.ravel(),
                             "ux_cm": 100 * D[..., 0].ravel(), "uy_cm": 100 * D[..., 1].ravel()})
               for label, D in fields.items()]).to_csv(data / "vector_field.csv", index=False)
    pd.concat([pd.DataFrame({"controller": label, "start": k, "step": range(len(p)),
                             "x_cm": 100 * (p[:, 0] - START[0]), "y_cm": 100 * (p[:, 1] - START[1])})
               for label, ps in paths.items() for k, p in enumerate(ps)]).to_csv(data / "trajectories.csv",
                                                                                index=False)
    final = {label: [float(100 * np.linalg.norm(p[-1] - START)) for p in ps] for label, ps in paths.items()}
    summary = {"which": args.which, "shown": shown, "spearman": rho, "axis_values_at_10cm": axes,
               "direction_vs_lqr": align, "path_gap_cm": {k: 100 * v for k, v in gaps.items()},
               "trajectory_final_distance_cm": final,
               "spearman_mean": {k: float(np.mean([r[k] for r in rho.values()])) for k in ("near", "far")}}
    (data / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {data / 'summary.json'} and the CSV files next to it")

    if args.wandb:
        import wandb

        run = wandb.init(project=args.project, name="e6 interpretability", job_type="analysis", group="figures",
                         config={"policies": list(models), "which": args.which, "shown": shown})
        run.log({p.stem: wandb.Image(str(p)) for p in written})
        run.log({"e6/per_seed": wandb.Table(dataframe=pd.DataFrame(
            [{"policy": n, "spearman_5cm": r["near"], "spearman_15cm": r["far"]} for n, r in rho.items()]))})
        run.summary.update({"e6/spearman_5cm_mean": summary["spearman_mean"]["near"],
                            "e6/spearman_15cm_mean": summary["spearman_mean"]["far"],
                            **{f"e6/direction_vs_lqr_{label.lower()}_{k}": v
                               for label, a in align.items() for k, v in a.items()},
                            **{f"e6/path_gap_cm_{k}": 100 * v for k, v in gaps.items()}})
        run.finish()


if __name__ == "__main__":
    main()
