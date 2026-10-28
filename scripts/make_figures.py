"""The figures of the reports, from the evaluation results and the learning curves.

Reads results/eval/ (written by evaluate.py and eval_summary.py: one folder per
controller and task, plus all_results.csv) and results/curves.csv (written by
export_curves.py), and writes PNG files to results/figures/:

    e1_learning_curves   validation success during training, every task and algorithm
    e1_test              test success and precision, against the PID and the LQR
    e1b_behaviour        go-to-point: where the target was vs how much the bird turned
    e2_heatmaps          success over the grid of wind x wing damage
    e2_robust_area       robust area, trained in normal vs randomised conditions
    e2_mass              success vs the bird's mass
    e2_failure           a wing suddenly losing 40%: deviation and survival
    e2_late_degradation  SAC and TD3 on hover: the best checkpoint vs the last one
    e3_finetuning        fine-tuning a normal-trained agent in randomised conditions
    e4_ablations         what each part of PPO (and SAC's alpha, a failure penalty) brings
    e5_transfer          navigation from scratch vs starting from the obstacles agent

    uv run python scripts/make_figures.py            # all of them
    uv run python scripts/make_figures.py --wandb    # and upload them to W&B (run "figures")

Agents are averaged over their seeds (each seed is also drawn, as a dot or a thin
line): the variability that matters is between training runs, not between episodes.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hummingbird_rl.analysis.plots import (  # noqa: E402
    ALGO,
    ALGO_LIGHT,
    ALGO_NAME,
    BASELINE,
    BLUES,
    INK,
    INK_2,
    INK_3,
    TASK_NAME,
    band,
    baseline_line,
    percent,
    save,
    steps_axis,
    style,
)

EVAL = ROOT / "results" / "eval"
OUT = ROOT / "results" / "figures"
SEEDS = range(1, 6)


# --- Data ------------------------------------------------------------------------------

class Data:
    def __init__(self, eval_dir: Path, curves: Path):
        self.eval = eval_dir
        self.all = pd.read_csv(eval_dir / "all_results.csv")
        self.curves = pd.read_csv(curves) if curves.exists() else None

    def row(self, name: str, task: str) -> pd.Series | None:
        r = self.all[(self.all["name"] == name) & (self.all["task"] == task)]
        return r.iloc[0] if len(r) else None

    def seeds(self, group: str, task: str, suffix: str = "") -> list[str]:
        """The names of the runs of a group that were evaluated on `task` (e.g. ppo_hover_normal_s1..s5)."""
        return [f"{group}_s{s}{suffix}" for s in SEEDS if self.row(f"{group}_s{s}{suffix}", task) is not None]

    def values(self, group: str, task: str, column: str, suffix: str = "") -> np.ndarray:
        return np.array([self.row(n, task)[column] for n in self.seeds(group, task, suffix)], dtype=float)

    def csv(self, name: str, task: str, suite: str) -> pd.DataFrame | None:
        path = self.eval / name / task / f"{suite}.csv"
        return pd.read_csv(path) if path.exists() else None

    def curve(self, group: str, metric: str = "success_rate") -> tuple[np.ndarray, np.ndarray] | None:
        """Validation curve of every seed of a group, on the steps they all have: (steps, seeds x steps)."""
        if self.curves is None:
            return None
        c = self.curves[(self.curves["metric"] == metric) & self.curves["name"].str.match(rf"^{group}_s[1-9]$")]
        if c.empty:
            return None
        table = c.pivot_table(index="step", columns="name", values="value").dropna()
        return table.index.to_numpy(), table.to_numpy().T


def mean_bar(ax, x, values, color, width=0.7, dots=True, fmt=None):
    """A bar at the mean over seeds, the seeds as dots on it, and the mean written above both."""
    if len(values) == 0:
        return
    ax.bar(x, values.mean(), width, color=color, zorder=2)
    if dots and len(values) > 1:
        jitter = np.linspace(-0.12, 0.12, len(values)) * width
        ax.scatter(x + jitter, values, s=12, color="white", edgecolor=INK, linewidth=0.7, zorder=3)
    if fmt:
        top = max(values.mean(), np.nanmax(values))
        ax.annotate(fmt.format(values.mean()), (x, top), xytext=(0, 4), textcoords="offset points", ha="center",
                    fontsize=7.5, color=INK_2)


# --- E1: benchmark ---------------------------------------------------------------------

ALGOS_FOR = {"hover": ["ppo", "sac", "td3"], "goto": ["ppo", "sac", "td3"], "obstacles": ["ppo", "sac"],
             "navigation": ["ppo", "sac"]}


def e1_learning_curves(d: Data) -> plt.Figure:
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    for ax, task in zip(axes, TASK_NAME):
        for algo in ALGOS_FOR[task]:
            c = d.curve(f"{algo}_{task}_normal")
            if c:
                band(ax, c[0], c[1], ALGO[algo], f"{ALGO_NAME[algo]} ({c[1].shape[0]} seeds)")
        refs = {b: d.row(b, task)["test/success_rate"] for b in ("pid", "lqr") if d.row(b, task) is not None}
        same = len(refs) == 2 and abs(refs["pid"] - refs["lqr"]) < 0.05
        for b, y in refs.items():
            baseline_line(ax, y, b, label=not same)
        if same:  # both at the same level: one label for the two lines
            ax.annotate("PID, LQR", (1.0, refs["pid"]), xycoords=("axes fraction", "data"), xytext=(3, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK_2)
        ax.set_title(TASK_NAME[task])
        ax.set_xlabel("environment steps")
        steps_axis(ax)
        percent(ax)
        ax.set_ylim(0, 1.03)
        ax.legend(loc="upper left" if task == "navigation" else "lower right")
    axes[0].set_ylabel("validation success")
    fig.suptitle("E1 · Learning curves in normal conditions (mean over seeds, band = range of seeds)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def e1_test(d: Data) -> plt.Figure:
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.4), gridspec_kw={"height_ratios": [1.3, 1]})
    for col, task in enumerate(TASK_NAME):
        names = ["pid", "lqr"] + ALGOS_FOR[task]
        for row, (column, fmt) in enumerate([("test/success_rate", "{:.0%}"), ("test/final_distance_cm", "{:.1f}")]):
            ax = axes[row, col]
            means = []
            for x, n in enumerate(names):
                if n in BASELINE:
                    r = d.row(n, task)
                    v = np.array([r[column]]) if r is not None else np.array([])
                    color = BASELINE[n]["color"]
                else:
                    v = d.values(f"{n}_{task}_normal", task, column)
                    color = ALGO[n]
                mean_bar(ax, x, v, color, fmt=fmt)
                means.append(v.mean() if len(v) else np.nan)
            ax.set_xticks(range(len(names)), [n.upper() for n in names])
            ax.grid(axis="x", visible=False)
            if row == 0:
                ax.set_title(TASK_NAME[task])
                percent(ax)
                ax.set_ylim(0, 1.12)
            else:
                ax.set_ylim(0, max(np.nanmax(means) * 1.3, 1))
        axes[0, 0].set_ylabel("test success (200 episodes)")
        axes[1, 0].set_ylabel("final distance (cm)")
    fig.suptitle("E1 · Test results in normal conditions (bars: mean over seeds, dots: each seed)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def e1b_behaviour(d: Data) -> plt.Figure:
    who = [("pid", "PID", INK_2), ("lqr", "LQR", INK_3), ("ppo_goto_normal_s1", "PPO #1", ALGO["ppo"]),
           ("sac_goto_normal_s1", "SAC #1", ALGO["sac"]), ("td3_goto_normal_s1", "TD3 #1", ALGO["td3"])]
    fig, axes = plt.subplots(1, len(who), figsize=(13, 3.0), sharey=True)
    for ax, (name, title, color) in zip(axes, who):
        b = d.csv(name, "goto", "behaviour")
        if b is None:
            ax.set_visible(False)
            continue
        ax.scatter(b["bearing_deg"], b["net_turn_deg"], s=9, color=color, alpha=0.75, linewidth=0)
        ax.plot([-180, 180], [-180, 180], color=INK_3, linewidth=0.9, linestyle=":")
        ax.axhline(0, color=INK_3, linewidth=0.9, linestyle=":")
        r = d.row(name, "goto")
        behind = r["behaviour/p_turn_target_behind"] if r is not None else np.nan
        ax.set_title(f"{title} · turns for a target behind: {behind:.0%}", fontsize=9)
        ax.set_xlim(-185, 185)
        ax.set_ylim(-200, 200)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_xlabel("target bearing at the start (°)")
    axes[0].set_ylabel("net turn during the flight (°)")
    axes[0].annotate("turns to face it", (150, 150), fontsize=7.5, color=INK_3, ha="right", va="bottom")
    axes[0].annotate("does not turn", (-175, 22), fontsize=7.5, color=INK_3)
    fig.suptitle("E1b · Go-to-point: does the bird turn towards the target, or fly sideways and backwards?",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


# --- E2: robustness --------------------------------------------------------------------

def grid_mean(d: Data, names: list[str], task: str) -> pd.DataFrame | None:
    grids = [g for n in names if (g := d.csv(n, task, "grid")) is not None]
    if not grids:
        return None
    g = pd.concat(grids).groupby(["damage", "wind"])["success_rate"].mean().unstack("wind")
    return g.sort_index(ascending=False)  # damage grows downwards


def robust_controllers(d: Data, task: str) -> list[tuple[str, list[str]]]:
    """Who is compared on robustness: the baselines, PPO both ways, SAC and TD3 randomised."""
    best = "_best" if task == "hover" else ""  # hover SAC/TD3 degrade late: their best checkpoint
    out = [("PID", ["pid"]), ("LQR", ["lqr"]),
           ("PPO normal", d.seeds("ppo_" + task + "_normal", task)),
           ("PPO randomised", d.seeds("ppo_" + task + "_randomised", task))]
    for algo in ("sac", "td3"):
        names = d.seeds(f"{algo}_{task}_randomised", task, best)
        if names:
            out.append((f"{ALGO_NAME[algo]} randomised" + (" (best)" if best else ""), names))
    return out


def e2_heatmaps(d: Data) -> plt.Figure:
    tasks = ["hover", "goto"]
    cols = max(len(robust_controllers(d, t)) for t in tasks)
    fig, axes = plt.subplots(2, cols, figsize=(2.3 * cols, 5.8), squeeze=False, layout="constrained")
    image = None
    for row, task in enumerate(tasks):
        for col in range(cols):
            ax = axes[row, col]
            ctrl = robust_controllers(d, task)
            if col >= len(ctrl):
                ax.set_visible(False)
                continue
            title, names = ctrl[col]
            g = grid_mean(d, names, task)
            if g is None:
                ax.set_visible(False)
                continue
            image = ax.imshow(g.to_numpy(), cmap=BLUES, vmin=0, vmax=1, aspect="auto")
            for (i, j), v in np.ndenumerate(g.to_numpy()):
                ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.55 else INK)
            area = np.mean(g.to_numpy())
            ax.set_title(f"{title}\nrobust area {area:.2f}", fontsize=8.5)
            ax.set_xticks(range(g.shape[1]), [f"{100 * w:.0f}" for w in g.columns] if row == 1 else [], fontsize=7)
            ax.set_yticks(range(g.shape[0]), [f"{100 * w:.0f}" for w in g.index], fontsize=7)
            ax.grid(False)
            if col == 0:
                ax.set_ylabel(f"{TASK_NAME[task]}\nwing force lost (%)")
            if row == 1:
                ax.set_xlabel("wind (% of weight)")
    cbar = fig.colorbar(image, ax=axes, shrink=0.6, pad=0.01)
    cbar.ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    cbar.set_label("success")
    fig.suptitle("E2 · Success over wind x wing damage (20 episodes per cell, mean over seeds; trained up to 30% wind)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    return fig


def e2_robust_area(d: Data) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, task in zip(axes, ["hover", "goto"]):
        best = "_best" if task == "hover" else ""
        x = 0
        ticks = []
        for algo in ("ppo", "sac", "td3"):
            sfx = "" if algo == "ppo" else best
            for regime, color in (("normal", ALGO_LIGHT[algo]), ("randomised", ALGO[algo])):
                v = d.values(f"{algo}_{task}_{regime}", task, "grid/robust_area", sfx)
                mean_bar(ax, x, v, color, fmt="{:.2f}")
                ticks.append((x, f"{ALGO_NAME[algo]}\n{regime}"))
                x += 1
            x += 0.5
        for b in ("pid", "lqr"):
            r = d.row(b, task)
            if r is not None:
                baseline_line(ax, r["grid/robust_area"], b)
        ax.set_xticks([t[0] for t in ticks], [t[1] for t in ticks], fontsize=7.5)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, 1.05)
        ax.set_title(TASK_NAME[task] + (" (SAC/TD3: best checkpoint)" if best else ""))
    axes[0].set_ylabel("robust area (mean success on the grid)")
    fig.suptitle("E2 · Training in randomised conditions: robust area, normal vs randomised",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def e2_mass(d: Data) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, task in zip(axes, ["hover", "goto"]):
        for b in ("pid", "lqr"):
            m = d.csv(b, task, "mass")
            if m is not None:
                ax.plot(m["mass_factor"], m["success_rate"], label=b.upper(), linewidth=1.6, **BASELINE[b])
        for regime, color in (("normal", ALGO_LIGHT["ppo"]), ("randomised", ALGO["ppo"])):
            ms = [m for n in d.seeds(f"ppo_{task}_{regime}", task) if (m := d.csv(n, task, "mass")) is not None]
            if ms:
                ys = np.array([m["success_rate"].to_numpy() for m in ms])
                band(ax, ms[0]["mass_factor"].to_numpy(), ys, color, f"PPO {regime}")
        ax.axvspan(0.75, 1.25, color=INK_3, alpha=0.08, linewidth=0)
        ax.annotate("randomised training range", (1.0, 0.03), ha="center", fontsize=7.5, color=INK_3)
        ax.set_title(TASK_NAME[task])
        ax.set_xlabel("mass (x normal)")
        percent(ax)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("success")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("E2 · Success vs the bird's mass (controllers are not told the mass)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def e2_failure(d: Data) -> plt.Figure:
    ctrl = robust_controllers(d, "hover")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    labels = [c[0] for c in ctrl]
    colors = [INK_2, INK_3, ALGO_LIGHT["ppo"], ALGO["ppo"], ALGO["sac"], ALGO["td3"]][: len(ctrl)]
    for ax, (column, title, fmt) in zip(axes, [
        ("failure/median_max_deviation_cm", "largest deviation after the failure (median, cm)", "{:.1f}"),
        ("failure/survival_rate", "flights still under control at the end", "{:.0%}"),
    ]):
        means = []
        for x, ((_, names), color) in enumerate(zip(ctrl, colors)):
            v = np.array([d.row(n, "hover")[column] for n in names if d.row(n, "hover") is not None], dtype=float)
            mean_bar(ax, x, v, color, fmt=fmt)
            means.append(np.nanmean(v) if len(v) else np.nan)
        ax.set_xticks(range(len(ctrl)), [lab.replace(" (best)", "\n(best)").replace(" ", "\n", 1) for lab in labels],
                      fontsize=7.5)
        ax.grid(axis="x", visible=False)
        ax.set_title(title, fontsize=9.5)
        if "survival" in column:
            percent(ax)
            ax.set_ylim(0, 1.12)
    fig.suptitle("E2 · Hover: one wing suddenly loses 40% of its force (between 1.5 and 3 s)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def e2_late_degradation(d: Data) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    for algo in ("sac", "td3"):
        for regime, style_ in (("normal", ":"), ("randomised", "-")):
            c = d.curve(f"{algo}_hover_{regime}")
            if c:
                for k, y in enumerate(c[1]):
                    ax.plot(c[0], y, color=ALGO[algo], linestyle=style_, linewidth=1.1, alpha=0.85,
                            label=f"{ALGO_NAME[algo]} {regime}" if k == 0 else None)
    c = d.curve("ppo_hover_normal")
    if c:
        ax.plot(c[0], c[1].mean(axis=0), color=ALGO["ppo"], linewidth=2, label="PPO normal (mean)")
    ax.set_title("Hover: validation success of every seed during training")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("validation success")
    steps_axis(ax)
    percent(ax)
    ax.set_ylim(0, 1.03)
    ax.legend(fontsize=7.5, loc="upper center", ncol=5, bbox_to_anchor=(0.5, -0.27))
    ax = axes[1]
    y = 0
    ticks = []
    for algo in ("sac", "td3"):
        for regime in ("normal", "randomised"):
            final = d.values(f"{algo}_hover_{regime}", "hover", "test/success_rate")
            best = d.values(f"{algo}_hover_{regime}", "hover", "test/success_rate", "_best")
            for f, b in zip(final, best):
                ax.plot([f, b], [y, y], color=ALGO_LIGHT[algo], linewidth=1.5, zorder=1)
                ax.scatter([f], [y], color="white", edgecolor=ALGO[algo], s=28, zorder=2)
                ax.scatter([b], [y], color=ALGO[algo], s=28, zorder=2)
                y += 1
            ticks.append((y - (len(final) + 1) / 2, f"{ALGO_NAME[algo]}\n{regime}"))
            y += 0.8
    ax.set_yticks([t[0] for t in ticks], [t[1] for t in ticks], fontsize=7.5)
    ax.invert_yaxis()
    percent(ax, "x")
    ax.set_xlim(-0.03, 1.05)
    ax.set_xlabel("test success: last model (open) vs best checkpoint (filled)")
    ax.set_title("Each seed, last vs best")
    fig.suptitle("E2 · The off-policy agents on hover: validation success swings during training",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


# --- E3: fine-tuning -------------------------------------------------------------------

def e3_finetuning(d: Data) -> plt.Figure:
    recipes = [("normal", "normal\n(E1)", "#9ec5f4"), ("finetuned", "fine-tuned\n+30%", "#5598e7"),
               ("finetuned_long", "fine-tuned\n+100%", "#256abf"),
               ("randomised", "randomised\nfrom scratch", "#104281")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    for k, task in enumerate(["hover", "goto"]):
        for j, (column, title, fmt) in enumerate([("grid/robust_area", "robust area", "{:.2f}"),
                                                  ("test/success_rate", "test success, normal conditions", "{:.0%}")]):
            ax = axes[2 * k + j]
            means = []
            for x, (recipe, _, color) in enumerate(recipes):
                v = d.values(f"ppo_{task}_{recipe}", task, column)
                mean_bar(ax, x, v, color, fmt=fmt)
                means.append(v.mean() if len(v) else np.nan)
            r = d.row("pid", task)
            if r is not None:
                baseline_line(ax, r[column], "pid")
            ax.set_xticks(range(len(recipes)), [rc[1] for rc in recipes], fontsize=7)
            ax.grid(axis="x", visible=False)
            ax.set_ylim(0, 1.12)
            ax.set_title(f"{TASK_NAME[task]}: {title}", fontsize=9.5)
            if "success" in column:
                percent(ax)
    fig.suptitle("E3 · PPO trained in normal conditions, then fine-tuned in randomised ones",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


# --- E4: ablations ---------------------------------------------------------------------

ABLATIONS = [
    ("ppo_hover_normal", "PPO, as trained", "hover", ALGO["ppo"]),
    ("ppo_hover_noclip", "no clipping", "hover", ALGO_LIGHT["ppo"]),
    ("ppo_hover_noadvnorm", "no advantage normalisation", "hover", ALGO_LIGHT["ppo"]),
    ("ppo_hover_norewnorm", "no reward normalisation", "hover", ALGO_LIGHT["ppo"]),
    ("ppo_hover_norewnorm_noadvnorm", "neither normalisation", "hover", ALGO_LIGHT["ppo"]),
    ("ppo_hover_lambda0", "GAE λ = 0", "hover", ALGO_LIGHT["ppo"]),
    ("ppo_hover_lambda1", "GAE λ = 1", "hover", ALGO_LIGHT["ppo"]),
    ("ppo_hover_nohoverinit", "no hover start", "hover", ALGO_LIGHT["ppo"]),
    ("sac_hover_normal", "SAC, learned α", "hover", ALGO["sac"]),
    ("sac_hover_fixedalpha", "SAC, fixed α = 0.2", "hover", ALGO_LIGHT["sac"]),
    ("sac_hover_annealed", "SAC, annealed learning rate", "hover", ALGO_LIGHT["sac"]),
    ("td3_hover_normal", "TD3, as trained", "hover", ALGO["td3"]),
    ("td3_hover_annealed", "TD3, annealed learning rate", "hover", ALGO_LIGHT["td3"]),
    ("ppo_navigation_normal", "PPO navigation", "navigation", ALGO["ppo"]),
    ("ppo_navigation_f50", "failure penalty −50", "navigation", ALGO_LIGHT["ppo"]),
]


def steps_to(c: tuple[np.ndarray, np.ndarray] | None, level: float = 0.9) -> np.ndarray:
    """For each seed, the first validation step at which success reached `level` (nan if never)."""
    if c is None:
        return np.array([])
    out = []
    for y in c[1]:
        hit = np.nonzero(y >= level)[0]
        out.append(c[0][hit[0]] if len(hit) else np.nan)
    return np.array(out, dtype=float)


def e4_ablations(d: Data) -> plt.Figure:
    # Only the variants that were trained and evaluated, in three blocks separated by a gap:
    # PPO on hover, the off-policy agents on hover, navigation.
    rows = [a for a in ABLATIONS if d.seeds(a[0], a[2])]
    block = [2 if task == "navigation" else (0 if group.startswith("ppo") else 1) for group, _, task, _ in rows]
    ys = np.zeros(len(rows))
    y = 0.0
    for k in range(len(rows) - 1, -1, -1):       # from the bottom up
        ys[k] = y
        y += 1.0 + (1.0 if k > 0 and block[k - 1] != block[k] else 0.0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 1.0 + 0.32 * (len(rows) + 2)), sharey=True)
    for y, (group, title, task, color) in zip(ys, rows):
        v = d.values(group, task, "test/success_rate")
        if len(v):
            axes[0].barh(y, v.mean(), 0.7, color=color, zorder=2)
            axes[0].scatter(v, np.full(len(v), y), s=12, color="white", edgecolor=INK, linewidth=0.7, zorder=3)
            axes[0].annotate(f"{v.mean():.0%}", (max(v.mean(), v.max()), y), xytext=(6, 0), textcoords="offset points",
                             va="center", fontsize=7.5, color=INK_2)
        s = steps_to(d.curve(group))
        if task == "hover" and len(s):
            reached = s[np.isfinite(s)]
            if len(reached):
                axes[1].barh(y, reached.mean(), 0.7, color=color, zorder=2)
                axes[1].scatter(reached, np.full(len(reached), y), s=12, color="white", edgecolor=INK,
                                linewidth=0.7, zorder=3)
            label = f"{len(reached)}/{len(s)} seeds reach it" if len(reached) < len(s) else ""
            if label:
                axes[1].annotate(label, (reached.max() if len(reached) else 0, y), xytext=(6, 0),
                                 textcoords="offset points", va="center", fontsize=7.5, color=INK_2)
        elif task != "hover":
            axes[1].annotate("(hover only)", (0, y), xytext=(4, 0), textcoords="offset points", va="center",
                             fontsize=7.5, color=INK_3)
    axes[0].set_yticks(ys, [a[1] for a in rows], fontsize=8)
    percent(axes[0], "x")
    axes[0].set_xlim(0, 1.12)
    axes[0].set_title("test success (200 episodes)")
    axes[0].grid(axis="y", visible=False)
    axes[1].set_title("hover: steps until validation success first reaches 90%")
    steps_axis(axes[1])
    axes[1].grid(axis="y", visible=False)
    fig.suptitle("E4 · Ablations: each part removed or changed, the rest as trained (3 seeds each)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


# --- E5: transfer ----------------------------------------------------------------------

def e5_transfer(d: Data) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1, 1.5]})
    ax = axes[0]
    pairs = []
    for s in SEEDS:
        a = d.row(f"ppo_navigation_normal_s{s}", "navigation")
        b = d.row(f"ppo_navigation_from_obstacles_s{s}", "navigation")
        if a is not None and b is not None:
            pairs.append((s, a["test/success_rate"], b["test/success_rate"]))
    for s, a, b in pairs:
        ax.plot([0, 1], [a, b], color=ALGO["ppo"] if b > a else INK_3, linewidth=1.3, marker="o", markersize=5)
        ax.annotate(f"#{s}", (1, b), xytext=(6, 0), textcoords="offset points", va="center", fontsize=7.5, color=INK_2)
    if len(pairs) >= 2:
        a = np.array([p[1] for p in pairs])
        b = np.array([p[2] for p in pairs])
        t, pval = stats.ttest_rel(b, a)
        ax.plot([0, 1], [a.mean(), b.mean()], color=INK, linewidth=2.5, label="mean")
        ax.set_title(f"Test success, paired by seed: +{100 * (b - a).mean():.0f} points\n"
                     f"{int((b > a).sum())}/{len(pairs)} seeds better · paired t-test p = {pval:.2f}", fontsize=9)
    r = d.row("pid", "navigation")
    if r is not None:
        baseline_line(ax, r["test/success_rate"], "pid")
    ax.set_xticks([0, 1], ["from scratch", "from obstacles"])
    ax.set_xlim(-0.3, 1.3)
    percent(ax)
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    for group, color, label in (("ppo_navigation_normal", ALGO_LIGHT["ppo"], "from scratch"),
                                ("ppo_navigation_from_obstacles", ALGO["ppo"], "from the obstacles agent")):
        c = d.curve(group)
        if c:
            band(ax, c[0], c[1], color, f"{label} ({c[1].shape[0]} seeds)")
    ax.set_title("Validation success during training on navigation")
    ax.set_xlabel("environment steps")
    steps_axis(ax)
    percent(ax)
    ax.legend(loc="upper left")
    fig.suptitle("E5 · Transfer: navigation learned from scratch vs starting from the obstacles agent",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


FIGURES = {
    "e1_learning_curves": e1_learning_curves, "e1_test": e1_test, "e1b_behaviour": e1b_behaviour,
    "e2_heatmaps": e2_heatmaps, "e2_robust_area": e2_robust_area, "e2_mass": e2_mass, "e2_failure": e2_failure,
    "e2_late_degradation": e2_late_degradation, "e3_finetuning": e3_finetuning, "e4_ablations": e4_ablations,
    "e5_transfer": e5_transfer,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=sorted(FIGURES), help="make only these figures")
    ap.add_argument("--eval", type=Path, default=EVAL)
    ap.add_argument("--curves", type=Path, default=ROOT / "results" / "curves.csv")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--wandb", action="store_true", help="also upload the figures to W&B (run 'figures')")
    ap.add_argument("--project", default="hummingbird-rl")
    args = ap.parse_args()
    style()
    d = Data(args.eval, args.curves)
    paths = {}
    for name in args.only or FIGURES:
        paths[name] = save(FIGURES[name](d), args.out, name)
        print(f"wrote {paths[name]}")
    if args.wandb:
        import wandb

        run = wandb.init(project=args.project, name="figures", job_type="analysis", group="figures")
        run.log({name: wandb.Image(str(p)) for name, p in paths.items()})
        run.log({"all_results": wandb.Table(dataframe=d.all)})
        run.finish()


if __name__ == "__main__":
    main()
