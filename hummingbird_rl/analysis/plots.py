"""The look shared by every figure of the reports, and the helpers they use.

Colours follow what they mean, the same in every figure:
  - each algorithm has one hue (PPO blue, SAC orange, TD3 aqua), checked to stay
    distinct for colour-blind readers; a lighter step of the same hue is the agent
    trained in normal conditions, the full one the agent trained in randomised
    conditions (or the variant under study);
  - the classical controllers are references, not competitors among the series:
    neutral ink, PID solid and LQR dashed, always labelled where they are drawn;
  - magnitudes on a grid (success rates) use one blue ramp, light to dark.
Identity is never carried by colour alone: every figure with several series has a
legend or direct labels.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"  # secondary text, and the PID
INK_3 = "#8c8a83"  # muted: the LQR, grid labels
GRID = "#e6e5e0"

ALGO = {"ppo": "#2a78d6", "sac": "#eb6834", "td3": "#1baf7a"}
ALGO_LIGHT = {"ppo": "#86b6ef", "sac": "#f4a47f", "td3": "#7fd3b4"}  # the same hues, lighter
ALGO_NAME = {"ppo": "PPO", "sac": "SAC", "td3": "TD3"}
BASELINE = {"pid": dict(color=INK_2, linestyle="-"), "lqr": dict(color=INK_3, linestyle="--")}
TASK_NAME = {"hover": "Hover & recover", "goto": "Go-to-point", "obstacles": "Obstacles", "navigation": "Navigation"}

# Sequential blue ramp (steps 100 to 700), for success rates from 0 to 1.
BLUES = LinearSegmentedColormap.from_list(
    "hb_blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)


def font() -> str:
    """The first of our fonts this computer has (Matplotlib always has DejaVu Sans). Asking for a
    missing one would make Matplotlib search for it, and warn, at every piece of text."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in ("Inter", "Segoe UI", "Helvetica", "Arial") if f in available), "DejaVu Sans")


def style() -> None:
    """Thin marks, recessive axes and grid, one font."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": font(), "font.size": 9.5,
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "lines.linewidth": 2.0, "legend.frameon": False, "legend.fontsize": 8.5,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def save(fig: plt.Figure, folder: Path, name: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def baseline_line(ax: plt.Axes, y: float, which: str, label: bool = True) -> None:
    """A horizontal reference line for the PID or the LQR, labelled at its right end."""
    ax.axhline(y, linewidth=1.4, zorder=1, **BASELINE[which])
    if label:
        ax.annotate(which.upper(), (1.0, y), xycoords=("axes fraction", "data"), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=8, color=BASELINE[which]["color"])


def band(ax: plt.Axes, x: np.ndarray, ys: np.ndarray, color: str, label: str) -> None:
    """The mean over seeds as a line, and the range of the seeds as a light band."""
    ax.fill_between(x, ys.min(axis=0), ys.max(axis=0), color=color, alpha=0.18, linewidth=0)
    ax.plot(x, ys.mean(axis=0), color=color, label=label)


def percent(ax: plt.Axes, axis: str = "y") -> None:
    fmt = mpl.ticker.PercentFormatter(1.0, decimals=0)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def steps_axis(ax: plt.Axes) -> None:
    """Environment steps in thousands or millions."""
    def fmt(v, _):
        return f"{v / 1e6:g}M" if v >= 1e6 else f"{v / 1e3:g}k"

    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
