"""Extract the wing-kinematics curves of the Dryad dataset into data/vanderbilt/.

    uv run python scripts/extract_dryad.py <folder with the .fig files> [output folder]

The data of Song, Tobalske, Powers, Hedrick & Luo (2016, R. Soc. Open Sci.
3:160230; Dryad doi:10.5061/dryad.8ch1b) are MATLAB figures (.fig, which are
MAT-files version 5 holding the whole graphics tree), not tables. This script
walks each figure, collects every plotted line, and writes one CSV per figure
with two columns per curve (curve1_x, curve1_y, curve2_x, ...); shorter curves
leave blank cells. The Dryad archive contains Data.tar: unpack it first
(`tar xf Data.tar`), then point this script at the folder of .fig files.

The CSVs in data/vanderbilt were extracted this way in the earlier project.
The viewer uses four of these curves (chord angle at the tip and mid-span,
twist at the tip, and the stroke-plane angle), resampled to 64 points per
wingbeat in web/src/view/wingData.json.
"""

import argparse
import csv
from itertools import zip_longest
from pathlib import Path

import numpy as np
import scipy.io

ROOT = Path(__file__).resolve().parents[1]

# What each figure shows, to make the printed summary readable.
DESCRIPTIONS = {
    "tip_trajectory": "path of the wingtip over a beat",
    "tip_velocity": "wingtip speed vs phase (7 span sections)",
    "aoa_phase_d": "angle of attack vs phase, downstroke (deg)",
    "aoa_phase_p": "angle of attack vs phase, upstroke (deg)",
    "chord_angle_phase_d": "chord angle vs phase, downstroke (deg)",
    "chord_angle_phase_p": "chord angle vs phase, upstroke (deg)",
    "wing_twist": "wing twist vs phase (deg)",
    "stroke_plane_angle_animal": "stroke-plane angle, this bird vs other species",
    "C_z_phase": "vertical force coefficient vs phase",
    "C_z_phase_body": "vertical force coefficient of the body vs phase",
    "C_d_phase_body": "drag coefficient of the body vs phase",
    "C_p_phase": "power coefficient vs phase",
    "C_t_phase": "thrust coefficient vs phase",
    "mesh_convergence_wing_f": "CFD mesh convergence (not kinematics)",
}


def _prop(obj, name: str):
    """A property of a MATLAB graphics object, or None."""
    props = getattr(obj, "properties", None)
    return getattr(props, name, None) if props is not None else None


def curves(figure) -> list[tuple[np.ndarray, np.ndarray]]:
    """Every plotted line (x, y) of a figure, in the order MATLAB stores them."""
    found = []

    def walk(obj) -> None:
        kind = getattr(obj, "type", "")
        if kind == "line" or "lineseries" in kind:
            x, y = _prop(obj, "XData"), _prop(obj, "YData")
            if x is not None and y is not None:
                found.append((np.atleast_1d(x).astype(float), np.atleast_1d(y).astype(float)))
        for child in np.atleast_1d(getattr(obj, "children", [])):
            if hasattr(child, "type"):
                walk(child)

    walk(figure)
    return found


def load_figure(path: Path):
    data = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    return data[next(k for k in data if not k.startswith("__"))]   # the one non-metadata entry


def write_csv(path: Path, lines: list[tuple[np.ndarray, np.ndarray]]) -> None:
    header = [f"curve{i + 1}_{axis}" for i in range(len(lines)) for axis in "xy"]
    columns = [[f"{v:.8f}" for v in values] for x, y in lines for values in (x, y)]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(zip_longest(*columns, fillvalue=""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figures", type=Path, help="folder holding the .fig files of the Dryad archive")
    ap.add_argument("out", type=Path, nargs="?", default=ROOT / "data" / "vanderbilt")
    args = ap.parse_args()

    paths = sorted(args.figures.glob("*.fig"))
    if not paths:
        raise SystemExit(f"no .fig file in {args.figures} (did you unpack Data.tar?)")
    args.out.mkdir(parents=True, exist_ok=True)
    for path in paths:
        lines = curves(load_figure(path))
        write_csv(args.out / f"{path.stem}.csv", lines)
        print(f"{path.stem:28s} {len(lines):3d} curves  {DESCRIPTIONS.get(path.stem, '')}")
    print(f"wrote {len(paths)} files to {args.out}")


if __name__ == "__main__":
    main()
