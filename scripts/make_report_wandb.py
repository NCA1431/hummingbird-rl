"""Build the W&B report: the text of docs/wandb_report.md, with the charts drawn by W&B itself.

    uv run python scripts/make_report_wandb.py --dry-run     # check everything, send nothing
    uv run python scripts/make_report_wandb.py               # create the report, print its link
    uv run python scripts/make_report_wandb.py --url <link>  # rebuild that report in place
    uv run python scripts/make_report_wandb.py --show <link> # list the custom charts of a report

The text lives in docs/wandb_report.md so that it can be edited and versioned like the code. There,
"#" lines become headings, a paragraph between two `$$` lines a LaTeX block (W&B's Markdown blocks
do not render equations), the other paragraphs Markdown blocks, `<!-- toc -->` the table of contents,
and each `<!-- panels: name -->` the charts defined below under that name. The charts read the runs
logged by scripts/log_results_wandb.py: one run per controller and task in the group "results"
(scores, sweeps, learning curves, tagged E1 ... E5) and one run "results tables" holding the tables
and the figures W&B cannot draw (E6). The E1b episodes and the E2 heatmaps are custom charts: our
Vega-Lite specs of docs/wandb_charts/, saved in W&B as hummingbird-e1b-scatter and
hummingbird-e2-heatmap. W&B's chart list shows the names, but a report needs the id W&B stored, which
for the heatmap begins with an underscore: --show prints the ids of a report's custom charts.
Run log_results_wandb.py first, with --replace, so that the report sees the latest results.

Each chart groups the runs by config.controller ("PPO normal", "SAC randomised (best)", "PID"...), which
gives the mean over seeds and the range of the seeds as a band, like the PNG figures, and the colours
follow the figures too: one hue per algorithm, PID and LQR in greys. Text and charts can still be edited
in W&B afterwards; rebuilding with --url replaces them with the file's version.

Uses the W&B login of this computer (uv run wandb login): no key appears here.
"""

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "wandb_report.md"

# Colours of the figures: one hue per algorithm (full, darker for randomised training, lighter for
# the other variants), the classical controllers in greys.
BASE = {"PPO": "#2a78d6", "SAC": "#eb6834", "TD3": "#1baf7a"}
DARK = {"PPO": "#184f95", "SAC": "#b4461a", "TD3": "#0f7a52"}
LIGHT = {"PPO": "#86b6ef", "SAC": "#f4a47f", "TD3": "#7fd3b4"}
REFERENCE = {"PID": "#52514e", "LQR": "#8c8a83"}
LABELS = ["PID", "LQR"] + [f"{a} {v}{b}" for a in BASE for b in ("", " (best)") for v in (
    "normal", "randomised", "fine-tuned +30%", "fine-tuned +100%", "no clipping", "no advantage normalisation",
    "no reward normalisation", "neither normalisation", "GAE λ = 0", "GAE λ = 1", "no hover start",
    "fixed α = 0.2", "annealed learning rate", "failure penalty −50", "from obstacles")]


def colour(label: str) -> str:
    if label in REFERENCE:
        return REFERENCE[label]
    algo, _, variant = label.partition(" ")
    variant = variant.removesuffix(" (best)")
    return BASE[algo] if variant == "normal" else DARK[algo] if variant == "randomised" else LIGHT[algo]


# --- The charts ------------------------------------------------------------------------------

def charts(wr, ex, entity: str, project: str) -> dict:
    """Every `panels:` name of the text -> its panel grids."""
    def runset(name, *filters, group="results", by_label=True):
        return wr.Runset(entity=entity, project=project, name=name, groupby=["config.controller"] if by_label else [],
                         filters=[ex.Metric("group") == group, *filters])

    def grid(rs, *panels, hide=False):
        colours = {} if not rs.groupby else {
            wr.RunsetGroup(runset_name=rs.name, keys=(wr.RunsetGroupKey(key=wr.Config("controller"), value=lab),)):
                colour(lab) for lab in LABELS}
        return wr.PanelGrid(runsets=[rs], panels=list(panels), custom_run_colors=colours, hide_run_sets=hide)

    def line(title, x, y, x_title, y_title, w=16, h=8, grouped=True):
        return wr.LinePlot(title=title, x=x, y=[y], title_x=x_title, title_y=y_title, smoothing_type="none",
                           groupby="config.controller" if grouped else None,
                           groupby_aggfunc="mean" if grouped else None,
                           groupby_rangefunc="minmax" if grouped else None, layout=wr.Layout(w=w, h=h))

    def bar(title, metric, w=8, h=8):
        return wr.BarPlot(title=title, metrics=[metric], groupby="config.controller", groupby_aggfunc="mean",
                          groupby_rangefunc="minmax", layout=wr.Layout(w=w, h=h))

    def images(name, *keys, h=14):
        shown = [wr.MediaBrowser(title=k.split("/")[-1], media_keys=[k], num_columns=1, layout=wr.Layout(w=24, h=h))
                 for k in keys]
        return grid(runset(name, group="results_tables", by_label=False), *shown, hide=True)

    def custom(title, table, preset, fields, w=24, h=12):
        """A custom chart: a W&B table drawn by a Vega-Lite spec saved in W&B under the name `preset`.
        Ours are versioned in docs/wandb_charts/ and saved once by hand in W&B's chart editor (Save as,
        publicly accessible), since the API can use a saved chart but not create one. They read the
        table's columns by name, so they need no field mapping."""
        return wr.CustomChart(query={"summaryTable": {"tableKey": table}}, chart_name=preset, chart_fields=fields,
                              chart_strings={"title": title}, layout=wr.Layout(w=w, h=h))

    task = lambda t: ex.Config("task") == t          # noqa: E731
    tag = lambda e: ex.Tags().isin([e])              # noqa: E731
    learning = lambda title: line(title, "val/env_steps", "val/success_rate", "environment steps",  # noqa: E731
                                  "validation success")
    names = {"hover": "Hover", "goto": "Go-to-point", "obstacles": "Obstacles", "navigation": "Navigation"}

    out = {}
    for t, name in names.items():
        out[f"e1_{t}"] = [grid(runset(f"E1 {name}", task(t), tag("E1")),
                               learning(f"{name}: validation success during training (PID and LQR: test score)"),
                               bar(f"{name}: test success (200 episodes)", "test/success_rate"))]
    out["e1b"] = [grid(runset("E1b go-to-point", task("goto"), tag("E1b")),
                       bar("Target behind: episodes where the bird turned", "behaviour/p_turn_target_behind"),
                       bar("Target behind: share of the transit flown head-first", "behaviour/forward_target_behind"),
                       bar("Target ahead: episodes where the bird turned", "behaviour/p_turn_target_ahead")),
                  grid(runset("E1b episodes", task("goto"), tag("E1b"), by_label=False),
                       custom("Go-to-point: net turn vs target bearing, one dot per test episode",
                              "behaviour/turn_vs_bearing_table", f"{entity}/hummingbird-e1b-scatter", {}, h=13))]
    out["e2_area"] = [grid(runset(f"E2 {names[t]}", task(t), tag("E2")),
                           bar(f"{names[t]}: robust area (whole grid)", "grid/robust_area", w=12),
                           bar(f"{names[t]}: robust area, winds within the training range",
                               "grid/robust_area_trained_range", w=12)) for t in ("hover", "goto")]
    out["e2_heatmaps"] = [grid(runset("E2 grid", group="results_tables", by_label=False),
                               custom("Success over wind x wing damage (mean over seeds)", "grid_mean",
                                      f"{entity}/_hummingbird-e2-heatmap", {}, h=20), hide=True)]
    out["e2_sweeps"] = [grid(runset(f"E2 sweeps {names[t]}", task(t), tag("E2")),
                             line(f"{names[t]}: success vs wind (mean over damage levels)", "wind/wind_pct",
                                  "wind/success", "wind (% of the weight)", "success", w=8),
                             line(f"{names[t]}: success vs wing damage (mean over winds)", "damage/damage_pct",
                                  "damage/success", "left wing force lost (%)", "success", w=8),
                             line(f"{names[t]}: success vs mass", "mass/mass_factor", "mass/success", "mass (x normal)",
                                  "success", w=8)) for t in ("hover", "goto")]
    out["e2_failure"] = [grid(runset("E2 sudden failure", task("hover"), tag("E2")),
                              bar("Largest deviation after the failure (median, cm)",
                                  "failure/median_max_deviation_cm"),
                              bar("Flights still under control at the end", "failure/survival_rate"),
                              bar("Time to settle back within 3 cm (median, s)", "failure/median_recovery_s"))]
    offpolicy = runset("Off-policy agents on hover, every seed", task("hover"), ex.Config("algo").isin(["sac", "td3"]),
                       ex.Config("checkpoint") == "last", by_label=False)
    out["e2_offpolicy"] = [grid(offpolicy,
                                line("SAC and TD3 on hover: validation success of every seed", "val/env_steps",
                                     "val/success_rate", "environment steps", "validation success", grouped=False),
                                wr.ScatterPlot(title="Test success: final model (x) vs best checkpoint (y)",
                                               x="test/success_rate", y="best_checkpoint/test_success_rate",
                                               layout=wr.Layout(w=8, h=8)))]
    out["e3"] = [grid(runset(f"E3 {names[t]}", task(t), tag("E3"), ex.Config("algo") == "ppo"),
                      bar(f"{names[t]}: robust area", "grid/robust_area", w=12),
                      bar(f"{names[t]}: test success in normal conditions", "test/success_rate", w=12))
                 for t in ("hover", "goto")]
    out["e4"] = [grid(runset("E4 hover", task("hover"), tag("E4")),
                      bar("Hover: test success", "test/success_rate", w=12, h=10),
                      bar("Hover: steps until validation success first reaches 90%", "train/steps_to_90pct",
                          w=12, h=10),
                      line("Hover: validation success during training", "val/env_steps", "val/success_rate",
                           "environment steps", "validation success", w=24)),
                 grid(runset("E4 navigation", task("navigation"), tag("E4")),
                      bar("Navigation: test success, failure penalty -10 vs -50", "test/success_rate", w=12),
                      bar("Navigation: episodes ending in a collision", "test/end_collision", w=12))]
    out["e5"] = [grid(runset("E5 navigation", task("navigation"), tag("E5"), ex.Config("algo") == "ppo"),
                      learning("Navigation: from scratch vs from the Obstacles agent"),
                      bar("Navigation: test success", "test/success_rate"))]
    out["e6"] = [images("E6 figures", "figures/e6_value_landscape", "figures/e6_vector_field", h=10)]

    return out


def report_link(url: str) -> str:
    """The link in the form the W&B library reads: .../entity/project/reports/title--id. Only the id
    at the end is used. Links on W&B's newer address (forge.coreweave.com/wandb/...) have one more
    part in front, which the library cannot split, so we keep the last four parts only."""
    link = urlparse(url)
    parts = link.path.strip("/").split("/")
    if parts and parts[-1] == "edit":
        # the link of a report open in W&B's editor: .../title--id/edit?draftId=<draft id>. The draft is
        # what the editor shows (not yet published), so we read that one.
        parts = parts[:-1]
        draft = parse_qs(link.query).get("draftId")
        if draft and "--" in parts[-1]:
            parts[-1] = parts[-1].rsplit("--", 1)[0] + "--" + draft[0].rstrip("=")
    if len(parts) < 4 or parts[-2] != "reports" or "--" not in parts[-1]:
        sys.exit(f"this does not look like the link of a W&B report: {url}")
    return "https://wandb.ai/" + "/".join(parts[-4:])


def show_custom_charts(report, wr) -> None:
    """Print the custom charts of a report as W&B stores them: the name of the chart definition
    (a preset saved in W&B, or one of its own such as wandb/scatter/v0), the table it reads and the
    field mapping. This is how we learn the name W&B gave a preset saved in its chart editor."""
    found = 0
    for block in report.blocks:
        for panel in getattr(block, "panels", []) if isinstance(block, wr.PanelGrid) else []:
            if isinstance(panel, wr.CustomChart):
                found += 1
                table = panel.query.get("summaryTable", {}).get("tableKey", panel.query)
                print(f"chart {found}: definition {panel.chart_name!r}, table {table!r}, fields {panel.chart_fields}")
    if not found:
        print("no custom chart in this report")


# --- The text --------------------------------------------------------------------------------

def blocks_from(text: str, wr, panels: dict) -> tuple[str, str, list]:
    """(title, description, blocks) from the Markdown source."""
    meta = dict(re.findall(r"<!--\s*(title|description):\s*(.*?)\s*-->", text))
    text = re.sub(r"<!--\s*(title|description):.*?-->", "", text)
    text = re.sub(r"<!--(?!\s*(?:panels:|toc))[\s\S]*?-->", "", text)       # explanatory comments
    blocks, buffer = [], []

    def flush():
        chunk = "\n".join(buffer).strip()
        if chunk:
            blocks.append(wr.MarkdownBlock(text=chunk))
        buffer.clear()

    lines = iter(text.splitlines())
    for line in lines:
        heading = re.match(r"^(#{1,3}) (.*)", line)
        marker = re.match(r"^<!--\s*(panels:\s*(\S+)|toc)\s*-->$", line.strip())
        if heading:
            flush()
            blocks.append({1: wr.H1, 2: wr.H2, 3: wr.H3}[len(heading.group(1))](text=heading.group(2).strip()))
        elif line.strip().startswith("$$"):
            # An equation on its own: `$$ ... $$` on one line, or `$$`, the lines, `$$`.
            flush()
            body = line.strip()[2:]
            while not body.rstrip().endswith("$$"):
                body += "\n" + next(lines)
            blocks.append(wr.LatexBlock(text=body.rstrip()[:-2].strip()))
        elif marker:
            flush()
            if marker.group(1) == "toc":
                blocks.append(wr.TableOfContents())
            else:
                name = marker.group(2)
                if name not in panels:
                    sys.exit(f"the text asks for charts {name!r}, which are not defined in this script")
                blocks.extend(panels[name])
        else:
            buffer.append(line)
    flush()
    return meta.get("title", "Hummingbird RL"), meta.get("description", ""), blocks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--project", default="hummingbird-rl")
    ap.add_argument("--entity", default=None, help="W&B user or team (default: the one you are logged in as)")
    ap.add_argument("--url", default=None, help="an existing report to rebuild in place, instead of a new one")
    ap.add_argument("--dry-run", action="store_true", help="build and check the report without W&B")
    ap.add_argument("--show", default=None, metavar="LINK", help="only list the custom charts of that report")
    args = ap.parse_args()

    import wandb_workspaces.expr as ex
    import wandb_workspaces.reports.v2 as wr

    if args.show:
        show_custom_charts(wr.Report.from_url(report_link(args.show)), wr)
        return

    if args.dry_run:
        # Converting run sets asks W&B for the project's id: answer with a fake one.
        import wandb_workspaces.reports.v2.interface as itf
        itf._get_api = lambda: None
        itf.execute_graphql = lambda api, query, variables: {"project": {"internalId": "UHJvamVjdDox"}}
        entity = args.entity or "me"
    else:
        import wandb
        entity = args.entity or wandb.Api().default_entity

    title, description, blocks = blocks_from(args.source.read_text(encoding="utf-8"), wr,
                                             charts(wr, ex, entity, args.project))
    if args.url:
        report = wr.Report.from_url(report_link(args.url))
        report.title, report.description, report.blocks = title, description, blocks
    else:
        report = wr.Report(project=args.project, entity=entity, title=title, description=description,
                           blocks=blocks, width="fixed")
    n_grids = sum(isinstance(b, wr.PanelGrid) for b in blocks)
    n_panels = sum(len(b.panels) for b in blocks if isinstance(b, wr.PanelGrid))
    print(f"{title}\n{len(blocks)} blocks, {n_grids} chart grids, {n_panels} charts")
    if args.dry_run:
        report._to_model()          # the same validation as a save, without sending anything
        print("dry run: the report is valid; nothing was sent")
        return
    report.save()
    print(f"saved: {report.url}")


if __name__ == "__main__":
    main()
