"""Write the files the web viewer reads: recorded episodes, policies, and the golden test files.

    uv run python scripts/export_web.py golden
    uv run python scripts/export_web.py episodes --policy pid --tasks hover goto obstacles navigation
    uv run python scripts/export_web.py episodes --policy ppo_hover_normal_s1 --suite failure --episodes 5
    uv run python scripts/export_web.py policy --policy ppo_hover_normal_s1

golden    web/tests/golden/*.json, for the parity tests of the TypeScript simulator,
          networks, tasks and controllers, and web/src/control/lqr_gains.json, the
          LQR gain the browser uses (to regenerate only if the physics, the tasks or
          the controllers change)
episodes  web/public/episodes/<policy>/<task>/<suite>_<seed>.json + index.json; the
          episodes are the first ones of the evaluation suite (same seeds, same
          conditions), so the viewer shows exactly what was measured
policy    web/public/policies/<policy>.json + index.json, the agent's network for
          the live mode

--policy works as in evaluate.py: "pid", "lqr", a run name or a model .zip.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate import load_policy  # noqa: E402  (scripts/evaluate.py, next to this file)

from hummingbird_rl.envs.hummingbird_env import HummingbirdEnv  # noqa: E402
from hummingbird_rl.evaluation.suites import SEEDS  # noqa: E402
from hummingbird_rl.export.trajectories import build_index, record_episode, save_json  # noqa: E402
from hummingbird_rl.export.web_export import (  # noqa: E402
    build_policy_index,
    golden_dynamics,
    golden_episodes,
    golden_observations,
    golden_policies,
    lqr_dict,
    policy_to_dict,
)

WEB = ROOT / "web"
REGIMES = {"test": "normal", "failure": "failure"}    # the conditions of each evaluation suite


def export_golden() -> None:
    out = WEB / "tests" / "golden"
    for name, make in [("dynamics", golden_dynamics), ("observations", golden_observations),
                       ("policies", golden_policies), ("episodes", golden_episodes)]:
        data = make()
        save_json(data, out / f"{name}.json")
        print(f"wrote {out / name}.json ({len(data['cases'])} cases)")
    path = WEB / "src" / "control" / "lqr_gains.json"
    save_json(lqr_dict(), path)
    print(f"wrote {path}")


def export_episodes(args) -> None:
    policy, meta = load_policy(args.policy, args.which)
    tasks = args.tasks or ([meta["task"]] if meta["task"] else None)
    if not tasks:
        sys.exit("--tasks is required for pid and lqr")
    folder = WEB / "public" / "episodes"
    for task in tasks:
        env = HummingbirdEnv(task, REGIMES[args.suite], **meta["env"])
        for seed in range(SEEDS[args.suite], SEEDS[args.suite] + args.episodes):
            ep = record_episode(env, policy, seed)
            ep.update(policy=meta["name"], suite=args.suite)
            save_json(ep, folder / meta["name"] / task / f"{args.suite}_{seed}.json")
            r = ep["result"]
            print(f"[{meta['name']} | {task} | {args.suite} {seed}] {r['reason']}, {r['length']} steps")
    save_json({"episodes": build_index(folder)}, folder / "index.json")


def export_policy(args) -> None:
    policy, meta = load_policy(args.policy, args.which)
    if args.policy in ("pid", "lqr"):
        sys.exit("pid and lqr are not networks; the viewer has its own copy of them")
    folder = WEB / "public" / "policies"
    path = folder / f"{meta['name']}.json"
    save_json({**policy_to_dict(policy, meta["name"]), "task": meta["task"], "which": args.which}, path)
    save_json({"policies": build_policy_index(folder)}, folder / "index.json")
    print(f"wrote {path} ({path.stat().st_size / 1e3:.0f} kB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    sub.add_parser("golden")
    ep = sub.add_parser("episodes")
    ep.add_argument("--policy", required=True)
    ep.add_argument("--which", choices=["final", "best"], default="final")
    ep.add_argument("--tasks", nargs="+", default=None)
    ep.add_argument("--suite", choices=sorted(REGIMES), default="test")
    ep.add_argument("--episodes", type=int, default=3)
    pol = sub.add_parser("policy")
    pol.add_argument("--policy", required=True)
    pol.add_argument("--which", choices=["final", "best"], default="final")
    args = ap.parse_args()

    if args.what == "golden":
        export_golden()
    elif args.what == "episodes":
        export_episodes(args)
    else:
        export_policy(args)


if __name__ == "__main__":
    main()
