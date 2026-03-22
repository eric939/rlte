#!/usr/bin/env python3
"""Run the interventional causal estimator across a grid of simulator settings."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-envs", nargs="+", default=["noise", "flow", "strategic"])
    parser.add_argument("--volumes", nargs="+", type=int, default=[20])
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--deltas", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--terminal-time", type=int, default=150)
    parser.add_argument("--time-delta", type=int, default=15)
    parser.add_argument("--delta-units", choices=["normalized", "lots"], default="normalized")
    parser.add_argument("--action-index", type=int, default=0)
    parser.add_argument("--slack-index", type=int, default=-1)
    parser.add_argument("--adaptive-intervention", action="store_true")
    parser.add_argument("--policy-kind", choices=["inactive", "fixed", "heuristic_sell"], default="fixed")
    parser.add_argument("--fixed-action", type=str, default="0.2,0.1,0.1,0.1,0.1,0.1,0.3")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--output-root", type=str, default="outputs/causal_grid")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).resolve().parent / "run_interventional_ground_truth.py"
    commands: list[list[str]] = []
    for market_env in args.market_envs:
        for volume in args.volumes:
            for horizon in args.horizons:
                for delta in args.deltas:
                    delta_tag = f"{delta:g}".replace(".", "p")
                    output_dir = root / f"{market_env}_v{volume}_h{horizon}_d{delta_tag}_{args.policy_kind}"
                    cmd = [
                        args.python_bin,
                        str(script_path),
                        "--market-env", market_env,
                        "--volume", str(volume),
                        "--terminal-time", str(args.terminal_time),
                        "--time-delta", str(args.time_delta),
                        "--seed-start", str(args.seed_start),
                        "--num-episodes", str(args.num_episodes),
                        "--intervention-mode", "random",
                        "--delta", str(delta),
                        "--delta-units", args.delta_units,
                        "--horizon", str(horizon),
                        "--action-index", str(args.action_index),
                        "--slack-index", str(args.slack_index),
                        "--policy-kind", args.policy_kind,
                        "--output-dir", str(output_dir),
                    ]
                    if args.adaptive_intervention:
                        cmd.append("--adaptive-intervention")
                    if args.policy_kind == "fixed":
                        cmd.extend(["--fixed-action", args.fixed_action])
                    commands.append(cmd)

    for cmd in commands:
        print("")
        print("running:")
        print(" ".join(shlex.quote(token) for token in cmd))
        subprocess.run(cmd, check=True)

    print("")
    print(f"completed {len(commands)} runs under {root}")


if __name__ == "__main__":
    main()
