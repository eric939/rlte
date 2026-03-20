#!/usr/bin/env python3
"""Run paired interventional ground-truth experiments on the existing simulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from causal.counterfactual_runner import run_paired_intervention
from causal.policy import InactivePolicy, TorchPolicyAdapter
from simulation.market_gym import Market


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-env", choices=["noise", "flow", "strategic"], default="noise")
    parser.add_argument("--volume", type=int, default=20)
    parser.add_argument("--terminal-time", type=int, default=150)
    parser.add_argument("--time-delta", type=int, default=15)
    parser.add_argument("--drop-feature", choices=["volume", "order_info", "drift", "none"], default="none")

    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated list of seeds.")
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--num-episodes", type=int, default=1)

    parser.add_argument("--intervention-mode", choices=["fixed", "random"], default="fixed")
    parser.add_argument("--intervention-time", type=int, default=0)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--delta-units", choices=["normalized", "lots"], default="normalized")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--action-index", type=int, default=0)
    parser.add_argument("--slack-index", type=int, default=-1)

    parser.add_argument(
        "--policy-kind",
        choices=["inactive", "logistic_normal", "logistic_normal_learn_std", "dirichlet"],
        default="inactive",
    )
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--output-dir", type=str, default="outputs/interventional_ground_truth")
    return parser.parse_args()


def build_base_config(args: argparse.Namespace) -> dict:
    drop_feature = None if args.drop_feature == "none" else args.drop_feature
    return {
        "market_env": args.market_env,
        "execution_agent": "rl_agent",
        "volume": args.volume,
        "seed": args.seed_start,
        "terminal_time": args.terminal_time,
        "time_delta": args.time_delta,
        "drop_feature": drop_feature,
    }


def parse_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return [int(token.strip()) for token in args.seeds.split(",") if token.strip()]
    return list(range(args.seed_start, args.seed_start + args.num_episodes))


def build_policy(args: argparse.Namespace, base_config: dict):
    if args.policy_kind == "inactive":
        probe_env = Market(base_config)
        return InactivePolicy(action_size=probe_env.action_space.shape[0])
    if args.model_path is None:
        raise ValueError("--model-path is required for non-inactive policies")
    return TorchPolicyAdapter.from_model_path(
        base_config=base_config,
        model_path=args.model_path,
        policy_kind=args.policy_kind,
        device=args.device,
        deterministic=True,
    )


def main() -> None:
    args = parse_args()
    base_config = build_base_config(args)
    seeds = parse_seeds(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy = build_policy(args, base_config)

    summary_rows = []
    detail_frames = []
    rng = np.random.default_rng(args.seed_start)

    for seed in seeds:
        selected_t0 = None if args.intervention_mode == "random" else args.intervention_time
        result = run_paired_intervention(
            base_config=base_config,
            seed=seed,
            intervention_time=selected_t0,
            delta=args.delta,
            horizon=args.horizon,
            policy=policy,
            delta_units=args.delta_units,
            action_index=args.action_index,
            slack_index=args.slack_index,
            rng=rng,
        )
        summary_row = dict(result.summary)
        summary_row["warnings"] = ";".join(result.warnings)
        summary_rows.append(summary_row)

        for label, frame in (
            ("baseline", result.baseline_log),
            ("plus", result.plus_log),
            ("minus", result.minus_log),
        ):
            detail = frame.copy()
            detail["pair_seed"] = seed
            detail["pair_t0"] = result.intervention_time
            detail["pair_label"] = label
            detail_frames.append(detail)

    summary_df = pd.DataFrame(summary_rows)
    if detail_frames:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            details_df = pd.concat(detail_frames, ignore_index=True)
    else:
        details_df = pd.DataFrame()

    summary_path = output_dir / "summary.csv"
    details_path = output_dir / "trajectory_logs.csv"
    config_path = output_dir / "config_snapshot.json"
    summary_json_path = output_dir / "summary.json"

    summary_df.to_csv(summary_path, index=False)
    details_df.to_csv(details_path, index=False)
    summary_df.to_json(summary_json_path, orient="records", indent=2)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "base_config": base_config,
                "policy_kind": args.policy_kind,
                "model_path": args.model_path,
                "seeds": seeds,
                "intervention_mode": args.intervention_mode,
                "intervention_time": args.intervention_time,
                "delta": args.delta,
                "delta_units": args.delta_units,
                "horizon": args.horizon,
                "action_index": args.action_index,
                "slack_index": args.slack_index,
                "device": args.device,
            },
            handle,
            indent=2,
        )

    if summary_df.empty:
        print("No runs were executed.")
        return

    print(f"saved summary to {summary_path}")
    print(f"saved detailed trajectory logs to {details_path}")
    print(f"saved config snapshot to {config_path}")
    print("")
    print(summary_df[["seed", "t0", "delta", "X_plus", "X_minus", "DeltaP_plus", "DeltaP_minus", "beta_true_hat"]].to_string(index=False))


if __name__ == "__main__":
    main()
