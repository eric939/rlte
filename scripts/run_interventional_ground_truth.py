#!/usr/bin/env python3
"""Run paired interventional ground-truth experiments on the existing simulator."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from causal.counterfactual_runner import run_paired_intervention
from causal.policy import FixedActionPolicy, HeuristicSellPolicy, InactivePolicy, TorchPolicyAdapter
from simulation.market_gym import Market


class ProgressReporter:
    """Terminal-first progress reporter with a plain-text fallback."""

    def __init__(self, total: int) -> None:
        self.total = int(total)
        self.use_tqdm = tqdm is not None
        self.bar = None
        if self.use_tqdm:
            self.bar = tqdm(
                total=self.total,
                desc="Causal runs",
                unit="seed",
                dynamic_ncols=True,
                smoothing=0.05,
                mininterval=0.2,
                leave=True,
                file=sys.stdout,
            )

    def update(self, n: int = 1) -> None:
        if self.bar is not None:
            self.bar.update(n)

    def write(self, text: str) -> None:
        if self.bar is not None:
            self.bar.write(text)
        else:
            print(text)

    def set_postfix(self, **kwargs) -> None:
        if self.bar is not None:
            self.bar.set_postfix(**kwargs, refresh=True)
        else:
            filtered = ", ".join(f"{key}={value}" for key, value in kwargs.items())
            print(f"[progress] {filtered}")

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()


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
    parser.add_argument(
        "--intervention-target",
        choices=["component", "market"],
        default="market",
        help="Intervene on a generic action component or specifically on submitted market-sell lots.",
    )
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument(
        "--min-abs-treatment-diff",
        type=float,
        default=0.0,
        help="Minimum absolute realized treatment difference |X_plus - X_minus| required for primary estimation.",
    )
    parser.add_argument("--action-index", type=int, default=0)
    parser.add_argument("--slack-index", type=int, default=-1)
    parser.add_argument(
        "--adaptive-intervention",
        action="store_true",
        help="Choose a feasible action/slack pair from the baseline action at the intervention time.",
    )
    parser.add_argument(
        "--replay-from-seed",
        action="store_true",
        help="Disable branching from the exact t0 simulator state and rerun plus/minus from seed instead.",
    )

    parser.add_argument(
        "--policy-kind",
        choices=["inactive", "fixed", "heuristic_sell", "logistic_normal", "logistic_normal_learn_std", "dirichlet"],
        default="inactive",
    )
    parser.add_argument(
        "--fixed-action",
        type=str,
        default=None,
        help="Comma-separated non-negative action weights for --policy-kind fixed.",
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


def parse_fixed_action(action_text: str, action_size: int) -> np.ndarray:
    values = [float(token.strip()) for token in action_text.split(",") if token.strip()]
    if len(values) != action_size:
        raise ValueError(
            f"--fixed-action must contain exactly {action_size} comma-separated values; got {len(values)}"
        )
    action = np.asarray(values, dtype=np.float64)
    if np.any(action < 0):
        raise ValueError("--fixed-action values must be non-negative")
    if float(action.sum()) <= 0:
        raise ValueError("--fixed-action must contain positive mass")
    return action


def build_policy(args: argparse.Namespace, base_config: dict):
    probe_env = Market(base_config)
    if args.policy_kind == "inactive":
        return InactivePolicy(action_size=probe_env.action_space.shape[0])
    if args.policy_kind == "fixed":
        if args.fixed_action is None:
            raise ValueError("--fixed-action is required for --policy-kind fixed")
        action = parse_fixed_action(args.fixed_action, probe_env.action_space.shape[0])
        return FixedActionPolicy(action=action)
    if args.policy_kind == "heuristic_sell":
        return HeuristicSellPolicy(action_size=probe_env.action_space.shape[0])
    if args.model_path is None:
        raise ValueError("--model-path is required for non-inactive policies")
    return TorchPolicyAdapter.from_model_path(
        base_config=base_config,
        model_path=args.model_path,
        policy_kind=args.policy_kind,
        device=args.device,
        deterministic=True,
    )


def _bootstrap_mean_ci(series: pd.Series, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    clean = series.dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        return float("nan"), float("nan")
    if len(clean) == 1:
        value = float(clean[0])
        return value, value
    rng = np.random.default_rng(seed)
    samples = rng.choice(clean, size=(n_boot, len(clean)), replace=True)
    means = samples.mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _pooled_slope(df: pd.DataFrame, x_col: str, y_col: str) -> float:
    valid = df.loc[df[x_col].notna() & df[y_col].notna() & (df[x_col] != 0)].copy()
    if valid.empty:
        return float("nan")
    x = valid[x_col].to_numpy(dtype=float)
    y = valid[y_col].to_numpy(dtype=float)
    denom = float(np.dot(x, x))
    if denom == 0:
        return float("nan")
    return float(np.dot(x, y) / denom)


def _bootstrap_pooled_slope_ci(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    valid = df.loc[df[x_col].notna() & df[y_col].notna() & (df[x_col] != 0)].copy()
    if valid.empty:
        return float("nan"), float("nan")
    if len(valid) == 1:
        value = _pooled_slope(valid, x_col=x_col, y_col=y_col)
        return value, value
    rng = np.random.default_rng(seed)
    indices = np.arange(len(valid))
    boot = []
    for _ in range(n_boot):
        sample_idx = rng.choice(indices, size=len(indices), replace=True)
        sample = valid.iloc[sample_idx]
        boot.append(_pooled_slope(sample, x_col=x_col, y_col=y_col))
    low, high = np.quantile(np.asarray(boot, dtype=float), [0.025, 0.975])
    return float(low), float(high)


def summarize_run(
    summary_df: pd.DataFrame,
    min_abs_treatment_diff: float = 0.0,
) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    if summary_df.empty:
        empty = pd.DataFrame(columns=["metric", "value"])
        return empty, {}, pd.DataFrame(), pd.DataFrame()

    working = summary_df.copy()
    working["clipped_any"] = working["plus_clipped"].fillna(False) | working["minus_clipped"].fillna(False)
    working["usable_pair"] = ~working["clipped_any"]
    usable = working.loc[working["usable_pair"]].copy()
    support_threshold = max(0.0, float(min_abs_treatment_diff))
    working["support_pair"] = working["usable_pair"] & (
        working["local_treatment_difference"].abs() >= support_threshold
    )
    support = working.loc[working["support_pair"]].copy()

    def _float_or_nan(series: pd.Series) -> float:
        return float(series.mean()) if len(series) else float("nan")

    def _stderr(series: pd.Series) -> float:
        clean = series.dropna()
        if len(clean) <= 1:
            return float("nan")
        return float(clean.std(ddof=1) / math.sqrt(len(clean)))

    def _estimand_stats(column: str, prefix: str) -> dict[str, float]:
        mean_value = _float_or_nan(support[column])
        stderr = _stderr(support[column])
        boot_low, boot_high = _bootstrap_mean_ci(support[column], seed=0)
        mean_all = _float_or_nan(working[column])
        median_value = float(support[column].median()) if len(support) else float("nan")
        nonzero_rate = float((support[column] != 0).mean()) if len(support) else float("nan")
        return {
            f"mean_{prefix}": mean_value,
            f"median_{prefix}": median_value,
            f"{prefix}_stderr": stderr,
            f"{prefix}_ci95_low": mean_value - 1.96 * stderr if not math.isnan(stderr) else float("nan"),
            f"{prefix}_ci95_high": mean_value + 1.96 * stderr if not math.isnan(stderr) else float("nan"),
            f"{prefix}_boot_ci95_low": boot_low,
            f"{prefix}_boot_ci95_high": boot_high,
            f"mean_{prefix}_all_runs": mean_all,
            f"nonzero_{prefix}_rate": nonzero_rate,
        }

    usable_rate = float(working["usable_pair"].mean())
    support_rate = float(working["support_pair"].mean())
    treatment_changed_rate = float((working["local_treatment_difference"] != 0).mean())
    outcome_changed_rate = float((working["local_outcome_difference"] != 0).mean())
    zero_outcome_given_treatment = float(
        (
            (working["local_treatment_difference"] != 0) & (working["local_outcome_difference"] == 0)
        ).mean()
    )

    beta_true_mean = _float_or_nan(support["beta_true_hat"])
    beta_true_stderr = _stderr(support["beta_true_hat"])
    beta_true_boot_low, beta_true_boot_high = _bootstrap_mean_ci(support["beta_true_hat"], seed=1)
    beta_exec_pool_hat = _pooled_slope(support, x_col="local_treatment_difference", y_col="local_outcome_difference")
    beta_exec_pool_low, beta_exec_pool_high = _bootstrap_pooled_slope_ci(
        support,
        x_col="local_treatment_difference",
        y_col="local_outcome_difference",
        seed=2,
    )

    aggregate_stats = {
        "n_runs": int(len(working)),
        "n_usable_pairs": int(working["usable_pair"].sum()),
        "n_support_pairs": int(working["support_pair"].sum()),
        "usable_pair_rate": usable_rate,
        "support_pair_rate": support_rate,
        "min_abs_treatment_diff": support_threshold,
        **_estimand_stats("beta_action_hat", "beta_action_hat"),
        **_estimand_stats("beta_exec_hat", "beta_exec_hat"),
        "beta_exec_pool_hat": beta_exec_pool_hat,
        "beta_exec_pool_boot_ci95_low": beta_exec_pool_low,
        "beta_exec_pool_boot_ci95_high": beta_exec_pool_high,
        "mean_beta_true_hat": beta_true_mean,
        "median_beta_true_hat": float(support["beta_true_hat"].median()) if len(support) else float("nan"),
        "beta_true_hat_stderr": beta_true_stderr,
        "beta_true_hat_ci95_low": beta_true_mean - 1.96 * beta_true_stderr if not math.isnan(beta_true_stderr) else float("nan"),
        "beta_true_hat_ci95_high": beta_true_mean + 1.96 * beta_true_stderr if not math.isnan(beta_true_stderr) else float("nan"),
        "beta_true_hat_boot_ci95_low": beta_true_boot_low,
        "beta_true_hat_boot_ci95_high": beta_true_boot_high,
        "mean_local_treatment_difference": _float_or_nan(support["local_treatment_difference"]),
        "mean_local_outcome_difference": _float_or_nan(support["local_outcome_difference"]),
        "mean_X_plus": _float_or_nan(support["X_plus"]),
        "mean_X_minus": _float_or_nan(support["X_minus"]),
        "mean_planned_market_sell_plus": _float_or_nan(support["planned_market_sell_plus"]),
        "mean_planned_market_sell_minus": _float_or_nan(support["planned_market_sell_minus"]),
        "mean_DeltaP_plus": _float_or_nan(support["DeltaP_plus"]),
        "mean_DeltaP_minus": _float_or_nan(support["DeltaP_minus"]),
        "treatment_changed_rate": treatment_changed_rate,
        "outcome_changed_rate": outcome_changed_rate,
        "zero_outcome_given_treatment_change_rate": zero_outcome_given_treatment,
        "plus_clip_rate": float(working["plus_clipped"].fillna(False).mean()),
        "minus_clip_rate": float(working["minus_clipped"].fillna(False).mean()),
        "mean_t0": float(working["t0"].mean()),
        "min_t0": int(working["t0"].min()),
        "max_t0": int(working["t0"].max()),
    }

    stats_df = pd.DataFrame(
        [{"metric": metric, "value": value} for metric, value in aggregate_stats.items()]
    )
    t0_rows = []
    for t0_value, group in working.groupby("t0", dropna=False):
        support_group = group.loc[group["support_pair"]]
        t0_rows.append(
            {
                "t0": t0_value,
                "n_runs": int(len(group)),
                "n_usable_pairs": int(group["usable_pair"].sum()),
                "usable_pair_rate": float(group["usable_pair"].mean()),
                "n_support_pairs": int(group["support_pair"].sum()),
                "support_pair_rate": float(group["support_pair"].mean()),
                "mean_beta_exec_hat": float(support_group["beta_exec_hat"].mean()) if len(support_group) else float("nan"),
                "median_beta_exec_hat": float(support_group["beta_exec_hat"].median()) if len(support_group) else float("nan"),
                "beta_exec_pool_hat": _pooled_slope(
                    support_group,
                    x_col="local_treatment_difference",
                    y_col="local_outcome_difference",
                ),
                "mean_local_treatment_difference": float(support_group["local_treatment_difference"].mean()) if len(support_group) else float("nan"),
                "outcome_changed_rate": float((group["local_outcome_difference"] != 0).mean()),
                "treatment_changed_rate": float((group["local_treatment_difference"] != 0).mean()),
            }
        )
    t0_summary = pd.DataFrame(t0_rows).sort_values("t0")
    if "strategic_direction" in working.columns:
        latent_rows = []
        grouped = working.loc[working["strategic_direction"].notna()].groupby("strategic_direction", dropna=False)
        for direction, group in grouped:
            support_group = group.loc[group["support_pair"]]
            pooled_hat = _pooled_slope(support_group, x_col="local_treatment_difference", y_col="local_outcome_difference")
            pooled_low, pooled_high = _bootstrap_pooled_slope_ci(
                support_group,
                x_col="local_treatment_difference",
                y_col="local_outcome_difference",
                seed=3,
            )
            latent_rows.append(
                {
                    "strategic_direction": direction,
                    "n_runs": int(len(group)),
                    "n_usable_pairs": int(group["usable_pair"].sum()),
                    "usable_pair_rate": float(group["usable_pair"].mean()),
                    "n_support_pairs": int(group["support_pair"].sum()),
                    "support_pair_rate": float(group["support_pair"].mean()),
                    "mean_beta_exec_hat": float(support_group["beta_exec_hat"].mean()) if len(support_group) else float("nan"),
                    "median_beta_exec_hat": float(support_group["beta_exec_hat"].median()) if len(support_group) else float("nan"),
                    "beta_exec_pool_hat": pooled_hat,
                    "beta_exec_pool_boot_ci95_low": pooled_low,
                    "beta_exec_pool_boot_ci95_high": pooled_high,
                    "mean_local_treatment_difference": float(support_group["local_treatment_difference"].mean()) if len(support_group) else float("nan"),
                    "treatment_changed_rate": float((group["local_treatment_difference"] != 0).mean()),
                    "outcome_changed_rate": float((group["local_outcome_difference"] != 0).mean()),
                }
            )
        latent_direction_summary = pd.DataFrame(latent_rows).sort_values("strategic_direction")
    else:
        latent_direction_summary = pd.DataFrame()
    return stats_df, aggregate_stats, t0_summary, latent_direction_summary


def main() -> None:
    args = parse_args()
    base_config = build_base_config(args)
    seeds = parse_seeds(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy = build_policy(args, base_config)

    summary_rows = []
    detail_frames = []
    skipped_runs: list[dict[str, object]] = []
    rng = np.random.default_rng(args.seed_start)
    completed_runs = 0
    usable_runs = 0
    plus_clipped_runs = 0
    minus_clipped_runs = 0
    progress = ProgressReporter(total=len(seeds))

    for seed in seeds:
        selected_t0 = None if args.intervention_mode == "random" else args.intervention_time
        try:
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
                intervention_target=args.intervention_target,
                adaptive_intervention=args.adaptive_intervention,
                branch_from_state=not args.replay_from_seed,
                rng=rng,
            )
        except ValueError as exc:
            skipped_runs.append({"seed": int(seed), "reason": str(exc)})
            progress.update(1)
            progress.write(f"skipping seed {seed}: {exc}")
            progress.set_postfix(
                done=completed_runs,
                usable=usable_runs,
                skipped=len(skipped_runs),
                plus_clip=plus_clipped_runs,
                minus_clip=minus_clipped_runs,
                last="skipped",
            )
            continue
        summary_row = dict(result.summary)
        progress.update(1)
        completed_runs += 1
        plus_clipped = bool(result.summary["plus_clipped"])
        minus_clipped = bool(result.summary["minus_clipped"])
        if plus_clipped:
            plus_clipped_runs += 1
        if minus_clipped:
            minus_clipped_runs += 1
        if not (plus_clipped or minus_clipped):
            usable_runs += 1
        summary_row.update(
            {
                "market_env": args.market_env,
                "volume": args.volume,
                "terminal_time": args.terminal_time,
                "time_delta": args.time_delta,
                "drop_feature": None if args.drop_feature == "none" else args.drop_feature,
                "policy_kind": args.policy_kind,
                "fixed_action": args.fixed_action,
                "action_index": args.action_index,
                "slack_index": args.slack_index,
                "adaptive_intervention": bool(args.adaptive_intervention),
                "intervention_target": args.intervention_target,
                "branch_from_state": not args.replay_from_seed,
                "intervention_mode": args.intervention_mode,
                "usable_pair": not (result.summary["plus_clipped"] or result.summary["minus_clipped"]),
                "absolute_beta_action_hat": abs(float(result.summary["beta_action_hat"])),
                "absolute_beta_exec_hat": abs(float(result.summary["beta_exec_hat"])) if not math.isnan(float(result.summary["beta_exec_hat"])) else float("nan"),
                "absolute_beta_true_hat": abs(float(result.summary["beta_true_hat"])),
            }
        )
        summary_row["warnings"] = ";".join(result.warnings)
        summary_rows.append(summary_row)

        beta_exec = float(result.summary["beta_exec_hat"])
        last_beta = "nan" if math.isnan(beta_exec) else f"{beta_exec:.3f}"
        progress.set_postfix(
            done=completed_runs,
            usable=usable_runs,
            skipped=len(skipped_runs),
            plus_clip=plus_clipped_runs,
            minus_clip=minus_clipped_runs,
            t0=int(result.intervention_time),
            last_beta=last_beta,
        )

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

    progress.close()

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
    aggregate_stats_path = output_dir / "aggregate_stats.json"
    aggregate_stats_csv_path = output_dir / "aggregate_stats.csv"
    t0_summary_path = output_dir / "t0_summary.csv"
    latent_direction_summary_path = output_dir / "latent_direction_summary.csv"

    summary_df.to_csv(summary_path, index=False)
    details_df.to_csv(details_path, index=False)
    summary_df.to_json(summary_json_path, orient="records", indent=2)
    aggregate_stats_df, aggregate_stats, t0_summary_df, latent_direction_summary_df = summarize_run(
        summary_df,
        min_abs_treatment_diff=args.min_abs_treatment_diff,
    )
    aggregate_stats_df.to_csv(aggregate_stats_csv_path, index=False)
    t0_summary_df.to_csv(t0_summary_path, index=False)
    latent_direction_summary_df.to_csv(latent_direction_summary_path, index=False)
    with aggregate_stats_path.open("w", encoding="utf-8") as handle:
        json.dump(aggregate_stats, handle, indent=2)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "base_config": base_config,
                "policy_kind": args.policy_kind,
                "fixed_action": args.fixed_action,
                "model_path": args.model_path,
                "seeds": seeds,
                "intervention_mode": args.intervention_mode,
                "intervention_time": args.intervention_time,
                "delta": args.delta,
                "delta_units": args.delta_units,
                "min_abs_treatment_diff": args.min_abs_treatment_diff,
                "intervention_target": args.intervention_target,
                "horizon": args.horizon,
                "action_index": args.action_index,
                "slack_index": args.slack_index,
                "adaptive_intervention": bool(args.adaptive_intervention),
                "branch_from_state": not args.replay_from_seed,
                "device": args.device,
                "skipped_runs": skipped_runs,
            },
            handle,
            indent=2,
        )

    if summary_df.empty:
        print("No runs were executed.")
        return

    print(f"saved summary to {summary_path}")
    print(f"saved detailed trajectory logs to {details_path}")
    print(f"saved aggregate stats to {aggregate_stats_path}")
    print(f"saved intervention-time summary to {t0_summary_path}")
    if not latent_direction_summary_df.empty:
        print(f"saved latent-direction summary to {latent_direction_summary_path}")
    print(f"saved config snapshot to {config_path}")
    print("")
    print(summary_df[["seed", "t0", "delta", "X_plus", "X_minus", "DeltaP_plus", "DeltaP_minus", "beta_action_hat", "beta_exec_hat"]].to_string(index=False))


if __name__ == "__main__":
    main()
