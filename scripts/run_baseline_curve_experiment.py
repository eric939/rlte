#!/usr/bin/env python3
"""Run the baseline burn-in curve-estimation experiment on the simulator."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from causal.counterfactual_runner import run_intervention_curve
from causal.policy import FixedActionPolicy, HeuristicSellPolicy, InactivePolicy, TorchPolicyAdapter
from simulation.market_gym import Market


class ProgressReporter:
    """Terminal-first progress reporter with a plain-text fallback."""

    def __init__(self, total: int) -> None:
        self.total = int(total)
        self.bar = None
        if tqdm is not None:
            self.bar = tqdm(
                total=self.total,
                desc="Baseline curve runs",
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

    def set_postfix(self, **kwargs) -> None:
        if self.bar is not None:
            self.bar.set_postfix(**kwargs, refresh=True)
        else:
            filtered = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            print(f"[progress] {filtered}")

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-env", choices=["noise", "flow", "strategic"], default="strategic")
    parser.add_argument("--volume", type=int, default=20)
    parser.add_argument("--terminal-time", type=int, default=150)
    parser.add_argument("--time-delta", type=int, default=15)
    parser.add_argument("--drop-feature", choices=["volume", "order_info", "drift", "none"], default="none")

    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated list of seeds.")
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--num-episodes", type=int, default=300)

    parser.add_argument("--intervention-mode", choices=["fixed", "random"], default="random")
    parser.add_argument("--intervention-time", type=int, default=3)
    parser.add_argument(
        "--burn-in-steps",
        type=int,
        default=2,
        help="Minimum decision index before intervention-time selection begins.",
    )
    parser.add_argument("--base-delta", type=float, default=1.0, help="Base intervention size in lots or normalized units.")
    parser.add_argument(
        "--delta-multipliers",
        type=str,
        default="-3,-2,-1,0,1,2,3",
        help="Comma-separated multipliers applied to --base-delta to form the curve levels.",
    )
    parser.add_argument(
        "--curve-mode",
        choices=["one_sided", "symmetric"],
        default="symmetric",
        help="One-sided baseline-to-higher-intensity ladder or symmetric local ladder around baseline.",
    )
    parser.add_argument("--delta-units", choices=["normalized", "lots"], default="lots")
    parser.add_argument("--intervention-target", choices=["component", "market"], default="market")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument(
        "--min-abs-treatment-diff",
        type=float,
        default=1.0,
        help="Support threshold applied when aggregating the curve.",
    )
    parser.add_argument("--action-index", type=int, default=0)
    parser.add_argument("--slack-index", type=int, default=-1)
    parser.add_argument("--replay-from-seed", action="store_true")
    parser.add_argument(
        "--impulse-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After the intervention step, freeze future actions to the baseline path to isolate impulse response.",
    )
    parser.add_argument("--min-inventory-before", type=float, default=4.0)
    parser.add_argument("--min-bid-depth-before", type=float, default=1.0)
    parser.add_argument("--max-spread-before", type=float, default=None)
    parser.add_argument("--max-abs-imbalance-before", type=float, default=None)
    parser.add_argument(
        "--placebo",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run a placebo arm (perturb passive limit levels) for falsification.",
    )

    parser.add_argument(
        "--policy-kind",
        choices=["inactive", "fixed", "heuristic_sell", "logistic_normal", "logistic_normal_learn_std", "dirichlet"],
        default="heuristic_sell",
    )
    parser.add_argument("--fixed-action", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--output-dir", type=str, default="outputs/baseline_curve_main")
    return parser.parse_args()


def build_base_config(args: argparse.Namespace) -> dict:
    if args.market_env != "strategic":
        raise ValueError("the default ground-truth benchmark is locked to the strategic regime")
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
        raise ValueError("--model-path is required for non-heuristic learned policies")
    return TorchPolicyAdapter.from_model_path(
        base_config=base_config,
        model_path=args.model_path,
        policy_kind=args.policy_kind,
        device=args.device,
        deterministic=True,
    )


def parse_delta_levels(args: argparse.Namespace) -> list[float]:
    multipliers = [float(token.strip()) for token in args.delta_multipliers.split(",") if token.strip()]
    if not multipliers:
        raise ValueError("--delta-multipliers must contain at least one value")
    levels = sorted({float(args.base_delta) * multiplier for multiplier in multipliers})
    if 0.0 not in levels:
        levels = [0.0] + levels
    if args.curve_mode == "one_sided" and any(level < 0 for level in levels):
        raise ValueError("one-sided baseline curve estimation requires non-negative delta levels")
    return levels


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
    idx = np.arange(len(valid))
    boot = []
    for _ in range(n_boot):
        sample = valid.iloc[rng.choice(idx, size=len(idx), replace=True)]
        boot.append(_pooled_slope(sample, x_col=x_col, y_col=y_col))
    low, high = np.quantile(np.asarray(boot, dtype=float), [0.025, 0.975])
    return float(low), float(high)


def aggregate_curve(summary_df: pd.DataFrame, min_abs_treatment_diff: float) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for delta, group in summary_df.groupby("delta", sort=True):
        working = group.copy()
        working["support_pair"] = (~working["clipped"].fillna(False)) & (
            working["local_treatment_difference"].abs() >= float(min_abs_treatment_diff)
        )
        support = working.loc[working["support_pair"]].copy()
        seed_offset = int(round(abs(float(delta)) * 1000))
        immediate_low, immediate_high = _bootstrap_mean_ci(
            support["local_immediate_level_difference"], seed=11 + seed_offset
        )
        immediate_slope_low, immediate_slope_high = _bootstrap_pooled_slope_ci(
            support,
            x_col="local_treatment_difference",
            y_col="local_immediate_level_difference",
            seed=13 + seed_offset,
        )
        outcome_low, outcome_high = _bootstrap_mean_ci(support["local_level_difference"], seed=17 + seed_offset)
        slope_low, slope_high = _bootstrap_pooled_slope_ci(
            support,
            x_col="local_treatment_difference",
            y_col="local_level_difference",
            seed=23 + seed_offset,
        )
        rows.append(
            {
                "delta": float(delta),
                "n_runs": int(len(working)),
                "n_support_pairs": int(len(support)),
                "support_pair_rate": float(len(support) / len(working)) if len(working) else float("nan"),
                "mean_X_level": float(working["X_level"].mean()),
                "mean_DeltaP_level": float(working["DeltaP_level"].mean()),
                "mean_immediate_midprice_level": float(working["immediate_midprice_level"].mean()),
                "mean_future_midprice_level": float(working["future_midprice_level"].mean()),
                "mean_local_treatment_difference": float(working["local_treatment_difference"].mean()),
                "mean_local_immediate_level_difference": float(working["local_immediate_level_difference"].mean()),
                "immediate_level_effect_ci95_low": immediate_low,
                "immediate_level_effect_ci95_high": immediate_high,
                "beta_exec_immediate_pool_hat": _pooled_slope(
                    support,
                    x_col="local_treatment_difference",
                    y_col="local_immediate_level_difference",
                ),
                "beta_exec_immediate_pool_ci95_low": immediate_slope_low,
                "beta_exec_immediate_pool_ci95_high": immediate_slope_high,
                "mean_local_level_difference": float(working["local_level_difference"].mean()),
                "level_effect_ci95_low": outcome_low,
                "level_effect_ci95_high": outcome_high,
                "beta_exec_level_pool_hat": _pooled_slope(
                    support,
                    x_col="local_treatment_difference",
                    y_col="local_level_difference",
                ),
                "beta_exec_level_pool_ci95_low": slope_low,
                "beta_exec_level_pool_ci95_high": slope_high,
                # Backward-compatible aliases for downstream scripts.
                "mean_local_outcome_difference": float(working["local_level_difference"].mean()),
                "local_outcome_ci95_low": outcome_low,
                "local_outcome_ci95_high": outcome_high,
                "beta_exec_pool_hat": _pooled_slope(
                    support,
                    x_col="local_treatment_difference",
                    y_col="local_level_difference",
                ),
                "beta_exec_pool_ci95_low": slope_low,
                "beta_exec_pool_ci95_high": slope_high,
            }
        )
    return pd.DataFrame(rows).sort_values("delta").reset_index(drop=True)


def aggregate_curve_by_direction(summary_df: pd.DataFrame, min_abs_treatment_diff: float) -> pd.DataFrame:
    if "strategic_direction" not in summary_df.columns:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for direction, group in summary_df.groupby("strategic_direction", sort=True):
        if pd.isna(direction):
            continue
        aggregate_df = aggregate_curve(group.copy(), min_abs_treatment_diff=min_abs_treatment_diff)
        if aggregate_df.empty:
            continue
        aggregate_df.insert(0, "strategic_direction", str(direction))
        frames.append(aggregate_df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def baseline_state_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "seed",
        "t0",
        "strategic_direction",
        "inventory_before",
        "midprice_before",
        "spread_before",
        "imbalance_before",
        "bid_depth_before",
        "ask_depth_before",
        "best_bid_before",
        "best_ask_before",
        "baseline_X",
        "baseline_immediate_midprice",
        "baseline_DeltaP",
        "baseline_future_midprice",
        "planned_market_sell_baseline",
    ]
    available = [column for column in columns if column in summary_df.columns]
    if not available:
        return pd.DataFrame()
    state_df = summary_df.loc[summary_df["delta"] == 0.0, available].copy()
    return state_df.sort_values(["seed"]).reset_index(drop=True)


def _decision_midprice_path(df: pd.DataFrame) -> np.ndarray:
    if df.empty:
        return np.asarray([], dtype=float)
    path = df["midprice_before"].to_numpy(dtype=float)
    final_value = float(df["midprice_after"].iloc[-1])
    return np.concatenate([path, np.asarray([final_value], dtype=float)])


def extract_response_paths(
    logs_by_level: dict[float, pd.DataFrame],
    seed: int,
    t0: int,
    time_delta: int,
) -> pd.DataFrame:
    baseline_path = _decision_midprice_path(logs_by_level[0.0])
    if baseline_path.size == 0:
        return pd.DataFrame()
    baseline_origin = float(baseline_path[int(t0)])
    rows: list[dict[str, float | int]] = []
    for delta_level, frame in sorted(logs_by_level.items()):
        level_path = _decision_midprice_path(frame)
        if level_path.size == 0 or int(t0) >= len(level_path):
            continue
        level_origin = float(level_path[int(t0)])
        max_h = min(len(level_path), len(baseline_path)) - int(t0) - 1
        for step_ahead in range(max_h + 1):
            idx = int(t0) + step_ahead
            delta_p_baseline = float(baseline_path[idx] - baseline_origin)
            delta_p_level = float(level_path[idx] - level_origin)
            rows.append(
                {
                    "seed": int(seed),
                    "t0": int(t0),
                    "delta": float(delta_level),
                    "step_ahead": int(step_ahead),
                    "clock_ahead": float(step_ahead * time_delta),
                    "baseline_midprice": float(baseline_path[idx]),
                    "level_midprice": float(level_path[idx]),
                    "baseline_delta_p": delta_p_baseline,
                    "level_delta_p": delta_p_level,
                    "response_vs_baseline": float(delta_p_level - delta_p_baseline),
                    "level_effect_vs_baseline": float(level_path[idx] - baseline_path[idx]),
                }
            )
    return pd.DataFrame(rows)


def aggregate_response_paths(path_df: pd.DataFrame) -> pd.DataFrame:
    if path_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, float | int]] = []
    grouped = path_df.groupby(["delta", "step_ahead", "clock_ahead"], sort=True)
    for (delta, step_ahead, clock_ahead), group in grouped:
        seed_offset = int(round(abs(float(delta)) * 1000))
        resp_low, resp_high = _bootstrap_mean_ci(group["level_effect_vs_baseline"], seed=101 + seed_offset + int(step_ahead))
        level_low, level_high = _bootstrap_mean_ci(group["level_delta_p"], seed=151 + seed_offset + int(step_ahead))
        rows.append(
            {
                "delta": float(delta),
                "step_ahead": int(step_ahead),
                "clock_ahead": float(clock_ahead),
                "n_runs": int(len(group)),
                "mean_response_vs_baseline": float(group["response_vs_baseline"].mean()),
                "mean_level_effect_vs_baseline": float(group["level_effect_vs_baseline"].mean()),
                "response_ci95_low": resp_low,
                "response_ci95_high": resp_high,
                "mean_level_delta_p": float(group["level_delta_p"].mean()),
                "level_delta_p_ci95_low": level_low,
                "level_delta_p_ci95_high": level_high,
                "mean_baseline_delta_p": float(group["baseline_delta_p"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["delta", "step_ahead"]).reset_index(drop=True)


def summarize_response_paths(path_aggregate_df: pd.DataFrame) -> pd.DataFrame:
    if path_aggregate_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, float | int]] = []
    for delta, group in path_aggregate_df.groupby("delta", sort=True):
        ordered = group.sort_values("step_ahead").reset_index(drop=True)
        peak_idx = int(ordered["mean_response_vs_baseline"].abs().idxmax())
        peak_row = ordered.loc[peak_idx]
        rows.append(
            {
                "delta": float(delta),
                "n_points": int(len(ordered)),
                "impact_at_horizon": float(ordered["mean_level_effect_vs_baseline"].iloc[-1]),
                "peak_abs_response": float(abs(peak_row["mean_response_vs_baseline"])),
                "peak_response_signed": float(peak_row["mean_response_vs_baseline"]),
                "peak_abs_level_effect": float(abs(ordered["mean_level_effect_vs_baseline"]).max()),
                "peak_clock_ahead": float(peak_row["clock_ahead"]),
                "cumulative_response": float(np.trapezoid(ordered["mean_level_effect_vs_baseline"], ordered["clock_ahead"])),
                "cumulative_abs_response": float(np.trapezoid(np.abs(ordered["mean_level_effect_vs_baseline"]), ordered["clock_ahead"])),
            }
        )
    return pd.DataFrame(rows).sort_values("delta").reset_index(drop=True)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


def render_curve_figure(aggregate_df: pd.DataFrame, output_dir: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)

    x = aggregate_df["delta"].to_numpy(dtype=float)
    y = aggregate_df["mean_local_immediate_level_difference"].to_numpy(dtype=float)
    y_low = aggregate_df["immediate_level_effect_ci95_low"].to_numpy(dtype=float)
    y_high = aggregate_df["immediate_level_effect_ci95_high"].to_numpy(dtype=float)
    axes[0].axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
    axes[0].plot(x, y, marker="o", color="#1b4f72", linewidth=1.8)
    axes[0].fill_between(x, y_low, y_high, color="#7fb3d5", alpha=0.30)
    axes[0].set_xlabel("Intervention Size (lots)")
    axes[0].set_ylabel(r"Immediate level effect vs baseline")
    axes[0].set_title("Immediate Impact Curve")

    positive = aggregate_df.loc[aggregate_df["delta"] > 0].copy()
    x_beta = positive["delta"].to_numpy(dtype=float)
    beta = positive["beta_exec_immediate_pool_hat"].to_numpy(dtype=float)
    beta_low = positive["beta_exec_immediate_pool_ci95_low"].to_numpy(dtype=float)
    beta_high = positive["beta_exec_immediate_pool_ci95_high"].to_numpy(dtype=float)
    axes[1].axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
    axes[1].plot(x_beta, beta, marker="o", color="#7d3c98", linewidth=1.8)
    if len(x_beta):
        axes[1].fill_between(x_beta, beta_low, beta_high, color="#c39bd3", alpha=0.30)
    axes[1].set_xlabel("Intervention Size (lots)")
    axes[1].set_ylabel(r"$\hat{\beta}^{\mathrm{pool,imm}}_{\mathrm{exec}}$")
    axes[1].set_title("Pooled Immediate-Impact Slope")

    fig.savefig(output_dir / "curve_figure.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "curve_figure.png", bbox_inches="tight")
    plt.close(fig)


def render_response_path_figure(path_aggregate_df: pd.DataFrame, output_dir: Path) -> None:
    if path_aggregate_df.empty:
        return
    apply_style()
    positive_levels = sorted(delta for delta in path_aggregate_df["delta"].unique() if float(delta) != 0.0)
    cmap = plt.get_cmap("viridis", len(positive_levels))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)

    for idx, delta_level in enumerate(positive_levels):
        subset = path_aggregate_df.loc[path_aggregate_df["delta"] == delta_level].sort_values("step_ahead")
        x = subset["clock_ahead"].to_numpy(dtype=float)
        y = subset["mean_level_effect_vs_baseline"].to_numpy(dtype=float)
        low = subset["response_ci95_low"].to_numpy(dtype=float)
        high = subset["response_ci95_high"].to_numpy(dtype=float)
        color = cmap(idx)
        label = f"{delta_level:g} lot" if float(delta_level) == 1.0 else f"{delta_level:g} lots"
        axes[0].plot(x, y, color=color, linewidth=1.8, marker="o", markersize=3.5, label=label)
        axes[0].fill_between(x, low, high, color=color, alpha=0.18)

        y_abs = subset["mean_level_delta_p"].to_numpy(dtype=float)
        low_abs = subset["level_delta_p_ci95_low"].to_numpy(dtype=float)
        high_abs = subset["level_delta_p_ci95_high"].to_numpy(dtype=float)
        axes[1].plot(x, y_abs, color=color, linewidth=1.8, marker="o", markersize=3.5, label=label)
        axes[1].fill_between(x, low_abs, high_abs, color=color, alpha=0.18)

    axes[0].axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
    axes[1].axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
    axes[0].set_xlabel("Seconds After Intervention")
    axes[0].set_ylabel(r"Level effect vs baseline")
    axes[0].set_title("Immediate and Persistent Level Effect")
    axes[1].set_xlabel("Seconds After Intervention")
    axes[1].set_ylabel(r"$\Delta P$ from intervention point")
    axes[1].set_title("Absolute Post-Intervention Path")
    axes[1].legend(frameon=False, fontsize=8, loc="best")

    fig.savefig(output_dir / "curve_path_figure.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "curve_path_figure.png", bbox_inches="tight")
    plt.close(fig)


def _filter_support(summary_df: pd.DataFrame, min_abs_treatment_diff: float) -> pd.DataFrame:
    """Filter to non-baseline, non-clipped, above-threshold support pairs."""
    non_baseline = summary_df.loc[summary_df["delta"] != 0.0].copy()
    return non_baseline.loc[
        (~non_baseline["clipped"].fillna(False))
        & (non_baseline["local_treatment_difference"].abs() >= float(min_abs_treatment_diff))
    ].copy()


def _ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS coefficients via least-squares. X should include intercept column."""
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coeffs


def _bootstrap_ols_ci(
    X: np.ndarray,
    y: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> np.ndarray:
    """Bootstrap 95% CIs for each OLS coefficient. Returns (n_coeffs, 2) array."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    boot_coeffs = []
    for _ in range(n_boot):
        sample = rng.choice(idx, size=len(idx), replace=True)
        boot_coeffs.append(_ols_fit(X[sample], y[sample]))
    boot_arr = np.asarray(boot_coeffs, dtype=float)
    return np.column_stack([
        np.quantile(boot_arr, 0.025, axis=0),
        np.quantile(boot_arr, 0.975, axis=0),
    ])


def _pooled_ols(summary_df: pd.DataFrame, min_abs_treatment_diff: float) -> dict:
    """Pool all non-baseline (ΔX, ΔP) pairs into three models:

    1. Simple:      ΔP = α + β·ΔX
    2. Adjusted:    ΔP = α + β·ΔX + γ₁·spread + γ₂·imbalance + γ₃·inventory
    3. Stratified:  per strategic_direction (buy/sell) simple OLS

    Returns point estimates and bootstrap 95% CIs.
    """
    support = _filter_support(summary_df, min_abs_treatment_diff)
    result: dict[str, object] = {"n_pooled": int(len(support))}

    nan_keys = (
        "beta_ols", "alpha_ols",
        "beta_ols_ci95_low", "beta_ols_ci95_high",
        "alpha_ols_ci95_low", "alpha_ols_ci95_high",
        "beta_adj", "alpha_adj",
        "beta_adj_ci95_low", "beta_adj_ci95_high",
        "gamma_spread", "gamma_imbalance", "gamma_inventory",
    )
    if len(support) < 3:
        for key in nan_keys:
            result[key] = float("nan")
        result["stratified"] = {}
        return result

    x = support["local_treatment_difference"].to_numpy(dtype=float)
    y = support["local_immediate_level_difference"].to_numpy(dtype=float)

    # --- Model 1: Simple OLS ---
    X_simple = np.column_stack([np.ones(len(x)), x])
    coeffs_simple = _ols_fit(X_simple, y)
    ci_simple = _bootstrap_ols_ci(X_simple, y, seed=42)
    result["alpha_ols"] = float(coeffs_simple[0])
    result["beta_ols"] = float(coeffs_simple[1])
    result["alpha_ols_ci95_low"] = float(ci_simple[0, 0])
    result["alpha_ols_ci95_high"] = float(ci_simple[0, 1])
    result["beta_ols_ci95_low"] = float(ci_simple[1, 0])
    result["beta_ols_ci95_high"] = float(ci_simple[1, 1])

    # --- Model 2: Covariate-adjusted OLS ---
    covariate_cols = ["spread_before", "imbalance_before", "inventory_before"]
    has_covariates = all(col in support.columns for col in covariate_cols)
    if has_covariates and len(support) >= 6:
        covariates = support[covariate_cols].to_numpy(dtype=float)
        X_adj = np.column_stack([np.ones(len(x)), x, covariates])
        coeffs_adj = _ols_fit(X_adj, y)
        ci_adj = _bootstrap_ols_ci(X_adj, y, seed=43)
        result["alpha_adj"] = float(coeffs_adj[0])
        result["beta_adj"] = float(coeffs_adj[1])
        result["beta_adj_ci95_low"] = float(ci_adj[1, 0])
        result["beta_adj_ci95_high"] = float(ci_adj[1, 1])
        result["gamma_spread"] = float(coeffs_adj[2])
        result["gamma_imbalance"] = float(coeffs_adj[3])
        result["gamma_inventory"] = float(coeffs_adj[4])
    else:
        for key in ("beta_adj", "alpha_adj", "beta_adj_ci95_low", "beta_adj_ci95_high",
                     "gamma_spread", "gamma_imbalance", "gamma_inventory"):
            result[key] = float("nan")

    # --- Model 3: Stratified by strategic direction ---
    stratified: dict[str, dict[str, float]] = {}
    if "strategic_direction" in support.columns:
        for direction, group in support.groupby("strategic_direction", sort=True):
            if len(group) < 3:
                stratified[str(direction)] = {
                    "n": int(len(group)),
                    "beta_ols": float("nan"),
                    "beta_ols_ci95_low": float("nan"),
                    "beta_ols_ci95_high": float("nan"),
                }
                continue
            x_s = group["local_treatment_difference"].to_numpy(dtype=float)
            y_s = group["local_immediate_level_difference"].to_numpy(dtype=float)
            X_s = np.column_stack([np.ones(len(x_s)), x_s])
            coeffs_s = _ols_fit(X_s, y_s)
            ci_s = _bootstrap_ols_ci(X_s, y_s, seed=44 + hash(str(direction)) % 100)
            stratified[str(direction)] = {
                "n": int(len(group)),
                "beta_ols": float(coeffs_s[1]),
                "beta_ols_ci95_low": float(ci_s[1, 0]),
                "beta_ols_ci95_high": float(ci_s[1, 1]),
                "alpha_ols": float(coeffs_s[0]),
            }
    result["stratified"] = stratified
    return result


def run_curve_pass(
    args: argparse.Namespace,
    base_config: dict,
    policy,
    delta_levels: list[float],
    seeds: list[int],
    output_dir: Path,
    *,
    placebo: bool = False,
    label: str = "curve",
) -> None:
    """Run one full pass of the curve experiment (real or placebo)."""
    branch_from_state = not args.replay_from_seed

    summary_frames: list[pd.DataFrame] = []
    trajectory_frames: list[pd.DataFrame] = []
    response_path_frames: list[pd.DataFrame] = []
    warnings_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []

    desc = f"Placebo {label}" if placebo else f"Baseline {label}"
    reporter = ProgressReporter(total=len(seeds))
    reporter_desc = desc
    if reporter.bar is not None:
        reporter.bar.set_description(reporter_desc)
    selection_rng = np.random.default_rng(args.seed_start)
    n_skipped = 0
    for idx, seed in enumerate(seeds, start=1):
        try:
            result = run_intervention_curve(
                base_config=base_config,
                seed=seed,
                intervention_time=args.intervention_time if args.intervention_mode == "fixed" else None,
                deltas=delta_levels,
                horizon=args.horizon,
                policy=policy,
                delta_units=args.delta_units,
                action_index=args.action_index,
                slack_index=args.slack_index,
                intervention_target=args.intervention_target,
                branch_from_state=branch_from_state,
                burn_in_steps=args.burn_in_steps,
                curve_mode=args.curve_mode,
                impulse_mode=args.impulse_mode,
                min_inventory_before=args.min_inventory_before,
                min_bid_depth_before=args.min_bid_depth_before,
                max_spread_before=args.max_spread_before,
                max_abs_imbalance_before=args.max_abs_imbalance_before,
                rng=selection_rng,
                placebo=placebo,
            )
            summary_frames.append(result.summary_df)
            for delta_level, frame in result.logs_by_level.items():
                logged = frame.copy()
                logged["curve_delta"] = float(delta_level)
                trajectory_frames.append(logged)
            response_path_frames.append(
                extract_response_paths(
                    result.logs_by_level,
                    seed=seed,
                    t0=result.intervention_time,
                    time_delta=args.time_delta,
                )
            )
            for warning_text in result.warnings:
                warnings_rows.append(
                    {
                        "seed": int(seed),
                        "t0": int(result.intervention_time),
                        "warning": str(warning_text),
                    }
                )
            support_mask = result.summary_df["local_treatment_difference"].abs() >= float(args.min_abs_treatment_diff)
            support_rate = float(support_mask.mean()) if len(result.summary_df) else float("nan")
            t0_value = result.intervention_time
        except ValueError as exc:
            n_skipped += 1
            skipped_rows.append({"seed": int(seed), "reason": str(exc)})
            support_rate = float("nan")
            t0_value = "NA"
        reporter.update(1)
        reporter.set_postfix(
            seed=seed,
            done=f"{idx}/{len(seeds)}",
            t0=t0_value,
            support="NA" if math.isnan(support_rate) else f"{support_rate:.2f}",
            skipped=n_skipped,
        )

    reporter.close()

    if not summary_frames:
        raise RuntimeError(f"no {label} runs completed")

    prefix = f"placebo_" if placebo else ""

    summary_df = pd.concat(summary_frames, ignore_index=True)
    aggregate_df = aggregate_curve(summary_df, min_abs_treatment_diff=args.min_abs_treatment_diff)
    aggregate_direction_df = aggregate_curve_by_direction(
        summary_df,
        min_abs_treatment_diff=args.min_abs_treatment_diff,
    )
    state_df = baseline_state_summary(summary_df)
    non_empty_paths = [frame for frame in response_path_frames if not frame.empty]
    path_df = pd.concat(non_empty_paths, ignore_index=True) if non_empty_paths else pd.DataFrame()
    path_aggregate_df = aggregate_response_paths(path_df)
    path_summary_df = summarize_response_paths(path_aggregate_df)

    # Pooled OLS: ΔP = α + β·ΔX
    ols_result = _pooled_ols(summary_df, min_abs_treatment_diff=args.min_abs_treatment_diff)

    completed_seed_ids = sorted(int(seed) for seed in summary_df["seed"].unique())
    baseline_seed_ids = sorted(int(seed) for seed in summary_df.loc[summary_df["delta"] == 0.0, "seed"].unique())
    state_seed_ids = sorted(int(seed) for seed in state_df["seed"].unique()) if not state_df.empty else []
    if completed_seed_ids != baseline_seed_ids:
        raise RuntimeError("curve summary integrity failure: baseline rows do not match completed seeds")
    if state_seed_ids and state_seed_ids != completed_seed_ids:
        raise RuntimeError("curve summary integrity failure: baseline_state_summary seeds do not match completed seeds")

    summary_df.to_csv(output_dir / f"{prefix}curve_summary.csv", index=False)
    aggregate_df.to_csv(output_dir / f"{prefix}curve_aggregate.csv", index=False)
    if not aggregate_direction_df.empty:
        aggregate_direction_df.to_csv(output_dir / f"{prefix}curve_aggregate_by_direction.csv", index=False)
    if not state_df.empty:
        state_df.to_csv(output_dir / f"{prefix}baseline_state_summary.csv", index=False)
    if not path_df.empty:
        path_df.to_csv(output_dir / f"{prefix}curve_response_paths.csv", index=False)
    if not path_aggregate_df.empty:
        path_aggregate_df.to_csv(output_dir / f"{prefix}curve_response_path_aggregate.csv", index=False)
    if not path_summary_df.empty:
        path_summary_df.to_csv(output_dir / f"{prefix}curve_response_path_summary.csv", index=False)
    (output_dir / f"{prefix}curve_aggregate.json").write_text(
        json.dumps(aggregate_df.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    if not aggregate_direction_df.empty:
        (output_dir / f"{prefix}curve_aggregate_by_direction.json").write_text(
            json.dumps(aggregate_direction_df.to_dict(orient="records"), indent=2),
            encoding="utf-8",
        )
    (output_dir / f"{prefix}pooled_ols.json").write_text(
        json.dumps(ols_result, indent=2),
        encoding="utf-8",
    )
    non_empty_trajectories = [frame for frame in trajectory_frames if not frame.empty]
    if non_empty_trajectories:
        pd.concat(non_empty_trajectories, ignore_index=True).to_csv(output_dir / f"{prefix}curve_logs.csv", index=False)
    if warnings_rows:
        pd.DataFrame(warnings_rows).to_csv(output_dir / f"{prefix}curve_warnings.csv", index=False)
    if skipped_rows:
        pd.DataFrame(skipped_rows).to_csv(output_dir / f"{prefix}curve_skipped_seeds.csv", index=False)

    config_snapshot = {
        "experiment_name": f"{'placebo' if placebo else 'baseline'}_curve",
        "base_config": base_config,
        "policy_kind": args.policy_kind,
        "delta_levels": delta_levels,
        "curve_mode": args.curve_mode,
        "delta_units": args.delta_units,
        "intervention_target": args.intervention_target,
        "intervention_mode": args.intervention_mode,
        "intervention_time": args.intervention_time if args.intervention_mode == "fixed" else None,
        "burn_in_steps": args.burn_in_steps,
        "impulse_mode": bool(args.impulse_mode),
        "placebo": placebo,
        "min_inventory_before": args.min_inventory_before,
        "min_bid_depth_before": args.min_bid_depth_before,
        "max_spread_before": args.max_spread_before,
        "max_abs_imbalance_before": args.max_abs_imbalance_before,
        "horizon": args.horizon,
        "min_abs_treatment_diff": args.min_abs_treatment_diff,
        "branch_from_state": branch_from_state,
        "primary_benchmark_object": "instantaneous_post_execution_level_effect",
        "primary_benchmark_reporting": "pooled_and_direction_stratified",
        "secondary_response_object": "future_level_effect_path",
        "num_episodes": len(seeds),
        "seeds": seeds,
        "completed_seeds": completed_seed_ids,
        "skipped_seeds": [int(row["seed"]) for row in skipped_rows],
        "n_completed_seeds": int(len(completed_seed_ids)),
        "n_skipped_seeds": int(n_skipped),
        "pooled_ols": ols_result,
    }
    (output_dir / f"{prefix}config_snapshot.json").write_text(json.dumps(config_snapshot, indent=2), encoding="utf-8")

    render_curve_figure(aggregate_df, output_dir)
    render_response_path_figure(path_aggregate_df, output_dir)

    tag = "PLACEBO" if placebo else "MAIN"
    print(f"\n[{tag}] {label} outputs:")
    print(f"  - {output_dir / f'{prefix}curve_summary.csv'}")
    print(f"  - {output_dir / f'{prefix}curve_aggregate.csv'}")
    if not aggregate_direction_df.empty:
        print(f"  - {output_dir / f'{prefix}curve_aggregate_by_direction.csv'}")
    print(f"  - {output_dir / f'{prefix}pooled_ols.json'}")
    print(f"  Simple OLS:   β̂ = {ols_result['beta_ols']:.6f}  "
          f"[{ols_result['beta_ols_ci95_low']:.6f}, {ols_result['beta_ols_ci95_high']:.6f}]  "
          f"(n={ols_result['n_pooled']})")
    if not math.isnan(float(ols_result.get("beta_adj", float("nan")))):
        print(f"  Adjusted OLS: β̂ = {ols_result['beta_adj']:.6f}  "
              f"[{ols_result['beta_adj_ci95_low']:.6f}, {ols_result['beta_adj_ci95_high']:.6f}]")
    stratified = ols_result.get("stratified", {})
    for direction, stats in stratified.items():
        if not math.isnan(float(stats.get("beta_ols", float("nan")))):
            print(f"  Stratified ({direction:>4s}): β̂ = {stats['beta_ols']:.6f}  "
                  f"[{stats['beta_ols_ci95_low']:.6f}, {stats['beta_ols_ci95_high']:.6f}]  "
                  f"(n={stats['n']})")


def main() -> None:
    args = parse_args()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = build_base_config(args)
    policy = build_policy(args, base_config)
    delta_levels = parse_delta_levels(args)
    seeds = parse_seeds(args)

    # Main experiment
    run_curve_pass(args, base_config, policy, delta_levels, seeds, output_dir, placebo=False, label="curve")

    # Placebo falsification arm
    if args.placebo:
        print("\n--- Running placebo falsification arm ---")
        run_curve_pass(args, base_config, policy, delta_levels, seeds, output_dir, placebo=True, label="curve")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        main()
