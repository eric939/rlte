#!/usr/bin/env python3
"""Analyze whether local interventional impact depends on pre-intervention book state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STATE_COLUMNS = [
    "bid_depth_before",
    "ask_depth_before",
    "spread_before",
    "imbalance_before",
    "inventory_before",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=str, default="outputs/baseline_curve_main")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--beta-column", type=str, default="beta_exec_immediate_hat")
    parser.add_argument("--min-abs-treatment-diff", type=float, default=1.0)
    parser.add_argument("--n-bins", type=int, default=4)
    parser.add_argument("--include-deltas", type=str, default=None, help="Optional comma-separated delta levels.")
    return parser.parse_args()


def load_curve_summary(input_dir: Path) -> pd.DataFrame:
    curve_path = input_dir / "curve_summary.csv"
    if not curve_path.exists():
        raise FileNotFoundError(f"missing {curve_path}")
    return pd.read_csv(curve_path)


def filter_analysis_rows(
    df: pd.DataFrame,
    beta_column: str,
    min_abs_treatment_diff: float,
    include_deltas: list[float] | None,
) -> pd.DataFrame:
    filtered = df.copy()
    filtered = filtered[filtered["delta"] != 0.0]
    filtered = filtered[~filtered["clipped"].astype(bool)]
    filtered = filtered[np.isfinite(filtered[beta_column])]
    filtered = filtered[filtered["local_treatment_difference"].abs() >= float(min_abs_treatment_diff)]
    if include_deltas is not None:
        filtered = filtered[filtered["delta"].isin(include_deltas)]
    filtered = filtered.reset_index(drop=True)
    if filtered.empty:
        raise ValueError("no rows remain after filtering")
    return filtered


def fit_ols(y: np.ndarray, X: np.ndarray, feature_names: list[str]) -> tuple[pd.DataFrame, float]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0, ddof=0)
    x_std = np.where(x_std > 0.0, x_std, 1.0)
    X_scaled = (X - x_mean) / x_std

    X_design = np.column_stack([np.ones(len(X_scaled)), X_scaled])
    names = ["intercept"] + feature_names
    beta_scaled, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    y_hat = X_design @ beta_scaled
    resid = y - y_hat
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 0.0 if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot

    dof = max(len(y) - X_design.shape[1], 1)
    sigma2 = ss_res / dof
    xtx_inv = np.linalg.pinv(X_design.T @ X_design)
    cov_scaled = sigma2 * xtx_inv

    transform = np.eye(len(names))
    transform[0, 1:] = -x_mean / x_std
    transform[1:, 1:] = np.diag(1.0 / x_std)
    beta_hat = transform @ beta_scaled
    cov_orig = transform @ cov_scaled @ transform.T
    se = np.sqrt(np.clip(np.diag(cov_orig), a_min=0.0, a_max=None))
    t_stat = np.divide(beta_hat, se, out=np.full_like(beta_hat, np.nan), where=se > 0)

    table = pd.DataFrame(
        {
            "term": names,
            "coef": beta_hat,
            "std_err": se,
            "t_stat": t_stat,
        }
    )
    return table, r2


def summarize_by_bins(df: pd.DataFrame, value_col: str, state_col: str, n_bins: int) -> pd.DataFrame:
    work = df[[value_col, state_col]].dropna().copy()
    unique = work[state_col].nunique()
    q = min(int(n_bins), int(unique))
    if q < 2:
        return pd.DataFrame()
    work["bin"] = pd.qcut(work[state_col], q=q, duplicates="drop")
    summary = (
        work.groupby("bin", observed=True)
        .agg(
            n=(value_col, "size"),
            state_mean=(state_col, "mean"),
            value_mean=(value_col, "mean"),
            value_std=(value_col, "std"),
        )
        .reset_index()
    )
    summary["value_se"] = summary["value_std"] / np.sqrt(summary["n"].clip(lower=1))
    summary["ci95_low"] = summary["value_mean"] - 1.96 * summary["value_se"]
    summary["ci95_high"] = summary["value_mean"] + 1.96 * summary["value_se"]
    summary["state_column"] = state_col
    summary["value_column"] = value_col
    summary["bin_label"] = summary["bin"].astype(str)
    return summary.drop(columns=["bin"])


def build_candidate_normalizations(df: pd.DataFrame) -> dict[str, pd.Series]:
    bid = df["bid_depth_before"].clip(lower=1e-8)
    ask = df["ask_depth_before"].clip(lower=1e-8)
    spread = df["spread_before"].clip(lower=1e-8)
    imbalance = df["imbalance_before"]
    mean_depth = 0.5 * (bid + ask)
    return {
        "inv_bid_depth": 1.0 / bid,
        "inv_mean_depth": 1.0 / mean_depth.clip(lower=1e-8),
        "spread_over_bid_depth": spread / bid,
        "sell_pressure_depth": (1.0 - imbalance) / bid,
    }


def evaluate_normalizations(df: pd.DataFrame, beta_column: str) -> tuple[pd.DataFrame, str]:
    rows: list[dict[str, float | str]] = []
    state_matrix = df[STATE_COLUMNS].to_numpy(dtype=float)
    raw_y = df[beta_column].to_numpy(dtype=float)
    _, raw_r2 = fit_ols(raw_y, state_matrix, STATE_COLUMNS)

    candidates = build_candidate_normalizations(df)
    best_name = ""
    best_score = np.inf
    for name, g in candidates.items():
        normalized = raw_y / g.to_numpy(dtype=float)
        reg_table, r2 = fit_ols(normalized, state_matrix, STATE_COLUMNS)
        corr_depth = float(np.corrcoef(normalized, df["bid_depth_before"])[0, 1])
        corr_imbalance = float(np.corrcoef(normalized, df["imbalance_before"])[0, 1])
        score = abs(corr_depth) + abs(corr_imbalance) + max(r2, 0.0)
        rows.append(
            {
                "candidate": name,
                "normalized_std": float(np.std(normalized, ddof=1)),
                "normalized_mean": float(np.mean(normalized)),
                "normalized_abs_mean": float(np.mean(np.abs(normalized))),
                "normalized_r2_on_state": float(r2),
                "raw_r2_on_state": float(raw_r2),
                "corr_with_bid_depth": corr_depth,
                "corr_with_imbalance": corr_imbalance,
                "score_lower_is_better": score,
                "intercept": float(reg_table.loc[reg_table["term"] == "intercept", "coef"].iloc[0]),
            }
        )
        if score < best_score:
            best_score = score
            best_name = name
    table = pd.DataFrame(rows).sort_values("score_lower_is_better").reset_index(drop=True)
    return table, best_name


def plot_binned_effects(
    summary_df: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    states = list(summary_df["state_column"].drop_duplicates())
    n = len(states)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.5), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, state_col in zip(axes, states):
        sub = summary_df[summary_df["state_column"] == state_col].copy()
        ax.errorbar(
            sub["state_mean"],
            sub["value_mean"],
            yerr=1.96 * sub["value_se"],
            fmt="o-",
            lw=1.4,
            capsize=3,
        )
        ax.axhline(0.0, color="black", lw=0.8, alpha=0.6)
        ax.set_title(state_col.replace("_before", ""))
        ax.set_xlabel("bin mean")
        ax.set_ylabel(sub["value_column"].iloc[0])
    fig.suptitle(title, fontsize=11)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "state_dependence"
    output_dir.mkdir(parents=True, exist_ok=True)

    include_deltas = None
    if args.include_deltas:
        include_deltas = [float(x) for x in args.include_deltas.split(",") if x.strip()]

    curve = load_curve_summary(input_dir)
    filtered = filter_analysis_rows(
        curve,
        beta_column=args.beta_column,
        min_abs_treatment_diff=args.min_abs_treatment_diff,
        include_deltas=include_deltas,
    )
    filtered.to_csv(output_dir / "analysis_sample.csv", index=False)

    reg_table, raw_r2 = fit_ols(
        filtered[args.beta_column].to_numpy(dtype=float),
        filtered[STATE_COLUMNS].to_numpy(dtype=float),
        STATE_COLUMNS,
    )
    reg_table["r2"] = raw_r2
    reg_table.to_csv(output_dir / "state_dependence_regression.csv", index=False)

    bin_rows = []
    for state_col in STATE_COLUMNS:
        bin_summary = summarize_by_bins(filtered, args.beta_column, state_col, n_bins=args.n_bins)
        if not bin_summary.empty:
            bin_rows.append(bin_summary)
    beta_bins = pd.concat(bin_rows, ignore_index=True) if bin_rows else pd.DataFrame()
    if not beta_bins.empty:
        beta_bins.to_csv(output_dir / "beta_bin_summary.csv", index=False)
        plot_binned_effects(
            beta_bins,
            output_dir / "beta_state_bins.png",
            title="Raw Local Beta vs Pre-Intervention State",
        )

    norm_table, best_name = evaluate_normalizations(filtered, args.beta_column)
    norm_table.to_csv(output_dir / "normalization_candidates.csv", index=False)

    normalized_bins = pd.DataFrame()
    if best_name:
        best_g = build_candidate_normalizations(filtered)[best_name]
        filtered = filtered.copy()
        filtered["beta_normalized_best"] = filtered[args.beta_column] / best_g.to_numpy(dtype=float)
        norm_bin_rows = []
        for state_col in STATE_COLUMNS:
            bin_summary = summarize_by_bins(filtered, "beta_normalized_best", state_col, n_bins=args.n_bins)
            if not bin_summary.empty:
                norm_bin_rows.append(bin_summary)
        normalized_bins = pd.concat(norm_bin_rows, ignore_index=True) if norm_bin_rows else pd.DataFrame()
        if not normalized_bins.empty:
            normalized_bins.to_csv(output_dir / "beta_normalized_bin_summary.csv", index=False)
            plot_binned_effects(
                normalized_bins,
                output_dir / "beta_normalized_state_bins.png",
                title=f"Normalized Beta vs State (best g = {best_name})",
            )

    summary = {
        "input_dir": str(input_dir),
        "n_rows_filtered": int(len(filtered)),
        "beta_column": args.beta_column,
        "min_abs_treatment_diff": float(args.min_abs_treatment_diff),
        "include_deltas": include_deltas,
        "raw_beta_r2_on_state": float(raw_r2),
        "best_normalization": best_name,
        "best_normalization_row": None if norm_table.empty else norm_table.iloc[0].to_dict(),
    }
    with open(output_dir / "analysis_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"wrote state-dependence analysis to {output_dir}")
    print(f"filtered rows: {len(filtered)}")
    print(f"raw beta R^2 on state: {raw_r2:.4f}")
    if best_name:
        print(f"best normalization candidate: {best_name}")


if __name__ == "__main__":
    main()
