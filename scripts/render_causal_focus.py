#!/usr/bin/env python3
"""Render a minimal paper-ready bundle for selected causal experiment directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Entries of the form label=path/to/run_dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for the focused outputs.",
    )
    return parser.parse_args()


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


def _fallback_pooled_stats(run_dir: Path) -> dict[str, float]:
    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        return {
            "beta_exec_pool_hat": float("nan"),
            "beta_exec_pool_boot_ci95_low": float("nan"),
            "beta_exec_pool_boot_ci95_high": float("nan"),
        }
    summary = pd.read_csv(summary_path)
    usable = summary.loc[
        (~summary["plus_clipped"].fillna(False)) & (~summary["minus_clipped"].fillna(False))
    ].copy()
    hat = _pooled_slope(usable, x_col="local_treatment_difference", y_col="local_outcome_difference")
    low, high = _bootstrap_pooled_slope_ci(
        usable,
        x_col="local_treatment_difference",
        y_col="local_outcome_difference",
        seed=11,
    )
    return {
        "beta_exec_pool_hat": hat,
        "beta_exec_pool_boot_ci95_low": low,
        "beta_exec_pool_boot_ci95_high": high,
    }


def _latent_direction_fields(run_dir: Path) -> dict[str, float]:
    latent_path = run_dir / "latent_direction_summary.csv"
    fields = {
        "buy_beta_exec_pool_hat": float("nan"),
        "buy_beta_exec_pool_ci95_low": float("nan"),
        "buy_beta_exec_pool_ci95_high": float("nan"),
        "sell_beta_exec_pool_hat": float("nan"),
        "sell_beta_exec_pool_ci95_low": float("nan"),
        "sell_beta_exec_pool_ci95_high": float("nan"),
    }
    if not latent_path.exists():
        return fields
    latent = pd.read_csv(latent_path)
    for direction in ("buy", "sell"):
        row = latent.loc[latent["strategic_direction"] == direction]
        if row.empty:
            continue
        record = row.iloc[0]
        fields[f"{direction}_beta_exec_pool_hat"] = float(record.get("beta_exec_pool_hat", float("nan")))
        fields[f"{direction}_beta_exec_pool_ci95_low"] = float(
            record.get("beta_exec_pool_boot_ci95_low", float("nan"))
        )
        fields[f"{direction}_beta_exec_pool_ci95_high"] = float(
            record.get("beta_exec_pool_boot_ci95_high", float("nan"))
        )
    return fields


def _t0_profile(run_dir: Path) -> pd.DataFrame:
    summary_path = run_dir / "summary.csv"
    config_path = run_dir / "config_snapshot.json"
    if not summary_path.exists():
        return pd.DataFrame()
    summary = pd.read_csv(summary_path)
    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    min_abs_treatment_diff = float(config.get("min_abs_treatment_diff", 0.0))
    working = summary.copy()
    working["usable_pair"] = (~working["plus_clipped"].fillna(False)) & (~working["minus_clipped"].fillna(False))
    working["support_pair"] = working["usable_pair"] & (
        working["local_treatment_difference"].abs() >= min_abs_treatment_diff
    )

    rows: list[dict[str, float | int | str]] = []
    for t0_value, group in working.groupby("t0", dropna=False):
        support = group.loc[group["support_pair"]].copy()
        if support.empty:
            continue
        overall_low, overall_high = _bootstrap_pooled_slope_ci(
            support,
            x_col="local_treatment_difference",
            y_col="local_outcome_difference",
            seed=31 + int(t0_value),
        )
        rows.append(
            {
                "t0": int(t0_value),
                "direction": "overall",
                "beta_exec_pool_hat": _pooled_slope(
                    support,
                    x_col="local_treatment_difference",
                    y_col="local_outcome_difference",
                ),
                "beta_exec_pool_ci95_low": overall_low,
                "beta_exec_pool_ci95_high": overall_high,
                "n_support_pairs": int(len(support)),
            }
        )
        if "strategic_direction" in support.columns:
            for direction, sub in support.loc[support["strategic_direction"].notna()].groupby("strategic_direction"):
                low, high = _bootstrap_pooled_slope_ci(
                    sub,
                    x_col="local_treatment_difference",
                    y_col="local_outcome_difference",
                    seed=97 + int(t0_value),
                )
                rows.append(
                    {
                        "t0": int(t0_value),
                        "direction": str(direction),
                        "beta_exec_pool_hat": _pooled_slope(
                            sub,
                            x_col="local_treatment_difference",
                            y_col="local_outcome_difference",
                        ),
                        "beta_exec_pool_ci95_low": low,
                        "beta_exec_pool_ci95_high": high,
                        "n_support_pairs": int(len(sub)),
                    }
                )
    return pd.DataFrame(rows).sort_values(["t0", "direction"])


def load_row(label: str, run_dir: Path) -> dict:
    aggregate = json.loads((run_dir / "aggregate_stats.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config_snapshot.json").read_text(encoding="utf-8"))
    base = config.get("base_config", {})
    if "beta_exec_pool_hat" not in aggregate:
        aggregate.update(_fallback_pooled_stats(run_dir))
    return {
        "label": label,
        "run_dir": str(run_dir),
        "market_env": base.get("market_env"),
        "volume": base.get("volume"),
        "terminal_time": base.get("terminal_time"),
        "time_delta": base.get("time_delta"),
        "policy_kind": config.get("policy_kind"),
        "intervention_mode": config.get("intervention_mode"),
        "intervention_target": config.get("intervention_target"),
        "branch_from_state": config.get("branch_from_state"),
        "delta": config.get("delta"),
        "horizon": config.get("horizon"),
        "n_runs": aggregate.get("n_runs"),
        "n_usable_pairs": aggregate.get("n_usable_pairs"),
        "usable_pair_rate": aggregate.get("usable_pair_rate"),
        "n_support_pairs": aggregate.get("n_support_pairs", aggregate.get("n_usable_pairs")),
        "support_pair_rate": aggregate.get("support_pair_rate", aggregate.get("usable_pair_rate")),
        "min_abs_treatment_diff": aggregate.get("min_abs_treatment_diff", 0.0),
        "beta_exec_pool_hat": aggregate.get("beta_exec_pool_hat"),
        "beta_exec_pool_ci95_low": aggregate.get("beta_exec_pool_boot_ci95_low", aggregate.get("beta_exec_pool_ci95_low")),
        "beta_exec_pool_ci95_high": aggregate.get("beta_exec_pool_boot_ci95_high", aggregate.get("beta_exec_pool_ci95_high")),
        "mean_beta_exec_hat": aggregate.get("mean_beta_exec_hat"),
        "beta_exec_hat_ci95_low": aggregate.get("beta_exec_hat_boot_ci95_low", aggregate.get("beta_exec_hat_ci95_low")),
        "beta_exec_hat_ci95_high": aggregate.get("beta_exec_hat_boot_ci95_high", aggregate.get("beta_exec_hat_ci95_high")),
        "mean_beta_action_hat": aggregate.get("mean_beta_action_hat"),
        "beta_action_hat_ci95_low": aggregate.get("beta_action_hat_ci95_low"),
        "beta_action_hat_ci95_high": aggregate.get("beta_action_hat_ci95_high"),
        "outcome_changed_rate": aggregate.get("outcome_changed_rate"),
        "treatment_changed_rate": aggregate.get("treatment_changed_rate"),
        "plus_clip_rate": aggregate.get("plus_clip_rate"),
        "minus_clip_rate": aggregate.get("minus_clip_rate"),
        "clip_any_rate": 1.0 - float(aggregate.get("usable_pair_rate")),
        **_latent_direction_fields(run_dir),
    }


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


def render_figure(table: pd.DataFrame, output_dir: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), constrained_layout=True)
    labels = table["label"].tolist()
    horizon_sorted = table.sort_values("horizon").reset_index(drop=True)
    pooled = horizon_sorted["beta_exec_pool_hat"].to_numpy(dtype=float)
    pooled_low = horizon_sorted["beta_exec_pool_ci95_low"].to_numpy(dtype=float)
    pooled_high = horizon_sorted["beta_exec_pool_ci95_high"].to_numpy(dtype=float)
    pooled_err = [pooled - pooled_low, pooled_high - pooled]

    if len(horizon_sorted["horizon"].unique()) == len(horizon_sorted):
        x_profile = horizon_sorted["horizon"].to_numpy(dtype=float)
        axes[0].errorbar(
            x_profile,
            pooled,
            yerr=pooled_err,
            fmt="o-",
            color="#1F4E79",
            ecolor="black",
            elinewidth=1.1,
            capsize=4,
            linewidth=1.4,
            markersize=5,
        )
        axes[0].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        axes[0].set_xlabel("Horizon")
        axes[0].set_ylabel(r"Pooled $\hat{\beta}_{exec}$")
        axes[0].set_title("Primary Pooled Estimand")
    else:
        y = list(range(len(table)))
        axes[0].errorbar(
            pooled,
            y,
            xerr=pooled_err,
            fmt="o",
            color="#1F4E79",
            ecolor="black",
            elinewidth=1.2,
            capsize=4,
            markersize=6,
        )
        axes[0].axvline(0.0, color="black", linestyle="--", linewidth=0.8)
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(labels)
        axes[0].invert_yaxis()
        axes[0].set_xlabel(r"Pooled $\hat{\beta}_{exec}$")
        axes[0].set_title("Primary Pooled Estimand")

    if len(horizon_sorted["horizon"].unique()) == len(horizon_sorted):
        x_profile = horizon_sorted["horizon"].to_numpy(dtype=float)
        for direction, color, marker in (
            ("buy", "#2E6F40", "o"),
            ("sell", "#8A3B3B", "s"),
        ):
            values = horizon_sorted[f"{direction}_beta_exec_pool_hat"].to_numpy(dtype=float)
            lows = horizon_sorted[f"{direction}_beta_exec_pool_ci95_low"].to_numpy(dtype=float)
            highs = horizon_sorted[f"{direction}_beta_exec_pool_ci95_high"].to_numpy(dtype=float)
            mask = ~np.isnan(values)
            if not np.any(mask):
                continue
            errs = [values[mask] - lows[mask], highs[mask] - values[mask]]
            axes[1].errorbar(
                x_profile[mask],
                values[mask],
                yerr=errs,
                fmt=f"{marker}-",
                color=color,
                ecolor=color,
                linewidth=1.3,
                elinewidth=1.0,
                capsize=3,
                markersize=4.5,
                label=direction.capitalize(),
            )
        axes[1].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        axes[1].set_xlabel("Horizon")
        axes[1].set_ylabel(r"Pooled $\hat{\beta}_{exec}$")
        axes[1].set_title("Latent-Direction Split")
        axes[1].legend(frameon=False, loc="best")
    else:
        x = np.arange(len(horizon_sorted), dtype=float)
        width = 0.34
        buy_vals = horizon_sorted["buy_beta_exec_pool_hat"].to_numpy(dtype=float)
        sell_vals = horizon_sorted["sell_beta_exec_pool_hat"].to_numpy(dtype=float)
        axes[1].bar(x - width / 2, np.nan_to_num(buy_vals, nan=0.0), width=width, color="#2E6F40", label="Buy")
        axes[1].bar(x + width / 2, np.nan_to_num(sell_vals, nan=0.0), width=width, color="#8A3B3B", label="Sell")
        axes[1].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        axes[1].set_xticks(list(x))
        axes[1].set_xticklabels(labels)
        axes[1].set_ylabel(r"Pooled $\hat{\beta}_{exec}$")
        axes[1].set_title("Latent-Direction Split")
        axes[1].legend(frameon=False, loc="best")

    width = 0.22
    support = table["support_pair_rate"].to_numpy(dtype=float)
    treatment = table["treatment_changed_rate"].to_numpy(dtype=float)
    outcome = table["outcome_changed_rate"].to_numpy(dtype=float)
    x = pd.RangeIndex(len(table)).to_numpy(dtype=float)
    axes[2].bar(x - width, support, width=width, color="#3F6C8A", label="Support pairs")
    axes[2].bar(x, treatment, width=width, color="#6C8E47", label="Treatment changed")
    axes[2].bar(x + width, outcome, width=width, color="#9E7C3B", label="Outcome changed")
    axes[2].set_xticks(list(x))
    axes[2].set_xticklabels(labels)
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_ylabel("Rate")
    axes[2].set_title("Diagnostics")
    axes[2].legend(loc="upper left", frameon=False)

    stem = output_dir / "focus_figure"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def render_t0_figure(table: pd.DataFrame, output_dir: Path) -> None:
    apply_style()
    run_count = len(table)
    fig, axes = plt.subplots(1, run_count, figsize=(6.0 * run_count, 4.2), constrained_layout=True)
    if run_count == 1:
        axes = [axes]

    styles = {
        "overall": {"color": "#1F4E79", "marker": "o", "label": "Overall"},
        "buy": {"color": "#2E6F40", "marker": "o", "label": "Latent buy"},
        "sell": {"color": "#8A3B3B", "marker": "s", "label": "Latent sell"},
    }

    for ax, row in zip(axes, table.to_dict(orient="records")):
        profile = _t0_profile(Path(row["run_dir"]))
        if profile.empty:
            ax.set_visible(False)
            continue
        for direction in ("overall", "buy", "sell"):
            sub = profile.loc[profile["direction"] == direction].sort_values("t0")
            if sub.empty:
                continue
            style = styles[direction]
            x = sub["t0"].to_numpy(dtype=float)
            y = sub["beta_exec_pool_hat"].to_numpy(dtype=float)
            low = sub["beta_exec_pool_ci95_low"].to_numpy(dtype=float)
            high = sub["beta_exec_pool_ci95_high"].to_numpy(dtype=float)
            n_support = sub["n_support_pairs"].to_numpy(dtype=float)
            yerr = [y - low, high - y]
            marker_sizes = 3.5 + 0.35 * np.sqrt(n_support)
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="none",
                ecolor=style["color"],
                elinewidth=0.9 if direction == "overall" else 0.7,
                capsize=2.5,
                alpha=0.55 if direction == "overall" else 0.35,
                zorder=2,
            )
            ax.plot(
                x,
                y,
                color=style["color"],
                linewidth=1.5 if direction == "overall" else 1.1,
                alpha=1.0 if direction == "overall" else 0.9,
                label=style["label"],
                zorder=3,
            )
            ax.scatter(
                x,
                y,
                s=np.square(marker_sizes),
                marker=style["marker"],
                color=style["color"],
                alpha=1.0 if direction == "overall" else 0.9,
                zorder=4,
            )
            if direction == "overall":
                for xi, yi, ni in zip(x, y, n_support):
                    ax.annotate(
                        f"{int(ni)}",
                        (xi, yi),
                        textcoords="offset points",
                        xytext=(0, 6),
                        ha="center",
                        fontsize=7,
                        color="#444444",
                    )
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel(r"Intervention Time $t_0$")
        ax.set_ylabel(r"Pooled $\hat{\beta}_{exec}$")
        ax.set_title(f"{row['label']} (h={int(row['horizon'])}, threshold={row['min_abs_treatment_diff']:.0f})")
        ax.set_xticks(sorted(profile["t0"].unique()))
        ax.legend(frameon=False, loc="best")

    stem = output_dir / "focus_t0_figure"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_outputs(table: pd.DataFrame, output_dir: Path) -> None:
    csv_path = output_dir / "focus_results.csv"
    tex_path = output_dir / "focus_results.tex"

    display_columns = [
        "label",
        "market_env",
        "policy_kind",
        "intervention_target",
        "delta",
        "horizon",
        "n_runs",
        "n_usable_pairs",
        "n_support_pairs",
        "support_pair_rate",
        "min_abs_treatment_diff",
        "beta_exec_pool_hat",
        "beta_exec_pool_ci95_low",
        "beta_exec_pool_ci95_high",
        "buy_beta_exec_pool_hat",
        "sell_beta_exec_pool_hat",
        "treatment_changed_rate",
        "outcome_changed_rate",
        "clip_any_rate",
    ]
    display_table = table.loc[:, display_columns].copy()
    display_table.to_csv(csv_path, index=False)
    tex_path.write_text(
        display_table.to_latex(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for item in args.runs:
        if "=" not in item:
            raise ValueError(f"Invalid --runs entry: {item!r}. Expected label=path.")
        label, path_text = item.split("=", 1)
        rows.append(load_row(label, Path(path_text)))

    table = pd.DataFrame(rows)
    write_outputs(table, output_dir)
    render_figure(table, output_dir)
    render_t0_figure(table, output_dir)

    print(f"saved focused table to {output_dir / 'focus_results.csv'}")
    print(f"saved focused figure to {output_dir / 'focus_figure.pdf'}")
    print(f"saved time-profile figure to {output_dir / 'focus_t0_figure.pdf'}")


if __name__ == "__main__":
    main()
