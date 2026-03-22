#!/usr/bin/env python3
"""Render publication-style comparisons across multiple causal experiment directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ENV_ORDER = ["noise", "flow", "strategic"]
METRIC_SPECS = [
    ("mean_beta_action_hat", r"Mean $\hat{\beta}_{action}$"),
    ("mean_beta_exec_hat", r"Mean $\hat{\beta}_{exec}$"),
    ("outcome_changed_rate", "Outcome-change rate"),
    ("usable_pair_rate", "Usable-pair rate"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dirs",
        type=str,
        nargs="+",
        required=True,
        help="One or more run directories containing summary.csv and config_snapshot.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/causal_comparison",
        help="Directory for comparison tables and figures.",
    )
    parser.add_argument(
        "--output-mode",
        choices=["compact", "full"],
        default="compact",
        help="Whether to write only a compact paper-ready bundle or the full artifact set.",
    )
    return parser.parse_args()


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _stderr(series: pd.Series) -> float:
    clean = series.dropna()
    if len(clean) <= 1:
        return float("nan")
    return float(clean.std(ddof=1) / np.sqrt(len(clean)))


def _estimand_aggregate(summary: pd.DataFrame, column: str, prefix: str) -> dict[str, float]:
    clipped_any = summary["plus_clipped"].fillna(False) | summary["minus_clipped"].fillna(False)
    usable = summary.loc[~clipped_any].copy()
    mean_value = _safe_float(usable[column].mean())
    stderr = _stderr(usable[column])
    return {
        f"mean_{prefix}": mean_value,
        f"median_{prefix}": _safe_float(usable[column].median()),
        f"{prefix}_stderr": stderr,
        f"{prefix}_ci95_low": mean_value - 1.96 * stderr if not np.isnan(stderr) else float("nan"),
        f"{prefix}_ci95_high": mean_value + 1.96 * stderr if not np.isnan(stderr) else float("nan"),
        f"mean_{prefix}_all_runs": _safe_float(summary[column].mean()),
        f"nonzero_{prefix}_rate": float((usable[column] != 0).mean()) if len(usable) else float("nan"),
    }


def _compute_aggregate(summary: pd.DataFrame) -> dict[str, float]:
    working = summary.copy()
    if "beta_action_hat" not in working.columns and "beta_true_hat" in working.columns:
        working["beta_action_hat"] = working["beta_true_hat"]
    if "beta_exec_hat" not in working.columns:
        working["beta_exec_hat"] = np.where(
            working["local_treatment_difference"] != 0,
            working["local_outcome_difference"] / working["local_treatment_difference"],
            np.nan,
        )
    clipped_any = working["plus_clipped"].fillna(False) | working["minus_clipped"].fillna(False)
    usable = working.loc[~clipped_any].copy()
    return {
        "n_runs": int(len(working)),
        "n_usable_pairs": int((~clipped_any).sum()),
        "usable_pair_rate": float((~clipped_any).mean()) if len(working) else float("nan"),
        **_estimand_aggregate(working, "beta_action_hat", "beta_action_hat"),
        **_estimand_aggregate(working, "beta_exec_hat", "beta_exec_hat"),
        "mean_local_treatment_difference": _safe_float(usable["local_treatment_difference"].mean()),
        "mean_local_outcome_difference": _safe_float(usable["local_outcome_difference"].mean()),
        "outcome_changed_rate": float((working["local_outcome_difference"] != 0).mean()) if len(working) else float("nan"),
        "treatment_changed_rate": float((working["local_treatment_difference"] != 0).mean()) if len(working) else float("nan"),
        "plus_clip_rate": float(working["plus_clipped"].fillna(False).mean()) if len(working) else float("nan"),
        "minus_clip_rate": float(working["minus_clipped"].fillna(False).mean()) if len(working) else float("nan"),
    }


def load_run(run_dir: Path) -> tuple[pd.DataFrame, dict]:
    summary = pd.read_csv(run_dir / "summary.csv")
    config = json.loads((run_dir / "config_snapshot.json").read_text(encoding="utf-8"))
    aggregate_path = run_dir / "aggregate_stats.json"
    aggregate = None
    if aggregate_path.exists():
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if aggregate is None or "mean_beta_action_hat" not in aggregate or "mean_beta_exec_hat" not in aggregate:
        aggregate = _compute_aggregate(summary)
    return summary, {**config, **aggregate}


def build_run_table(run_dirs: list[Path]) -> pd.DataFrame:
    rows: list[dict] = []
    for run_dir in run_dirs:
        _summary, stats = load_run(run_dir)
        base = stats.get("base_config", {})
        rows.append(
            {
                "run_dir": str(run_dir),
                "market_env": base.get("market_env"),
                "volume": base.get("volume"),
                "terminal_time": base.get("terminal_time"),
                "time_delta": base.get("time_delta"),
                "drop_feature": base.get("drop_feature"),
                "policy_kind": stats.get("policy_kind"),
                "fixed_action": stats.get("fixed_action"),
                "model_path": stats.get("model_path"),
                "intervention_mode": stats.get("intervention_mode"),
                "intervention_time": stats.get("intervention_time"),
                "delta": stats.get("delta"),
                "delta_units": stats.get("delta_units"),
                "horizon": stats.get("horizon"),
                "action_index": stats.get("action_index"),
                "slack_index": stats.get("slack_index"),
                "n_runs": stats.get("n_runs"),
                "n_usable_pairs": stats.get("n_usable_pairs"),
                "usable_pair_rate": stats.get("usable_pair_rate"),
                "mean_beta_action_hat": stats.get("mean_beta_action_hat"),
                "beta_action_hat_ci95_low": stats.get("beta_action_hat_ci95_low"),
                "beta_action_hat_ci95_high": stats.get("beta_action_hat_ci95_high"),
                "mean_beta_exec_hat": stats.get("mean_beta_exec_hat"),
                "beta_exec_hat_ci95_low": stats.get("beta_exec_hat_ci95_low"),
                "beta_exec_hat_ci95_high": stats.get("beta_exec_hat_ci95_high"),
                "nonzero_beta_action_hat_rate": stats.get("nonzero_beta_action_hat_rate"),
                "nonzero_beta_exec_hat_rate": stats.get("nonzero_beta_exec_hat_rate"),
                "mean_local_treatment_difference": stats.get("mean_local_treatment_difference"),
                "mean_local_outcome_difference": stats.get("mean_local_outcome_difference"),
                "treatment_changed_rate": stats.get("treatment_changed_rate"),
                "outcome_changed_rate": stats.get("outcome_changed_rate"),
                "plus_clip_rate": stats.get("plus_clip_rate"),
                "minus_clip_rate": stats.get("minus_clip_rate"),
            }
        )
    table = pd.DataFrame(rows)
    sort_key = pd.Categorical(table["market_env"], categories=ENV_ORDER, ordered=True)
    table = (
        table.assign(_env_order=sort_key)
        .sort_values(["volume", "_env_order", "horizon", "delta"])
        .drop(columns="_env_order")
        .reset_index(drop=True)
    )
    return table


def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


def _render_heatmap(ax, matrix: np.ndarray, x_labels: list[str], y_labels: list[str], title: str, cmap: str, fmt: str = "{:.3f}") -> None:
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            label = "NA" if np.isnan(value) else fmt.format(value)
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color="black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def render_volume_figures(table: pd.DataFrame, output_dir: Path) -> None:
    apply_paper_style()
    for volume, volume_table in table.groupby("volume", sort=True):
        deltas = sorted(volume_table["delta"].dropna().unique().tolist())
        horizons = sorted(volume_table["horizon"].dropna().unique().tolist())
        x_labels = [f"{delta:g}" for delta in deltas]
        y_labels = [f"h={h}" for h in horizons]

        fig, axes = plt.subplots(len(ENV_ORDER), len(METRIC_SPECS), figsize=(16, 9), constrained_layout=True)
        if len(ENV_ORDER) == 1:
            axes = np.asarray([axes])

        for row_idx, env in enumerate(ENV_ORDER):
            env_table = volume_table.loc[volume_table["market_env"] == env].copy()
            for col_idx, (metric, title) in enumerate(METRIC_SPECS):
                pivot = (
                    env_table.pivot_table(index="horizon", columns="delta", values=metric, aggfunc="first")
                    .reindex(index=horizons, columns=deltas)
                    .to_numpy(dtype=float)
                )
                cmap = "RdBu_r" if "beta" in metric else "YlGnBu"
                fmt = "{:.3f}" if "rate" not in metric else "{:.2f}"
                _render_heatmap(
                    axes[row_idx, col_idx],
                    pivot,
                    x_labels,
                    y_labels,
                    f"{env.capitalize()}: {title}",
                    cmap=cmap,
                    fmt=fmt,
                )
                if row_idx == len(ENV_ORDER) - 1:
                    axes[row_idx, col_idx].set_xlabel(r"$\delta$")
                if col_idx == 0:
                    axes[row_idx, col_idx].set_ylabel("Horizon")

        fig.suptitle(f"Causal Interventional Comparison, Volume {volume}", fontsize=14, fontweight="bold")
        stem = output_dir / f"comparison_volume_{volume}"
        fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def render_effect_curve_figure(table: pd.DataFrame, output_dir: Path) -> None:
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    metric_specs = [
        ("mean_beta_action_hat", r"Mean $\hat{\beta}_{action}$"),
        ("mean_beta_exec_hat", r"Mean $\hat{\beta}_{exec}$"),
    ]
    for ax, (metric, ylabel) in zip(axes, metric_specs):
        for env in ENV_ORDER:
            env_table = table.loc[table["market_env"] == env].copy()
            for volume in sorted(env_table["volume"].dropna().unique()):
                subset = env_table.loc[(env_table["volume"] == volume) & (env_table["delta"] == env_table["delta"].min())]
                subset = subset.sort_values("horizon")
                if subset.empty:
                    continue
                ax.plot(
                    subset["horizon"],
                    subset[metric],
                    marker="o",
                    linewidth=1.3,
                    label=f"{env}, v={volume}",
                )
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Horizon")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " vs Horizon")
    axes[1].legend(loc="best")
    fig.suptitle("Horizon Profiles at Smallest Delta", fontsize=14, fontweight="bold")
    stem = output_dir / "comparison_horizon_profiles"
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def render_compact_figure(table: pd.DataFrame, output_dir: Path) -> None:
    apply_paper_style()
    deltas = sorted(table["delta"].dropna().unique().tolist())
    horizons = sorted(table["horizon"].dropna().unique().tolist())
    x_labels = [f"{delta:g}" for delta in deltas]
    y_labels = [f"h={h}" for h in horizons]

    fig, axes = plt.subplots(2, len(ENV_ORDER), figsize=(12, 6), constrained_layout=True)
    if len(ENV_ORDER) == 1:
        axes = np.asarray([[axes[0]], [axes[1]]])

    metric_rows = [
        ("mean_beta_exec_hat", r"Mean $\hat{\beta}_{exec}$", "RdBu_r", "{:.3f}"),
        ("outcome_changed_rate", "Outcome-change rate", "YlGnBu", "{:.2f}"),
    ]

    for col_idx, env in enumerate(ENV_ORDER):
        env_table = table.loc[table["market_env"] == env].copy()
        for row_idx, (metric, title, cmap, fmt) in enumerate(metric_rows):
            pivot = (
                env_table.pivot_table(index="horizon", columns="delta", values=metric, aggfunc="first")
                .reindex(index=horizons, columns=deltas)
                .to_numpy(dtype=float)
            )
            _render_heatmap(
                axes[row_idx, col_idx],
                pivot,
                x_labels,
                y_labels,
                f"{env.capitalize()}: {title}",
                cmap=cmap,
                fmt=fmt,
            )
            if row_idx == len(metric_rows) - 1:
                axes[row_idx, col_idx].set_xlabel(r"$\delta$")
            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel("Horizon")

    fig.suptitle("Primary Causal Results", fontsize=14, fontweight="bold")
    stem = output_dir / "main_figure"
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_tables(table: pd.DataFrame, output_dir: Path) -> None:
    table_path = output_dir / "comparison_table.csv"
    latex_path = output_dir / "comparison_table.tex"
    display_columns = [
        "market_env",
        "policy_kind",
        "volume",
        "horizon",
        "delta",
        "n_runs",
        "n_usable_pairs",
        "usable_pair_rate",
        "mean_beta_action_hat",
        "beta_action_hat_ci95_low",
        "beta_action_hat_ci95_high",
        "mean_beta_exec_hat",
        "beta_exec_hat_ci95_low",
        "beta_exec_hat_ci95_high",
        "mean_local_treatment_difference",
        "mean_local_outcome_difference",
        "treatment_changed_rate",
        "outcome_changed_rate",
    ]
    display_table = table.loc[:, display_columns].copy()
    table.to_csv(table_path, index=False)
    latex_table = display_table.to_latex(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x))
    latex_path.write_text(latex_table, encoding="utf-8")


def save_compact_tables(table: pd.DataFrame, output_dir: Path) -> None:
    csv_path = output_dir / "main_results.csv"
    latex_path = output_dir / "main_results.tex"
    compact_columns = [
        "market_env",
        "volume",
        "horizon",
        "delta",
        "n_runs",
        "n_usable_pairs",
        "usable_pair_rate",
        "mean_beta_exec_hat",
        "beta_exec_hat_ci95_low",
        "beta_exec_hat_ci95_high",
        "mean_beta_action_hat",
        "beta_action_hat_ci95_low",
        "beta_action_hat_ci95_high",
        "outcome_changed_rate",
    ]
    compact_table = table.loc[:, compact_columns].copy()
    compact_table.to_csv(csv_path, index=False)
    latex_table = compact_table.to_latex(
        index=False,
        float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x),
    )
    latex_path.write_text(latex_table, encoding="utf-8")


def write_summary_markdown(table: pd.DataFrame, output_dir: Path) -> None:
    report_path = output_dir / "comparison_report.md"
    lines = [
        "# Causal Comparison Report",
        "",
        "## Key Takeaways",
        "",
    ]
    for _, row in table.iterrows():
        beta_action = _safe_float(row.get("mean_beta_action_hat"))
        beta_exec = _safe_float(row.get("mean_beta_exec_hat"))
        usable = _safe_float(row.get("usable_pair_rate"))
        outcome_changed = _safe_float(row.get("outcome_changed_rate"))
        lines.append(
            f"- `{row['market_env']} | v={row['volume']} | h={row['horizon']} | d={row['delta']}`: "
            f"`beta_action_hat={beta_action:.4f}`, "
            f"`beta_exec_hat={beta_exec:.4f}`, "
            f"`usable={100 * usable:.1f}%`, "
            f"`outcome_changed={100 * outcome_changed:.1f}%`."
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `comparison_table.csv`",
            "- `comparison_table.tex`",
            "- `comparison_volume_<volume>.pdf` and `.png`",
            "- `comparison_horizon_profiles.pdf` and `.png`",
            "",
            "## Full Table",
            "",
            "```text",
            table.to_string(index=False),
            "```",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_compact_summary(table: pd.DataFrame, output_dir: Path) -> None:
    report_path = output_dir / "main_summary.md"
    lines = [
        "# Main Causal Results",
        "",
        "Primary estimand: `beta_exec_hat = (DeltaP_plus - DeltaP_minus) / (X_plus - X_minus)`.",
        "Secondary diagnostic: `beta_action_hat = (DeltaP_plus - DeltaP_minus) / (2 * delta)`.",
        "",
        "## Highest-signal cells by |mean beta_exec_hat|",
        "",
    ]
    ranking = table.reindex(table["mean_beta_exec_hat"].abs().sort_values(ascending=False).index)
    for _, row in ranking.head(6).iterrows():
        lines.append(
            f"- `{row['market_env']} | h={row['horizon']} | d={row['delta']}`: "
            f"`beta_exec_hat={_safe_float(row['mean_beta_exec_hat']):.4f}` "
            f"(95% CI {_safe_float(row['beta_exec_hat_ci95_low']):.4f}, {_safe_float(row['beta_exec_hat_ci95_high']):.4f}), "
            f"`outcome_changed_rate={100 * _safe_float(row['outcome_changed_rate']):.1f}%`, "
            f"`usable_pair_rate={100 * _safe_float(row['usable_pair_rate']):.1f}%`."
        )

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `main_results.csv`",
            "- `main_results.tex`",
            "- `main_figure.pdf` and `main_figure.png`",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dirs = [Path(path) for path in args.input_dirs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = build_run_table(run_dirs)
    if args.output_mode == "compact":
        save_compact_tables(table, output_dir)
        render_compact_figure(table, output_dir)
        write_compact_summary(table, output_dir)
        print(f"saved compact table to {output_dir / 'main_results.csv'}")
        print(f"saved compact figure to {output_dir / 'main_figure.pdf'}")
        print(f"saved compact summary to {output_dir / 'main_summary.md'}")
    else:
        save_tables(table, output_dir)
        render_volume_figures(table, output_dir)
        render_effect_curve_figure(table, output_dir)
        write_summary_markdown(table, output_dir)
        print(f"saved comparison table to {output_dir / 'comparison_table.csv'}")
        print(f"saved comparison heatmaps under {output_dir}")
        print(f"saved comparison report to {output_dir / 'comparison_report.md'}")


if __name__ == "__main__":
    main()
