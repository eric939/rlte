#!/usr/bin/env python3
"""Render a compact report for paired causal intervention outputs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing summary.csv and trajectory_logs.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for the rendered report. Defaults to <input-dir>/report.",
    )
    return parser.parse_args()


def _safe_float(value, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_data(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(input_dir / "summary.csv")
    trajectories = pd.read_csv(input_dir / "trajectory_logs.csv")
    return summary, trajectories


def intervention_rows(trajectories: pd.DataFrame) -> pd.DataFrame:
    rows = trajectories.loc[trajectories["decision_index"] == trajectories["pair_t0"]].copy()
    rows = rows.sort_values(["pair_seed", "pair_label"]).reset_index(drop=True)
    return rows


def compute_metrics(summary: pd.DataFrame, intervention_df: pd.DataFrame) -> dict[str, object]:
    total_runs = int(len(summary))
    plus_clipped = int(summary["plus_clipped"].fillna(False).sum())
    minus_clipped = int(summary["minus_clipped"].fillna(False).sum())
    any_clipped_mask = summary["plus_clipped"].fillna(False) | summary["minus_clipped"].fillna(False)
    valid_mask = ~any_clipped_mask
    valid_summary = summary.loc[valid_mask].copy()

    warning_counts = (
        summary["warnings"]
        .fillna("")
        .str.split(";")
        .explode()
        .str.strip()
    )
    warning_counts = warning_counts[warning_counts != ""].value_counts()

    plus_rows = intervention_df.loc[intervention_df["pair_label"] == "plus"].copy()
    minus_rows = intervention_df.loc[intervention_df["pair_label"] == "minus"].copy()

    def _stat(frame: pd.DataFrame, column: str) -> float:
        if frame.empty:
            return float("nan")
        return float(frame[column].mean())

    def _estimand_metrics(column: str, prefix: str) -> dict[str, float]:
        return {
            f"mean_{prefix}_all": _safe_float(summary[column].mean()),
            f"median_{prefix}_all": _safe_float(summary[column].median()),
            f"mean_{prefix}_valid": _safe_float(valid_summary[column].mean()),
            f"median_{prefix}_valid": _safe_float(valid_summary[column].median()),
        }

    metrics = {
        "total_runs": total_runs,
        "valid_runs": int(valid_mask.sum()),
        "invalid_runs": int(any_clipped_mask.sum()),
        "plus_clip_rate": plus_clipped / total_runs if total_runs else float("nan"),
        "minus_clip_rate": minus_clipped / total_runs if total_runs else float("nan"),
        "any_clip_rate": any_clipped_mask.mean() if total_runs else float("nan"),
        **_estimand_metrics("beta_action_hat", "beta_action_hat"),
        **_estimand_metrics("beta_exec_hat", "beta_exec_hat"),
        "mean_beta_all": _safe_float(summary["beta_true_hat"].mean()),
        "median_beta_all": _safe_float(summary["beta_true_hat"].median()),
        "mean_beta_valid": _safe_float(valid_summary["beta_true_hat"].mean()),
        "median_beta_valid": _safe_float(valid_summary["beta_true_hat"].median()),
        "mean_outcome_diff_all": _safe_float(summary["local_outcome_difference"].mean()),
        "mean_outcome_diff_valid": _safe_float(valid_summary["local_outcome_difference"].mean()),
        "mean_x_plus": _stat(plus_rows, "signed_executed_volume"),
        "mean_x_minus": _stat(minus_rows, "signed_executed_volume"),
        "mean_delta_p_plus": _stat(plus_rows, "delta_p_horizon"),
        "mean_delta_p_minus": _stat(minus_rows, "delta_p_horizon"),
        "warning_counts": warning_counts,
        "valid_summary": valid_summary,
    }
    return metrics


def build_findings(summary: pd.DataFrame, metrics: dict[str, object]) -> list[str]:
    findings: list[str] = []
    total_runs = int(metrics["total_runs"])
    valid_runs = int(metrics["valid_runs"])
    if total_runs == 0:
        return ["No runs were found in summary.csv."]

    any_clip_rate = float(metrics["any_clip_rate"])
    if valid_runs == 0:
        findings.append(
            "No unclipped intervention pairs were produced, so the current run is not a clean symmetric finite-difference estimate."
        )
    elif any_clip_rate > 0.25:
        findings.append(
            f"Only {valid_runs}/{total_runs} runs were unclipped, so clipping materially limits the usable sample."
        )
    else:
        findings.append(
            f"{valid_runs}/{total_runs} runs were unclipped, so most intervention pairs are usable for summary statistics."
        )

    warning_counts = metrics["warning_counts"]
    if len(warning_counts) > 0:
        top_warning = warning_counts.index[0]
        top_warning_count = int(warning_counts.iloc[0])
        findings.append(f"Most common warning: `{top_warning}` in {top_warning_count}/{total_runs} runs.")

    mean_beta_action_valid = _safe_float(metrics["mean_beta_action_hat_valid"])
    mean_beta_exec_valid = _safe_float(metrics["mean_beta_exec_hat_valid"])
    mean_beta_all = _safe_float(metrics["mean_beta_all"])
    if not math.isnan(mean_beta_action_valid):
        findings.append(f"Mean unclipped `beta_action_hat`: {mean_beta_action_valid:.4f}.")
    else:
        findings.append(f"Mean all-run `beta_true_hat`: {mean_beta_all:.4f}, but that average includes clipped runs.")
    if not math.isnan(mean_beta_exec_valid):
        findings.append(f"Mean unclipped `beta_exec_hat`: {mean_beta_exec_valid:.4f}.")

    mean_outcome_diff_valid = _safe_float(metrics["mean_outcome_diff_valid"])
    if not math.isnan(mean_outcome_diff_valid):
        findings.append(f"Mean unclipped outcome difference `DeltaP_plus - DeltaP_minus`: {mean_outcome_diff_valid:.4f}.")

    if summary["beta_action_hat"].nunique(dropna=True) <= 1:
        findings.append("The action-level effect estimate is nearly constant across seeds, which often indicates a degenerate setup rather than a stable structural result.")

    return findings


def render_dashboard(
    summary: pd.DataFrame,
    intervention_df: pd.DataFrame,
    metrics: dict[str, object],
    findings: list[str],
    output_path: Path,
) -> None:
    plt.style.use("default")
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25])

    ax_text = fig.add_subplot(grid[0, 0])
    ax_clip = fig.add_subplot(grid[0, 1])
    ax_beta = fig.add_subplot(grid[1, 0])
    ax_outcome = fig.add_subplot(grid[1, 1])

    ax_text.axis("off")
    valid_runs = int(metrics["valid_runs"])
    total_runs = int(metrics["total_runs"])
    lines = [
        "Most Important Findings",
        "",
        f"Runs: {total_runs}",
        f"Unclipped usable pairs: {valid_runs}/{total_runs}",
        f"Any clipping rate: {100 * float(metrics['any_clip_rate']):.1f}%",
        f"Mean beta_action (all): {_safe_float(metrics['mean_beta_action_hat_all']):.4f}",
    ]
    mean_beta_action_valid = _safe_float(metrics["mean_beta_action_hat_valid"])
    mean_beta_exec_valid = _safe_float(metrics["mean_beta_exec_hat_valid"])
    if not math.isnan(mean_beta_action_valid):
        lines.append(f"Mean beta_action (unclipped): {mean_beta_action_valid:.4f}")
    if not math.isnan(mean_beta_exec_valid):
        lines.append(f"Mean beta_exec (unclipped): {mean_beta_exec_valid:.4f}")
    lines.append("")
    for finding in findings[:5]:
        lines.append(f"- {finding}")
    ax_text.text(
        0.0,
        1.0,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
    )

    clip_rates = pd.Series(
        {
            "plus clipped": 100 * float(metrics["plus_clip_rate"]),
            "minus clipped": 100 * float(metrics["minus_clip_rate"]),
            "any clipped": 100 * float(metrics["any_clip_rate"]),
        }
    )
    colors = ["#2D6A4F", "#BC4749", "#7C6A0A"]
    ax_clip.bar(clip_rates.index, clip_rates.values, color=colors)
    ax_clip.set_ylim(0, 100)
    ax_clip.set_ylabel("Percent of runs")
    ax_clip.set_title("Clipping Rates")
    for idx, value in enumerate(clip_rates.values):
        ax_clip.text(idx, value + 2, f"{value:.1f}%", ha="center", va="bottom", fontsize=10)

    valid_summary = metrics["valid_summary"]
    if len(valid_summary) > 0:
        beta_action_values = valid_summary["beta_action_hat"].to_numpy(dtype=float)
        beta_exec_values = valid_summary["beta_exec_hat"].dropna().to_numpy(dtype=float)
        bins = min(20, max(5, len(beta_action_values)))
        ax_beta.hist(beta_action_values, bins=bins, color="#457B9D", alpha=0.70, label="beta_action_hat")
        if len(beta_exec_values) > 0:
            ax_beta.hist(beta_exec_values, bins=min(20, max(5, len(beta_exec_values))), color="#B85C38", alpha=0.55, label="beta_exec_hat")
            ax_beta.axvline(float(np.mean(beta_exec_values)), color="#8A3B12", linestyle=":", linewidth=2)
        ax_beta.axvline(float(np.mean(beta_action_values)), color="#1D3557", linestyle="--", linewidth=2)
        ax_beta.set_title("Estimand Distributions (Unclipped Runs)")
        ax_beta.set_xlabel("Estimated local effect")
        ax_beta.set_ylabel("Count")
        ax_beta.legend(loc="upper right")
    else:
        ax_beta.axis("off")
        ax_beta.text(
            0.5,
            0.5,
            "No unclipped runs.\nUse a trained policy or a feasible intervention setup.",
            ha="center",
            va="center",
            fontsize=12,
        )

    plot_df = summary.copy()
    plot_df["clipped"] = plot_df["plus_clipped"].fillna(False) | plot_df["minus_clipped"].fillna(False)
    for clipped, color, label in [
        (False, "#2A9D8F", "unclipped"),
        (True, "#E76F51", "clipped"),
    ]:
        subset = plot_df.loc[plot_df["clipped"] == clipped]
        if subset.empty:
            continue
        ax_outcome.scatter(
            subset["DeltaP_minus"],
            subset["DeltaP_plus"],
            color=color,
            alpha=0.85,
            s=50,
            label=label,
        )
    combined = pd.concat([plot_df["DeltaP_minus"], plot_df["DeltaP_plus"]], ignore_index=True)
    if not combined.empty:
        lo = float(combined.min())
        hi = float(combined.max())
        pad = max(0.25, 0.05 * (hi - lo if hi > lo else 1.0))
        ax_outcome.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", color="#6C757D", linewidth=1)
        ax_outcome.set_xlim(lo - pad, hi + pad)
        ax_outcome.set_ylim(lo - pad, hi + pad)
    ax_outcome.set_title("Outcome Response: Minus vs Plus")
    ax_outcome.set_xlabel("DeltaP_minus")
    ax_outcome.set_ylabel("DeltaP_plus")
    ax_outcome.legend(loc="best")

    fig.suptitle("Causal Intervention Report", fontsize=16, fontweight="bold")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_markdown_report(
    input_dir: Path,
    output_dir: Path,
    summary: pd.DataFrame,
    metrics: dict[str, object],
    findings: list[str],
) -> None:
    report_path = output_dir / "report.md"
    warning_counts = metrics["warning_counts"]

    lines = [
        "# Causal Intervention Report",
        "",
        "## Setup",
        "",
        f"- Input directory: `{input_dir}`",
        f"- Runs: `{int(metrics['total_runs'])}`",
        f"- Unclipped usable pairs: `{int(metrics['valid_runs'])}`",
        f"- Any clipping rate: `{100 * float(metrics['any_clip_rate']):.1f}%`",
        f"- Mean beta_action (all runs): `{_safe_float(metrics['mean_beta_action_hat_all']):.6f}`",
    ]

    mean_beta_action_valid = _safe_float(metrics["mean_beta_action_hat_valid"])
    mean_beta_exec_valid = _safe_float(metrics["mean_beta_exec_hat_valid"])
    if not math.isnan(mean_beta_action_valid):
        lines.append(f"- Mean beta_action (unclipped only): `{mean_beta_action_valid:.6f}`")
    if not math.isnan(mean_beta_exec_valid):
        lines.append(f"- Mean beta_exec (unclipped only): `{mean_beta_exec_valid:.6f}`")

    lines.extend(
        [
            "",
            "## Most Important Findings",
            "",
        ]
    )
    lines.extend([f"- {finding}" for finding in findings])

    lines.extend(
        [
            "",
            "## Warnings",
            "",
        ]
    )
    if len(warning_counts) == 0:
        lines.append("- None")
    else:
        for warning, count in warning_counts.items():
            lines.append(f"- `{warning}`: `{int(count)}` runs")

    lines.extend(
        [
            "",
            "## Output",
            "",
            "- Dashboard: `dashboard.png`",
            "- This report: `report.md`",
            "",
            "## Summary Table Preview",
            "",
            "```text",
            summary.head(10).to_string(index=False),
            "```",
            "",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "report"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary, trajectories = load_data(input_dir)
    intervention_df = intervention_rows(trajectories)
    metrics = compute_metrics(summary, intervention_df)
    findings = build_findings(summary, metrics)

    render_dashboard(summary, intervention_df, metrics, findings, output_dir / "dashboard.png")
    render_markdown_report(input_dir, output_dir, summary, metrics, findings)

    print(f"saved dashboard to {output_dir / 'dashboard.png'}")
    print(f"saved markdown report to {output_dir / 'report.md'}")
    print("")
    print("Top findings:")
    for finding in findings[:5]:
        print(f"- {finding}")


if __name__ == "__main__":
    main()
