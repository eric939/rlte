#!/usr/bin/env python3
"""Interactive entrypoint for the main causal estimation experiment."""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNNER = PROJECT_ROOT / "scripts" / "run_interventional_ground_truth.py"
FOCUS = PROJECT_ROOT / "scripts" / "render_causal_focus.py"


def _prompt_text(label: str, default: str) -> str:
    raw = input(f"{label} [{default}]: ").strip()
    return raw or default


def _prompt_int(label: str, default: int) -> int:
    return int(_prompt_text(label, str(default)))


def _prompt_float(label: str, default: float) -> float:
    return float(_prompt_text(label, str(default)))


def _prompt_bool(label: str, default: bool) -> bool:
    default_text = "Y/n" if default else "y/N"
    raw = input(f"{label} [{default_text}]: ").strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes", "1", "true"}


def _prompt_horizons(label: str, default: list[int]) -> list[int]:
    raw = _prompt_text(label, ",".join(str(x) for x in default))
    values = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("at least one horizon is required")
    return values


def _run(cmd: list[str]) -> None:
    print("")
    print("running:")
    print(" ".join(shlex.quote(token) for token in cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    print("Main causal estimation experiment")
    print("Default setup: strategic regime, heuristic state-dependent sell policy, adaptive market-volume intervention, branching from the exact t0 state.")
    print("Default paper output: horizon profile with pooled execution-effect estimates, latent-direction splits, and a 1-lot realized-treatment support filter.")
    print("")

    market_env = _prompt_text("Market regime", "strategic")
    volume = _prompt_int("Execution volume", 20)
    seed_start = _prompt_int("Seed start", 100)
    num_episodes = _prompt_int("Number of episodes", 200)
    delta_lots = _prompt_float("Intervention size in lots", 1.0)
    min_abs_treatment_diff = _prompt_float("Minimum |X_plus - X_minus| for primary estimation", 1.0)
    run_profile = _prompt_bool("Run horizon profile instead of main/robust pair?", True)
    if run_profile:
        horizons = _prompt_horizons("Horizons", [1, 2, 3, 5])
        main_horizon = None
        run_robust = False
        robust_horizon = None
    else:
        main_horizon = _prompt_int("Main horizon", 3)
        run_robust = _prompt_bool("Also run robustness horizon?", True)
        robust_horizon = _prompt_int("Robustness horizon", 5) if run_robust else None
    output_root = _prompt_text("Output directory", "outputs/causal_final")

    python_bin = sys.executable
    output_root_path = PROJECT_ROOT / output_root
    main_output = output_root_path / f"{market_env}_main"
    robust_output = output_root_path / f"{market_env}_robust"
    focus_output = output_root_path / "focus"

    common = [
        python_bin,
        str(RUNNER),
        "--market-env",
        market_env,
        "--volume",
        str(volume),
        "--terminal-time",
        "150",
        "--time-delta",
        "15",
        "--seed-start",
        str(seed_start),
        "--num-episodes",
        str(num_episodes),
        "--intervention-mode",
        "random",
        "--delta",
        str(delta_lots),
        "--delta-units",
        "lots",
        "--min-abs-treatment-diff",
        str(min_abs_treatment_diff),
        "--intervention-target",
        "market",
        "--policy-kind",
        "heuristic_sell",
        "--adaptive-intervention",
        "--slack-index",
        "-1",
    ]

    focus_runs: list[str] = []
    if run_profile:
        for horizon in horizons:
            run_output = output_root_path / f"{market_env}_h{horizon}"
            run_cmd = common + ["--horizon", str(horizon), "--output-dir", str(run_output)]
            _run(run_cmd)
            focus_runs.append(f"h{horizon}={run_output}")
    else:
        main_cmd = common + ["--horizon", str(main_horizon), "--output-dir", str(main_output)]
        _run(main_cmd)
        focus_runs.append(f"main={main_output}")
        if run_robust and robust_horizon is not None:
            robust_cmd = common + ["--horizon", str(robust_horizon), "--output-dir", str(robust_output)]
            _run(robust_cmd)
            focus_runs.append(f"robust={robust_output}")

    render_cmd = [python_bin, str(FOCUS), "--runs", *focus_runs, "--output-dir", str(focus_output)]
    _run(render_cmd)

    print("")
    print("final outputs:")
    print(f"- {focus_output / 'focus_results.csv'}")
    print(f"- {focus_output / 'focus_results.tex'}")
    print(f"- {focus_output / 'focus_figure.pdf'}")


if __name__ == "__main__":
    main()
