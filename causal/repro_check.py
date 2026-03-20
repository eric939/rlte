"""Utilities for rollout reproducibility checks."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from causal.sim_wrapper import MarketSimulatorWrapper


@dataclass
class ReproCheckResult:
    """Structured reproducibility report."""

    exact_match: bool
    mismatched_fields: list[str]
    first_log: pd.DataFrame
    second_log: pd.DataFrame


def compare_logged_dataframes(first: pd.DataFrame, second: pd.DataFrame) -> list[str]:
    """Return field names that differ between two logged rollouts."""
    ignored_fields = {"episode_id"}

    left = first.drop(columns=[column for column in ignored_fields if column in first.columns])
    right = second.drop(columns=[column for column in ignored_fields if column in second.columns])

    if list(left.columns) != list(right.columns):
        return ["columns"]
    if len(left) != len(right):
        return ["length"]

    mismatches: list[str] = []
    for column in left.columns:
        if not left[column].equals(right[column]):
            mismatches.append(column)
    return mismatches


def run_reproducibility_check(base_config: dict, seed: int, policy=None, horizon: int = 1) -> ReproCheckResult:
    """Run two identical episodes and compare the logged trajectories exactly."""
    first_wrapper = MarketSimulatorWrapper(base_config=base_config, policy=policy)
    second_wrapper = MarketSimulatorWrapper(base_config=base_config, policy=policy)

    first_log = first_wrapper.run_episode_with_logging(seed=seed, run_label="baseline", horizon=horizon)
    second_log = second_wrapper.run_episode_with_logging(seed=seed, run_label="baseline", horizon=horizon)

    mismatched_fields = compare_logged_dataframes(first_log, second_log)
    return ReproCheckResult(
        exact_match=len(mismatched_fields) == 0,
        mismatched_fields=mismatched_fields,
        first_log=first_log,
        second_log=second_log,
    )
