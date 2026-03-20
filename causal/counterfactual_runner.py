"""Paired interventional counterfactual runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from causal.intervention import InterventionSpec
from causal.sim_wrapper import MarketSimulatorWrapper


@dataclass
class PairedInterventionResult:
    """Container for one baseline / plus / minus coupled experiment."""

    seed: int
    intervention_time: int
    delta: float
    delta_units: str
    horizon: int
    baseline_log: pd.DataFrame
    plus_log: pd.DataFrame
    minus_log: pd.DataFrame
    summary: dict
    warnings: list[str]


def _select_intervention_time(
    baseline_log: pd.DataFrame,
    intervention_time: int | None,
    horizon: int,
    rng: np.random.Generator | None,
) -> int:
    if baseline_log.empty:
        raise ValueError("baseline episode produced no decision rows")
    max_valid = len(baseline_log) - 1
    if intervention_time is not None:
        if not 0 <= intervention_time <= max_valid:
            raise ValueError(f"intervention_time {intervention_time} outside valid range [0, {max_valid}]")
        return int(intervention_time)

    rng = np.random.default_rng(0) if rng is None else rng
    upper = max_valid if horizon <= 0 else max_valid
    return int(rng.integers(0, upper + 1))


def _prefix_mismatch_fields(reference: pd.DataFrame, counterfactual: pd.DataFrame, until: int) -> list[str]:
    key_fields = [
        "clock_time",
        "inventory_before",
        "inventory_after",
        "midprice_before",
        "midprice_after",
        "best_bid_before",
        "best_ask_before",
        "best_bid_after",
        "best_ask_after",
    ]
    mismatches: list[str] = []
    if until <= 0:
        return mismatches
    left = reference.loc[reference["decision_index"] < until, key_fields].reset_index(drop=True)
    right = counterfactual.loc[counterfactual["decision_index"] < until, key_fields].reset_index(drop=True)
    if len(left) != len(right):
        return ["prefix_length"]
    for field in key_fields:
        if not left[field].equals(right[field]):
            mismatches.append(field)
    return mismatches


def run_paired_intervention(
    base_config: dict,
    seed: int,
    intervention_time: int | None,
    delta: float,
    horizon: int,
    policy=None,
    delta_units: str = "normalized",
    action_index: int = 0,
    slack_index: int = -1,
    scripted_actions: dict[int, np.ndarray] | None = None,
    rng: np.random.Generator | None = None,
) -> PairedInterventionResult:
    """Run baseline, plus, and minus episodes with shared seed and shared policy."""
    baseline_wrapper = MarketSimulatorWrapper(base_config=base_config, policy=policy)
    baseline_log = baseline_wrapper.run_episode_with_logging(
        seed=seed,
        intervention=None,
        run_label="baseline",
        scripted_actions=scripted_actions,
        horizon=horizon,
    )

    chosen_t0 = _select_intervention_time(
        baseline_log=baseline_log,
        intervention_time=intervention_time,
        horizon=horizon,
        rng=rng,
    )

    plus_spec = InterventionSpec(
        intervention_time=chosen_t0,
        delta=delta,
        direction="plus",
        action_index=action_index,
        slack_index=slack_index,
        units=delta_units,
    )
    minus_spec = InterventionSpec(
        intervention_time=chosen_t0,
        delta=delta,
        direction="minus",
        action_index=action_index,
        slack_index=slack_index,
        units=delta_units,
    )

    plus_wrapper = MarketSimulatorWrapper(base_config=base_config, policy=policy)
    minus_wrapper = MarketSimulatorWrapper(base_config=base_config, policy=policy)

    plus_log = plus_wrapper.run_episode_with_logging(
        seed=seed,
        intervention=plus_spec,
        run_label="plus",
        scripted_actions=scripted_actions,
        horizon=horizon,
    )
    minus_log = minus_wrapper.run_episode_with_logging(
        seed=seed,
        intervention=minus_spec,
        run_label="minus",
        scripted_actions=scripted_actions,
        horizon=horizon,
    )

    baseline_row = baseline_log.loc[baseline_log["decision_index"] == chosen_t0].iloc[0]
    plus_row = plus_log.loc[plus_log["decision_index"] == chosen_t0].iloc[0]
    minus_row = minus_log.loc[minus_log["decision_index"] == chosen_t0].iloc[0]

    delta_p_plus = float(plus_row["delta_p_horizon"])
    delta_p_minus = float(minus_row["delta_p_horizon"])
    outcome_difference = delta_p_plus - delta_p_minus
    beta_true_hat = np.nan if delta == 0 else outcome_difference / (2.0 * float(delta))
    local_treatment_difference = float(plus_row["signed_executed_volume"] - minus_row["signed_executed_volume"])

    warnings: list[str] = []
    prefix_mismatch_plus = _prefix_mismatch_fields(baseline_log, plus_log, until=chosen_t0)
    prefix_mismatch_minus = _prefix_mismatch_fields(baseline_log, minus_log, until=chosen_t0)
    if prefix_mismatch_plus:
        warnings.append(f"baseline_vs_plus_prefix_mismatch={','.join(prefix_mismatch_plus)}")
    if prefix_mismatch_minus:
        warnings.append(f"baseline_vs_minus_prefix_mismatch={','.join(prefix_mismatch_minus)}")
    if bool(plus_row["clipped"]):
        warnings.append("plus_intervention_clipped")
    if bool(minus_row["clipped"]):
        warnings.append("minus_intervention_clipped")

    summary = {
        "seed": int(seed),
        "t0": int(chosen_t0),
        "delta": float(delta),
        "delta_units": delta_units,
        "horizon": int(horizon),
        "baseline_signed_executed_volume": float(baseline_row["signed_executed_volume"]),
        "X_plus": float(plus_row["signed_executed_volume"]),
        "X_minus": float(minus_row["signed_executed_volume"]),
        "local_treatment_difference": local_treatment_difference,
        "DeltaP_plus": delta_p_plus,
        "DeltaP_minus": delta_p_minus,
        "local_outcome_difference": float(outcome_difference),
        "beta_true_hat": float(beta_true_hat),
        "plus_clipped": bool(plus_row["clipped"]),
        "minus_clipped": bool(minus_row["clipped"]),
    }

    return PairedInterventionResult(
        seed=int(seed),
        intervention_time=int(chosen_t0),
        delta=float(delta),
        delta_units=delta_units,
        horizon=int(horizon),
        baseline_log=baseline_log,
        plus_log=plus_log,
        minus_log=minus_log,
        summary=summary,
        warnings=warnings,
    )
