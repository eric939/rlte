"""Paired interventional counterfactual runner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from causal.intervention import InterventionSpec, choose_adaptive_pair
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
    candidate_times: list[int] | None = None,
) -> int:
    if baseline_log.empty:
        raise ValueError("baseline episode produced no decision rows")
    max_valid = len(baseline_log) - 1
    valid_times = list(range(max_valid + 1)) if candidate_times is None else sorted(set(int(x) for x in candidate_times))
    if intervention_time is not None:
        if intervention_time not in valid_times:
            raise ValueError(f"intervention_time {intervention_time} outside valid range [0, {max_valid}]")
        return int(intervention_time)

    rng = np.random.default_rng(0) if rng is None else rng
    feasible_times = [t for t in valid_times if 0 <= t <= max_valid and (horizon <= 0 or t <= max_valid - int(horizon))]
    if not feasible_times:
        raise ValueError("no feasible intervention times available")
    return int(rng.choice(feasible_times))


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
    intervention_target: str = "market",
    adaptive_intervention: bool = False,
    branch_from_state: bool = True,
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

    candidate_times = None
    preferred_action_index = 0 if intervention_target == "market" else None
    if adaptive_intervention:
        candidate_times = []
        for _, row in baseline_log.iterrows():
            try:
                baseline_action = np.asarray(json.loads(row["actual_action"]), dtype=np.float64)
                choose_adaptive_pair(
                    action=baseline_action,
                    delta=delta,
                    units=delta_units,
                    remaining_inventory=float(row["inventory_before"]),
                    inactive_index=slack_index,
                    preferred_action_index=preferred_action_index,
                )
                candidate_times.append(int(row["decision_index"]))
            except ValueError:
                continue

    chosen_t0 = _select_intervention_time(
        baseline_log=baseline_log,
        intervention_time=intervention_time,
        horizon=horizon,
        rng=rng,
        candidate_times=candidate_times,
    )
    baseline_row = baseline_log.loc[baseline_log["decision_index"] == chosen_t0].iloc[0]
    remaining_inventory = float(baseline_row["inventory_before"])

    selected_action_index = int(action_index)
    selected_slack_index = int(slack_index)
    if adaptive_intervention:
        baseline_action = np.asarray(json.loads(baseline_row["actual_action"]), dtype=np.float64)
        selected_action_index, selected_slack_index = choose_adaptive_pair(
            action=baseline_action,
            delta=delta,
            units=delta_units,
            remaining_inventory=remaining_inventory,
            inactive_index=slack_index,
            preferred_action_index=preferred_action_index,
        )
    elif intervention_target == "market":
        selected_action_index = 0

    plus_spec = InterventionSpec(
        intervention_time=chosen_t0,
        delta=delta,
        direction="plus",
        action_index=selected_action_index,
        slack_index=selected_slack_index,
        units=delta_units,
        target=intervention_target,
    )
    minus_spec = InterventionSpec(
        intervention_time=chosen_t0,
        delta=delta,
        direction="minus",
        action_index=selected_action_index,
        slack_index=selected_slack_index,
        units=delta_units,
        target=intervention_target,
    )
    if branch_from_state:
        prefix_wrapper = MarketSimulatorWrapper(base_config=base_config, policy=policy)
        prefix_wrapper.reset(seed=seed, run_label="baseline")
        while prefix_wrapper.get_current_decision_index() < chosen_t0 and not prefix_wrapper.terminated:
            prefix_wrapper.step()
        baseline_wrapper = prefix_wrapper.clone_current_state(run_label="baseline")
        plus_wrapper = prefix_wrapper.clone_current_state(run_label="plus")
        minus_wrapper = prefix_wrapper.clone_current_state(run_label="minus")
        baseline_wrapper.continue_episode(intervention=None, run_label="baseline", scripted_actions=scripted_actions)
        plus_wrapper.continue_episode(intervention=plus_spec, run_label="plus", scripted_actions=scripted_actions)
        minus_wrapper.continue_episode(intervention=minus_spec, run_label="minus", scripted_actions=scripted_actions)
        baseline_log = baseline_wrapper.get_logged_dataframe(horizon=horizon)
        plus_log = plus_wrapper.get_logged_dataframe(horizon=horizon)
        minus_log = minus_wrapper.get_logged_dataframe(horizon=horizon)
    else:
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
    local_treatment_difference = float(plus_row["signed_executed_volume"] - minus_row["signed_executed_volume"])
    beta_action_hat = np.nan if delta == 0 else outcome_difference / (2.0 * float(delta))
    beta_exec_hat = np.nan if local_treatment_difference == 0 else outcome_difference / local_treatment_difference

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
        "intervention_target": intervention_target,
        "branch_from_state": bool(branch_from_state),
        "horizon": int(horizon),
        "strategic_direction": baseline_row.get("strategic_direction"),
        "baseline_signed_executed_volume": float(baseline_row["signed_executed_volume"]),
        "baseline_planned_market_sell": float(baseline_row["planned_market_sell_after"]),
        "X_plus": float(plus_row["signed_executed_volume"]),
        "X_minus": float(minus_row["signed_executed_volume"]),
        "planned_market_sell_plus": float(plus_row["planned_market_sell_after"]),
        "planned_market_sell_minus": float(minus_row["planned_market_sell_after"]),
        "local_treatment_difference": local_treatment_difference,
        "DeltaP_plus": delta_p_plus,
        "DeltaP_minus": delta_p_minus,
        "local_outcome_difference": float(outcome_difference),
        "beta_action_hat": float(beta_action_hat),
        "beta_exec_hat": float(beta_exec_hat),
        "beta_true_hat": float(beta_action_hat),
        "selected_action_index": int(selected_action_index),
        "selected_slack_index": int(selected_slack_index),
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
