"""Paired interventional counterfactual runner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from causal.intervention import InterventionSpec, choose_adaptive_pair, choose_increasing_pair
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


@dataclass
class CurveInterventionResult:
    """Container for one baseline plus multi-level intervention curve experiment."""

    seed: int
    intervention_time: int
    horizon: int
    delta_levels: list[float]
    logs_by_level: dict[float, pd.DataFrame]
    summary_df: pd.DataFrame
    warnings: list[str]


def _select_intervention_time(
    baseline_log: pd.DataFrame,
    intervention_time: int | None,
    horizon: int,
    rng: np.random.Generator | None,
    candidate_times: list[int] | None = None,
    burn_in_steps: int = 0,
) -> int:
    if baseline_log.empty:
        raise ValueError("baseline episode produced no decision rows")
    max_valid = len(baseline_log) - 1
    valid_times = list(range(max_valid + 1)) if candidate_times is None else sorted(set(int(x) for x in candidate_times))
    if intervention_time is not None:
        horizon_ok = horizon <= 0 or int(intervention_time) <= max_valid - int(horizon)
        burn_in_ok = int(intervention_time) >= int(burn_in_steps)
        if intervention_time not in valid_times or not horizon_ok or not burn_in_ok:
            raise ValueError(
                f"intervention_time {intervention_time} outside feasible range after burn-in={burn_in_steps} "
                f"and horizon={horizon}"
            )
        return int(intervention_time)

    rng = np.random.default_rng(0) if rng is None else rng
    feasible_times = [
        t
        for t in valid_times
        if burn_in_steps <= t <= max_valid and (horizon <= 0 or t <= max_valid - int(horizon))
    ]
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


def _future_scripted_actions_from_log(
    baseline_log: pd.DataFrame,
    start_after_t0: int,
) -> dict[int, np.ndarray]:
    scripted: dict[int, np.ndarray] = {}
    for _, row in baseline_log.iterrows():
        decision_index = int(row["decision_index"])
        if decision_index <= int(start_after_t0):
            continue
        action_json = row.get("actual_action")
        if not isinstance(action_json, str) or not action_json:
            continue
        scripted[decision_index] = np.asarray(json.loads(action_json), dtype=np.float64)
    return scripted


def _passes_state_filters(
    row: pd.Series,
    min_inventory_before: float = 0.0,
    min_bid_depth_before: float = 0.0,
    max_spread_before: float | None = None,
    max_abs_imbalance_before: float | None = None,
) -> bool:
    if float(row.get("inventory_before", 0.0)) < float(min_inventory_before):
        return False
    if float(row.get("bid_depth_before", 0.0)) < float(min_bid_depth_before):
        return False
    if max_spread_before is not None and float(row.get("spread_before", np.inf)) > float(max_spread_before):
        return False
    if max_abs_imbalance_before is not None and abs(float(row.get("imbalance_before", 0.0))) > float(max_abs_imbalance_before):
        return False
    return True


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
    burn_in_steps: int = 0,
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
        burn_in_steps=burn_in_steps,
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


def run_intervention_curve(
    base_config: dict,
    seed: int,
    intervention_time: int | None,
    deltas: Iterable[float],
    horizon: int,
    policy=None,
    delta_units: str = "normalized",
    action_index: int = 0,
    slack_index: int = -1,
    intervention_target: str = "market",
    branch_from_state: bool = True,
    scripted_actions: dict[int, np.ndarray] | None = None,
    rng: np.random.Generator | None = None,
    burn_in_steps: int = 0,
    curve_mode: str = "one_sided",
    impulse_mode: bool = False,
    min_inventory_before: float = 0.0,
    min_bid_depth_before: float = 0.0,
    max_spread_before: float | None = None,
    max_abs_imbalance_before: float | None = None,
    placebo: bool = False,
) -> CurveInterventionResult:
    """Estimate an intervention-response curve from a common branched simulator state.

    The curve uses a baseline (delta = 0) and one-sided increases in submitted selling
    intensity at the same intervention time after a burn-in period.
    """
    delta_levels = sorted({float(delta) for delta in deltas})
    if curve_mode not in {"one_sided", "symmetric"}:
        raise ValueError(f"unknown curve_mode: {curve_mode}")
    if curve_mode == "one_sided":
        delta_levels = [delta for delta in delta_levels if delta >= 0.0]
        if not delta_levels:
            raise ValueError("one-sided curve estimation requires at least one non-negative delta level")
    else:
        if not delta_levels:
            raise ValueError("symmetric curve estimation requires at least one delta level")
        if 0.0 not in delta_levels:
            delta_levels = sorted(delta_levels + [0.0])

    max_delta = max(abs(delta) for delta in delta_levels)
    baseline_wrapper = MarketSimulatorWrapper(base_config=base_config, policy=policy)
    baseline_log = baseline_wrapper.run_episode_with_logging(
        seed=seed,
        intervention=None,
        run_label="baseline",
        scripted_actions=scripted_actions,
        horizon=horizon,
    )

    preferred_action_index = 0 if intervention_target == "market" else None
    candidate_times: list[int] = []
    for _, row in baseline_log.iterrows():
        try:
            if int(row["decision_index"]) < int(burn_in_steps):
                continue
            if not _passes_state_filters(
                row,
                min_inventory_before=min_inventory_before,
                min_bid_depth_before=min_bid_depth_before,
                max_spread_before=max_spread_before,
                max_abs_imbalance_before=max_abs_imbalance_before,
            ):
                continue
            baseline_action = np.asarray(json.loads(row["actual_action"]), dtype=np.float64)
            if curve_mode == "symmetric":
                choose_adaptive_pair(
                    action=baseline_action,
                    delta=max_delta,
                    units=delta_units,
                    remaining_inventory=float(row["inventory_before"]),
                    inactive_index=slack_index,
                    preferred_action_index=preferred_action_index,
                )
            else:
                choose_increasing_pair(
                    action=baseline_action,
                    delta=max_delta,
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
        burn_in_steps=burn_in_steps,
    )
    baseline_row = baseline_log.loc[baseline_log["decision_index"] == chosen_t0].iloc[0]
    remaining_inventory = float(baseline_row["inventory_before"])

    baseline_action = np.asarray(json.loads(baseline_row["actual_action"]), dtype=np.float64)
    if curve_mode == "symmetric":
        selected_action_index, selected_slack_index = choose_adaptive_pair(
            action=baseline_action,
            delta=max_delta,
            units=delta_units,
            remaining_inventory=remaining_inventory,
            inactive_index=slack_index,
            preferred_action_index=preferred_action_index,
        )
    else:
        selected_action_index, selected_slack_index = choose_increasing_pair(
            action=baseline_action,
            delta=max_delta,
            units=delta_units,
            remaining_inventory=remaining_inventory,
            inactive_index=slack_index,
            preferred_action_index=preferred_action_index,
        )
    if placebo:
        # Placebo arm: shift mass between two deep passive limit levels.
        # This exercises the full branching / replay machinery but produces
        # no meaningful change in realized market execution, validating
        # the identification strategy.
        n_components = len(baseline_action)
        selected_action_index = max(3, n_components - 3)
        selected_slack_index = max(4, n_components - 2)
        if selected_action_index == selected_slack_index:
            selected_slack_index = selected_action_index - 1
    elif intervention_target == "market":
        selected_action_index = 0
    else:
        selected_action_index = int(action_index)

    future_scripted_actions = None
    if impulse_mode:
        future_scripted_actions = _future_scripted_actions_from_log(baseline_log, start_after_t0=chosen_t0)

    if branch_from_state:
        prefix_wrapper = MarketSimulatorWrapper(base_config=base_config, policy=policy)
        prefix_wrapper.reset(seed=seed, run_label="baseline")
        while prefix_wrapper.get_current_decision_index() < chosen_t0 and not prefix_wrapper.terminated:
            prefix_wrapper.step()
        logs_by_level: dict[float, pd.DataFrame] = {0.0: baseline_log.copy()}
        warnings: list[str] = []
        for delta_level in delta_levels:
            if delta_level == 0.0:
                continue
            level_wrapper = prefix_wrapper.clone_current_state(run_label=f"delta_{delta_level:g}")
            level_spec = InterventionSpec(
                intervention_time=chosen_t0,
                delta=float(abs(delta_level)),
                direction="plus" if float(delta_level) > 0.0 else "minus",
                action_index=selected_action_index,
                slack_index=selected_slack_index,
                units=delta_units,
                target=intervention_target,
            )
            level_wrapper.continue_episode(
                intervention=level_spec,
                run_label=f"delta_{delta_level:g}",
                scripted_actions=future_scripted_actions if impulse_mode else scripted_actions,
            )
            logs_by_level[float(delta_level)] = level_wrapper.get_logged_dataframe(horizon=horizon)
            level_row = logs_by_level[float(delta_level)].loc[
                logs_by_level[float(delta_level)]["decision_index"] == chosen_t0
            ].iloc[0]
            if bool(level_row["clipped"]):
                warnings.append(f"delta_{delta_level:g}_clipped")
    else:
        baseline_branch = MarketSimulatorWrapper(base_config=base_config, policy=policy)
        logs_by_level = {
            0.0: baseline_branch.run_episode_with_logging(
                seed=seed,
                intervention=None,
                run_label="baseline",
                scripted_actions=scripted_actions,
                horizon=horizon,
            )
        }
        warnings = []
        for delta_level in delta_levels:
            if delta_level == 0.0:
                continue
            level_spec = InterventionSpec(
                intervention_time=chosen_t0,
                delta=float(abs(delta_level)),
                direction="plus" if float(delta_level) > 0.0 else "minus",
                action_index=selected_action_index,
                slack_index=selected_slack_index,
                units=delta_units,
                target=intervention_target,
            )
            level_wrapper = MarketSimulatorWrapper(base_config=base_config, policy=policy)
            logs_by_level[float(delta_level)] = level_wrapper.run_episode_with_logging(
                seed=seed,
                intervention=level_spec,
                run_label=f"delta_{delta_level:g}",
                scripted_actions=future_scripted_actions if impulse_mode else scripted_actions,
                horizon=horizon,
            )
            level_row = logs_by_level[float(delta_level)].loc[
                logs_by_level[float(delta_level)]["decision_index"] == chosen_t0
            ].iloc[0]
            if bool(level_row["clipped"]):
                warnings.append(f"delta_{delta_level:g}_clipped")

    baseline_t0_row = logs_by_level[0.0].loc[logs_by_level[0.0]["decision_index"] == chosen_t0].iloc[0]
    summary_rows: list[dict[str, float | int | str | bool | None]] = []
    for delta_level in delta_levels:
        level_log = logs_by_level[float(delta_level)]
        level_row = level_log.loc[level_log["decision_index"] == chosen_t0].iloc[0]
        local_treatment_difference = float(level_row["signed_executed_volume"] - baseline_t0_row["signed_executed_volume"])
        baseline_immediate_midprice = float(
            baseline_t0_row["midprice_after_execution"]
            if pd.notna(baseline_t0_row.get("midprice_after_execution"))
            else baseline_t0_row["midprice_after"]
        )
        level_immediate_midprice = float(
            level_row["midprice_after_execution"]
            if pd.notna(level_row.get("midprice_after_execution"))
            else level_row["midprice_after"]
        )
        local_immediate_level_difference = float(level_immediate_midprice - baseline_immediate_midprice)
        baseline_future_midprice = float(baseline_t0_row["future_midprice"])
        level_future_midprice = float(level_row["future_midprice"])
        local_level_difference = float(level_future_midprice - baseline_future_midprice)
        # With a shared branched pre-intervention state, this coincides with the
        # difference in forward returns. We log it explicitly as a price-level effect
        # to avoid ambiguity in downstream analysis.
        local_outcome_difference = local_level_difference
        beta_exec_hat = np.nan if local_treatment_difference == 0 else local_level_difference / local_treatment_difference
        beta_action_hat = np.nan if float(delta_level) == 0.0 else local_level_difference / float(delta_level)
        beta_exec_immediate_hat = (
            np.nan if local_treatment_difference == 0 else local_immediate_level_difference / local_treatment_difference
        )
        beta_action_immediate_hat = (
            np.nan if float(delta_level) == 0.0 else local_immediate_level_difference / float(delta_level)
        )
        summary_rows.append(
            {
                "seed": int(seed),
                "t0": int(chosen_t0),
                "delta": float(delta_level),
                "delta_units": delta_units,
                "horizon": int(horizon),
                "curve_mode": curve_mode,
                "impulse_mode": bool(impulse_mode),
                "strategic_direction": baseline_t0_row.get("strategic_direction"),
                "selected_action_index": int(selected_action_index),
                "selected_slack_index": int(selected_slack_index),
                "baseline_X": float(baseline_t0_row["signed_executed_volume"]),
                "baseline_immediate_midprice": baseline_immediate_midprice,
                "baseline_DeltaP": float(baseline_t0_row["delta_p_horizon"]),
                "baseline_future_midprice": baseline_future_midprice,
                "inventory_before": float(baseline_t0_row["inventory_before"]),
                "midprice_before": float(baseline_t0_row["midprice_before"]),
                "spread_before": float(baseline_t0_row["spread_before"]),
                "imbalance_before": float(baseline_t0_row["imbalance_before"]),
                "bid_depth_before": float(baseline_t0_row["bid_depth_before"]),
                "ask_depth_before": float(baseline_t0_row["ask_depth_before"]),
                "best_bid_before": float(baseline_t0_row["best_bid_before"]),
                "best_ask_before": float(baseline_t0_row["best_ask_before"]),
                "X_level": float(level_row["signed_executed_volume"]),
                "immediate_midprice_level": level_immediate_midprice,
                "DeltaP_level": float(level_row["delta_p_horizon"]),
                "future_midprice_level": level_future_midprice,
                "planned_market_sell_baseline": float(baseline_t0_row["planned_market_sell_after"]),
                "planned_market_sell_level": float(level_row["planned_market_sell_after"]),
                "local_treatment_difference": local_treatment_difference,
                "local_immediate_level_difference": local_immediate_level_difference,
                "local_level_difference": local_level_difference,
                "local_outcome_difference": local_outcome_difference,
                "beta_exec_immediate_hat": float(beta_exec_immediate_hat),
                "beta_exec_level_hat": float(beta_exec_hat),
                "beta_exec_hat": float(beta_exec_hat),
                "beta_action_immediate_hat": float(beta_action_immediate_hat),
                "beta_action_level_hat": float(beta_action_hat),
                "beta_action_hat": float(beta_action_hat),
                "clipped": bool(level_row["clipped"]),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["delta"]).reset_index(drop=True)
    return CurveInterventionResult(
        seed=int(seed),
        intervention_time=int(chosen_t0),
        horizon=int(horizon),
        delta_levels=delta_levels,
        logs_by_level=logs_by_level,
        summary_df=summary_df,
        warnings=warnings,
    )
