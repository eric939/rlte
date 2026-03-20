"""Structured logging primitives for causal simulator rollouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import pandas as pd

from causal.utils import array_to_json


@dataclass
class DecisionRecord:
    """One decision-bin observation for baseline or counterfactual runs."""

    episode_id: str
    run_label: str
    seed: int
    decision_index: int
    clock_time: float
    next_clock_time: float
    policy_name: str
    proposed_action: str | None
    actual_action: str | None
    inventory_before: float
    inventory_after: float
    active_volume_before: float
    active_volume_after: float
    executed_market_order_volume: float
    executed_limit_order_volume: float
    executed_market_buy_volume: float
    executed_limit_buy_volume: float
    signed_executed_volume: float
    reward_step: float
    reward_cumulative: float
    drift_after: float
    terminated: bool
    intervened: bool
    intervention_time: int | None
    delta: float
    delta_units: str
    direction: str
    clipped: bool
    clip_reason: str | None
    selected_action_index: int | None
    selected_action_before: float | None
    selected_action_after: float | None
    slack_action_index: int | None
    slack_action_before: float | None
    slack_action_after: float | None
    requested_delta: float
    realized_delta: float
    best_bid_before: float
    best_ask_before: float
    midprice_before: float
    spread_before: float
    imbalance_before: float
    bid_depth_before: float
    ask_depth_before: float
    best_bid_after: float
    best_ask_after: float
    midprice_after: float
    spread_after: float
    imbalance_after: float
    bid_depth_after: float
    ask_depth_after: float
    future_midprice: float | None = None
    delta_p_horizon: float | None = None
    horizon: int | None = None


def records_to_dataframe(records: Sequence[DecisionRecord], horizon: int | None = None) -> pd.DataFrame:
    """Convert records to a stable pandas export format."""
    rows = [asdict(record) for record in records]
    df = pd.DataFrame(rows)
    if horizon is not None:
        df = annotate_horizon_outcomes(df, horizon=horizon)
    return df


def annotate_horizon_outcomes(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Add DeltaP_t = midprice(t+horizon) - midprice(t) using decision-bin midprices."""
    if df.empty:
        annotated = df.copy()
        annotated["future_midprice"] = []
        annotated["delta_p_horizon"] = []
        annotated["horizon"] = []
        return annotated

    annotated = df.copy()
    decision_midprices = annotated["midprice_before"].tolist()
    decision_midprices.append(float(annotated["midprice_after"].iloc[-1]))

    future_midprices = []
    delta_ps = []
    last_index = len(decision_midprices) - 1
    for idx in range(len(annotated)):
        future_idx = min(idx + horizon, last_index)
        future_midprice = float(decision_midprices[future_idx])
        current_midprice = float(decision_midprices[idx])
        future_midprices.append(future_midprice)
        delta_ps.append(future_midprice - current_midprice)

    annotated["future_midprice"] = future_midprices
    annotated["delta_p_horizon"] = delta_ps
    annotated["horizon"] = int(horizon)
    return annotated


def prepare_action_fields(proposed_action, actual_action) -> tuple[str | None, str | None]:
    """Serialize action vectors for logging."""
    return array_to_json(proposed_action), array_to_json(actual_action)
