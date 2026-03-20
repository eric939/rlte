"""Action override logic for direct simulator interventions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from causal.utils import normalize_simplex, to_numpy_action


@dataclass(frozen=True)
class InterventionSpec:
    """Single-time perturbation of an action component."""

    intervention_time: int | None
    delta: float
    direction: str
    action_index: int = 0
    slack_index: int = -1
    units: str = "normalized"
    clip: bool = True

    def applies(self, decision_index: int) -> bool:
        return self.intervention_time is not None and decision_index == self.intervention_time


@dataclass(frozen=True)
class ActionOverride:
    """Result of transforming a proposed action under an intervention."""

    proposed_action: np.ndarray
    actual_action: np.ndarray
    intervened: bool
    clipped: bool
    clip_reason: str | None
    intervention_time: int | None
    requested_delta: float
    realized_delta: float
    direction: str
    action_index: int | None
    slack_index: int | None
    selected_before: float | None
    selected_after: float | None
    slack_before: float | None
    slack_after: float | None
    units: str


def _normalized_delta(delta: float, units: str, remaining_inventory: float | None) -> float:
    if units == "normalized":
        return float(delta)
    if units != "lots":
        raise ValueError(f"unknown intervention units: {units}")
    if remaining_inventory is None or remaining_inventory <= 0:
        return 0.0
    return float(delta) / float(remaining_inventory)


def apply_intervention(
    proposed_action,
    spec: InterventionSpec | None,
    decision_index: int,
    remaining_inventory: float | None = None,
) -> ActionOverride:
    """Shift probability mass between the selected action component and a slack component."""
    action = normalize_simplex(proposed_action)
    if spec is None or not spec.applies(decision_index) or spec.direction == "baseline":
        return ActionOverride(
            proposed_action=action.copy(),
            actual_action=action.copy(),
            intervened=False,
            clipped=False,
            clip_reason=None,
            intervention_time=None if spec is None else spec.intervention_time,
            requested_delta=0.0 if spec is None else float(spec.delta),
            realized_delta=0.0,
            direction="baseline" if spec is None else spec.direction,
            action_index=None if spec is None else int(spec.action_index),
            slack_index=None if spec is None else int(spec.slack_index),
            selected_before=None if spec is None else float(action[spec.action_index]),
            selected_after=None if spec is None else float(action[spec.action_index]),
            slack_before=None if spec is None else float(action[spec.slack_index]),
            slack_after=None if spec is None else float(action[spec.slack_index]),
            units="normalized" if spec is None else spec.units,
        )

    actual = action.copy()
    delta_normalized = _normalized_delta(spec.delta, spec.units, remaining_inventory)
    sign = 1.0 if spec.direction == "plus" else -1.0
    requested_delta = sign * delta_normalized

    action_index = int(spec.action_index)
    slack_index = int(spec.slack_index)
    if slack_index < 0:
        slack_index += len(actual)
    if action_index == slack_index:
        raise ValueError("action_index and slack_index must differ")

    selected_before = float(actual[action_index])
    slack_before = float(actual[slack_index])
    clip_reason = None
    clipped = False

    if requested_delta >= 0:
        transferable = float(actual[slack_index])
        realized_delta = min(requested_delta, transferable)
        if realized_delta < requested_delta:
            clipped = True
            clip_reason = "insufficient_slack_mass"
        actual[action_index] += realized_delta
        actual[slack_index] -= realized_delta
    else:
        transferable = float(actual[action_index])
        realized_delta = max(requested_delta, -transferable)
        if realized_delta > requested_delta:
            clipped = True
            clip_reason = "insufficient_selected_mass"
        actual[action_index] += realized_delta
        actual[slack_index] -= realized_delta

    if not spec.clip and clipped:
        raise ValueError(f"infeasible intervention without clipping: {clip_reason}")

    actual = normalize_simplex(actual)
    selected_after = float(actual[action_index])
    slack_after = float(actual[slack_index])

    if spec.units == "lots" and remaining_inventory is not None:
        realized_delta_units = realized_delta * float(remaining_inventory)
    else:
        realized_delta_units = realized_delta

    return ActionOverride(
        proposed_action=action,
        actual_action=actual,
        intervened=True,
        clipped=clipped,
        clip_reason=clip_reason,
        intervention_time=spec.intervention_time,
        requested_delta=float(spec.delta),
        realized_delta=float(realized_delta_units),
        direction=spec.direction,
        action_index=action_index,
        slack_index=slack_index,
        selected_before=selected_before,
        selected_after=selected_after,
        slack_before=slack_before,
        slack_after=slack_after,
        units=spec.units,
    )
