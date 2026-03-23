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
    target: str = "component"
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
    target: str


def _normalized_delta(delta: float, units: str, remaining_inventory: float | None) -> float:
    if units == "normalized":
        return float(delta)
    if units != "lots":
        raise ValueError(f"unknown intervention units: {units}")
    if remaining_inventory is None or remaining_inventory <= 0:
        return 0.0
    return float(delta) / float(remaining_inventory)


def target_volumes_from_action(action: np.ndarray, remaining_inventory: float) -> np.ndarray:
    """Mirror RLAgent's action-to-target-volume conversion for exact lot interventions."""
    inventory = int(round(float(remaining_inventory)))
    target_volumes = np.zeros(len(action), dtype=np.int64)
    available_volume = inventory
    for idx in range(len(action)):
        volume_on_level = min(int(np.round(float(action[idx]) * inventory)), available_volume)
        target_volumes[idx] = volume_on_level
        available_volume -= volume_on_level
    target_volumes[-1] += available_volume
    return target_volumes


def choose_adaptive_pair(
    action,
    delta: float,
    units: str,
    remaining_inventory: float | None,
    inactive_index: int = -1,
    preferred_action_index: int | None = None,
) -> tuple[int, int]:
    """Choose a feasible action/slack pair for symmetric plus/minus interventions.

    The policy uses the current baseline action and remaining inventory to find a
    pair of components with enough mass for both directions. We prefer moving
    volume from a less aggressive component into a more aggressive one.
    """
    action_array = normalize_simplex(action)
    n = len(action_array)
    inactive_idx = inactive_index if inactive_index >= 0 else n + inactive_index

    if units == "lots":
        if remaining_inventory is None or remaining_inventory <= 0:
            raise ValueError("adaptive lot-based intervention requires positive remaining inventory")
        component_mass = target_volumes_from_action(action_array, remaining_inventory).astype(np.float64)
        threshold = max(1.0, float(delta))
    else:
        component_mass = action_array
        threshold = float(delta)

    if threshold <= 0:
        raise ValueError("adaptive intervention requires positive delta")

    priorities = -np.arange(n, dtype=np.float64)
    priorities[inactive_idx] = -float(n)

    best_pair: tuple[float, tuple[int, int]] | None = None
    candidate_action_indices = range(n) if preferred_action_index is None else [int(preferred_action_index)]
    for action_index in candidate_action_indices:
        for slack_index in range(n):
            if action_index == slack_index:
                continue
            if component_mass[action_index] < threshold:
                continue
            if component_mass[slack_index] < threshold:
                continue
            aggressiveness_gap = priorities[action_index] - priorities[slack_index]
            inactive_bonus = 0.25 if slack_index == inactive_idx else 0.0
            tie_break = component_mass[action_index] + component_mass[slack_index]
            score = aggressiveness_gap + inactive_bonus + 1e-3 * tie_break
            if best_pair is None or score > best_pair[0]:
                best_pair = (score, (action_index, slack_index))

    if best_pair is None:
        raise ValueError("could not find a feasible adaptive intervention pair")
    return best_pair[1]


def choose_increasing_pair(
    action,
    delta: float,
    units: str,
    remaining_inventory: float | None,
    inactive_index: int = -1,
    preferred_action_index: int | None = None,
) -> tuple[int, int]:
    """Choose a feasible pair for one-sided baseline-to-higher-intensity interventions.

    For curve estimation we increase selling intensity from a common baseline state.
    This only requires enough slack mass on the donor component; unlike the symmetric
    plus/minus case, the selected component does not need enough mass to move back.
    """
    action_array = normalize_simplex(action)
    n = len(action_array)
    inactive_idx = inactive_index if inactive_index >= 0 else n + inactive_index

    if units == "lots":
        if remaining_inventory is None or remaining_inventory <= 0:
            raise ValueError("lot-based curve intervention requires positive remaining inventory")
        component_mass = target_volumes_from_action(action_array, remaining_inventory).astype(np.float64)
        threshold = max(1.0, float(delta))
    else:
        component_mass = action_array
        threshold = float(delta)

    if threshold <= 0:
        raise ValueError("curve intervention requires positive delta")

    priorities = -np.arange(n, dtype=np.float64)
    priorities[inactive_idx] = -float(n)

    best_pair: tuple[float, tuple[int, int]] | None = None
    candidate_action_indices = range(n) if preferred_action_index is None else [int(preferred_action_index)]
    for action_index in candidate_action_indices:
        for slack_index in range(n):
            if action_index == slack_index:
                continue
            if component_mass[slack_index] < threshold:
                continue
            inactive_bonus = 0.25 if slack_index == inactive_idx else 0.0
            tie_break = component_mass[slack_index] - priorities[slack_index]
            score = inactive_bonus + 1e-3 * tie_break
            if best_pair is None or score > best_pair[0]:
                best_pair = (score, (action_index, slack_index))

    if best_pair is None:
        raise ValueError("could not find a feasible one-sided intervention pair")
    return best_pair[1]


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
            target="component" if spec is None else spec.target,
        )

    actual = action.copy()
    sign = 1.0 if spec.direction == "plus" else -1.0

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

    if spec.units == "lots" and remaining_inventory is not None and remaining_inventory > 0:
        target_volumes = target_volumes_from_action(actual, float(remaining_inventory))
        requested_delta = sign * float(spec.delta)
        if requested_delta >= 0:
            transferable = int(target_volumes[slack_index])
            realized_delta_lots = min(int(round(requested_delta)), transferable)
            if realized_delta_lots < requested_delta:
                clipped = True
                clip_reason = "insufficient_slack_lots"
            target_volumes[action_index] += realized_delta_lots
            target_volumes[slack_index] -= realized_delta_lots
        else:
            transferable = int(target_volumes[action_index])
            realized_delta_lots = max(int(round(requested_delta)), -transferable)
            if realized_delta_lots > requested_delta:
                clipped = True
                clip_reason = "insufficient_selected_lots"
            target_volumes[action_index] += realized_delta_lots
            target_volumes[slack_index] -= realized_delta_lots

        actual = normalize_simplex(target_volumes.astype(np.float64) / float(remaining_inventory))
        realized_delta_units = float(realized_delta_lots)
        realized_delta = float(realized_delta_lots) / float(remaining_inventory)
    else:
        delta_normalized = _normalized_delta(spec.delta, spec.units, remaining_inventory)
        requested_delta = sign * delta_normalized

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

        actual = normalize_simplex(actual)
        selected_after = float(actual[action_index])
        slack_after = float(actual[slack_index])

        if spec.units == "lots" and remaining_inventory is not None:
            realized_delta_units = realized_delta * float(remaining_inventory)
        else:
            realized_delta_units = realized_delta

    if not spec.clip and clipped:
        raise ValueError(f"infeasible intervention without clipping: {clip_reason}")
    selected_after = float(actual[action_index])
    slack_after = float(actual[slack_index])

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
        target=spec.target,
    )
