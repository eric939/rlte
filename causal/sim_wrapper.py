"""Wrapper layer for deterministic, logged causal simulator rollouts."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from queue import PriorityQueue
from typing import Mapping
from uuid import uuid4

import numpy as np
import pandas as pd

from causal.feature_extraction import book_features_to_dict, extract_book_features
from causal.intervention import ActionOverride, InterventionSpec, apply_intervention, target_volumes_from_action
from causal.logging_utils import DecisionRecord, prepare_action_fields, records_to_dataframe
from causal.policy import InactivePolicy, PolicyLike
from causal.utils import normalize_simplex, set_global_seeds, to_numpy_action
from simulation.market_gym import Market


@dataclass(frozen=True)
class ReplayReadyState:
    """Replay descriptor used instead of deep-cloning simulator state."""

    base_config: dict
    seed: int
    action_history: list[list[float]]


class MarketSimulatorWrapper:
    """High-level causal interface over the existing `Market` gym environment."""

    def __init__(self, base_config: dict, policy: PolicyLike | None = None, depth_levels: int = 5) -> None:
        self.base_config = copy.deepcopy(base_config)
        self.base_config.setdefault("drop_feature", None)
        self.policy = policy
        self.depth_levels = depth_levels

        self.env: Market | None = None
        self.current_observation: np.ndarray | None = None
        self.current_info: dict | None = None
        self.current_seed: int | None = None
        self.current_episode_id: str | None = None
        self.current_run_label: str | None = None
        self.terminated: bool = False
        self.decision_index: int = 0
        self.records: list[DecisionRecord] = []
        self.action_history: list[list[float]] = []

    def _make_env(self, seed: int) -> Market:
        config = copy.deepcopy(self.base_config)
        config["seed"] = int(seed)
        return Market(config)

    def reset(self, seed: int, episode_id: str | None = None, run_label: str = "baseline"):
        """Create a fresh replay-coupled environment for the given seed."""
        set_global_seeds(seed)
        self.env = self._make_env(seed)
        observation, info = self.env.reset(seed=seed)
        if self.policy is None and self.env.execution_agent_id == "rl_agent":
            self.policy = InactivePolicy(self.env.action_space.shape[0])

        self.current_observation = np.asarray(observation, dtype=np.float32)
        self.current_info = dict(info)
        self.current_seed = int(seed)
        self.current_episode_id = episode_id or f"episode_{uuid4().hex[:8]}"
        self.current_run_label = run_label
        self.terminated = bool(info.get("terminated", False))
        self.decision_index = 0
        self.records = []
        self.action_history = []
        return observation, info

    def get_current_decision_index(self) -> int:
        return self.decision_index

    def _require_env(self) -> Market:
        if self.env is None:
            raise RuntimeError("reset() must be called before stepping the simulator")
        return self.env

    def _execution_agent(self):
        env = self._require_env()
        return env.agents[env.execution_agent_id]

    def get_state_snapshot(self) -> dict:
        """Extract the current execution state and book summary."""
        env = self._require_env()
        features = extract_book_features(env.lob, depth_levels=self.depth_levels)
        execution_agent = self._execution_agent()
        snapshot = {
            "time": float(self.current_info["time"]) if self.current_info is not None else float("nan"),
            "inventory": float(execution_agent.volume),
            "active_volume": float(execution_agent.active_volume),
            "market_buys": float(execution_agent.market_buys),
            "market_sells": float(execution_agent.market_sells),
            "limit_buys": float(execution_agent.limit_buys),
            "limit_sells": float(execution_agent.limit_sells),
            "reward_cumulative": float(execution_agent.cummulative_reward),
        }
        strategic_agent = env.agents.get("strategic_agent")
        snapshot["strategic_direction"] = None if strategic_agent is None else getattr(strategic_agent, "direction", None)
        snapshot.update(book_features_to_dict(features, "now"))
        return snapshot

    def clone_or_replay_ready_state(self) -> ReplayReadyState:
        """Return a deterministic replay descriptor instead of unsafe deep cloning."""
        if self.current_seed is None:
            raise RuntimeError("cannot create replay state before reset()")
        return ReplayReadyState(
            base_config=copy.deepcopy(self.base_config),
            seed=int(self.current_seed),
            action_history=copy.deepcopy(self.action_history),
        )

    def clone_current_state(self, run_label: str | None = None, episode_id: str | None = None) -> "MarketSimulatorWrapper":
        """Branch the simulator from the current decision state without replaying from seed."""
        env = self._require_env()
        cloned = copy.copy(self)
        cloned.base_config = copy.deepcopy(self.base_config)
        cloned.policy = self.policy
        cloned.env = copy.copy(env)
        cloned.env.agents = copy.deepcopy(env.agents)
        cloned.env.lob = copy.deepcopy(env.lob)
        cloned.env.pq = PriorityQueue()
        for item in list(env.pq.queue):
            cloned.env.pq.put(copy.deepcopy(item))
        cloned.current_observation = None if self.current_observation is None else np.asarray(self.current_observation, dtype=np.float32).copy()
        cloned.current_info = copy.deepcopy(self.current_info)
        cloned.current_seed = self.current_seed
        cloned.current_episode_id = episode_id or self.current_episode_id or f"episode_{uuid4().hex[:8]}"
        cloned.current_run_label = run_label or self.current_run_label
        cloned.terminated = self.terminated
        cloned.decision_index = self.decision_index
        cloned.records = copy.deepcopy(self.records)
        cloned.action_history = copy.deepcopy(self.action_history)
        return cloned

    def _policy_name(self) -> str:
        return "no_policy" if self.policy is None else getattr(self.policy, "name", self.policy.__class__.__name__)

    def _compute_policy_action(self) -> np.ndarray | None:
        env = self._require_env()
        if env.execution_agent_id != "rl_agent":
            return None
        if self.policy is None:
            raise RuntimeError("RL rollouts require a policy")
        action = self.policy.act(np.asarray(self.current_observation, dtype=np.float32))
        return normalize_simplex(action)

    def _build_record(
        self,
        pre_snapshot: dict,
        post_snapshot: dict,
        reward: float,
        proposed_action: np.ndarray | None,
        override: ActionOverride | None,
    ) -> DecisionRecord:
        env = self._require_env()
        execution_agent = self._execution_agent()
        actual_action = None if override is None else override.actual_action
        if override is None:
            actual_action = proposed_action

        proposed_action_json, actual_action_json = prepare_action_fields(proposed_action, actual_action)
        planned_market_sell_before = None
        planned_market_sell_after = None
        planned_active_sell_before = None
        planned_active_sell_after = None
        remaining_inventory = float(pre_snapshot["inventory"])
        if proposed_action is not None and remaining_inventory > 0:
            planned_before = target_volumes_from_action(normalize_simplex(proposed_action), remaining_inventory)
            planned_market_sell_before = float(planned_before[0])
            planned_active_sell_before = float(planned_before[:-1].sum())
        if actual_action is not None and remaining_inventory > 0:
            planned_after = target_volumes_from_action(normalize_simplex(actual_action), remaining_inventory)
            planned_market_sell_after = float(planned_after[0])
            planned_active_sell_after = float(planned_after[:-1].sum())

        executed_market_order_volume = post_snapshot["market_sells"] - pre_snapshot["market_sells"]
        executed_limit_order_volume = post_snapshot["limit_sells"] - pre_snapshot["limit_sells"]
        executed_market_buy_volume = post_snapshot["market_buys"] - pre_snapshot["market_buys"]
        executed_limit_buy_volume = post_snapshot["limit_buys"] - pre_snapshot["limit_buys"]
        # Paper convention: sells are negative signed execution, buys are positive.
        signed_executed_volume = (
            -executed_market_order_volume
            -executed_limit_order_volume
            + executed_market_buy_volume
            + executed_limit_buy_volume
        )

        if override is None:
            intervened = False
            direction = "baseline"
            delta = 0.0
            delta_units = "normalized"
            intervention_target = "component"
            clipped = False
            clip_reason = None
            selected_action_index = None
            selected_action_before = None
            selected_action_after = None
            slack_action_index = None
            slack_action_before = None
            slack_action_after = None
            requested_delta = 0.0
            realized_delta = 0.0
            intervention_time = None
        else:
            intervened = override.intervened
            direction = override.direction
            delta = override.requested_delta
            delta_units = override.units
            intervention_target = override.target
            clipped = override.clipped
            clip_reason = override.clip_reason
            selected_action_index = override.action_index
            selected_action_before = override.selected_before
            selected_action_after = override.selected_after
            slack_action_index = override.slack_index
            slack_action_before = override.slack_before
            slack_action_after = override.slack_after
            requested_delta = override.requested_delta
            realized_delta = override.realized_delta
            intervention_time = override.intervention_time

        execution_snapshot = getattr(env, "last_execution_snapshot", None) or {}

        return DecisionRecord(
            episode_id=str(self.current_episode_id),
            run_label=str(self.current_run_label),
            seed=int(self.current_seed),
            decision_index=int(self.decision_index),
            clock_time=float(pre_snapshot["time"]),
            next_clock_time=float(post_snapshot["time"]),
            policy_name=self._policy_name(),
            proposed_action=proposed_action_json,
            actual_action=actual_action_json,
            inventory_before=float(pre_snapshot["inventory"]),
            inventory_after=float(post_snapshot["inventory"]),
            active_volume_before=float(pre_snapshot["active_volume"]),
            active_volume_after=float(post_snapshot["active_volume"]),
            executed_market_order_volume=float(executed_market_order_volume),
            executed_limit_order_volume=float(executed_limit_order_volume),
            executed_market_buy_volume=float(executed_market_buy_volume),
            executed_limit_buy_volume=float(executed_limit_buy_volume),
            signed_executed_volume=float(signed_executed_volume),
            reward_step=float(reward),
            reward_cumulative=float(execution_agent.cummulative_reward),
            drift_after=float(self.current_info["drift"]),
            strategic_direction=pre_snapshot.get("strategic_direction"),
            terminated=bool(self.terminated),
            intervened=intervened,
            intervention_time=intervention_time,
            delta=float(delta),
            delta_units=delta_units,
            intervention_target=intervention_target,
            direction=direction,
            clipped=clipped,
            clip_reason=clip_reason,
            selected_action_index=selected_action_index,
            selected_action_before=selected_action_before,
            selected_action_after=selected_action_after,
            slack_action_index=slack_action_index,
            slack_action_before=slack_action_before,
            slack_action_after=slack_action_after,
            requested_delta=float(requested_delta),
            realized_delta=float(realized_delta),
            planned_market_sell_before=planned_market_sell_before,
            planned_market_sell_after=planned_market_sell_after,
            planned_active_sell_before=planned_active_sell_before,
            planned_active_sell_after=planned_active_sell_after,
            best_bid_before=float(pre_snapshot["best_bid_now"]),
            best_ask_before=float(pre_snapshot["best_ask_now"]),
            midprice_before=float(pre_snapshot["midprice_now"]),
            spread_before=float(pre_snapshot["spread_now"]),
            imbalance_before=float(pre_snapshot["imbalance_now"]),
            bid_depth_before=float(pre_snapshot["bid_depth_now"]),
            ask_depth_before=float(pre_snapshot["ask_depth_now"]),
            best_bid_after=float(post_snapshot["best_bid_now"]),
            best_ask_after=float(post_snapshot["best_ask_now"]),
            midprice_after=float(post_snapshot["midprice_now"]),
            best_bid_after_execution=float(execution_snapshot["best_bid_after_execution"]) if "best_bid_after_execution" in execution_snapshot else None,
            best_ask_after_execution=float(execution_snapshot["best_ask_after_execution"]) if "best_ask_after_execution" in execution_snapshot else None,
            midprice_after_execution=float(execution_snapshot["midprice_after_execution"]) if "midprice_after_execution" in execution_snapshot else None,
            spread_after_execution=float(execution_snapshot["spread_after_execution"]) if "spread_after_execution" in execution_snapshot else None,
            imbalance_after_execution=float(execution_snapshot["imbalance_after_execution"]) if "imbalance_after_execution" in execution_snapshot else None,
            bid_depth_after_execution=float(execution_snapshot["bid_depth_after_execution"]) if "bid_depth_after_execution" in execution_snapshot else None,
            ask_depth_after_execution=float(execution_snapshot["ask_depth_after_execution"]) if "ask_depth_after_execution" in execution_snapshot else None,
            spread_after=float(post_snapshot["spread_now"]),
            imbalance_after=float(post_snapshot["imbalance_now"]),
            bid_depth_after=float(post_snapshot["bid_depth_now"]),
            ask_depth_after=float(post_snapshot["ask_depth_now"]),
        )

    def step(
        self,
        action_override=None,
        record: bool = True,
        proposed_action=None,
    ):
        """Advance one decision step, optionally overriding the RL action."""
        env = self._require_env()
        if self.terminated:
            raise RuntimeError("cannot call step() after termination")

        pre_snapshot = self.get_state_snapshot()
        policy_action = to_numpy_action(proposed_action)
        if env.execution_agent_id == "rl_agent":
            if policy_action is None:
                policy_action = self._compute_policy_action()
            policy_action = normalize_simplex(policy_action)
            final_action = policy_action if action_override is None else normalize_simplex(action_override)
            observation, reward, terminated, truncated, info = env.step(final_action.astype(np.float32))
            self.action_history.append([float(x) for x in final_action])
        else:
            final_action = None
            observation, reward, terminated, truncated, info = env.step()

        self.current_observation = np.asarray(observation, dtype=np.float32)
        self.current_info = dict(info)
        self.terminated = bool(terminated)
        post_snapshot = self.get_state_snapshot()

        if record:
            record_override = None
            if env.execution_agent_id == "rl_agent":
                record_override = ActionOverride(
                    proposed_action=policy_action,
                    actual_action=final_action,
                    intervened=action_override is not None,
                    clipped=False,
                    clip_reason=None,
                    intervention_time=None,
                    requested_delta=0.0,
                    realized_delta=0.0,
                    direction=self.current_run_label or "baseline",
                    action_index=None,
                    slack_index=None,
                    selected_before=None,
                    selected_after=None,
                    slack_before=None,
                    slack_after=None,
                    units="normalized",
                    target="component",
                )
            self.records.append(self._build_record(pre_snapshot, post_snapshot, reward, policy_action, record_override))

        self.decision_index += 1
        return observation, reward, terminated, truncated, info

    def continue_episode(
        self,
        intervention: InterventionSpec | None = None,
        run_label: str | None = None,
        scripted_actions: Mapping[int, np.ndarray] | None = None,
    ) -> list[DecisionRecord]:
        """Continue from the current simulator state until termination."""
        if run_label is not None:
            self.current_run_label = run_label
        while not self.terminated:
            env = self._require_env()
            policy_action = self._compute_policy_action() if env.execution_agent_id == "rl_agent" else None
            override_action = None
            override_result = None
            if policy_action is not None:
                if scripted_actions is not None and self.decision_index in scripted_actions:
                    scripted_action = normalize_simplex(scripted_actions[self.decision_index])
                    override_result = ActionOverride(
                        proposed_action=policy_action,
                        actual_action=scripted_action,
                        intervened=True,
                        clipped=False,
                        clip_reason="scripted_action_override",
                        intervention_time=self.decision_index,
                        requested_delta=0.0,
                        realized_delta=0.0,
                        direction=run_label,
                        action_index=None,
                        slack_index=None,
                        selected_before=None,
                        selected_after=None,
                        slack_before=None,
                        slack_after=None,
                        units="normalized",
                        target="component",
                    )
                    override_action = scripted_action
                elif intervention is not None:
                    override_result = apply_intervention(
                        proposed_action=policy_action,
                        spec=intervention,
                        decision_index=self.decision_index,
                        remaining_inventory=self._execution_agent().volume,
                    )
                    override_action = override_result.actual_action if override_result.intervened else None

            pre_snapshot = self.get_state_snapshot()
            if env.execution_agent_id == "rl_agent":
                final_action = policy_action if override_action is None else override_action
                observation, reward, terminated, truncated, info = env.step(final_action.astype(np.float32))
                self.action_history.append([float(x) for x in final_action])
            else:
                observation, reward, terminated, truncated, info = env.step()
                final_action = None

            self.current_observation = np.asarray(observation, dtype=np.float32)
            self.current_info = dict(info)
            self.terminated = bool(terminated)
            post_snapshot = self.get_state_snapshot()
            self.records.append(self._build_record(pre_snapshot, post_snapshot, reward, policy_action, override_result))
            self.decision_index += 1

        return list(self.records)

    def run_episode(
        self,
        seed: int,
        intervention: InterventionSpec | None = None,
        run_label: str = "baseline",
        episode_id: str | None = None,
        scripted_actions: Mapping[int, np.ndarray] | None = None,
    ) -> list[DecisionRecord]:
        """Replay one episode from seed, optionally perturbing exactly one decision time."""
        self.reset(seed=seed, episode_id=episode_id, run_label=run_label)
        return self.continue_episode(intervention=intervention, run_label=run_label, scripted_actions=scripted_actions)

    def run_episode_with_logging(self, **kwargs) -> pd.DataFrame:
        """Convenience helper returning a dataframe directly."""
        horizon = kwargs.pop("horizon", None)
        self.run_episode(**kwargs)
        return self.get_logged_dataframe(horizon=horizon)

    def get_logged_dataframe(self, horizon: int | None = None) -> pd.DataFrame:
        """Return the current episode log as a dataframe."""
        return records_to_dataframe(self.records, horizon=horizon)
