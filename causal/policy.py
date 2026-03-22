"""Policy adapters for causal rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import gymnasium as gym
import numpy as np
import torch

from causal.utils import to_numpy_action
from simulation.market_gym import Market


class PolicyLike(Protocol):
    """Protocol used by the simulator wrapper."""

    name: str

    def act(self, observation: np.ndarray) -> np.ndarray:
        """Return a 1D action vector."""


@dataclass
class _DummyVecEnvSpec:
    """Minimal object matching the interface expected by the RL policy classes."""

    single_observation_space: gym.Space
    single_action_space: gym.Space


class InactivePolicy:
    """Policy that always keeps all remaining volume inactive."""

    def __init__(self, action_size: int, inactive_index: int = -1) -> None:
        self.name = "inactive_policy"
        self.action_size = int(action_size)
        self.inactive_index = inactive_index if inactive_index >= 0 else action_size + inactive_index

    def act(self, observation: np.ndarray) -> np.ndarray:
        action = np.zeros(self.action_size, dtype=np.float64)
        action[self.inactive_index] = 1.0
        return action


class FixedActionPolicy:
    """Policy that always returns the same normalized action vector."""

    def __init__(self, action: np.ndarray, name: str = "fixed_action_policy") -> None:
        action_array = np.asarray(action, dtype=np.float64).reshape(-1)
        if action_array.size == 0:
            raise ValueError("fixed action policy requires at least one action component")
        if np.any(action_array < 0):
            raise ValueError("fixed action policy requires non-negative action components")
        total = float(action_array.sum())
        if total <= 0:
            raise ValueError("fixed action policy requires action components with positive mass")
        self.name = name
        self.action = action_array / total

    def act(self, observation: np.ndarray) -> np.ndarray:
        return self.action.copy()


class HeuristicSellPolicy:
    """State-dependent sell policy using the RL observation features."""

    def __init__(self, action_size: int, name: str = "heuristic_sell_policy") -> None:
        if action_size < 3:
            raise ValueError("heuristic sell policy requires at least market, one limit level, and inactive")
        self.name = name
        self.action_size = int(action_size)
        self.num_limit_levels = self.action_size - 2

    @staticmethod
    def _clip(value: float, low: float, high: float) -> float:
        return float(np.clip(value, low, high))

    def act(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float64).reshape(-1)
        time_norm = self._clip(obs[0] if obs.size > 0 else 0.0, 0.0, 1.0)
        inventory_norm = self._clip(obs[1] if obs.size > 1 else 1.0, 0.0, 1.0)
        mid_drift = float(obs[3]) if obs.size > 3 else 0.0
        spread = max(float(obs[4]), 0.0) if obs.size > 4 else 0.1
        imbalance = self._clip(obs[5] if obs.size > 5 else 0.0, -1.0, 1.0)

        bid_depth = float(np.mean(obs[6:11])) if obs.size >= 11 else 0.0
        ask_depth = float(np.mean(obs[11:16])) if obs.size >= 16 else 0.0

        schedule_target = max(0.0, 1.0 - time_norm)
        schedule_gap = inventory_norm - schedule_target

        urgency = self._clip(
            0.50 * time_norm
            + 0.90 * max(0.0, schedule_gap)
            + 0.70 * max(0.0, -mid_drift)
            + 0.35 * max(0.0, -imbalance),
            0.0,
            2.0,
        )
        patience = self._clip(
            0.60 * max(0.0, -schedule_gap)
            + 0.45 * max(0.0, mid_drift)
            + 0.30 * max(0.0, imbalance),
            0.0,
            2.0,
        )

        market_score = 0.20 + 1.20 * urgency + 0.20 * max(0.0, ask_depth - bid_depth)
        inactive_score = max(0.05, 0.10 + 0.90 * patience - 0.60 * urgency)

        near_limit_total = max(
            0.10,
            1.20 + 0.80 * max(0.0, imbalance) + 0.25 * bid_depth + 0.20 * spread - 0.70 * urgency,
        )
        deep_limit_total = max(
            0.05,
            0.15 + 0.80 * patience + 0.10 * spread - 0.15 * max(0.0, -imbalance),
        )

        proximity = np.exp(-0.8 * np.arange(self.num_limit_levels, dtype=np.float64))
        proximity = proximity / proximity.sum()
        if self.num_limit_levels == 1:
            depth_bias = np.array([1.0], dtype=np.float64)
        else:
            depth_bias = np.linspace(0.0, 1.0, self.num_limit_levels, dtype=np.float64)
            depth_bias = depth_bias / depth_bias.sum()

        limit_scores = near_limit_total * proximity + deep_limit_total * depth_bias
        scores = np.concatenate(([market_score], limit_scores, [inactive_score])).astype(np.float64)
        scores = np.maximum(scores, 1e-6)
        return scores / scores.sum()


class CallablePolicy:
    """Wrap a plain callable into the policy protocol."""

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray], name: str = "callable_policy") -> None:
        self._fn = fn
        self.name = name

    def act(self, observation: np.ndarray) -> np.ndarray:
        return np.asarray(self._fn(observation), dtype=np.float64)


class TorchPolicyAdapter:
    """Thin adapter over the existing actor-critic policy classes."""

    def __init__(self, model: torch.nn.Module, device: str = "cpu", deterministic: bool = True, name: str = "torch_policy") -> None:
        self.model = model
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.name = name
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_model_path(
        cls,
        base_config: dict,
        model_path: str,
        policy_kind: str = "logistic_normal",
        device: str = "cpu",
        deterministic: bool = True,
    ) -> "TorchPolicyAdapter":
        from rl_files.actor_critic import AgentLogisticNormal, DirichletAgent

        config = dict(base_config)
        config.setdefault("drop_feature", None)
        probe_env = Market(config)
        env_spec = _DummyVecEnvSpec(
            single_observation_space=probe_env.observation_space,
            single_action_space=probe_env.action_space,
        )

        if policy_kind == "dirichlet":
            model = DirichletAgent(env_spec)
            adapter_name = "dirichlet_policy"
        elif policy_kind == "logistic_normal_learn_std":
            model = AgentLogisticNormal(env_spec, variance_scaling=False)
            adapter_name = "logistic_normal_learn_std_policy"
        else:
            model = AgentLogisticNormal(env_spec, variance_scaling=True)
            adapter_name = "logistic_normal_policy"

        state_dict = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(state_dict)
        return cls(model=model, device=device, deterministic=deterministic, name=adapter_name)

    def act(self, observation: np.ndarray) -> np.ndarray:
        obs_tensor = torch.as_tensor(np.asarray(observation, dtype=np.float32), device=self.device).reshape(1, -1)
        with torch.no_grad():
            if self.deterministic and hasattr(self.model, "deterministic_action"):
                action = self.model.deterministic_action(obs_tensor)
            else:
                action, _, _, _ = self.model.get_action_and_value(obs_tensor)
        return to_numpy_action(action.detach().cpu().numpy()[0])
