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
