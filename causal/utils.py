"""Small utilities shared by the causal modules."""

from __future__ import annotations

import json
import random
from typing import Iterable

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is present in the project runtime
    torch = None


def set_global_seeds(seed: int) -> None:
    """Best-effort global seeding for reproducible rollouts."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def to_numpy_action(action: Iterable[float] | np.ndarray | None) -> np.ndarray | None:
    """Convert an action-like input to a 1D float64 numpy array."""
    if action is None:
        return None
    array = np.asarray(action, dtype=np.float64).reshape(-1)
    return array


def normalize_simplex(action: Iterable[float] | np.ndarray) -> np.ndarray:
    """Project a non-negative vector back to the simplex by clipping and renormalizing."""
    array = to_numpy_action(action)
    if array is None:
        raise ValueError("cannot normalize a None action")
    clipped = np.clip(array, 0.0, None)
    total = float(np.sum(clipped))
    if total <= 0:
        raise ValueError("action must have positive mass after clipping")
    return clipped / total


def array_to_json(action: Iterable[float] | np.ndarray | None) -> str | None:
    """Serialize an action vector for stable CSV/JSON export."""
    if action is None:
        return None
    array = to_numpy_action(action)
    return json.dumps([float(x) for x in array], separators=(",", ":"))
