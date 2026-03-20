"""Feature extraction helpers for causal rollout logging."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class BookFeatures:
    """Compact state summary at a decision time."""

    best_bid: float
    best_ask: float
    midprice: float
    spread: float
    imbalance: float
    bid_depth: float
    ask_depth: float


def _safe_scalar(value: float) -> float:
    return float(value) if value is not None else float("nan")


def compute_midprice(best_bid: float, best_ask: float) -> float:
    """Return the midprice when both best prices are defined."""
    if np.isnan(best_bid) or np.isnan(best_ask):
        return float("nan")
    return float((best_bid + best_ask) / 2.0)


def compute_spread(best_bid: float, best_ask: float) -> float:
    """Return the quoted spread."""
    if np.isnan(best_bid) or np.isnan(best_ask):
        return float("nan")
    return float(best_ask - best_bid)


def compute_imbalance(bid_volumes: np.ndarray, ask_volumes: np.ndarray) -> float:
    """Use first-levels volume imbalance as a simple robust proxy."""
    total = float(np.sum(bid_volumes) + np.sum(ask_volumes))
    if total <= 0:
        return 0.0
    return float((np.sum(bid_volumes) - np.sum(ask_volumes)) / total)


def extract_book_features(lob, depth_levels: int = 5) -> BookFeatures:
    """Map current LOB state to a stable set of causal analysis features."""
    best_bid = _safe_scalar(lob.get_best_price("bid"))
    best_ask = _safe_scalar(lob.get_best_price("ask"))

    if lob.data.bid_volumes:
        bid_volumes = np.asarray(lob.data.bid_volumes[-1][:depth_levels], dtype=np.float64)
        ask_volumes = np.asarray(lob.data.ask_volumes[-1][:depth_levels], dtype=np.float64)
    else:
        bid_volumes = np.zeros(depth_levels, dtype=np.float64)
        ask_volumes = np.zeros(depth_levels, dtype=np.float64)

    # Replace NaN-only empty-book snapshots with zeros for the imbalance proxy.
    bid_volumes = np.nan_to_num(bid_volumes, nan=0.0)
    ask_volumes = np.nan_to_num(ask_volumes, nan=0.0)

    return BookFeatures(
        best_bid=best_bid,
        best_ask=best_ask,
        midprice=compute_midprice(best_bid, best_ask),
        spread=compute_spread(best_bid, best_ask),
        imbalance=compute_imbalance(bid_volumes, ask_volumes),
        bid_depth=float(np.sum(bid_volumes)),
        ask_depth=float(np.sum(ask_volumes)),
    )


def book_features_to_dict(features: BookFeatures, suffix: str) -> dict[str, float]:
    """Flatten feature dataclass fields with a suffix for before/after logging."""
    return {f"{key}_{suffix}": value for key, value in asdict(features).items()}
