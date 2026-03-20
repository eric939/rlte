import pandas as pd

from causal.counterfactual_runner import run_paired_intervention
from causal.policy import InactivePolicy
from causal.repro_check import compare_logged_dataframes, run_reproducibility_check
from causal.sim_wrapper import MarketSimulatorWrapper


BASE_CONFIG = {
    "market_env": "noise",
    "execution_agent": "rl_agent",
    "volume": 20,
    "seed": 0,
    "terminal_time": 150,
    "time_delta": 15,
    "drop_feature": None,
}


def _inactive_policy():
    return InactivePolicy(action_size=7)


def test_same_seed_reproduces_baseline_trajectory():
    result = run_reproducibility_check(base_config=BASE_CONFIG, seed=123, policy=_inactive_policy(), horizon=1)
    assert result.exact_match
    assert result.mismatched_fields == []


def test_intervention_is_logged_on_the_selected_row():
    result = run_paired_intervention(
        base_config=BASE_CONFIG,
        seed=123,
        intervention_time=0,
        delta=0.1,
        horizon=1,
        policy=_inactive_policy(),
    )

    plus_row = result.plus_log.loc[result.plus_log["decision_index"] == 0].iloc[0]
    minus_row = result.minus_log.loc[result.minus_log["decision_index"] == 0].iloc[0]

    assert bool(plus_row["intervened"])
    assert plus_row["direction"] == "plus"
    assert int(plus_row["intervention_time"]) == 0
    assert bool(minus_row["intervened"])
    assert minus_row["direction"] == "minus"


def test_beta_true_hat_is_computed():
    result = run_paired_intervention(
        base_config=BASE_CONFIG,
        seed=123,
        intervention_time=0,
        delta=0.1,
        horizon=1,
        policy=_inactive_policy(),
    )
    assert "beta_true_hat" in result.summary
    assert pd.notna(result.summary["beta_true_hat"])


def test_logging_schema_contains_required_columns():
    wrapper = MarketSimulatorWrapper(base_config=BASE_CONFIG, policy=_inactive_policy())
    log = wrapper.run_episode_with_logging(seed=123, run_label="baseline", horizon=1)
    required_columns = {
        "episode_id",
        "seed",
        "decision_index",
        "clock_time",
        "proposed_action",
        "actual_action",
        "inventory_before",
        "inventory_after",
        "executed_market_order_volume",
        "executed_limit_order_volume",
        "signed_executed_volume",
        "best_bid_before",
        "best_ask_before",
        "midprice_before",
        "spread_before",
        "imbalance_before",
        "reward_step",
        "run_label",
        "intervened",
        "intervention_time",
        "delta",
        "direction",
        "delta_p_horizon",
    }
    assert required_columns.issubset(set(log.columns))


def test_no_intervention_path_matches_baseline_behavior():
    wrapper_one = MarketSimulatorWrapper(base_config=BASE_CONFIG, policy=_inactive_policy())
    wrapper_two = MarketSimulatorWrapper(base_config=BASE_CONFIG, policy=_inactive_policy())
    baseline_one = wrapper_one.run_episode_with_logging(seed=123, run_label="baseline", horizon=1)
    baseline_two = wrapper_two.run_episode_with_logging(seed=123, run_label="baseline", horizon=1)
    assert compare_logged_dataframes(baseline_one, baseline_two) == []
