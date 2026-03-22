# Simulator Change Log

This file records simulator-side modifications made on top of the Cheridito--Weiss base implementation in this repository.

The goal is to maintain a clear distinction between:
- the original simulator design inherited from the base codebase
- project-specific modifications introduced for causal market-dynamics research

Future simulator edits should append a new dated entry here.

## 2026-03-22

### Time-varying latent alpha in the strategic regime

Files:
- `config/config.py`
- `simulation/agents.py`
- `simulation/market_gym.py`
- `causal/logging_utils.py`
- `causal/sim_wrapper.py`
- `causal/counterfactual_runner.py`
- `scripts/run_interventional_ground_truth.py`

Changes:
- Replaced the fixed hidden buy/sell direction of `StrategicAgent` with a persistent latent alpha process.
- Added strategic-alpha parameters:
  - `alpha_rho`
  - `alpha_sigma`
  - `alpha_init_scale`
  - `alpha_volume_sensitivity`
- The strategic direction is now determined by the sign of the latent alpha state at each strategic event.
- Strategic market and limit order volumes are now scaled by the magnitude of the latent alpha state.
- The strategic latent state is exposed to the causal layer for ex post diagnostics:
  - `strategic_direction`
  - `strategic_alpha`
- Added CLI support in `scripts/run_interventional_ground_truth.py` to override latent-alpha parameters from experiments.

Motivation:
- The original strategic regime used a constant hidden direction over the full episode.
- For the paper's causal setup, latent alpha should be time-varying and persistent rather than static.
- This change makes confounding and state dependence more realistic and more useful for later CIV/NIV benchmarking.

Verification:
- Direct simulator check confirmed that the latent state moves over time, can switch sign, and changes strategic order aggressiveness.

### Tactical flow/drift feedback in the noise-flow dynamics

Files:
- `config/config.py`
- `simulation/agents.py`
- `simulation/market_gym.py`
- `scripts/run_interventional_ground_truth.py`

Changes:
- Extended `NoiseAgent` so its reaction is no longer driven only by book imbalance.
- Added optional reactions to:
  - recent signed market-order flow
  - recent short-horizon drift
- Added new parameters:
  - `flow_reaction`
  - `flow_factor`
  - `drift_reaction`
  - `drift_factor`
  - `reaction_lookback`
- These richer reactions were implemented for exploratory use and can be activated in custom simulator branches.
- Added CLI support in `scripts/run_interventional_ground_truth.py` for these parameters.

Motivation:
- In the base simulator, tactical response was mainly routed through visible imbalance.
- For the causal market-impact project, indirect impact should also propagate through recent signed flow and recent returns.
- This makes endogenous feedback more realistic and gives a richer dynamic channel for causal effects.

Status:
- **Not part of the default paper setup after rescoping.**
- These changes remain exploratory simulator extensions and are not activated in the default `flow` or `strategic` environments used for the main causal analysis.

### Added a simple liquidity provider / market-maker agent

Files:
- `config/config.py`
- `simulation/agents.py`
- `simulation/market_gym.py`
- `scripts/run_interventional_ground_truth.py`

Changes:
- Added `LiquidityProviderAgent`, a simple scheduled market maker.
- The agent:
  - cancels stale quotes
  - replenishes visible depth on both sides of the book
  - widens quotes under directional pressure
  - skews posted volume away from the pressured side
- Added new configuration block `market_maker_config`.
- Added environment-level overrides:
  - `market_maker_max_volume`
  - `market_maker_time_delta`
  - `market_maker_levels`
  - `market_maker_inventory_skew`
  - `market_maker_widening_sensitivity`
- The liquidity provider was implemented as an exploratory extension rather than adopted into the paper-facing default simulator.

Motivation:
- The base simulator lacked an explicit liquidity provider that adjusts visible depth in response to state.
- A market maker is important for state-dependent impact, endogenous spread/depth, and a more realistic microstructure response to order flow.
- This is a simulator-side change aimed at enriching market dynamics, independent of any RL execution logic.

Status:
- **Not part of the default paper setup after rescoping.**
- The agent implementation is retained as an experimental branch, but it is not included in the main `flow` or `strategic` environments used for the paper results.

### Rescoping decision for the paper-facing simulator

Files:
- `simulation/market_gym.py`
- `scripts/run_interventional_ground_truth.py`

Changes:
- Removed the richer tactical flow/drift feedback and the liquidity provider from the default environment wiring.
- Removed their paper-facing CLI exposure from `scripts/run_interventional_ground_truth.py`.
- Kept the time-varying latent alpha process as the sole simulator modification in the main paper path.

Motivation:
- To preserve the independence of the Cheridito--Weiss simulator as much as possible.
- To ensure that the main simulator modification is directly tied to the methodological need of the paper: genuinely time-varying latent confounding.
- To avoid scope creep and reduce the risk that the simulator is tuned in ways that could be seen as favorable to the proposed causal methodology.
