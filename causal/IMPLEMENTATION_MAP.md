# Causal Implementation Map

## Core simulator modules

- `simulation/market_gym.py`
  - `Market.__init__`: builds the market, execution agent, and optional observation agent.
  - `Market.reset`: creates a fresh `LimitOrderBook`, resets agents, seeds the initial event queue, and runs to the first observation.
  - `Market.step` / `Market.transition`: drives the event queue until the next observation or termination and returns reward/info.
  - Causal hook point: wrap `reset` / `step` rather than rewriting transition logic. This keeps the existing simulator semantics intact.

- `simulation/agents.py`
  - `NoiseAgent.generate_order`: main exogenous order flow sampler. Randomness comes from `self.np_random`.
  - `StrategicAgent.generate_order` and `StrategicAgent.reset`: additional seeded exogenous flow in the strategic environment.
  - `ExecutionAgent.update_position_from_message_list`: updates remaining inventory, active volume, market fills, passive fills, and cumulative reward.
  - `RLAgent.generate_order`: action-conditioned execution logic. The action is a simplex over `[market, limit levels..., inactive]`.
  - `RLAgent.get_observation`: maps internal market state to the observation seen by the policy.
  - Causal hook point: intervene only in the action passed into `RLAgent.generate_order` at a chosen decision index.

- `limit_order_book/limit_order_book.py`
  - `LimitOrderBook.process_order_list`: applies order lists and returns confirmation/fill messages.
  - `LimitOrderBook.data`: stores time stamps, best prices, level-2 volumes, submitted order sizes, and cancellation volumes.
  - Causal hook point: use logged book state plus execution-agent counters to derive `X_t`, `DeltaP_t`, spread, and imbalance proxies.

## Action / policy path

- `rl_files/actor_critic.py`
  - `AgentLogisticNormal.deterministic_action` and `DirichletAgent.deterministic_action`: deterministic rollout helpers for trained RL policies.
  - Causal hook point: the causal layer can load these models without touching the training stack.

## Randomness and seed handling

- Existing simulator randomness is local to agent RNGs:
  - `NoiseAgent`: `np.random.default_rng(seed)`
  - `StrategicAgent`: `np.random.default_rng(seed)`
- Existing training scripts also seed Python / NumPy / torch, but the simulator itself did not fully re-plumb seeds on `reset`.
- Implemented change:
  - `Market.reset(seed=...)` now re-seeds the internal noise/strategic agent RNGs.
  - `Market.__init__` now uses local config copies instead of mutating shared module-level config dicts.
  - Observation-agent `terminal_time` and `time_delta` are now aligned with the requested environment config.

## Causal layer modules

- `causal/sim_wrapper.py`
  - High-level wrapper around `Market`.
  - Replays episodes from seed, computes policy actions, applies one-time overrides, and logs decision-bin records.
  - Uses deterministic replay from `(base_config, seed, action history)` instead of unsafe deep cloning.

- `causal/intervention.py`
  - Defines `InterventionSpec`.
  - Applies `plus` / `minus` perturbations by shifting mass between the selected action component and a slack component, defaulting to `[market] <-> [inactive]`.
  - Preserves simplex admissibility and surfaces clipping diagnostics.

- `causal/feature_extraction.py`
  - Extracts best bid/ask, midprice, spread, depth, and an imbalance proxy from the live LOB.

- `causal/logging_utils.py`
  - Defines the stable `DecisionRecord` schema.
  - Computes `DeltaP_t = midprice(t + horizon) - midprice(t)` on decision bins.

- `causal/counterfactual_runner.py`
  - Runs baseline / plus / minus trajectories with shared seed.
  - Produces the local finite-difference estimate
    - `beta_true_hat = (DeltaP_plus - DeltaP_minus) / (2 * delta)`
  - Surfaces warnings for clipping and reproducibility mismatches before the intervention time.

- `causal/repro_check.py`
  - Runs two identical baseline rollouts and compares logged series exactly.

## Main assumptions

- The causal intervention target is the RL execution agent only. Benchmark agents are left untouched.
- The action component of interest is the immediate market-sell share by default (`action_index=0`).
- The default perturbation preserves simplex mass by borrowing from / returning to the inactive component (`slack_index=-1`).
- `X_t` is defined as signed agent-attributable executed volume within one decision bin:
  - `market sells + passive limit sells - market buys - passive limit buys`
- `DeltaP_t` is defined on decision bins using logged midprices:
  - `midprice(t + horizon) - midprice(t)`

## Deterministic replay risks

- The simulator is event-driven and the intervention changes downstream state, so exact path-matching is only expected before the intervention time.
- Replay coupling is seed-based, not full state cloning. This is more robust than attempting to deep-copy the priority queue, agents, and order book mid-episode.
- If a policy is stochastic at inference time, replay will only be deterministic if the policy adapter is configured for deterministic actions.
- The repo depends on `sortedcontainers`; a small local fallback was added because the active runtime in this workspace did not have that dependency installed.
