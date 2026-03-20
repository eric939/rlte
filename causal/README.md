# Causal Interventional Layer

This directory implements only the first-stage causal infrastructure:

- simulator inspection and wrappering
- deterministic seed-coupled rollout access
- causal variable extraction
- paired `plus` / `minus` direct interventions for local ground-truth impact estimation

It does **not** implement observational estimators, IV/CIV/NIV/GMM, or any step beyond the direct intervention engine.

## Definitions

- `X_t`
  - Signed agent-attributable executed volume within one decision bin.
  - Implemented as:
    - `market sells + passive limit sells - market buys - passive limit buys`

- `DeltaP_t`
  - Decision-bin midprice response over a configurable horizon.
  - Implemented as:
    - `midprice(t + horizon) - midprice(t)`

- `beta_true_hat`
  - Local finite-difference ground-truth effect estimate:
    - `(DeltaP_plus - DeltaP_minus) / (2 * delta)`

## Main files

- `causal/sim_wrapper.py`
  - High-level wrapper over `simulation.market_gym.Market`.
- `causal/intervention.py`
  - Action perturbation and clipping logic.
- `causal/feature_extraction.py`
  - Stable book-state feature extraction.
- `causal/logging_utils.py`
  - Typed decision-bin logging schema.
- `causal/counterfactual_runner.py`
  - Baseline / plus / minus paired reruns and summary statistics.
- `causal/repro_check.py`
  - Same-seed reproducibility utility.
- `scripts/run_interventional_ground_truth.py`
  - CLI entrypoint for saving summaries and raw logs.

## Reproducibility check

Example:

```bash
/usr/local/bin/python3 - <<'PY'
from causal.policy import InactivePolicy
from causal.repro_check import run_reproducibility_check

base_config = {
    "market_env": "noise",
    "execution_agent": "rl_agent",
    "volume": 20,
    "seed": 0,
    "terminal_time": 150,
    "time_delta": 15,
    "drop_feature": None,
}

result = run_reproducibility_check(
    base_config=base_config,
    seed=123,
    policy=InactivePolicy(action_size=7),
    horizon=1,
)
print(result.exact_match, result.mismatched_fields)
PY
```

## Paired intervention run

Example:

```bash
/usr/local/bin/python3 scripts/run_interventional_ground_truth.py \
  --market-env noise \
  --volume 20 \
  --terminal-time 150 \
  --time-delta 15 \
  --seeds 123,124 \
  --intervention-mode fixed \
  --intervention-time 0 \
  --delta 0.1 \
  --horizon 1 \
  --policy-kind inactive \
  --output-dir outputs/interventional_ground_truth
```

## Current limitations

- Replay is implemented by rerunning from seed, not by cloning a mid-episode simulator state.
- Exact equality is expected before the intervention time, not after it.
- The default intervention targets the market-order action component and compensates via the inactive component.
- The active workspace interpreter was missing `sortedcontainers`, so a small local fallback is used when the external package is unavailable.
