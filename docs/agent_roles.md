# Agent Roles

Robin's current runtime is deterministic and local-template based. These roles are structured boundaries in the research loop and future extension points for external runtimes.

## Factor Research Agent

Proposes traditional and optional deep-learning factor hypotheses. The current implementation uses deterministic templates and recent experiment memory, making it runnable without an external LLM.

## Debate Agent

Adds a structured critique for every idea:

- bullish case,
- bearish case,
- conservative validation gate,
- verdict,
- required changes.

## Implementation Agent

Turns approved ideas into factor panels:

- Traditional ideas compile through `feature_dsl.py`.
- DL ideas use `dl_factors.py` when PyTorch is installed.

## Factor Validation Agent

Evaluates each factor panel with:

- IC and rank IC,
- OOS IC and OOS rank IC,
- decile spread,
- long-short Sharpe,
- drawdown,
- coverage,
- stability.

## Strategy Construction Agent

Turns accepted/watch factors into executable portfolio strategy candidates, mainly top-k rotations with rebalance, gross exposure, volatility target, and drawdown controls.

## Portfolio Backtest Agent

Backtests strategy candidates and measures:

- total and OOS return,
- OOS Sharpe,
- max drawdown,
- turnover,
- exposure,
- benchmark-relative excess.

## Strategy Promotion Agent

Selects the best portfolio strategy candidate. It marks strategies as `accepted`, `watch`, `rejected`, or `failed`.

## Supervisor

Selects the next research arm with a UCB-style reward policy and persists arm history.
