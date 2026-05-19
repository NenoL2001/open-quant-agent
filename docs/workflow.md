# Workflow

```text
Research
  -> Debate
  -> Implementation
  -> Factor Backtest
  -> Hybrid Fusion
  -> Strategy Construction
  -> Portfolio Backtest
  -> Strategy Promotion
  -> Memory Update
  -> Supervisor Update
  -> Checkpoint
```

## One Cycle

1. Load state and recent experiment memory.
2. Select a research arm.
3. Propose factor ideas.
4. Debate each idea and reject weak candidates before implementation.
5. Implement approved factors as panels.
6. Validate factors with in-sample and out-of-sample metrics.
7. Fuse available panels when multiple panels exist.
8. Convert accepted/watch factors into strategy candidates.
9. Backtest strategy candidates at the portfolio level.
10. Write JSONL, checkpoint, and Markdown memory.

## Research Arms

- `lstm_sequence`
- `transformer_attention`
- `temporal_fusion`
- `hybrid_stack`
- `vol_adjusted_momentum`
- `reversal_quality`

## Statuses

Factor statuses:

- `accepted`
- `watch`
- `rejected`
- `failed`

Strategy statuses:

- `accepted`
- `watch`
- `rejected`
- `failed`

The public default is conservative: a strategy with negative OOS excess remains on `watch` even if its absolute return and Sharpe look strong.
