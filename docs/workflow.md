# Workflow

```text
Session Create / Load
  -> Host Run
  -> Research
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

1. Create or load a `QuantResearchSession`.
2. `QuantHost` marks the session running and appends an audit event.
3. Load state and recent experiment memory.
4. Select a research arm.
5. Propose factor ideas.
6. Debate each idea and reject weak candidates before implementation.
7. Implement approved factors as panels.
8. Validate factors with in-sample and out-of-sample metrics.
9. Fuse available panels when multiple panels exist.
10. Convert accepted/watch factors into strategy candidates.
11. Backtest strategy candidates at the portfolio level.
12. Write JSONL, checkpoint, and Markdown memory inside the session directory.
13. Save session status and best factor/strategy results.

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

## CLI

Legacy direct run:

```bash
robin --offline-synthetic --max-iterations 1 --universe-size 24
```

Session-native run:

```bash
robin session create --goal "smoke" --offline-synthetic --no-network
robin session run qrs_xxx --max-iterations 1 --offline-synthetic
robin session inspect qrs_xxx
```
