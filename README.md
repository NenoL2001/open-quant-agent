<p align="center">
  <img src="docs/assets/robin-logo.svg" alt="Robin logo: an airy small lion with quant chart elements" width="520">
</p>

# Robin

Robin is a session-native, host-orchestrated agentic quant research platform. It generates factor hypotheses, critiques them, implements math or deep-learning factor panels, validates those panels, converts the best panels into executable portfolio strategy candidates, and promotes only strategies that pass out-of-sample gates.

This repository is a cleaned public extraction and refactor of the multi-agent quant work originally prototyped inside the `autotrade` project.

> Research only. This project is not financial advice and does not place trades.

## What It Does

- Research Agent proposes traditional formula factors and optional PyTorch sequence factors.
- Debate Agent writes bullish, bearish, and conservative validation cases before implementation.
- Implementation Agent compiles safe JSON-style feature expressions or trains compact sequence models.
- Validation Agent evaluates IC, rank IC, OOS rank IC, decile spread, Sharpe, drawdown, coverage, and stability.
- Fusion step combines factor panels with equal-weight, IC-weighted, or stacking logic.
- Strategy Agent builds top-k rotation strategies from accepted/watch factors.
- Portfolio Backtest Agent evaluates OOS return, OOS Sharpe, drawdown, turnover, exposure, and benchmark-relative excess.
- Memory layer writes JSONL experiments, checkpoints, and Markdown knowledge updates.
- Session Host wraps each run in an auditable research session with isolated events, transcripts, checkpoints, and artifacts.

## Install

```bash
git clone <your-fork-url> robin
cd robin
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For deep-learning factors:

```bash
pip install -e ".[dev,dl]"
```

Without PyTorch, DL factor attempts fail gracefully and the loop still evaluates traditional factors.

## Quick Start

Run a deterministic offline cycle:

```bash
robin --offline-synthetic --max-iterations 1 --universe-size 24
```

Or without installing the console script:

```bash
PYTHONPATH=src python3 -m open_quant_agent.cli --offline-synthetic --max-iterations 1
```

Run with yfinance data:

```bash
robin --max-iterations 1 --period 3y --interval 1d
```

The legacy command name remains available:

```bash
open-quant-agent --offline-synthetic --max-iterations 1
```

## Sessions

Create, run, and inspect an isolated research session:

```bash
robin session create --goal "Find robust semiconductor supply-chain alpha" --offline-synthetic --no-network
robin session run qrs_xxx --max-iterations 1 --offline-synthetic
robin session inspect qrs_xxx
```

## Outputs

The legacy loop writes local research artifacts:

- `.agent_state/multi_agent_quant_state.json` - supervisor arm state plus factor and strategy history.
- `.agent_state/checkpoints/cycle_XXXX.json` - full cycle checkpoints.
- `multi_agent_experiments.jsonl` - machine-readable experiment memory.
- `EXPERIMENTS_LOG.md` - cycle summaries.
- `RESULTS.md` - factor and strategy result summaries.
- `FACTOR_KNOWLEDGE_BASE.md` - factor memory.
- `STRATEGY_KNOWLEDGE_BASE.md` - strategy memory.

These files are ignored by git by default.

Session runs write isolated artifacts under:

- `runs/sessions/{session_id}/session.json`
- `runs/sessions/{session_id}/status.json`
- `runs/sessions/{session_id}/events.jsonl`
- `runs/sessions/{session_id}/transcript.jsonl`
- `runs/sessions/{session_id}/.agent_state/checkpoints/`
- `runs/sessions/{session_id}/multi_agent_experiments.jsonl`

## Strategy Promotion

The strategy layer intentionally separates "good absolute backtest" from "production-worthy candidate." A strategy is accepted only when it has positive OOS return, positive OOS excess versus the equal-weight benchmark, controlled drawdown, enough rebalances, and enough exposure.

That means a strategy can show strong OOS Sharpe but remain on `watch` if it still loses to the benchmark.

## Architecture

See:

- [docs/architecture.md](docs/architecture.md)
- [docs/agent_roles.md](docs/agent_roles.md)
- [docs/workflow.md](docs/workflow.md)

## Daemon Mode

For a simple local research daemon:

```bash
scripts/run_daemon.sh
```

Set cadence with:

```bash
ROBIN_SLEEP_SECONDS=900 scripts/run_daemon.sh
```

## Tests

```bash
pytest
```

## License

MIT. See [LICENSE](LICENSE).

## Attribution

Robin is derived from and substantially refactored out of the local `autotrade` research project. See [NOTICE](NOTICE.md).
