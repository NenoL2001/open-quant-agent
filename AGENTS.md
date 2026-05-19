# AGENTS

This repository is Robin, a research-only agentic quant platform.

## Ground Rules

- Preserve the legacy offline synthetic loop.
- Keep all changes testable and incremental.
- Do not add broker, exchange, or live order execution integrations.
- Do not commit `.env`, API keys, credentials, generated logs, or local checkpoints.
- Generated code must not run directly in the core process. Future generated ops must pass spec, tests, sandbox, and review gates before registration.
- Strategy promotion must remain benchmark-relative; high absolute Sharpe alone is not enough.
- Networked tools and crawlers must be disabled by default.

## Required Checks

Before finishing a change:

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src python3 -m open_quant_agent.cli --offline-synthetic --max-iterations 1 --universe-size 8
```

If the change touches sessions or host behavior, also verify:

```bash
PYTHONPATH=src python3 -m open_quant_agent.cli session create --goal "smoke" --offline-synthetic --no-network
PYTHONPATH=src python3 -m open_quant_agent.cli session run <session_id> --max-iterations 1 --offline-synthetic
PYTHONPATH=src python3 -m open_quant_agent.cli session inspect <session_id>
```
