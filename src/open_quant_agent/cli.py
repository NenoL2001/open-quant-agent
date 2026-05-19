from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

from open_quant_agent.orchestrator import MultiAgentQuantOrchestrator, OrchestratorConfig


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)sZ %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        force=True,
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TradingAgents/RD-Agent inspired multi-agent quant loop.")
    parser.add_argument("--max-iterations", type=int, default=4)
    parser.add_argument("--period", default="3y")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--offline-synthetic", action="store_true", help="Use deterministic synthetic OHLCV data.")
    parser.add_argument("--state-dir", default=".agent_state")
    parser.add_argument("--universe-size", type=int, default=48)
    parser.add_argument("--fusion-method", choices=["ic_weighted", "equal_weight", "stacking"], default="ic_weighted")
    parser.add_argument("--log", default="logs/multi_agent_quant_loop.log")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(Path(args.log))
    config = OrchestratorConfig(
        max_iterations=args.max_iterations,
        period=args.period,
        interval=args.interval,
        offline_synthetic=args.offline_synthetic,
        state_dir=Path(args.state_dir),
        universe_size=args.universe_size,
        fusion_method=args.fusion_method,
    )
    records = MultiAgentQuantOrchestrator(config).run(args.max_iterations)
    summary = {
        "iterations": len(records),
        "last_cycle": records[-1]["cycle"] if records else None,
        "best": records[-1].get("best") if records else None,
        "best_strategy": records[-1].get("best_strategy") if records else None,
        "state_dir": str(config.state_dir),
    }
    logging.info("multi_agent_quant_complete %s", summary)
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
