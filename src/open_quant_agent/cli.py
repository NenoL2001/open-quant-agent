from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

from open_quant_agent.host import HostConfig, QuantHost
from open_quant_agent.orchestrator import MultiAgentQuantOrchestrator, OrchestratorConfig
from open_quant_agent.sessions import SessionConfig


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
    parser = argparse.ArgumentParser(description="Run Robin, a session-native multi-agent quant research loop.")
    parser.add_argument("--max-iterations", type=int, default=4)
    parser.add_argument("--period", default="3y")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--offline-synthetic", action="store_true", help="Use deterministic synthetic OHLCV data.")
    parser.add_argument("--state-dir", default=".agent_state")
    parser.add_argument("--universe-size", type=int, default=48)
    parser.add_argument("--fusion-method", choices=["ic_weighted", "equal_weight", "stacking"], default="ic_weighted")
    parser.add_argument("--log", default="logs/multi_agent_quant_loop.log")
    return parser.parse_args(argv)


def parse_session_args(argv: list[str]) -> argparse.Namespace:
    sessions_root, cleaned_argv = _extract_sessions_root(argv)
    parser = argparse.ArgumentParser(description="Manage Robin quant research sessions.")
    subparsers = parser.add_subparsers(dest="session_command", required=True)

    create = subparsers.add_parser("create", help="Create a new research session.")
    create.add_argument("--goal", required=True)
    create.add_argument("--offline-synthetic", action="store_true", default=True)
    create.add_argument("--online", action="store_false", dest="offline_synthetic", help="Allow online data mode for later runs.")
    create.add_argument("--no-network", action="store_true", default=True)
    create.add_argument("--allow-network", action="store_false", dest="no_network")
    create.add_argument("--max-iterations", type=int, default=1)
    create.add_argument("--universe-size", type=int, default=48)
    create.add_argument("--period", default="3y")
    create.add_argument("--interval", default="1d")
    create.add_argument("--fusion-method", choices=["ic_weighted", "equal_weight", "stacking"], default="ic_weighted")

    run = subparsers.add_parser("run", help="Run an existing research session.")
    run.add_argument("session_id")
    run.add_argument("--max-iterations", type=int)
    run.add_argument("--offline-synthetic", action="store_true")
    run.add_argument("--online", action="store_true")
    run.add_argument("--universe-size", type=int)
    run.add_argument("--period")
    run.add_argument("--interval")
    run.add_argument("--fusion-method", choices=["ic_weighted", "equal_weight", "stacking"])

    inspect = subparsers.add_parser("inspect", help="Print session.json for a research session.")
    inspect.add_argument("session_id")

    subparsers.add_parser("list", help="List known sessions.")
    args = parser.parse_args(cleaned_argv)
    args.sessions_root = sessions_root
    return args


def _extract_sessions_root(argv: list[str]) -> tuple[str, list[str]]:
    sessions_root = "runs/sessions"
    cleaned: list[str] = []
    index = 0
    while index < len(argv):
        if argv[index] == "--sessions-root":
            if index + 1 >= len(argv):
                raise SystemExit("--sessions-root requires a value")
            sessions_root = argv[index + 1]
            index += 2
            continue
        cleaned.append(argv[index])
        index += 1
    return sessions_root, cleaned


def main(argv: Iterable[str] | None = None) -> None:
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    if argv_list and argv_list[0] == "session":
        handle_session_command(argv_list[1:])
        return

    args = parse_args(argv_list)
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


def handle_session_command(argv: list[str]) -> None:
    args = parse_session_args(argv)
    host = QuantHost(HostConfig(sessions_root=Path(args.sessions_root)))
    if args.session_command == "create":
        session_config = SessionConfig(
            offline_synthetic=bool(args.offline_synthetic),
            no_network=bool(args.no_network),
            max_iterations=args.max_iterations,
            universe_size=args.universe_size,
            period=args.period,
            interval=args.interval,
            fusion_method=args.fusion_method,
        )
        session = host.create_session(args.goal, session_config)
        print(
            json.dumps(
                {
                    "session_id": session.session_id,
                    "path": str(host.sessions.path_for(session.session_id)),
                    "status": session.status.value,
                },
                sort_keys=True,
            )
        )
        return

    if args.session_command == "run":
        session = host.sessions.load(args.session_id)
        if args.max_iterations is not None:
            session.config.max_iterations = args.max_iterations
        if args.offline_synthetic:
            session.config.offline_synthetic = True
            session.config.no_network = True
        if args.online:
            session.config.offline_synthetic = False
            session.config.no_network = False
        if args.universe_size is not None:
            session.config.universe_size = args.universe_size
        if args.period is not None:
            session.config.period = args.period
        if args.interval is not None:
            session.config.interval = args.interval
        if args.fusion_method is not None:
            session.config.fusion_method = args.fusion_method
        host.sessions.save(session)
        records = host.run_session(args.session_id)
        updated = host.sessions.load(args.session_id)
        print(
            json.dumps(
                {
                    "session_id": args.session_id,
                    "records": len(records),
                    "status": updated.status.value,
                },
                sort_keys=True,
            )
        )
        return

    if args.session_command == "inspect":
        print(json.dumps(host.inspect_session(args.session_id), indent=2, sort_keys=True))
        return

    if args.session_command == "list":
        print(json.dumps({"sessions": host.sessions.list_sessions()}, indent=2, sort_keys=True))
        return

    raise SystemExit(f"unknown session command: {args.session_command}")


if __name__ == "__main__":
    main()
