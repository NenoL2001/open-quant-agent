from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from open_quant_agent.host.config import HostConfig
from open_quant_agent.orchestrator import MultiAgentQuantOrchestrator, OrchestratorConfig
from open_quant_agent.sessions.models import QuantResearchSession, SessionConfig, SessionStatus
from open_quant_agent.sessions.store import SessionStore


class QuantHost:
    """Owns orchestration. Agents propose; the host executes and records."""

    def __init__(self, config: HostConfig | None = None) -> None:
        self.config = config or HostConfig()
        self.sessions = SessionStore(self.config.sessions_root)

    def create_session(self, goal: str, session_config: SessionConfig | None = None) -> QuantResearchSession:
        return self.sessions.create(goal, session_config)

    def run_session(self, session_id: str) -> list[dict[str, Any]]:
        session = self.sessions.load(session_id)
        session.status = SessionStatus.RUNNING
        session.error = None
        self.sessions.save(session)
        self.sessions.append_event(session_id, "session_run_started", {"config": session.config.__dict__})

        try:
            records = self._run_legacy_loop(session)
            session.status = SessionStatus.COMPLETED
            session.validation_results.extend([record.get("best") for record in records if record.get("best")])
            session.strategy_results.extend([record.get("best_strategy") for record in records if record.get("best_strategy")])
            self.sessions.append_event(session_id, "session_run_completed", {"records": len(records)})
            return records
        except Exception as exc:
            session.status = SessionStatus.FAILED
            session.error = {"type": type(exc).__name__, "message": str(exc)}
            self.sessions.append_event(session_id, "session_run_failed", session.error)
            raise
        finally:
            self.sessions.save(session)

    def inspect_session(self, session_id: str) -> dict[str, Any]:
        return self.sessions.load(session_id).to_dict()

    def _run_legacy_loop(self, session: QuantResearchSession) -> list[dict[str, Any]]:
        session_path = self.sessions.path_for(session.session_id)
        config = OrchestratorConfig(
            max_iterations=session.config.max_iterations,
            period=session.config.period,
            interval=session.config.interval,
            offline_synthetic=session.config.offline_synthetic,
            state_dir=session_path / ".agent_state",
            universe_size=session.config.universe_size,
            fusion_method=session.config.fusion_method,
        )
        with _working_directory(session_path):
            return MultiAgentQuantOrchestrator(config).run(session.config.max_iterations)


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    path.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)
