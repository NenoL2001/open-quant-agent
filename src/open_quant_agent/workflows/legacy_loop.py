from __future__ import annotations

from pathlib import Path
from typing import Any

from open_quant_agent.orchestrator import MultiAgentQuantOrchestrator, OrchestratorConfig
from open_quant_agent.sessions.models import QuantResearchSession


class LegacyLoopWorkflow:
    """Session-compatible wrapper over the current factor-to-strategy loop."""

    def run(self, session: QuantResearchSession, state_dir: Path) -> list[dict[str, Any]]:
        config = OrchestratorConfig(
            max_iterations=session.config.max_iterations,
            period=session.config.period,
            interval=session.config.interval,
            offline_synthetic=session.config.offline_synthetic,
            state_dir=state_dir,
            universe_size=session.config.universe_size,
            fusion_method=session.config.fusion_method,
        )
        return MultiAgentQuantOrchestrator(config).run(session.config.max_iterations)
