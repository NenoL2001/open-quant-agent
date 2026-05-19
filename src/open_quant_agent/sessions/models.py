from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from open_quant_agent.schemas import stable_id, utc_now


class SessionStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SessionConfig:
    offline_synthetic: bool = True
    no_network: bool = True
    max_iterations: int = 1
    universe_size: int = 48
    period: str = "3y"
    interval: str = "1d"
    fusion_method: str = "ic_weighted"
    runtime_name: str = "local_template"
    permission_profile: str = "research_safe"


@dataclass
class QuantResearchSession:
    session_id: str
    user_goal: str
    created_at: str
    updated_at: str
    status: SessionStatus = SessionStatus.CREATED
    config: SessionConfig = field(default_factory=SessionConfig)
    tags: list[str] = field(default_factory=list)
    universe_spec: dict[str, Any] = field(default_factory=dict)
    data_requirements: list[dict[str, Any]] = field(default_factory=list)
    factor_hypotheses: list[dict[str, Any]] = field(default_factory=list)
    planned_ops: list[dict[str, Any]] = field(default_factory=list)
    executed_ops: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    validation_results: list[dict[str, Any]] = field(default_factory=list)
    strategy_results: list[dict[str, Any]] = field(default_factory=list)
    memory_writes: list[dict[str, Any]] = field(default_factory=list)
    error: dict[str, Any] | None = None

    @classmethod
    def create(cls, user_goal: str, config: SessionConfig | None = None) -> "QuantResearchSession":
        now = utc_now()
        session_id = stable_id({"goal": user_goal, "created_at": now}, prefix="qrs_")
        return cls(
            session_id=session_id,
            user_goal=user_goal,
            created_at=now,
            updated_at=now,
            config=config or SessionConfig(),
        )

    def touch(self) -> None:
        self.updated_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload
