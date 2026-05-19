from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from open_quant_agent.memory import append_jsonl
from open_quant_agent.schemas import utc_now
from open_quant_agent.sessions.models import QuantResearchSession, SessionConfig, SessionStatus


class SessionStore:
    def __init__(self, root: Path = Path("runs/sessions")) -> None:
        self.root = root.expanduser().resolve()

    def create(self, user_goal: str, config: SessionConfig | None = None) -> QuantResearchSession:
        session = QuantResearchSession.create(user_goal=user_goal, config=config)
        self.ensure_layout(session.session_id)
        self.save(session)
        self.append_event(session.session_id, "session_created", {"goal": user_goal})
        self.append_transcript(session.session_id, "user", user_goal)
        return session

    def path_for(self, session_id: str) -> Path:
        return self.root / session_id

    def ensure_layout(self, session_id: str) -> None:
        base = self.path_for(session_id)
        for rel in [
            "checkpoints",
            "artifacts/raw_data",
            "artifacts/cleaned_data",
            "artifacts/features",
            "artifacts/factor_panels",
            "artifacts/validations",
            "artifacts/strategies",
            "artifacts/backtests",
            "artifacts/reports",
            "generated_ops",
            "generated_specs",
            "validation",
            "strategy",
            "reports",
        ]:
            (base / rel).mkdir(parents=True, exist_ok=True)

    def load(self, session_id: str) -> QuantResearchSession:
        path = self.path_for(session_id) / "session.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        config_data = data.pop("config", {})
        data["config"] = SessionConfig(**config_data)
        data["status"] = SessionStatus(data.get("status", "created"))
        allowed = {field.name for field in fields(QuantResearchSession)}
        return QuantResearchSession(**{key: value for key, value in data.items() if key in allowed})

    def save(self, session: QuantResearchSession) -> None:
        self.ensure_layout(session.session_id)
        session.touch()
        base = self.path_for(session.session_id)
        payload = session.to_dict()
        (base / "session.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        (base / "status.json").write_text(
            json.dumps(
                {
                    "session_id": session.session_id,
                    "status": session.status.value,
                    "updated_at": session.updated_at,
                    "error": session.error,
                },
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def append_event(self, session_id: str, event_type: str, payload: dict[str, Any]) -> None:
        append_jsonl(
            self.path_for(session_id) / "events.jsonl",
            {
                "timestamp": utc_now(),
                "event_type": event_type,
                "payload": payload,
            },
        )

    def append_transcript(self, session_id: str, role: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        append_jsonl(
            self.path_for(session_id) / "transcript.jsonl",
            {
                "timestamp": utc_now(),
                "role": role,
                "content": content,
                "metadata": metadata or {},
            },
        )

    def list_sessions(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        sessions: list[dict[str, Any]] = []
        for path in sorted(self.root.iterdir()):
            session_path = path / "session.json"
            if not session_path.exists():
                continue
            try:
                raw = json.loads(session_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            sessions.append(
                {
                    "session_id": raw.get("session_id", path.name),
                    "status": raw.get("status"),
                    "goal": raw.get("user_goal"),
                    "updated_at": raw.get("updated_at"),
                    "path": str(path),
                }
            )
        return sessions
