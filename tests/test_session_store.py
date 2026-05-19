from __future__ import annotations

import json
from pathlib import Path

from open_quant_agent.sessions import SessionConfig, SessionStatus, SessionStore


def test_session_create_layout_and_load(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    session = store.create("smoke goal", SessionConfig(universe_size=8))

    session_path = store.path_for(session.session_id)
    assert (session_path / "session.json").exists()
    assert (session_path / "status.json").exists()
    assert (session_path / "events.jsonl").exists()
    assert (session_path / "transcript.jsonl").exists()
    assert (session_path / "checkpoints").is_dir()
    assert (session_path / "artifacts/factor_panels").is_dir()
    assert (session_path / "artifacts/backtests").is_dir()

    event = json.loads((session_path / "events.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert event["event_type"] == "session_created"

    loaded = store.load(session.session_id)
    assert loaded.user_goal == "smoke goal"
    assert loaded.status == SessionStatus.CREATED
    assert loaded.config.universe_size == 8
