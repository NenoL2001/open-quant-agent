from __future__ import annotations

import json
from pathlib import Path

from open_quant_agent.cli import main


def test_session_cli_create_run_inspect(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    sessions_root = tmp_path / "runs/sessions"

    main(
        [
            "session",
            "--sessions-root",
            str(sessions_root),
            "create",
            "--goal",
            "cli session smoke",
            "--offline-synthetic",
            "--no-network",
            "--universe-size",
            "8",
        ]
    )
    created = json.loads(capsys.readouterr().out)
    session_id = created["session_id"]
    assert created["status"] == "created"

    main(
        [
            "session",
            "--sessions-root",
            str(sessions_root),
            "run",
            session_id,
            "--max-iterations",
            "1",
            "--offline-synthetic",
        ]
    )
    run_result = json.loads(capsys.readouterr().out)
    assert run_result == {"records": 1, "session_id": session_id, "status": "completed"}

    main(["session", "--sessions-root", str(sessions_root), "inspect", session_id])
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["session_id"] == session_id
    assert inspected["status"] == "completed"
    assert inspected["strategy_results"]
