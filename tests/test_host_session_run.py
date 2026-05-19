from __future__ import annotations

from pathlib import Path

from open_quant_agent.host import HostConfig, QuantHost
from open_quant_agent.sessions import SessionConfig, SessionStatus


def test_host_runs_legacy_loop_inside_session(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    host = QuantHost(HostConfig(sessions_root=tmp_path / "runs/sessions"))
    session = host.create_session(
        "offline session smoke",
        SessionConfig(offline_synthetic=True, no_network=True, max_iterations=1, universe_size=8),
    )

    records = host.run_session(session.session_id)
    loaded = host.sessions.load(session.session_id)
    session_path = host.sessions.path_for(session.session_id)

    assert len(records) == 1
    assert loaded.status == SessionStatus.COMPLETED
    assert loaded.validation_results
    assert (session_path / ".agent_state/checkpoints").is_dir()
    assert (session_path / "multi_agent_experiments.jsonl").exists()
    assert not (tmp_path / "multi_agent_experiments.jsonl").exists()
