from __future__ import annotations

import json
from pathlib import Path

from open_quant_agent.cli import main


def test_legacy_cli_still_works(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    main(
        [
            "--offline-synthetic",
            "--max-iterations",
            "1",
            "--universe-size",
            "8",
            "--state-dir",
            str(tmp_path / ".agent_state"),
            "--log",
            str(tmp_path / "logs/legacy.log"),
        ]
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["iterations"] == 1
    assert summary["last_cycle"] == 1
    assert summary["best"] is not None
