from __future__ import annotations

from pathlib import Path

from open_quant_agent import MultiAgentQuantOrchestrator, OrchestratorConfig
from open_quant_agent.data import synthetic_ohlcv_frames


def test_synthetic_data_shape() -> None:
    frames = synthetic_ohlcv_frames(universe_size=8, periods=120)
    assert "SPY" in frames
    assert frames
    for frame in frames.values():
        assert {"Open", "High", "Low", "Close", "Volume"}.issubset(frame.columns)


def test_one_offline_cycle(tmp_path: Path) -> None:
    config = OrchestratorConfig(
        max_iterations=1,
        offline_synthetic=True,
        universe_size=12,
        state_dir=tmp_path / ".agent_state",
    )
    records = MultiAgentQuantOrchestrator(config).run()
    assert len(records) == 1
    record = records[0]
    assert record["cycle"] == 1
    assert record["validations"]
    assert "strategies" in record
