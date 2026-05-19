from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from open_quant_agent.schemas import ArmStats, ResearchState, utc_now


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def read_jsonl(path: Path, limit: int = 50) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows[-limit:]


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=True, sort_keys=True, default=_json_default) + "\n")


class AgentMemory:
    def __init__(self, state_dir: Path = Path(".agent_state")) -> None:
        self.state_dir = state_dir
        self.state_path = state_dir / "multi_agent_quant_state.json"
        self.checkpoints_dir = state_dir / "checkpoints"
        self.experiments_jsonl = Path("multi_agent_experiments.jsonl")

    def load_state(self, arm_names: list[str]) -> ResearchState:
        if self.state_path.exists():
            try:
                state = ResearchState.from_dict(json.loads(self.state_path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                state = ResearchState()
        else:
            state = ResearchState()
        for arm in arm_names:
            state.arms.setdefault(arm, ArmStats())
        return state

    def save_state(self, state: ResearchState) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    def checkpoint(self, cycle: int, record: dict[str, Any], state: ResearchState) -> Path:
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoints_dir / f"cycle_{cycle:04d}.json"
        payload = {"record": record, "state": state.to_dict()}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
        return path

    def append_experiment(self, record: dict[str, Any]) -> None:
        append_jsonl(self.experiments_jsonl, record)

    def recent_experiments(self, limit: int = 20) -> list[dict[str, Any]]:
        return read_jsonl(self.experiments_jsonl, limit=limit)


def append_markdown(path: Path, heading: str, body: str) -> None:
    if not path.exists():
        path.write_text(f"# {heading}\n\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as file:
        file.write(body.rstrip() + "\n\n")


def record_iteration_markdown(record: dict[str, Any]) -> None:
    best = record.get("best") or {}
    best_line = "none"
    if best:
        metrics = best.get("metrics", {})
        best_line = (
            f"{best.get('factor_name')} ({best.get('kind')}) "
            f"rank_ic={float(metrics.get('rank_ic') or 0.0):.4f}, "
            f"oos_rank_ic={float(metrics.get('oos_rank_ic') or 0.0):.4f}, "
            f"sharpe={float(metrics.get('sharpe') or 0.0):.2f}, "
            f"max_dd={float(metrics.get('max_drawdown_pct') or 0.0):.2f}%"
        )
    best_strategy = record.get("best_strategy") or {}
    strategy_line = "none"
    if best_strategy:
        metrics = best_strategy.get("metrics", {})
        strategy_line = (
            f"{best_strategy.get('strategy_name')} from {best_strategy.get('source_factor')} "
            f"status={best_strategy.get('status')}, "
            f"return={float(metrics.get('return_pct') or 0.0):.2f}%, "
            f"oos_return={float(metrics.get('oos_return_pct') or 0.0):.2f}%, "
            f"oos_excess={float(metrics.get('oos_excess_return_pct') or 0.0):.2f}%, "
            f"oos_sharpe={float(metrics.get('oos_sharpe') or 0.0):.2f}, "
            f"max_dd={float(metrics.get('max_drawdown_pct') or 0.0):.2f}%, "
            f"reward={float(best_strategy.get('reward') or 0.0):.2f}"
        )
    body = (
        f"## Cycle {record.get('cycle')} - {record.get('timestamp', utc_now())}\n\n"
        f"- Selected arm: `{record.get('selected_arm')}`\n"
        f"- Ideas: {', '.join(item.get('name', '') for item in record.get('ideas', []))}\n"
        f"- Best: {best_line}\n"
        f"- Best strategy: {strategy_line}\n"
        f"- Decision: {record.get('supervisor_decision')}\n"
        f"- Checkpoint: `{record.get('checkpoint_path')}`"
    )
    append_markdown(Path("EXPERIMENTS_LOG.md"), "Experiments Log", body)

    result_body = (
        f"## Cycle {record.get('cycle')} Summary\n\n"
        f"- Timestamp: {record.get('timestamp')}\n"
        f"- Best factor: {best_line}\n"
        f"- Best strategy: {strategy_line}\n"
        f"- Status counts: {_status_counts(record.get('validations', []))}\n"
        f"- Strategy status counts: {_status_counts(record.get('strategies', []))}\n"
    )
    append_markdown(Path("RESULTS.md"), "Results", result_body)

    kb_body = (
        f"## Knowledge Update - Cycle {record.get('cycle')}\n\n"
        f"- Best observation: {best_line}\n"
        f"- Supervisor update: {record.get('supervisor_decision')}\n"
        "- Memory policy: retain successful and failed factors with OOS IC, Sharpe, drawdown, and debate notes."
    )
    append_markdown(Path("FACTOR_KNOWLEDGE_BASE.md"), "Factor Knowledge Base", kb_body)

    strategy_kb_body = (
        f"## Strategy Update - Cycle {record.get('cycle')}\n\n"
        f"- Best strategy: {strategy_line}\n"
        f"- Strategy status counts: {_status_counts(record.get('strategies', []))}\n"
        "- Memory policy: retain executable strategy specs with portfolio return, OOS Sharpe, drawdown, turnover, source factor, and promotion status."
    )
    append_markdown(Path("STRATEGY_KNOWLEDGE_BASE.md"), "Strategy Knowledge Base", strategy_kb_body)


def _status_counts(validations: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for validation in validations:
        status = str(validation.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return counts
