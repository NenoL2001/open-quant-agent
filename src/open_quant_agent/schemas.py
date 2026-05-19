from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal


FactorKind = Literal["traditional", "dl", "hybrid"]
DLModelType = Literal["lstm", "transformer", "temporal_fusion"]


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def stable_id(payload: dict[str, Any], prefix: str = "") -> str:
    raw = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}{digest}" if prefix else digest


@dataclass(frozen=True)
class FactorIdea:
    name: str
    kind: FactorKind
    family: str
    horizon: int
    hypothesis: str
    source_agent: str
    expr: dict[str, Any] | None = None
    model_type: DLModelType | None = None
    lookback: int = 32
    priority: float = 1.0
    constraints: dict[str, Any] = field(default_factory=dict)

    @property
    def idea_id(self) -> str:
        return stable_id(asdict(self), prefix="idea_")


@dataclass(frozen=True)
class DebateReport:
    idea_id: str
    bullish_case: str
    bearish_case: str
    conservative_case: str
    verdict: Literal["approve", "revise", "reject"]
    score: float
    required_changes: list[str] = field(default_factory=list)


@dataclass
class FactorArtifact:
    idea: FactorIdea
    panel_name: str
    kind: FactorKind
    status: Literal["implemented", "failed"]
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class ValidationResult:
    idea_id: str
    factor_name: str
    kind: FactorKind
    status: Literal["accepted", "watch", "rejected", "failed"]
    metrics: dict[str, Any]
    notes: str = ""

    @property
    def reward(self) -> float:
        rank_ic = float(self.metrics.get("oos_rank_ic") or self.metrics.get("rank_ic") or 0.0)
        sharpe = float(self.metrics.get("oos_sharpe") or self.metrics.get("sharpe") or 0.0)
        spread = float(self.metrics.get("oos_decile_spread_pct") or self.metrics.get("decile_spread_pct") or 0.0)
        dd = abs(float(self.metrics.get("oos_max_drawdown_pct") or self.metrics.get("max_drawdown_pct") or 0.0))
        return rank_ic * 220.0 + sharpe * 8.0 + spread * 0.6 - max(0.0, dd - 25.0) * 0.25


@dataclass(frozen=True)
class StrategyCandidate:
    name: str
    source_factor: str
    construction: Literal["topk_rotation", "long_short_spread"]
    rebalance_bars: int
    top_n: int
    bottom_n: int = 0
    max_gross: float = 1.0
    max_name_weight: float = 0.25
    annual_vol_target: float = 0.18
    min_score_quantile: float = 0.60
    drawdown_reduce: float = -0.10
    drawdown_pause: float = -0.18
    cash_on_negative_cross_section: bool = True
    thesis: str = ""

    @property
    def strategy_id(self) -> str:
        return stable_id(asdict(self), prefix="strategy_")


@dataclass
class StrategyValidationResult:
    strategy_id: str
    strategy_name: str
    source_factor: str
    status: Literal["accepted", "watch", "rejected", "failed"]
    metrics: dict[str, Any]
    spec: dict[str, Any]
    notes: str = ""

    @property
    def reward(self) -> float:
        oos_sharpe = float(self.metrics.get("oos_sharpe") or 0.0)
        oos_return = float(self.metrics.get("oos_return_pct") or 0.0)
        total_excess = float(self.metrics.get("excess_return_pct") or 0.0)
        oos_excess = float(self.metrics.get("oos_excess_return_pct") or 0.0)
        oos_calmar = float(self.metrics.get("oos_calmar") or 0.0)
        max_dd = abs(float(self.metrics.get("max_drawdown_pct") or 0.0))
        turnover = float(self.metrics.get("turnover_pct") or 0.0)
        return (
            oos_sharpe * 18.0
            + oos_return * 0.10
            + oos_excess * 0.45
            + total_excess * 0.05
            + oos_calmar * 5.0
            - max(0.0, max_dd - 30.0) * 0.5
            - turnover * 0.01
        )


@dataclass
class ArmStats:
    pulls: int = 0
    value: float = 0.0
    last_reward: float = 0.0

    def update(self, reward: float) -> None:
        self.pulls += 1
        self.last_reward = reward
        self.value += (reward - self.value) / self.pulls


@dataclass
class ResearchState:
    cycle: int = 0
    arms: dict[str, ArmStats] = field(default_factory=dict)
    best_history: list[dict[str, Any]] = field(default_factory=list)
    strategy_history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle": self.cycle,
            "arms": {name: asdict(stats) for name, stats in self.arms.items()},
            "best_history": self.best_history[-50:],
            "strategy_history": self.strategy_history[-50:],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ResearchState:
        arms = {
            name: ArmStats(
                pulls=int(stats.get("pulls", 0)),
                value=float(stats.get("value", 0.0)),
                last_reward=float(stats.get("last_reward", 0.0)),
            )
            for name, stats in dict(raw.get("arms") or {}).items()
            if isinstance(stats, dict)
        }
        return cls(
            cycle=int(raw.get("cycle", 0)),
            arms=arms,
            best_history=list(raw.get("best_history") or []),
            strategy_history=list(raw.get("strategy_history") or []),
        )


@dataclass
class IterationRecord:
    cycle: int
    timestamp: str
    selected_arm: str
    ideas: list[dict[str, Any]]
    debates: list[dict[str, Any]]
    validations: list[dict[str, Any]]
    best: dict[str, Any] | None
    supervisor_decision: str
    checkpoint_path: str
    strategies: list[dict[str, Any]] = field(default_factory=list)
    best_strategy: dict[str, Any] | None = None
