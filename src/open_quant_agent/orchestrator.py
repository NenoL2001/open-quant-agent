from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from open_quant_agent.agents import (
    BacktestValidationAgent,
    DebateCritiqueAgent,
    FactorResearchAgent,
    ImplementationAgent,
    SupervisorAgent,
)
from open_quant_agent.data import (
    all_download_tickers,
    baseline_benchmarks,
    download_ohlcv,
    expanded_liquid_universe,
    filter_usable_frames,
    synthetic_ohlcv_frames,
)
from open_quant_agent.hybrid_factors import fuse_factor_panels
from open_quant_agent.memory import AgentMemory, record_iteration_markdown
from open_quant_agent.schemas import (
    FactorArtifact,
    FactorIdea,
    IterationRecord,
    StrategyValidationResult,
    ValidationResult,
    utc_now,
)
from open_quant_agent.strategy_agents import (
    PortfolioBacktestAgent,
    StrategyConstructionAgent,
    StrategyPromotionAgent,
)


ARM_NAMES = [
    "lstm_sequence",
    "transformer_attention",
    "temporal_fusion",
    "hybrid_stack",
    "vol_adjusted_momentum",
    "reversal_quality",
]


@dataclass(frozen=True)
class OrchestratorConfig:
    max_iterations: int = 4
    period: str = "3y"
    interval: str = "1d"
    offline_synthetic: bool = False
    state_dir: Path = Path(".agent_state")
    universe_size: int = 48
    fusion_method: str = "ic_weighted"


class MultiAgentQuantOrchestrator:
    """Research -> Debate -> Implementation -> Backtest -> Feedback loop."""

    def __init__(self, config: OrchestratorConfig | None = None) -> None:
        self.config = config or OrchestratorConfig()
        self.memory = AgentMemory(self.config.state_dir)
        self.research_agent = FactorResearchAgent()
        self.debate_agent = DebateCritiqueAgent()
        self.implementation_agent = ImplementationAgent()
        self.validation_agent = BacktestValidationAgent()
        self.strategy_construction_agent = StrategyConstructionAgent()
        self.portfolio_backtest_agent = PortfolioBacktestAgent()
        self.strategy_promotion_agent = StrategyPromotionAgent()
        self.supervisor = SupervisorAgent(ARM_NAMES)

    def run(self, max_iterations: int | None = None) -> list[dict[str, Any]]:
        iterations = max_iterations or self.config.max_iterations
        state = self.memory.load_state(ARM_NAMES)
        frames = self._load_frames()
        universe = self._research_universe(frames)
        records: list[dict[str, Any]] = []
        for _ in range(iterations):
            state.cycle += 1
            selected_arm = self.supervisor.select_arm(state)
            recent = self.memory.recent_experiments(limit=20)
            ideas = self.research_agent.propose(selected_arm, state.cycle, recent)
            debates = [self.debate_agent.debate(idea, recent) for idea in ideas]
            approved = [idea for idea, debate in zip(ideas, debates, strict=False) if debate.verdict != "reject"]

            panels: dict[str, pd.DataFrame] = {}
            artifacts: list[FactorArtifact] = []
            validations: list[ValidationResult] = []
            for idea in approved:
                artifact, panel = self.implementation_agent.implement(idea, frames, universe)
                artifacts.append(artifact)
                if not panel.empty:
                    panels[artifact.panel_name] = panel
                validations.append(self.validation_agent.validate(artifact, panel, frames, universe))

            fusion_result = self._validate_fusion(selected_arm, panels, frames, universe, ideas)
            if fusion_result is not None:
                fusion_artifact, fused_panel, fusion_validation = fusion_result
                artifacts.append(fusion_artifact)
                if not fused_panel.empty:
                    panels[fusion_artifact.panel_name] = fused_panel
                validations.append(fusion_validation)

            best = self._best_validation(validations)
            decision = self.supervisor.update(state, selected_arm, best)
            strategy_results = self._validate_strategies(validations, panels, frames, universe, selected_arm)
            best_strategy = self.strategy_promotion_agent.select_best(strategy_results)
            strategy_decision = self.strategy_promotion_agent.decision(best_strategy)
            if best_strategy is not None:
                state.strategy_history.append(
                    {
                        "cycle": state.cycle,
                        "strategy": best_strategy.strategy_name,
                        "source_factor": best_strategy.source_factor,
                        "reward": best_strategy.reward,
                        "status": best_strategy.status,
                        "return_pct": best_strategy.metrics.get("return_pct"),
                        "oos_return_pct": best_strategy.metrics.get("oos_return_pct"),
                        "oos_sharpe": best_strategy.metrics.get("oos_sharpe"),
                        "max_drawdown_pct": best_strategy.metrics.get("max_drawdown_pct"),
                    }
                )
            decision = f"{decision} {strategy_decision}"
            record = self._record(
                state.cycle,
                selected_arm,
                ideas,
                debates,
                artifacts,
                validations,
                best,
                strategy_results,
                best_strategy,
                decision,
            )
            checkpoint = self.memory.checkpoint(state.cycle, record, state)
            record["checkpoint_path"] = str(checkpoint)
            checkpoint.write_text(json.dumps({"record": record, "state": state.to_dict()}, indent=2, sort_keys=True), encoding="utf-8")
            self.memory.save_state(state)
            self.memory.append_experiment(record)
            record_iteration_markdown(record)
            records.append(record)
            logging.info(
                "multi_agent_cycle=%s arm=%s best=%s best_strategy=%s",
                state.cycle,
                selected_arm,
                record.get("best"),
                record.get("best_strategy"),
            )
        return records

    def _validate_fusion(
        self,
        selected_arm: str,
        panels: dict[str, pd.DataFrame],
        frames: dict[str, pd.DataFrame],
        universe: list[str],
        ideas: list[FactorIdea],
    ) -> tuple[FactorArtifact, pd.DataFrame, ValidationResult] | None:
        if len(panels) < 2:
            return None
        horizon = max(idea.horizon for idea in ideas) if ideas else 21
        fused_panel, metadata = fuse_factor_panels(
            panels,
            frames,
            universe,
            horizon=horizon,
            method="stacking" if selected_arm == "hybrid_stack" else self.config.fusion_method,
        )
        idea = FactorIdea(
            name=f"hybrid_fused_{selected_arm}",
            kind="hybrid",
            family="factor_fusion",
            horizon=horizon,
            source_agent="ImplementationAgent",
            hypothesis="Fuse approved math and DL factor panels with IC-aware weights to improve robustness.",
            constraints={"fusion": metadata},
        )
        artifact = FactorArtifact(
            idea=idea,
            panel_name=idea.name,
            kind="hybrid",
            status="implemented" if not fused_panel.empty else "failed",
            metadata=metadata,
            error=None if not fused_panel.empty else metadata.get("status", "fusion_failed"),
        )
        return artifact, fused_panel, self.validation_agent.validate(artifact, fused_panel, frames, universe)

    def _validate_strategies(
        self,
        validations: list[ValidationResult],
        panels: dict[str, pd.DataFrame],
        frames: dict[str, pd.DataFrame],
        universe: list[str],
        selected_arm: str,
    ) -> list[StrategyValidationResult]:
        candidates = self.strategy_construction_agent.construct(validations, panels, selected_arm)
        results: list[StrategyValidationResult] = []
        for candidate in candidates:
            panel = panels.get(candidate.source_factor)
            if panel is None:
                continue
            results.append(self.portfolio_backtest_agent.validate(candidate, panel, frames, universe))
        return results

    def _best_validation(self, validations: list[ValidationResult]) -> ValidationResult | None:
        candidates = [item for item in validations if item.status in {"accepted", "watch"}]
        if not candidates:
            return None
        candidates.sort(key=lambda item: item.reward, reverse=True)
        return candidates[0]

    def _record(
        self,
        cycle: int,
        selected_arm: str,
        ideas: list[FactorIdea],
        debates,
        artifacts: list[FactorArtifact],
        validations: list[ValidationResult],
        best: ValidationResult | None,
        strategies: list[StrategyValidationResult],
        best_strategy: StrategyValidationResult | None,
        decision: str,
    ) -> dict[str, Any]:
        record = IterationRecord(
            cycle=cycle,
            timestamp=utc_now(),
            selected_arm=selected_arm,
            ideas=[asdict(idea) | {"idea_id": idea.idea_id} for idea in ideas],
            debates=[asdict(debate) for debate in debates],
            validations=[asdict(validation) | {"reward": validation.reward} for validation in validations],
            best=(asdict(best) | {"reward": best.reward}) if best is not None else None,
            supervisor_decision=decision,
            checkpoint_path="pending",
            strategies=[asdict(strategy) | {"reward": strategy.reward} for strategy in strategies],
            best_strategy=(asdict(best_strategy) | {"reward": best_strategy.reward}) if best_strategy is not None else None,
        )
        payload = asdict(record)
        payload["artifacts"] = [
            {
                "idea_id": artifact.idea.idea_id,
                "panel_name": artifact.panel_name,
                "kind": artifact.kind,
                "status": artifact.status,
                "metadata": artifact.metadata,
                "error": artifact.error,
            }
            for artifact in artifacts
        ]
        return payload

    def _load_frames(self) -> dict[str, pd.DataFrame]:
        if self.config.offline_synthetic:
            return synthetic_ohlcv_frames(self.config.universe_size)
        try:
            frames = filter_usable_frames(download_ohlcv(all_download_tickers(), period=self.config.period, interval=self.config.interval))
            missing = [ticker for ticker in baseline_benchmarks() if ticker not in frames]
            if missing:
                frames.update(filter_usable_frames(download_ohlcv(missing, period=self.config.period, interval=self.config.interval)))
            if frames:
                return frames
        except Exception:
            logging.exception("download_failed_falling_back_to_synthetic")
        return synthetic_ohlcv_frames(self.config.universe_size)

    def _research_universe(self, frames: dict[str, pd.DataFrame]) -> list[str]:
        preferred = [ticker for ticker in expanded_liquid_universe() if ticker in frames and ticker not in {"SPY", "QQQ", "SMH", "SOXX"}]
        if not preferred:
            preferred = [ticker for ticker in frames if ticker not in {"SPY", "QQQ", "SMH", "SOXX"}]
        return preferred[: self.config.universe_size]
