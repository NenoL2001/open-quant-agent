from __future__ import annotations

import math
from dataclasses import asdict, replace
from typing import Any

import pandas as pd

from open_quant_agent.feature_dsl import (
    FeatureSpec,
    build_feature_panel,
    normalize_feature_spec,
)
from open_quant_agent.dl_factors import DLFactorConfig, build_dl_factor_panel
from open_quant_agent.hybrid_factors import evaluate_factor_panel
from open_quant_agent.schemas import DebateReport, FactorArtifact, FactorIdea, ValidationResult


class FactorResearchAgent:
    """RD-Agent style hypothesis generator for math and deep learning factors."""

    def propose(self, selected_arm: str, cycle: int, recent_memory: list[dict[str, Any]]) -> list[FactorIdea]:
        suffix = f"{cycle}_{selected_arm}"
        if selected_arm == "lstm_sequence":
            ideas = [
                FactorIdea(
                    name=f"lstm_return_embedding_{suffix}",
                    kind="dl",
                    family="sequence_momentum",
                    horizon=21,
                    model_type="lstm",
                    lookback=32,
                    source_agent="FactorResearchAgent",
                    hypothesis="A shared LSTM can learn nonlinear return/volatility state transitions and rank assets by expected 21-day return.",
                ),
                self._residual_momentum(suffix),
            ]
        elif selected_arm == "transformer_attention":
            ideas = [
                FactorIdea(
                    name=f"transformer_cross_asset_proxy_{suffix}",
                    kind="dl",
                    family="attention_momentum",
                    horizon=21,
                    model_type="transformer",
                    lookback=40,
                    source_agent="FactorResearchAgent",
                    hypothesis="Temporal self-attention should detect regime-dependent continuation and reversal patterns missed by fixed windows.",
                ),
                self._volume_breakout(suffix),
            ]
        elif selected_arm == "temporal_fusion":
            ideas = [
                FactorIdea(
                    name=f"temporal_fusion_regime_{suffix}",
                    kind="dl",
                    family="regime_fusion",
                    horizon=21,
                    model_type="temporal_fusion",
                    lookback=48,
                    source_agent="FactorResearchAgent",
                    hypothesis="Gated temporal fusion can combine slow trend context with recent downside volatility to improve robustness.",
                ),
                self._drawdown_recovery(suffix),
            ]
        elif selected_arm == "reversal_quality":
            ideas = [self._drawdown_recovery(suffix), self._short_reversal(suffix)]
        elif selected_arm == "vol_adjusted_momentum":
            ideas = [self._vol_adjusted_momentum(suffix), self._residual_momentum(suffix)]
        else:
            ideas = [
                FactorIdea(
                    name=f"hybrid_lstm_math_stack_{suffix}",
                    kind="dl",
                    family="hybrid_stack_seed",
                    horizon=21,
                    model_type="lstm",
                    lookback=36,
                    source_agent="FactorResearchAgent",
                    hypothesis="Use a DL sequence factor as the nonlinear leg and fuse it with accepted math factors using IC-weighted stacking.",
                ),
                self._vol_adjusted_momentum(suffix),
                self._drawdown_recovery(suffix),
            ]
        return self._inject_memory(ideas, recent_memory)

    def _inject_memory(self, ideas: list[FactorIdea], recent_memory: list[dict[str, Any]]) -> list[FactorIdea]:
        if not recent_memory:
            return ideas
        best = next((row.get("best") for row in reversed(recent_memory) if row.get("best")), None)
        if not isinstance(best, dict):
            return ideas
        factor_name = str(best.get("factor_name") or "unknown")
        status = str(best.get("status") or "unknown")
        metrics = best.get("metrics") if isinstance(best.get("metrics"), dict) else {}
        memory_hint = (
            f" Memory prior: recent best `{factor_name}` was {status} with "
            f"oos_rank_ic={float(metrics.get('oos_rank_ic') or 0.0):.4f}; preserve useful signal diversity and avoid repeating weak failure modes."
        )
        injected = []
        for idea in ideas:
            constraints = dict(idea.constraints)
            constraints["memory_prior_factor"] = factor_name
            injected.append(replace(idea, hypothesis=(idea.hypothesis + memory_hint)[:900], constraints=constraints))
        return injected

    def _vol_adjusted_momentum(self, suffix: str) -> FactorIdea:
        return FactorIdea(
            name=f"math_vol_adjusted_momentum_{suffix}",
            kind="traditional",
            family="momentum",
            horizon=21,
            expr={
                "op": "div",
                "left": {"op": "roc", "window": 63},
                "right": {"op": "atr_pct", "window": 21},
            },
            source_agent="FactorResearchAgent",
            hypothesis="Medium-term momentum scaled by ATR should retain trend persistence while penalizing fragile high-volatility winners.",
        )

    def _residual_momentum(self, suffix: str) -> FactorIdea:
        return FactorIdea(
            name=f"math_beta_residual_{suffix}",
            kind="traditional",
            family="relative_strength",
            horizon=21,
            expr={"op": "beta_residual", "window": 126, "benchmark": "SPY"},
            source_agent="FactorResearchAgent",
            hypothesis="Positive residual performance after removing market beta should isolate idiosyncratic leadership.",
        )

    def _volume_breakout(self, suffix: str) -> FactorIdea:
        return FactorIdea(
            name=f"math_volume_breakout_{suffix}",
            kind="traditional",
            family="volume",
            horizon=21,
            expr={
                "op": "add",
                "left": {"op": "donchian_position", "window": 55},
                "right": {"op": "volume_z", "window": 30},
            },
            source_agent="FactorResearchAgent",
            hypothesis="Range breakouts confirmed by abnormal volume should have more durable follow-through.",
        )

    def _drawdown_recovery(self, suffix: str) -> FactorIdea:
        return FactorIdea(
            name=f"math_drawdown_recovery_{suffix}",
            kind="traditional",
            family="reversal",
            horizon=21,
            expr={
                "op": "sub",
                "left": {"op": "price_position", "window": 63},
                "right": {"op": "drawdown", "window": 63},
            },
            source_agent="FactorResearchAgent",
            hypothesis="Assets recovering within their recent range after controlled drawdowns may combine reversal and trend continuation.",
        )

    def _short_reversal(self, suffix: str) -> FactorIdea:
        return FactorIdea(
            name=f"math_short_reversal_{suffix}",
            kind="traditional",
            family="reversal",
            horizon=10,
            expr={
                "op": "sub",
                "left": {"op": "rsi", "window": 5},
                "right": {"op": "roc", "window": 5},
            },
            source_agent="FactorResearchAgent",
            hypothesis="Short-term overextension should partially mean-revert when ranked cross-sectionally.",
            constraints={"direction": -1},
        )


class DebateCritiqueAgent:
    """TradingAgents style optimistic vs conservative critique."""

    def debate(self, idea: FactorIdea, recent_memory: list[dict[str, Any]]) -> DebateReport:
        bullish = (
            f"{idea.name} has a clear predictive thesis and is diverse versus plain price momentum."
            if idea.kind == "traditional"
            else f"{idea.name} can learn nonlinear temporal interactions that the current formula DSL cannot express."
        )
        bearish = (
            "The factor may be redundant with existing momentum features and needs correlation and OOS IC checks."
            if idea.kind == "traditional"
            else "The DL factor can overfit short histories, so it must be trained on an early split and judged on OOS dates only."
        )
        conservative = "Approve only if coverage is broad, OOS rank IC is positive, drawdown is controlled, and the hybrid stack improves over the math baseline."
        score = 0.72 if idea.kind == "dl" else 0.66
        if "short_reversal" in idea.name:
            score -= 0.08
        verdict = "approve" if score >= 0.58 else "revise"
        changes = []
        if idea.kind == "dl":
            changes.append("Use chronological train/OOS split and rank-normalize predictions cross-sectionally.")
        else:
            changes.append("Shift/normalize through existing FeatureSpec DSL to avoid lookahead.")
        return DebateReport(
            idea_id=idea.idea_id,
            bullish_case=bullish,
            bearish_case=bearish,
            conservative_case=conservative,
            verdict=verdict,
            score=score,
            required_changes=changes,
        )


class ImplementationAgent:
    """Compiles approved factor ideas into panels compatible with validation."""

    def implement(
        self,
        idea: FactorIdea,
        frames: dict[str, pd.DataFrame],
        universe: list[str],
    ) -> tuple[FactorArtifact, pd.DataFrame]:
        try:
            if idea.kind == "traditional":
                spec = self._feature_spec(idea)
                panel = build_feature_panel(spec, frames, universe)
                artifact = FactorArtifact(
                    idea=idea,
                    panel_name=spec.name,
                    kind=idea.kind,
                    status="implemented",
                    metadata={"feature_spec": asdict(spec), "implementation": "open_quant_agent.feature_dsl.build_feature_panel"},
                )
                return artifact, panel
            if idea.kind == "dl":
                config = DLFactorConfig(
                    model_type=str(idea.model_type),
                    lookback=idea.lookback,
                    horizon=idea.horizon,
                    seed=int(idea.idea_id.rsplit("_", 1)[-1][:8], 16) % 10000,
                )
                panel, metadata = build_dl_factor_panel(frames, universe, config)
                artifact = FactorArtifact(
                    idea=idea,
                    panel_name=idea.name,
                    kind=idea.kind,
                    status="implemented" if not panel.empty else "failed",
                    metadata={"dl_config": metadata, "implementation": "open_quant_agent.dl_factors"},
                    error=None if not panel.empty else str(metadata.get("status")),
                )
                return artifact, panel
        except Exception as exc:
            return (
                FactorArtifact(
                    idea=idea,
                    panel_name=idea.name,
                    kind=idea.kind,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                ),
                pd.DataFrame(),
            )
        return (
            FactorArtifact(idea=idea, panel_name=idea.name, kind=idea.kind, status="failed", error="unsupported factor kind"),
            pd.DataFrame(),
        )

    def _feature_spec(self, idea: FactorIdea) -> FeatureSpec:
        raw = {
            "name": idea.name,
            "family": idea.family,
            "expr": idea.expr or {"op": "roc", "window": 21},
            "normalize": "cross_sectional_zscore",
            "direction": int(idea.constraints.get("direction", 1)),
            "horizon": idea.horizon,
            "thesis": idea.hypothesis,
        }
        spec, reasons = normalize_feature_spec(raw, "multi_agent_research")
        if spec is None:
            raise ValueError("; ".join(reasons))
        return spec


class BacktestValidationAgent:
    """Evaluates factor panels through IC, spread, OOS, Sharpe, and drawdown checks."""

    def validate(
        self,
        artifact: FactorArtifact,
        panel: pd.DataFrame,
        frames: dict[str, pd.DataFrame],
        universe: list[str],
    ) -> ValidationResult:
        if artifact.status == "failed" or panel.empty:
            return ValidationResult(
                idea_id=artifact.idea.idea_id,
                factor_name=artifact.panel_name,
                kind=artifact.kind,
                status="failed",
                metrics={"status_reason": artifact.error or "empty panel"},
                notes="Implementation failed before validation.",
            )
        metrics = evaluate_factor_panel(panel, frames, universe, artifact.idea.horizon)
        status = self._status(metrics)
        notes = (
            f"rank_ic={metrics.get('rank_ic', 0.0):.4f}, "
            f"oos_rank_ic={metrics.get('oos_rank_ic', 0.0):.4f}, "
            f"sharpe={metrics.get('sharpe', 0.0):.2f}, "
            f"max_dd={metrics.get('max_drawdown_pct', 0.0):.2f}%"
        )
        return ValidationResult(
            idea_id=artifact.idea.idea_id,
            factor_name=artifact.panel_name,
            kind=artifact.kind,
            status=status,
            metrics=metrics,
            notes=notes,
        )

    def _status(self, metrics: dict[str, Any]) -> str:
        if metrics.get("status_reason") != "ok":
            return "failed"
        oos_rank_ic = float(metrics.get("oos_rank_ic") or 0.0)
        rank_ic = float(metrics.get("rank_ic") or 0.0)
        coverage = float(metrics.get("coverage_pct") or 0.0)
        sharpe = float(metrics.get("sharpe") or 0.0)
        max_dd = float(metrics.get("max_drawdown_pct") or 0.0)
        if coverage >= 20.0 and oos_rank_ic > 0.005 and sharpe > 0.15 and max_dd >= -35.0:
            return "accepted"
        if coverage >= 10.0 and (oos_rank_ic > 0.0 or rank_ic > 0.0):
            return "watch"
        return "rejected"


class SupervisorAgent:
    """Adaptive controller with a simple UCB-style arm selector."""

    def __init__(self, arm_names: list[str]) -> None:
        self.arm_names = arm_names

    def select_arm(self, state) -> str:
        for arm in self.arm_names:
            if state.arms[arm].pulls == 0:
                return arm
        total = sum(stats.pulls for stats in state.arms.values()) + 1
        scored = []
        for arm in self.arm_names:
            stats = state.arms[arm]
            bonus = math.sqrt(2.0 * math.log(total) / max(stats.pulls, 1))
            scored.append((stats.value + bonus, arm))
        scored.sort(reverse=True)
        return scored[0][1]

    def update(self, state, selected_arm: str, best: ValidationResult | None) -> str:
        reward = best.reward if best is not None else -5.0
        state.arms[selected_arm].update(reward)
        if best is None:
            return f"No valid factor; decrease confidence in `{selected_arm}` and keep exploring."
        state.best_history.append(
            {
                "factor": best.factor_name,
                "kind": best.kind,
                "reward": reward,
                "status": best.status,
                "rank_ic": best.metrics.get("rank_ic"),
                "oos_rank_ic": best.metrics.get("oos_rank_ic"),
                "sharpe": best.metrics.get("sharpe"),
            }
        )
        if best.status == "accepted":
            return f"Promote `{best.factor_name}` to knowledge base and continue adjacent exploration from `{selected_arm}`."
        return f"Keep `{best.factor_name}` on watchlist; require stronger OOS IC before promotion."
