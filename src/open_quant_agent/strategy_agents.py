from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from open_quant_agent.hybrid_factors import close_panel
from open_quant_agent.schemas import StrategyCandidate, StrategyValidationResult, ValidationResult


class StrategyConstructionAgent:
    """Turns validated factor panels into executable portfolio strategy specs."""

    def construct(
        self,
        factor_validations: list[ValidationResult],
        panels: dict[str, pd.DataFrame],
        selected_arm: str,
    ) -> list[StrategyCandidate]:
        candidates: list[StrategyCandidate] = []
        usable = [item for item in factor_validations if item.status in {"accepted", "watch"} and item.factor_name in panels]
        usable.sort(key=lambda item: item.reward, reverse=True)
        for validation in usable[:4]:
            metrics = validation.metrics
            oos_rank_ic = float(metrics.get("oos_rank_ic") or 0.0)
            oos_sharpe = float(metrics.get("oos_sharpe") or 0.0)
            top_n = 5 if validation.kind in {"hybrid", "dl"} else 6
            if oos_rank_ic > 0.05 or oos_sharpe > 1.0:
                top_n = max(3, top_n - 1)
            base_name = _safe_name(validation.factor_name)
            candidates.append(
                StrategyCandidate(
                    name=f"{base_name}_top{top_n}_rotation",
                    source_factor=validation.factor_name,
                    construction="topk_rotation",
                    rebalance_bars=max(5, min(21, int(metrics.get("oos_observations") or 21) // 9 or 10)),
                    top_n=top_n,
                    max_gross=1.0,
                    max_name_weight=0.28 if top_n <= 4 else 0.22,
                    annual_vol_target=0.18 if selected_arm != "reversal_quality" else 0.14,
                    min_score_quantile=0.62,
                    drawdown_reduce=-0.10,
                    drawdown_pause=-0.18,
                    thesis=f"Top-k rotation using {validation.factor_name}; promoted from factor status {validation.status}.",
                )
            )
            if validation.kind in {"dl", "hybrid"}:
                candidates.append(
                    StrategyCandidate(
                        name=f"{base_name}_risk_balanced_rotation",
                        source_factor=validation.factor_name,
                        construction="topk_rotation",
                        rebalance_bars=10,
                        top_n=6,
                        max_gross=0.85,
                        max_name_weight=0.18,
                        annual_vol_target=0.14,
                        min_score_quantile=0.58,
                        drawdown_reduce=-0.08,
                        drawdown_pause=-0.15,
                        thesis=f"Lower gross risk-balanced rotation using {validation.factor_name}.",
                    )
                )
        return candidates


class PortfolioBacktestAgent:
    """Backtests factor-driven portfolio strategies with OOS and drawdown gates."""

    def validate(
        self,
        candidate: StrategyCandidate,
        panel: pd.DataFrame,
        frames: dict[str, pd.DataFrame],
        universe: list[str],
    ) -> StrategyValidationResult:
        try:
            metrics = backtest_factor_strategy(candidate, panel, frames, universe)
            status = self._status(metrics)
            notes = (
                f"return={metrics.get('return_pct', 0.0):.2f}%, "
                f"oos_return={metrics.get('oos_return_pct', 0.0):.2f}%, "
                f"oos_excess={metrics.get('oos_excess_return_pct', 0.0):.2f}%, "
                f"sharpe={metrics.get('sharpe', 0.0):.2f}, "
                f"oos_sharpe={metrics.get('oos_sharpe', 0.0):.2f}, "
                f"max_dd={metrics.get('max_drawdown_pct', 0.0):.2f}%"
            )
        except Exception as exc:
            metrics = {"status_reason": f"{type(exc).__name__}: {exc}"}
            status = "failed"
            notes = "Strategy backtest failed."
        return StrategyValidationResult(
            strategy_id=candidate.strategy_id,
            strategy_name=candidate.name,
            source_factor=candidate.source_factor,
            status=status,
            metrics=metrics,
            spec=asdict(candidate),
            notes=notes,
        )

    def _status(self, metrics: dict[str, Any]) -> str:
        if metrics.get("status_reason") != "ok":
            return "failed"
        oos_sharpe = float(metrics.get("oos_sharpe") or 0.0)
        oos_return = float(metrics.get("oos_return_pct") or 0.0)
        oos_excess = float(metrics.get("oos_excess_return_pct") or 0.0)
        max_dd = float(metrics.get("max_drawdown_pct") or 0.0)
        rebalances = int(metrics.get("num_rebalances") or 0)
        exposure = float(metrics.get("exposure_pct") or 0.0)
        if oos_sharpe >= 0.75 and oos_return > 0.0 and oos_excess > 0.0 and max_dd >= -30.0 and rebalances >= 8 and exposure >= 10.0:
            return "accepted"
        if (oos_sharpe > 0.0 and oos_return > 0.0) or oos_excess > 0.0:
            return "watch"
        return "rejected"


class StrategyPromotionAgent:
    """Selects the best strategy candidate and writes a human-readable decision."""

    def select_best(self, results: list[StrategyValidationResult]) -> StrategyValidationResult | None:
        candidates = [item for item in results if item.status in {"accepted", "watch"}]
        if not candidates:
            return None
        candidates.sort(key=lambda item: item.reward, reverse=True)
        return candidates[0]

    def decision(self, best: StrategyValidationResult | None) -> str:
        if best is None:
            return "No strategy promoted; factor research continues but portfolio construction needs better OOS evidence."
        if best.status == "accepted":
            return f"Promote strategy `{best.strategy_name}` for paper-strategy tracking; keep validating before real trading."
        return f"Keep strategy `{best.strategy_name}` on watchlist; require stronger OOS return and drawdown before promotion."


def backtest_factor_strategy(
    candidate: StrategyCandidate,
    panel: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    universe: list[str],
    train_fraction: float = 0.70,
) -> dict[str, Any]:
    closes = close_panel(frames, universe)
    if closes.empty or panel.empty:
        return _empty_strategy_metrics("empty_panel_or_closes")
    common_columns = [column for column in panel.columns if column in closes.columns]
    if len(common_columns) < max(4, candidate.top_n):
        return _empty_strategy_metrics("insufficient_common_columns")
    factor = panel[common_columns].replace([math.inf, -math.inf], np.nan).astype("float64")
    closes = closes[common_columns].reindex(factor.index).ffill().astype("float64")
    returns = closes.pct_change().replace([math.inf, -math.inf], np.nan).fillna(0.0)
    trailing_vol = returns.rolling(20).std() * math.sqrt(252.0)
    dates = list(factor.index)
    if len(dates) < candidate.rebalance_bars * 4:
        return _empty_strategy_metrics("insufficient_dates")

    equity = 1.0
    equity_curve: list[float] = []
    daily_returns: list[float] = []
    exposure_values: list[float] = []
    weights: dict[str, float] = {}
    turnover = 0.0
    num_rebalances = 0
    unique_tickers: set[str] = set()
    peak = 1.0

    for index, date in enumerate(dates):
        if index > 0 and weights:
            day_return = 0.0
            row_returns = returns.loc[date]
            for ticker, weight in weights.items():
                value = row_returns.get(ticker, 0.0)
                if pd.notna(value):
                    day_return += weight * float(value)
            equity *= max(0.01, 1.0 + day_return)
            daily_returns.append(day_return)
        else:
            daily_returns.append(0.0)

        peak = max(peak, equity)
        current_drawdown = equity / peak - 1.0
        if index >= 25 and index % candidate.rebalance_bars == 0:
            new_weights = _target_weights(candidate, factor.loc[date], trailing_vol.loc[date], current_drawdown)
            turnover += _turnover(weights, new_weights)
            weights = new_weights
            unique_tickers.update(weights)
            num_rebalances += 1

        exposure_values.append(sum(abs(value) for value in weights.values()))
        equity_curve.append(equity)

    split = max(1, min(len(equity_curve) - 1, int(len(equity_curve) * train_fraction)))
    all_metrics = _portfolio_metrics(equity_curve, daily_returns, exposure_values, 0, len(equity_curve))
    train_metrics = _portfolio_metrics(equity_curve, daily_returns, exposure_values, 0, split)
    oos_metrics = _portfolio_metrics(equity_curve, daily_returns, exposure_values, split, len(equity_curve))
    benchmark_metrics = _benchmark_metrics(closes, dates)
    metrics = {
        **all_metrics,
        "train_return_pct": train_metrics["return_pct"],
        "train_sharpe": train_metrics["sharpe"],
        "train_max_drawdown_pct": train_metrics["max_drawdown_pct"],
        "oos_return_pct": oos_metrics["return_pct"],
        "oos_sharpe": oos_metrics["sharpe"],
        "oos_max_drawdown_pct": oos_metrics["max_drawdown_pct"],
        "oos_calmar": oos_metrics["calmar"],
        "num_rebalances": num_rebalances,
        "unique_tickers": len(unique_tickers),
        "turnover_pct": turnover * 100.0,
        "avg_gross_exposure_pct": float(np.mean(exposure_values) * 100.0) if exposure_values else 0.0,
        "max_gross_exposure_pct": float(np.max(exposure_values) * 100.0) if exposure_values else 0.0,
        **benchmark_metrics,
        "status_reason": "ok",
    }
    metrics["excess_return_pct"] = metrics["return_pct"] - metrics.get("equal_weight_return_pct", 0.0)
    metrics["oos_excess_return_pct"] = metrics["oos_return_pct"] - metrics.get("oos_equal_weight_return_pct", 0.0)
    return metrics


def _target_weights(
    candidate: StrategyCandidate,
    scores: pd.Series,
    annual_vol: pd.Series,
    current_drawdown: float,
) -> dict[str, float]:
    clean = scores.dropna().astype("float64")
    if clean.empty or current_drawdown <= candidate.drawdown_pause:
        return {}
    ranks = clean.rank(pct=True)
    selected = ranks[ranks >= candidate.min_score_quantile].sort_values(ascending=False).head(candidate.top_n).index.tolist()
    if candidate.cash_on_negative_cross_section:
        selected = [ticker for ticker in selected if clean[ticker] > 0.0]
    if not selected:
        return {}
    selected_vol = annual_vol.reindex(selected).dropna()
    est_vol = float(selected_vol.mean()) if not selected_vol.empty else candidate.annual_vol_target
    if est_vol <= 0 or math.isnan(est_vol):
        est_vol = candidate.annual_vol_target
    gross = min(candidate.max_gross, max(0.20, candidate.annual_vol_target / est_vol))
    if current_drawdown <= candidate.drawdown_reduce:
        gross *= 0.45
    per_name = min(candidate.max_name_weight, gross / len(selected))
    weights = {ticker: per_name for ticker in selected}
    gross_used = sum(weights.values())
    if gross_used > candidate.max_gross and gross_used > 0:
        scale = candidate.max_gross / gross_used
        weights = {ticker: weight * scale for ticker, weight in weights.items()}
    return weights


def _turnover(old: dict[str, float], new: dict[str, float]) -> float:
    tickers = set(old) | set(new)
    return sum(abs(new.get(ticker, 0.0) - old.get(ticker, 0.0)) for ticker in tickers)


def _portfolio_metrics(
    equity_curve: list[float],
    daily_returns: list[float],
    exposure_values: list[float],
    start: int,
    end: int,
) -> dict[str, float]:
    curve = pd.Series(equity_curve[start:end], dtype="float64")
    returns = pd.Series(daily_returns[start:end], dtype="float64")
    exposure = pd.Series(exposure_values[start:end], dtype="float64")
    if len(curve) < 2 or curve.iloc[0] <= 0:
        return {"return_pct": 0.0, "sharpe": 0.0, "max_drawdown_pct": 0.0, "calmar": 0.0, "exposure_pct": 0.0}
    normalized = curve / curve.iloc[0]
    total_return = float((normalized.iloc[-1] - 1.0) * 100.0)
    drawdowns = normalized / normalized.cummax() - 1.0
    max_drawdown = float(drawdowns.min() * 100.0)
    std = returns.std()
    sharpe = float(returns.mean() / std * math.sqrt(252.0)) if std and std > 0 else 0.0
    periods = max(len(curve), 1)
    annual_return = (normalized.iloc[-1] ** (252.0 / periods) - 1.0) * 100.0 if normalized.iloc[-1] > 0 else -100.0
    calmar = float(annual_return / abs(max_drawdown)) if max_drawdown < 0 else 0.0
    return {
        "return_pct": total_return,
        "annual_return_pct": float(annual_return),
        "sharpe": sharpe,
        "max_drawdown_pct": max_drawdown,
        "calmar": calmar,
        "exposure_pct": float((exposure > 0).mean() * 100.0) if len(exposure) else 0.0,
    }


def _benchmark_metrics(closes: pd.DataFrame, dates: list[Any], train_fraction: float = 0.70) -> dict[str, float]:
    if closes.empty:
        return {"equal_weight_return_pct": 0.0, "oos_equal_weight_return_pct": 0.0}
    normalized = closes.reindex(dates).ffill().dropna(how="all")
    normalized = normalized / normalized.iloc[0]
    equal_weight = normalized.mean(axis=1).dropna()
    if len(equal_weight) < 2:
        return {"equal_weight_return_pct": 0.0, "oos_equal_weight_return_pct": 0.0}
    split = max(1, min(len(equal_weight) - 1, int(len(equal_weight) * train_fraction)))
    total = float((equal_weight.iloc[-1] / equal_weight.iloc[0] - 1.0) * 100.0)
    oos = float((equal_weight.iloc[-1] / equal_weight.iloc[split] - 1.0) * 100.0)
    return {"equal_weight_return_pct": total, "oos_equal_weight_return_pct": oos}


def _empty_strategy_metrics(reason: str) -> dict[str, Any]:
    return {
        "return_pct": 0.0,
        "oos_return_pct": 0.0,
        "sharpe": 0.0,
        "oos_sharpe": 0.0,
        "max_drawdown_pct": 0.0,
        "oos_max_drawdown_pct": 0.0,
        "num_rebalances": 0,
        "unique_tickers": 0,
        "turnover_pct": 0.0,
        "status_reason": reason,
    }


def _safe_name(value: str) -> str:
    chars = []
    for char in value.lower():
        if char.isalnum() or char == "_":
            chars.append(char)
        elif char in {"-", " "}:
            chars.append("_")
    return ("".join(chars).strip("_") or "strategy")[:72]
