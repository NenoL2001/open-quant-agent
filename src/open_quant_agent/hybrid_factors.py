from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def close_panel(frames: dict[str, pd.DataFrame], universe: list[str]) -> pd.DataFrame:
    closes = {}
    for ticker in universe:
        frame = frames.get(ticker)
        if frame is not None and not frame.empty and "Close" in frame.columns:
            closes[ticker] = frame["Close"].astype("float64")
    return pd.concat(closes, axis=1).sort_index() if closes else pd.DataFrame()


def future_return_panel(frames: dict[str, pd.DataFrame], universe: list[str], horizon: int) -> pd.DataFrame:
    closes = close_panel(frames, universe)
    if closes.empty:
        return pd.DataFrame()
    return closes.shift(-horizon) / closes - 1.0


def next_return_panel(frames: dict[str, pd.DataFrame], universe: list[str]) -> pd.DataFrame:
    closes = close_panel(frames, universe)
    if closes.empty:
        return pd.DataFrame()
    return closes.pct_change().shift(-1)


def evaluate_factor_panel(
    panel: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    universe: list[str],
    horizon: int,
    train_fraction: float = 0.70,
) -> dict[str, Any]:
    if panel.empty:
        return _empty_metrics("empty_panel")
    future_returns = future_return_panel(frames, universe, horizon)
    next_returns = next_return_panel(frames, universe)
    common_columns = [column for column in panel.columns if column in future_returns.columns]
    if not common_columns:
        return _empty_metrics("no_common_columns")
    factor = panel[common_columns].replace([math.inf, -math.inf], np.nan).astype("float64")
    future = future_returns[common_columns].reindex(factor.index).astype("float64")
    next_ret = next_returns[common_columns].reindex(factor.index).astype("float64")
    valid = factor.notna() & future.notna()
    coverage = float(valid.sum().sum()) / max(valid.size, 1)
    if coverage <= 0:
        return _empty_metrics("no_valid_cells")

    dates = list(factor.index)
    split = max(1, min(len(dates) - 1, int(len(dates) * train_fraction)))
    train_dates = set(dates[:split])
    oos_dates = set(dates[split:])

    all_metrics = _cross_sectional_metrics(factor, future, next_ret, None)
    train_metrics = _cross_sectional_metrics(factor, future, next_ret, train_dates)
    oos_metrics = _cross_sectional_metrics(factor, future, next_ret, oos_dates)
    metrics = {
        **all_metrics,
        "coverage_pct": coverage * 100.0,
        "train_rank_ic": train_metrics["rank_ic"],
        "train_ic": train_metrics["ic"],
        "train_decile_spread_pct": train_metrics["decile_spread_pct"],
        "oos_rank_ic": oos_metrics["rank_ic"],
        "oos_ic": oos_metrics["ic"],
        "oos_decile_spread_pct": oos_metrics["decile_spread_pct"],
        "oos_sharpe": oos_metrics["sharpe"],
        "oos_max_drawdown_pct": oos_metrics["max_drawdown_pct"],
        "oos_return_pct": oos_metrics["return_pct"],
        "oos_observations": oos_metrics["observations"],
        "status_reason": "ok",
    }
    return metrics


def fuse_factor_panels(
    panels: dict[str, pd.DataFrame],
    frames: dict[str, pd.DataFrame],
    universe: list[str],
    horizon: int,
    method: str = "ic_weighted",
    train_fraction: float = 0.70,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    usable = {name: panel for name, panel in panels.items() if not panel.empty}
    if not usable:
        return pd.DataFrame(), {"method": method, "status": "empty"}
    common_index = sorted(set.intersection(*(set(panel.index) for panel in usable.values())))
    common_columns = sorted(set.intersection(*(set(panel.columns) for panel in usable.values())))
    if not common_index or not common_columns:
        return pd.DataFrame(), {"method": method, "status": "no_overlap"}
    aligned = {name: panel.reindex(index=common_index, columns=common_columns).astype("float64") for name, panel in usable.items()}
    if method == "equal_weight":
        weights = {name: 1.0 / len(aligned) for name in aligned}
    elif method == "stacking":
        weights = _stacking_weights(aligned, frames, common_columns, horizon, train_fraction)
    else:
        weights = _ic_weights(aligned, frames, common_columns, horizon, train_fraction)
    fused = sum(aligned[name].fillna(0.0) * weight for name, weight in weights.items())
    fused = fused.where(pd.concat([panel.notna() for panel in aligned.values()]).groupby(level=0).any())
    fused = fused.rank(axis=1, pct=True) - 0.5
    return fused.astype("float64"), {"method": method, "status": "ok", "weights": weights}


def _cross_sectional_metrics(
    factor: pd.DataFrame,
    future: pd.DataFrame,
    next_ret: pd.DataFrame,
    allowed_dates: set[Any] | None,
) -> dict[str, Any]:
    ic_values: list[float] = []
    rank_ic_values: list[float] = []
    spread_values: list[float] = []
    daily_returns: list[float] = []
    metric_dates: list[Any] = []
    for date in factor.index:
        if allowed_dates is not None and date not in allowed_dates:
            continue
        aligned = pd.concat([factor.loc[date].rename("factor"), future.loc[date].rename("future")], axis=1).dropna()
        if len(aligned) >= 6:
            ic = _safe_corr(aligned["factor"], aligned["future"])
            rank_ic = _safe_corr(aligned["factor"].rank(), aligned["future"].rank())
            ranks = aligned["factor"].rank(pct=True)
            top = aligned.loc[ranks >= 0.8, "future"]
            bottom = aligned.loc[ranks <= 0.2, "future"]
            if pd.notna(ic):
                ic_values.append(float(ic))
            if pd.notna(rank_ic):
                rank_ic_values.append(float(rank_ic))
                metric_dates.append(date)
            if not top.empty and not bottom.empty:
                spread_values.append(float(top.mean() - bottom.mean()))

        trade_aligned = pd.concat([factor.loc[date].rename("factor"), next_ret.loc[date].rename("next")], axis=1).dropna()
        if len(trade_aligned) >= 6:
            ranks = trade_aligned["factor"].rank(pct=True)
            long_ret = trade_aligned.loc[ranks >= 0.8, "next"]
            short_ret = trade_aligned.loc[ranks <= 0.2, "next"]
            if not long_ret.empty and not short_ret.empty:
                daily_returns.append(float(long_ret.mean() - short_ret.mean()))

    curve = (pd.Series(daily_returns, dtype="float64") + 1.0).cumprod() if daily_returns else pd.Series(dtype="float64")
    if curve.empty:
        max_drawdown = 0.0
        total_return = 0.0
        sharpe = 0.0
    else:
        drawdowns = curve / curve.cummax() - 1.0
        max_drawdown = float(drawdowns.min() * 100.0)
        total_return = float((curve.iloc[-1] - 1.0) * 100.0)
        returns = pd.Series(daily_returns, dtype="float64")
        sharpe = float(returns.mean() / returns.std() * math.sqrt(252.0)) if returns.std() and returns.std() > 0 else 0.0

    if rank_ic_values:
        by_year = pd.Series(rank_ic_values, index=pd.to_datetime(metric_dates)).groupby(lambda value: value.year).mean()
        stability = float((by_year > 0).mean() * 100.0) if len(by_year) else 0.0
    else:
        stability = 0.0
    return {
        "ic": _mean(ic_values),
        "rank_ic": _mean(rank_ic_values),
        "decile_spread_pct": _mean(spread_values) * 100.0,
        "stability_pct": stability,
        "sharpe": sharpe,
        "return_pct": total_return,
        "max_drawdown_pct": max_drawdown,
        "observations": len(rank_ic_values),
        "trade_observations": len(daily_returns),
    }


def _ic_weights(
    panels: dict[str, pd.DataFrame],
    frames: dict[str, pd.DataFrame],
    universe: list[str],
    horizon: int,
    train_fraction: float,
) -> dict[str, float]:
    raw_weights = {}
    for name, panel in panels.items():
        metrics = evaluate_factor_panel(panel, frames, universe, horizon, train_fraction)
        raw_weights[name] = max(0.0, float(metrics.get("train_rank_ic") or 0.0))
    if sum(raw_weights.values()) <= 0:
        return {name: 1.0 / len(raw_weights) for name in raw_weights}
    total = sum(raw_weights.values())
    return {name: value / total for name, value in raw_weights.items()}


def _stacking_weights(
    panels: dict[str, pd.DataFrame],
    frames: dict[str, pd.DataFrame],
    universe: list[str],
    horizon: int,
    train_fraction: float,
) -> dict[str, float]:
    names = list(panels)
    target = future_return_panel(frames, universe, horizon)
    dates = list(next(iter(panels.values())).index)
    split = max(1, min(len(dates) - 1, int(len(dates) * train_fraction)))
    train_index = dates[:split]
    rows = []
    y = []
    for date in train_index:
        values = [panels[name].loc[date] for name in names if date in panels[name].index]
        if len(values) != len(names) or date not in target.index:
            continue
        frame = pd.concat([series.rename(name) for series, name in zip(values, names, strict=False)] + [target.loc[date].rename("target")], axis=1).dropna()
        if len(frame) < 6:
            continue
        rows.append(frame[names].to_numpy())
        y.append(frame["target"].to_numpy())
    if not rows:
        return {name: 1.0 / len(names) for name in names}
    x_matrix = np.vstack(rows)
    y_vector = np.concatenate(y)
    ridge = 1e-3 * np.eye(len(names))
    try:
        coef = np.linalg.solve(x_matrix.T @ x_matrix + ridge, x_matrix.T @ y_vector)
    except np.linalg.LinAlgError:
        return {name: 1.0 / len(names) for name in names}
    coef = np.clip(coef, 0.0, None)
    if coef.sum() <= 0:
        return {name: 1.0 / len(names) for name in names}
    coef = coef / coef.sum()
    return {name: float(weight) for name, weight in zip(names, coef, strict=False)}


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _safe_corr(left: pd.Series, right: pd.Series) -> float | None:
    if left.std() <= 0 or right.std() <= 0:
        return None
    value = left.corr(right)
    return float(value) if pd.notna(value) else None


def _empty_metrics(reason: str) -> dict[str, Any]:
    return {
        "ic": 0.0,
        "rank_ic": 0.0,
        "decile_spread_pct": 0.0,
        "coverage_pct": 0.0,
        "stability_pct": 0.0,
        "sharpe": 0.0,
        "return_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "oos_rank_ic": 0.0,
        "oos_ic": 0.0,
        "oos_decile_spread_pct": 0.0,
        "oos_sharpe": 0.0,
        "oos_max_drawdown_pct": 0.0,
        "oos_return_pct": 0.0,
        "observations": 0,
        "oos_observations": 0,
        "status_reason": reason,
    }
