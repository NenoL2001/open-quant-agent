from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


ALLOWED_FEATURE_FAMILIES = {
    "momentum",
    "reversal",
    "volatility",
    "volume",
    "relative_strength",
    "regime",
    "quality_proxy",
    "trend_quality",
    "liquidity",
    "seasonality",
    "pairs",
}
ALLOWED_FEATURE_NORMALIZATIONS = {"none", "zscore", "rank", "cross_sectional_zscore", "cross_sectional_rank"}
ALLOWED_FEATURE_OPS = {
    "roc",
    "log_return",
    "rolling_mean",
    "rolling_std",
    "realized_vol",
    "downside_vol",
    "rolling_skew",
    "efficiency_ratio",
    "drawdown",
    "gap",
    "price_position",
    "ema_gap",
    "rsi",
    "atr_pct",
    "bb_width",
    "donchian_position",
    "volume_z",
    "dollar_volume_z",
    "relative_strength",
    "beta_residual",
    "rank",
    "zscore",
    "add",
    "sub",
    "mul",
    "div",
    "clip",
}
ALLOWED_FEATURE_BENCHMARKS = {"SOXX", "SMH", "QQQ", "SPY"}
MAX_FEATURE_EXPR_DEPTH = 4
MAX_FEATURE_REFS = 6


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    family: str
    expr: dict[str, Any]
    normalize: str = "cross_sectional_zscore"
    direction: int = 1
    horizon: int = 21
    source: str = "local_factory"
    thesis: str = ""


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def read_jsonl(path: Path, limit: int = 20) -> list[dict[str, Any]]:
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
        file.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def safe_name(value: str) -> str:
    chars = []
    for char in value.lower():
        if char.isalnum() or char == "_":
            chars.append(char)
        elif char in {"-", " "}:
            chars.append("_")
    cleaned = "".join(chars).strip("_")
    return (cleaned or "feature")[:64]


def usable_frame(frame: pd.DataFrame | None) -> bool:
    if frame is None or frame.empty:
        return False
    if not all(column in frame.columns for column in ("High", "Low", "Close")):
        return False
    return bool(frame["Close"].dropna().size)


def rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def build_feature_proposal_prompt(recent_results: list[dict[str, Any]], lessons: str, count: int) -> str:
    schema = {
        "features": [
            {
                "name": "mom_63_vol_adj",
                "family": "momentum|reversal|volatility|volume|relative_strength|regime|quality_proxy|trend_quality|liquidity|seasonality|pairs",
                "thesis": "63-day momentum divided by realized volatility captures persistent trend quality.",
                "expr": {
                    "op": "div",
                    "left": {"op": "roc", "window": 63},
                    "right": {"op": "rolling_std", "window": 20},
                },
                "normalize": "cross_sectional_zscore",
                "direction": 1,
                "horizon": 21,
            }
        ]
    }
    return (
        f"Generate {count} JSON feature specs for liquid US equities and ETFs. "
        "These are alpha research features, not trading instructions. Use only the schema below.\n"
        "Allowed ops: roc, log_return, rolling_mean, rolling_std, realized_vol, downside_vol, rolling_skew, "
        "efficiency_ratio, drawdown, gap, price_position, ema_gap, rsi, atr_pct, bb_width, "
        "donchian_position, volume_z, dollar_volume_z, relative_strength, beta_residual, rank, zscore, "
        "add, sub, mul, div, clip. Arithmetic ops must use left/right child expressions; clip uses child, low, high.\n"
        "Hard constraints: expression depth <= 4, windows 2-252, horizon 5-63, direction must be 1 or -1, "
        "no Python code, no column names, no files, no URLs, no shell commands. Prefer features with clear economic logic, "
        "expected temporal stability, low redundancy, and lower drawdown when used in cross-sectional rotation.\n\n"
        f"Schema example:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
        f"Recent results:\n{json.dumps(recent_results[-8:], ensure_ascii=False)[:6000]}\n\n"
        f"Lessons:\n{lessons[-4000:]}\n"
    )


def normalize_feature_spec(raw: dict[str, Any], source: str) -> tuple[FeatureSpec | None, list[str]]:
    reasons: list[str] = []
    name = safe_name(str(raw.get("name") or f"{source}_feature"))
    family = str(raw.get("family") or "momentum").lower()
    normalize = str(raw.get("normalize") or "cross_sectional_zscore").lower()
    expr = raw.get("expr")
    if not isinstance(expr, dict):
        reasons.append("expr must be an object")
        expr = {"op": "roc", "window": 21}
    try:
        direction = int(raw.get("direction", 1))
    except Exception:
        reasons.append("direction must be 1 or -1")
        direction = 1
    try:
        horizon = int(raw.get("horizon", 21))
    except Exception:
        reasons.append("horizon must be an integer")
        horizon = 21
    spec = FeatureSpec(
        name=name,
        family=family,
        expr=expr,
        normalize=normalize,
        direction=direction,
        horizon=horizon,
        source=source,
        thesis=str(raw.get("thesis") or "")[:400],
    )
    reasons.extend(validate_feature_spec(spec))
    return (None if reasons else spec), reasons


def validate_feature_spec(spec: FeatureSpec) -> list[str]:
    reasons: list[str] = []
    if spec.family not in ALLOWED_FEATURE_FAMILIES:
        reasons.append(f"feature family not allowed: {spec.family}")
    if spec.normalize not in ALLOWED_FEATURE_NORMALIZATIONS:
        reasons.append(f"feature normalization not allowed: {spec.normalize}")
    if spec.direction not in {-1, 1}:
        reasons.append("direction must be 1 or -1")
    if not (5 <= spec.horizon <= 63):
        reasons.append("horizon outside allowed range")
    reasons.extend(validate_feature_expr(spec.expr, 1))
    return reasons


def feature_number(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except Exception:
        return default


def feature_window(expr: dict[str, Any], name: str = "window", default: int | None = None) -> int | None:
    value = expr.get(name, default)
    try:
        window = int(value)
    except Exception:
        return None
    return window if 2 <= window <= 252 else None


def validate_feature_expr(expr: Any, depth: int) -> list[str]:
    reasons: list[str] = []
    if depth > MAX_FEATURE_EXPR_DEPTH:
        return ["feature expression exceeds max depth"]
    if not isinstance(expr, dict):
        return ["feature expression node must be an object"]
    allowed_keys = {"op", "window", "fast", "slow", "benchmark", "left", "right", "child", "low", "high", "z"}
    for key, value in expr.items():
        if key not in allowed_keys:
            reasons.append(f"feature expression key not allowed: {key}")
        if isinstance(value, str) and key not in {"op", "benchmark"}:
            reasons.append(f"feature expression string value not allowed for {key}")
    op = str(expr.get("op") or "").lower()
    if op not in ALLOWED_FEATURE_OPS:
        reasons.append(f"feature op not allowed: {op}")
        return reasons
    if op in {
        "roc",
        "log_return",
        "rolling_mean",
        "rolling_std",
        "realized_vol",
        "downside_vol",
        "rolling_skew",
        "efficiency_ratio",
        "drawdown",
        "gap",
        "price_position",
        "rsi",
        "atr_pct",
        "bb_width",
        "donchian_position",
        "volume_z",
        "dollar_volume_z",
        "relative_strength",
        "beta_residual",
    } and feature_window(expr) is None:
        reasons.append(f"{op} window outside allowed range")
    if op == "ema_gap":
        fast = feature_window(expr, "fast")
        slow = feature_window(expr, "slow")
        if fast is None or slow is None or fast >= slow:
            reasons.append("ema_gap requires 2 <= fast < slow <= 252")
    if op in {"relative_strength", "beta_residual"}:
        benchmark = str(expr.get("benchmark") or "SPY").upper()
        if benchmark not in ALLOWED_FEATURE_BENCHMARKS:
            reasons.append(f"benchmark not allowed in feature expr: {benchmark}")
    if op in {"add", "sub", "mul", "div"}:
        reasons.extend(validate_feature_expr(expr.get("left"), depth + 1))
        reasons.extend(validate_feature_expr(expr.get("right"), depth + 1))
    if op in {"clip", "rank", "zscore"}:
        reasons.extend(validate_feature_expr(expr.get("child"), depth + 1))
    if op == "clip":
        low = feature_number(expr.get("low"))
        high = feature_number(expr.get("high"))
        if low is None or high is None or low >= high:
            reasons.append("clip requires numeric low < high")
    if op in {"rank", "zscore"} and feature_window(expr, default=63) is None:
        reasons.append(f"{op} window outside allowed range")
    return reasons


def feature_expr_complexity(expr: Any) -> int:
    if not isinstance(expr, dict):
        return 1
    op = str(expr.get("op") or "").lower()
    if op in {"add", "sub", "mul", "div"}:
        return 1 + feature_expr_complexity(expr.get("left")) + feature_expr_complexity(expr.get("right"))
    if op in {"clip", "rank", "zscore"}:
        return 1 + feature_expr_complexity(expr.get("child"))
    return 1


def feature_id(spec: FeatureSpec) -> str:
    payload = asdict(spec)
    payload.pop("source", None)
    payload.pop("thesis", None)
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def safe_divide(left: pd.Series, right: pd.Series) -> pd.Series:
    denominator = right.where(right.abs() > 1e-12)
    return left / denominator


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([math.inf, -math.inf], pd.NA).astype("float64")


def compute_feature_expr(expr: dict[str, Any], frame: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> pd.Series:
    op = str(expr.get("op") or "").lower()
    close = frame["Close"].astype("float64")
    high = frame["High"].astype("float64")
    low = frame["Low"].astype("float64")
    open_ = frame["Open"].astype("float64") if "Open" in frame.columns else close
    volume = frame["Volume"].astype("float64") if "Volume" in frame.columns else pd.Series(0.0, index=frame.index)
    window = feature_window(expr, default=21) or 21
    returns = close.pct_change()
    if op == "roc":
        return close / close.shift(window) - 1.0
    if op == "log_return":
        ratio = close / close.shift(window)
        return ratio.where(ratio > 0).apply(lambda value: math.log(value) if pd.notna(value) else pd.NA)
    if op == "rolling_mean":
        return returns.rolling(window).mean()
    if op == "rolling_std":
        return returns.rolling(window).std()
    if op == "realized_vol":
        return returns.rolling(window).std()
    if op == "downside_vol":
        return returns.where(returns < 0, 0.0).rolling(window).std()
    if op == "rolling_skew":
        return returns.rolling(window).skew()
    if op == "efficiency_ratio":
        direction = (close - close.shift(window)).abs()
        path = close.diff().abs().rolling(window).sum()
        return safe_divide(direction, path) - 0.5
    if op == "drawdown":
        rolling_high = close.rolling(window).max()
        return close / rolling_high - 1.0
    if op == "gap":
        return open_ / close.shift(1) - 1.0
    if op == "price_position":
        highest = high.shift(1).rolling(window).max()
        lowest = low.shift(1).rolling(window).min()
        return safe_divide(close - lowest, highest - lowest) - 0.5
    if op == "ema_gap":
        fast = feature_window(expr, "fast", 12) or 12
        slow = feature_window(expr, "slow", 48) or 48
        return close.ewm(span=fast, adjust=False).mean() / close.ewm(span=slow, adjust=False).mean() - 1.0
    if op == "rsi":
        return rsi(close, window) / 100.0 - 0.5
    if op == "atr_pct":
        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        return tr.rolling(window).mean() / close
    if op == "bb_width":
        z_value = feature_number(expr.get("z"), 2.0) or 2.0
        mid = close.rolling(window).mean()
        return safe_divide(close.rolling(window).std() * z_value * 2.0, mid)
    if op == "donchian_position":
        highest = high.shift(1).rolling(window).max()
        lowest = low.shift(1).rolling(window).min()
        return safe_divide(close - lowest, highest - lowest) - 0.5
    if op == "volume_z":
        mean = volume.rolling(window).mean()
        std = volume.rolling(window).std()
        return safe_divide(volume - mean, std)
    if op == "dollar_volume_z":
        dollar_volume = close * volume
        mean = dollar_volume.rolling(window).mean()
        std = dollar_volume.rolling(window).std()
        return safe_divide(dollar_volume - mean, std)
    if op == "relative_strength":
        benchmark = str(expr.get("benchmark") or "SPY").upper()
        benchmark_frame = frames.get(benchmark)
        if not usable_frame(benchmark_frame):
            return pd.Series(pd.NA, index=frame.index, dtype="float64")
        benchmark_close = benchmark_frame["Close"].reindex(frame.index).ffill().astype("float64")
        return (close / close.shift(window) - 1.0) - (benchmark_close / benchmark_close.shift(window) - 1.0)
    if op == "beta_residual":
        benchmark = str(expr.get("benchmark") or "SPY").upper()
        benchmark_frame = frames.get(benchmark)
        if not usable_frame(benchmark_frame):
            return pd.Series(pd.NA, index=frame.index, dtype="float64")
        benchmark_returns = benchmark_frame["Close"].reindex(frame.index).ffill().astype("float64").pct_change()
        beta = returns.rolling(window).cov(benchmark_returns) / benchmark_returns.rolling(window).var()
        return returns.rolling(window).sum() - beta * benchmark_returns.rolling(window).sum()
    if op in {"add", "sub", "mul", "div"}:
        left = compute_feature_expr(expr["left"], frame, frames)
        right = compute_feature_expr(expr["right"], frame, frames)
        if op == "add":
            return left + right
        if op == "sub":
            return left - right
        if op == "mul":
            return left * right
        return safe_divide(left, right)
    if op == "clip":
        child = compute_feature_expr(expr["child"], frame, frames)
        return child.clip(float(expr["low"]), float(expr["high"]))
    if op == "zscore":
        child = compute_feature_expr(expr["child"], frame, frames)
        mean = child.rolling(window).mean()
        std = child.rolling(window).std()
        return safe_divide(child - mean, std)
    if op == "rank":
        child = compute_feature_expr(expr["child"], frame, frames)
        return child.rolling(window).rank(pct=True) - 0.5
    return pd.Series(pd.NA, index=frame.index, dtype="float64")


def normalize_feature_panel(panel: pd.DataFrame, normalize: str) -> pd.DataFrame:
    if panel.empty or normalize == "none":
        return panel
    if normalize == "cross_sectional_zscore":
        row_mean = panel.mean(axis=1)
        row_std = panel.std(axis=1).replace(0, pd.NA)
        return panel.sub(row_mean, axis=0).div(row_std, axis=0)
    if normalize in {"cross_sectional_rank", "rank"}:
        return panel.rank(axis=1, pct=True) - 0.5
    if normalize == "zscore":
        return panel.apply(lambda series: safe_divide(series - series.rolling(63).mean(), series.rolling(63).std()))
    return panel


def build_feature_panel(spec: FeatureSpec, frames: dict[str, pd.DataFrame], universe: list[str]) -> pd.DataFrame:
    series_by_ticker: dict[str, pd.Series] = {}
    for ticker in universe:
        frame = frames.get(ticker)
        if not usable_frame(frame):
            continue
        series = numeric_series(compute_feature_expr(spec.expr, frame, frames)).shift(1)
        if series.notna().sum() >= max(20, spec.horizon * 2):
            series_by_ticker[ticker] = series
    if not series_by_ticker:
        return pd.DataFrame()
    panel = pd.concat(series_by_ticker, axis=1).sort_index()
    panel = normalize_feature_panel(panel, spec.normalize)
    return panel.replace([math.inf, -math.inf], pd.NA).astype("float64") * spec.direction


def close_panel(frames: dict[str, pd.DataFrame], universe: list[str]) -> pd.DataFrame:
    closes = {}
    for ticker in universe:
        frame = frames.get(ticker)
        if usable_frame(frame):
            closes[ticker] = frame["Close"].astype("float64")
    return pd.concat(closes, axis=1).sort_index() if closes else pd.DataFrame()


def evaluate_feature_spec(spec: FeatureSpec, frames: dict[str, pd.DataFrame], universe: list[str]) -> tuple[dict[str, Any], pd.DataFrame]:
    panel = build_feature_panel(spec, frames, universe)
    closes = close_panel(frames, universe)
    if panel.empty or closes.empty:
        return empty_feature_result(spec, "insufficient_data"), panel
    common_columns = [column for column in panel.columns if column in closes.columns]
    panel = panel[common_columns]
    closes = closes[common_columns].reindex(panel.index).ffill()
    future_returns = closes.shift(-spec.horizon) / closes - 1.0
    valid = panel.notna() & future_returns.notna()
    total_cells = max(valid.size, 1)
    coverage = float(valid.sum().sum()) / total_cells
    ic_values: list[float] = []
    rank_ic_values: list[float] = []
    spread_values: list[float] = []
    ic_dates: list[Any] = []
    for date in panel.index:
        aligned = pd.concat([panel.loc[date].rename("feature"), future_returns.loc[date].rename("future")], axis=1).dropna()
        if len(aligned) < 8:
            continue
        ic = aligned["feature"].corr(aligned["future"])
        rank_ic = aligned["feature"].rank().corr(aligned["future"].rank())
        ranks = aligned["feature"].rank(pct=True)
        top = aligned.loc[ranks >= 0.8, "future"]
        bottom = aligned.loc[ranks <= 0.2, "future"]
        if pd.notna(ic):
            ic_values.append(float(ic))
        if pd.notna(rank_ic):
            rank_ic_values.append(float(rank_ic))
            ic_dates.append(date)
        if not top.empty and not bottom.empty:
            spread_values.append(float(top.mean() - bottom.mean()))
    mean_ic = sum(ic_values) / len(ic_values) if ic_values else 0.0
    mean_rank_ic = sum(rank_ic_values) / len(rank_ic_values) if rank_ic_values else 0.0
    decile_spread = sum(spread_values) / len(spread_values) if spread_values else 0.0
    if rank_ic_values:
        by_year = pd.Series(rank_ic_values, index=pd.to_datetime(ic_dates)).groupby(lambda value: value.year).mean()
        stability = float((by_year > 0).mean() * 100.0) if len(by_year) else 0.0
    else:
        stability = 0.0
    complexity = feature_expr_complexity(spec.expr)
    score_value = mean_rank_ic * 1800.0 + mean_ic * 700.0 + decile_spread * 1500.0 + stability * 0.35 + coverage * 18.0 - complexity * 1.5
    status = "accepted" if coverage >= 0.25 and mean_rank_ic > 0.0 and decile_spread > 0.0 and stability >= 35.0 else "weak"
    record = {
        "feature_id": feature_id(spec),
        "feature": spec.name,
        "family": spec.family,
        "source": spec.source,
        "status": status,
        "coverage_pct": coverage * 100.0,
        "ic": mean_ic,
        "rank_ic": mean_rank_ic,
        "decile_spread_pct": decile_spread * 100.0,
        "stability_pct": stability,
        "complexity": complexity,
        "score": score_value,
        "horizon": spec.horizon,
        "normalize": spec.normalize,
        "thesis": spec.thesis,
    }
    return record, panel


def empty_feature_result(spec: FeatureSpec, status: str) -> dict[str, Any]:
    return {
        "feature_id": feature_id(spec),
        "feature": spec.name,
        "family": spec.family,
        "source": spec.source,
        "status": status,
        "coverage_pct": 0.0,
        "ic": 0.0,
        "rank_ic": 0.0,
        "decile_spread_pct": 0.0,
        "stability_pct": 0.0,
        "complexity": feature_expr_complexity(spec.expr),
        "score": -999.0,
        "horizon": spec.horizon,
        "normalize": spec.normalize,
        "thesis": spec.thesis,
    }


def local_feature_factory(cycle: int, count: int) -> list[dict[str, Any]]:
    seeds = [
        {
            "name": f"mom_63_vol_adj_{cycle}",
            "family": "momentum",
            "expr": {
                "op": "div",
                "left": {"op": "roc", "window": 63},
                "right": {"op": "rolling_std", "window": 20},
            },
            "normalize": "cross_sectional_zscore",
            "direction": 1,
            "horizon": 21,
            "thesis": "Risk-adjusted medium-term momentum should select trend leaders without overpaying for volatility.",
        },
        {
            "name": f"rs_qqq_42_low_vol_{cycle}",
            "family": "relative_strength",
            "expr": {
                "op": "sub",
                "left": {"op": "relative_strength", "window": 42, "benchmark": "QQQ"},
                "right": {"op": "atr_pct", "window": 20},
            },
            "normalize": "cross_sectional_zscore",
            "direction": 1,
            "horizon": 21,
            "thesis": "QQQ-relative strength penalized by ATR targets smoother leadership.",
        },
        {
            "name": f"volume_confirmed_breakout_{cycle}",
            "family": "volume",
            "expr": {
                "op": "add",
                "left": {"op": "donchian_position", "window": 55},
                "right": {"op": "volume_z", "window": 30},
            },
            "normalize": "cross_sectional_zscore",
            "direction": 1,
            "horizon": 21,
            "thesis": "Breakouts confirmed by abnormal volume should persist better than price-only breakouts.",
        },
        {
            "name": f"low_vol_quality_proxy_{cycle}",
            "family": "quality_proxy",
            "expr": {
                "op": "sub",
                "left": {"op": "ema_gap", "fast": 21, "slow": 126},
                "right": {"op": "bb_width", "window": 40},
            },
            "normalize": "cross_sectional_zscore",
            "direction": 1,
            "horizon": 21,
            "thesis": "Uptrend with compressed bands approximates quality momentum and should lower drawdowns.",
        },
        {
            "name": f"short_reversal_stable_{cycle}",
            "family": "reversal",
            "expr": {
                "op": "sub",
                "left": {"op": "rsi", "window": 5},
                "right": {"op": "roc", "window": 5},
            },
            "normalize": "cross_sectional_zscore",
            "direction": -1,
            "horizon": 10,
            "thesis": "Short-term overextension often mean-reverts when normalized across liquid names.",
        },
        {
            "name": f"trend_efficiency_84_{cycle}",
            "family": "trend_quality",
            "expr": {
                "op": "add",
                "left": {"op": "efficiency_ratio", "window": 84},
                "right": {"op": "ema_gap", "fast": 21, "slow": 126},
            },
            "normalize": "cross_sectional_zscore",
            "direction": 1,
            "horizon": 21,
            "thesis": "Efficient directional trends should persist better than noisy momentum.",
        },
        {
            "name": f"drawdown_recovery_63_{cycle}",
            "family": "reversal",
            "expr": {
                "op": "sub",
                "left": {"op": "price_position", "window": 63},
                "right": {"op": "drawdown", "window": 63},
            },
            "normalize": "cross_sectional_zscore",
            "direction": 1,
            "horizon": 21,
            "thesis": "Names recovering within their range after controlled drawdowns may show healthier continuation.",
        },
        {
            "name": f"downside_vol_penalty_{cycle}",
            "family": "volatility",
            "expr": {
                "op": "sub",
                "left": {"op": "roc", "window": 63},
                "right": {"op": "downside_vol", "window": 30},
            },
            "normalize": "cross_sectional_zscore",
            "direction": 1,
            "horizon": 21,
            "thesis": "Momentum penalized by downside volatility should lower crash exposure.",
        },
        {
            "name": f"gap_followthrough_10_{cycle}",
            "family": "seasonality",
            "expr": {
                "op": "add",
                "left": {"op": "gap", "window": 5},
                "right": {"op": "price_position", "window": 20},
            },
            "normalize": "cross_sectional_zscore",
            "direction": 1,
            "horizon": 10,
            "thesis": "Positive gaps that hold near the session range high proxy post-news drift.",
        },
        {
            "name": f"market_residual_momentum_{cycle}",
            "family": "regime",
            "expr": {"op": "beta_residual", "window": 63, "benchmark": "SPY"},
            "normalize": "cross_sectional_zscore",
            "direction": 1,
            "horizon": 21,
            "thesis": "Positive market-beta residual momentum seeks idiosyncratic alpha beyond SPY.",
        },
    ]
    specs = [dict(item) for item in seeds[:count]]
    for index in range(max(0, count - len(specs))):
        window = 21 + ((cycle + index) % 8) * 14
        specs.append(
            {
                "name": f"feature_variant_{cycle}_{index}",
                "family": "momentum" if index % 2 == 0 else "volatility",
                "expr": {
                    "op": "div",
                    "left": {"op": "roc", "window": window},
                    "right": {"op": "atr_pct", "window": 14 + index % 10},
                },
                "normalize": "cross_sectional_zscore",
                "direction": 1,
                "horizon": 21,
                "thesis": "Deterministic feature mutation over momentum and volatility windows.",
            }
        )
    return specs


def max_panel_correlation(panel: pd.DataFrame, accepted_panels: list[pd.DataFrame]) -> float:
    if panel.empty or not accepted_panels:
        return 0.0
    left = panel.stack().dropna()
    max_corr = 0.0
    for accepted in accepted_panels:
        right = accepted.stack().dropna()
        joined = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
        if len(joined) < 100:
            continue
        corr = joined["left"].corr(joined["right"])
        if pd.notna(corr):
            max_corr = max(max_corr, abs(float(corr)))
    return max_corr


def build_feature_pool(
    raw_features: list[tuple[dict[str, Any], str]],
    frames: dict[str, pd.DataFrame],
    candidates_path: Path,
    universe: list[str],
) -> tuple[dict[str, FeatureSpec], list[dict[str, Any]]]:
    accepted: dict[str, FeatureSpec] = {}
    accepted_panels: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    seen_ids = {str(row.get("feature_id")) for row in read_jsonl(candidates_path, limit=5000)}
    for raw, source in raw_features:
        spec, reasons = normalize_feature_spec(raw, source)
        if spec is None:
            rejected_id = hashlib.sha256(json.dumps(raw, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
            record = {
                "timestamp": utc_now(),
                "feature_id": f"rejected_{rejected_id}",
                "status": "rejected",
                "source": source,
                "reasons": reasons,
                "spec": raw,
            }
            append_jsonl(candidates_path, record)
            records.append(record)
            continue
        fid = feature_id(spec)
        if fid in seen_ids:
            record = {
                "timestamp": utc_now(),
                "feature_id": fid,
                "feature": spec.name,
                "status": "duplicate",
                "source": source,
                "reasons": ["duplicate"],
                "spec": asdict(spec),
            }
            append_jsonl(candidates_path, record)
            records.append(record)
            continue
        result, panel = evaluate_feature_spec(spec, frames, universe)
        corr = max_panel_correlation(panel, accepted_panels)
        result["max_abs_corr_to_pool"] = corr
        result["spec"] = asdict(spec)
        result["timestamp"] = utc_now()
        if result["status"] == "accepted" and corr <= 0.92:
            accepted[spec.name] = spec
            accepted_panels.append(panel)
        elif result["status"] == "accepted":
            result["status"] = "rejected_redundant"
            result["reasons"] = ["max_abs_corr_to_pool > 0.92"]
        append_jsonl(candidates_path, result)
        records.append(result)
        seen_ids.add(fid)
    return accepted, records


def feature_pool_context(records: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    accepted = [record for record in records if record.get("status") == "accepted"]
    accepted.sort(key=lambda item: float(item.get("score") or -999.0), reverse=True)
    return [
        {
            "name": record.get("feature"),
            "family": record.get("family"),
            "score": record.get("score"),
            "rank_ic": record.get("rank_ic"),
            "decile_spread_pct": record.get("decile_spread_pct"),
            "stability_pct": record.get("stability_pct"),
            "thesis": record.get("thesis"),
        }
        for record in accepted[:limit]
    ]


def enrich_strategy_candidates_with_features(
    raw_candidates: list[tuple[dict[str, Any], str]], feature_context: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], str]]:
    feature_names = [str(item.get("name")) for item in feature_context if item.get("name")]
    if not feature_names:
        return raw_candidates
    enriched = []
    for index, (raw, source) in enumerate(raw_candidates):
        item = dict(raw)
        if not item.get("feature_refs"):
            width = min(3, len(feature_names))
            refs = [feature_names[(index + offset) % len(feature_names)] for offset in range(width)]
            item["feature_refs"] = refs
            item["rank_terms"] = refs[:2]
            item["entry_filters"] = refs[:1]
        enriched.append((item, source))
    return enriched
