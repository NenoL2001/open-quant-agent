from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd


BASELINE_BENCHMARKS = ["SPY", "QQQ"]

DEFAULT_UNIVERSE = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AVGO",
    "AMD",
    "MU",
    "MRVL",
    "QCOM",
    "TXN",
    "AMAT",
    "LRCX",
    "KLAC",
    "ADI",
    "ON",
    "MPWR",
    "MCHP",
    "NXPI",
    "INTC",
    "ARM",
    "TER",
    "SNPS",
    "CDNS",
    "SMCI",
    "DELL",
    "TSM",
    "ASML",
    "COHR",
    "LITE",
    "AAOI",
    "FN",
    "GLW",
    "CIEN",
    "CRDO",
    "ANET",
    "MTSI",
    "XLK",
    "XLV",
    "XLF",
    "XLY",
    "XLI",
    "XLC",
    "XLP",
    "XLE",
    "XLU",
    "XLRE",
    "XLB",
    "SMH",
    "SOXX",
    "IWM",
    "DIA",
    "RSP",
]


def expanded_liquid_universe() -> list[str]:
    return list(DEFAULT_UNIVERSE)


def baseline_benchmarks() -> list[str]:
    return list(BASELINE_BENCHMARKS)


def all_download_tickers() -> list[str]:
    return sorted(dict.fromkeys(expanded_liquid_universe() + ["SPY", "QQQ", "SMH", "SOXX"]))


def filter_usable_frames(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {
        ticker: frame
        for ticker, frame in frames.items()
        if frame is not None and not frame.empty and all(column in frame.columns for column in ("High", "Low", "Close"))
    }


def download_ohlcv(tickers: Iterable[str], period: str = "3y", interval: str = "1d") -> dict[str, pd.DataFrame]:
    try:
        import yfinance as yf
    except Exception as exc:  # pragma: no cover - depends on optional runtime package
        raise RuntimeError("yfinance is required for online data downloads") from exc

    unique_tickers = sorted(dict.fromkeys(str(ticker).upper() for ticker in tickers if str(ticker).strip()))
    if not unique_tickers:
        return {}
    raw = yf.download(
        unique_tickers,
        period=period,
        interval=interval,
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True,
    )
    frames: dict[str, pd.DataFrame] = {}
    if raw.empty:
        return frames
    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in unique_tickers:
            if ticker not in raw.columns.get_level_values(0):
                continue
            frame = raw[ticker].copy()
            frames[ticker] = _normalize_ohlcv_frame(frame)
    else:
        frames[unique_tickers[0]] = _normalize_ohlcv_frame(raw.copy())
    usable = filter_usable_frames(frames)
    logging.info("downloaded_ohlcv tickers=%s usable=%s", len(unique_tickers), len(usable))
    return usable


def synthetic_ohlcv_frames(universe_size: int = 48, periods: int = 720, seed: int = 123) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    tickers = expanded_liquid_universe()[:universe_size]
    tickers = sorted(dict.fromkeys(tickers + ["SPY", "QQQ", "SMH", "SOXX"]))
    dates = pd.bdate_range(end=pd.Timestamp("2026-05-18"), periods=periods)
    market = rng.normal(0.00045, 0.010, size=periods)
    regime = np.sin(np.linspace(0, 9.0 * np.pi, periods)) * 0.003
    tech = rng.normal(0.00055, 0.013, size=periods) + regime
    frames: dict[str, pd.DataFrame] = {}
    for index, ticker in enumerate(tickers):
        beta = 0.65 + (index % 9) * 0.06
        sector_beta = 0.25 + (index % 5) * 0.05
        quality = (index % 7 - 3) * 0.00008
        noise = rng.normal(0.0, 0.014 + (index % 6) * 0.002, size=periods)
        returns = beta * market + sector_beta * tech + quality + noise
        for step in range(2, periods):
            returns[step] += 0.10 * returns[step - 1] - 0.04 * returns[step - 2]
            if step % 87 == index % 23:
                returns[step] += 0.035 + (index % 4) * 0.004
        close = 80.0 * np.exp(np.cumsum(returns))
        open_ = np.r_[close[0], close[:-1]] * (1.0 + rng.normal(0.0, 0.0025, size=periods))
        intraday = np.abs(rng.normal(0.010, 0.004, size=periods))
        high = np.maximum(open_, close) * (1.0 + intraday)
        low = np.minimum(open_, close) * (1.0 - intraday)
        volume = rng.lognormal(mean=15.0 + (index % 5) * 0.08, sigma=0.35, size=periods)
        frames[ticker] = pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
            index=dates,
        )
    return frames


def _normalize_ohlcv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rename = {column: str(column).title().replace(" ", "") for column in frame.columns}
    normalized = frame.rename(columns=rename)
    if "AdjClose" in normalized.columns and "Close" not in normalized.columns:
        normalized["Close"] = normalized["AdjClose"]
    columns = [column for column in ["Open", "High", "Low", "Close", "Volume"] if column in normalized.columns]
    return normalized[columns].dropna(how="all").sort_index()
