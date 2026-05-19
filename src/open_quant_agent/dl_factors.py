from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - exercised only when torch is missing
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


FEATURE_COLUMNS = [
    "ret_1",
    "ret_5",
    "ret_21",
    "realized_vol_10",
    "downside_vol_10",
    "range_pct",
    "volume_z_20",
]


@dataclass(frozen=True)
class DLFactorConfig:
    model_type: str
    lookback: int = 32
    horizon: int = 21
    hidden_size: int = 24
    num_layers: int = 1
    max_epochs: int = 2
    batch_size: int = 256
    learning_rate: float = 1e-3
    train_fraction: float = 0.70
    max_train_samples: int = 4096
    seed: int = 17
    min_samples: int = 160
    device: str = "cpu"


class LSTMFactorModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :]).squeeze(-1)


class TransformerFactorModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_size, hidden_size)
        heads = 4 if hidden_size % 4 == 0 else 2
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 2,
            dropout=0.05,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, num_layers))
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.proj(x))
        return self.head(encoded[:, -1, :]).squeeze(-1)


class TemporalFusionLiteModel(nn.Module):
    """Compact Temporal Fusion style model with gated sequence and context paths."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.var_gate = nn.Sequential(nn.Linear(input_size, input_size), nn.Sigmoid())
        self.sequence = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.context = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.fusion_gate = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.Sigmoid())
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = x * self.var_gate(x)
        sequence_output, _ = self.sequence(gated)
        seq_last = sequence_output[:, -1, :]
        context = self.context(gated.mean(dim=1))
        gate = self.fusion_gate(torch.cat([seq_last, context], dim=1))
        fused = gate * seq_last + (1.0 - gate) * context
        return self.head(fused).squeeze(-1)


def build_dl_factor_panel(
    frames: dict[str, pd.DataFrame],
    universe: list[str],
    config: DLFactorConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if torch is None or nn is None or DataLoader is None or TensorDataset is None:
        raise RuntimeError("PyTorch is required for DL factor generation")
    if config.model_type not in {"lstm", "transformer", "temporal_fusion"}:
        raise ValueError(f"unsupported model_type={config.model_type}")

    rng = np.random.default_rng(config.seed)
    dates = _all_dates(frames, universe)
    if len(dates) < config.lookback + config.horizon + 20:
        return pd.DataFrame(), {"status": "insufficient_dates", "dates": len(dates)}
    cutoff_index = max(config.lookback + 1, min(len(dates) - config.horizon - 1, int(len(dates) * config.train_fraction)))
    cutoff_date = dates[cutoff_index]

    samples, targets, meta = _build_samples(frames, universe, config, cutoff_date)
    if len(samples) < config.min_samples:
        return pd.DataFrame(), {"status": "insufficient_samples", "samples": len(samples), "cutoff_date": str(cutoff_date)}

    train_mask = np.array([item["date"] <= cutoff_date for item in meta], dtype=bool)
    if train_mask.sum() < config.min_samples:
        return pd.DataFrame(), {"status": "insufficient_train_samples", "samples": int(train_mask.sum())}

    train_x = samples[train_mask]
    train_y = targets[train_mask]
    if len(train_x) > config.max_train_samples:
        selected = rng.choice(len(train_x), size=config.max_train_samples, replace=False)
        train_x = train_x[selected]
        train_y = train_y[selected]

    mean = train_x.reshape(-1, train_x.shape[-1]).mean(axis=0)
    std = train_x.reshape(-1, train_x.shape[-1]).std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    train_x = (train_x - mean) / std
    all_x = (samples - mean) / std

    torch.manual_seed(config.seed)
    model = _make_model(config, input_size=samples.shape[-1]).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    dataset = TensorDataset(
        torch.tensor(train_x, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    losses: list[float] = []
    model.train()
    for _ in range(config.max_epochs):
        epoch_losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(config.device)
            batch_y = batch_y.to(config.device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x)
            loss = loss_fn(prediction, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)) if epoch_losses else math.nan)

    predictions = _predict(model, all_x, config)
    panel = _predictions_to_panel(predictions, meta)
    panel = panel.where(panel.notna()).rank(axis=1, pct=True) - 0.5
    metadata = {
        "status": "ok",
        "model_type": config.model_type,
        "config": asdict(config),
        "features": FEATURE_COLUMNS,
        "samples": int(len(samples)),
        "train_samples": int(train_mask.sum()),
        "prediction_cells": int(panel.notna().sum().sum()),
        "cutoff_date": str(cutoff_date),
        "training_loss": losses,
        "oos_only_after": str(cutoff_date),
    }
    return panel.astype("float64"), metadata


def _make_model(config: DLFactorConfig, input_size: int) -> nn.Module:
    if config.model_type == "lstm":
        return LSTMFactorModel(input_size, config.hidden_size, config.num_layers)
    if config.model_type == "transformer":
        return TransformerFactorModel(input_size, config.hidden_size, config.num_layers)
    return TemporalFusionLiteModel(input_size, config.hidden_size, config.num_layers)


def _predict(model: nn.Module, samples: np.ndarray, config: DLFactorConfig) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(samples), config.batch_size * 4):
            batch = torch.tensor(samples[start : start + config.batch_size * 4], dtype=torch.float32).to(config.device)
            preds.append(model(batch).detach().cpu().numpy())
    return np.concatenate(preds) if preds else np.array([], dtype="float64")


def _predictions_to_panel(predictions: np.ndarray, meta: list[dict[str, Any]]) -> pd.DataFrame:
    values: dict[str, pd.Series] = {}
    by_ticker: dict[str, dict[Any, float]] = {}
    for value, item in zip(predictions, meta, strict=False):
        by_ticker.setdefault(str(item["ticker"]), {})[item["date"]] = float(value)
    for ticker, ticker_values in by_ticker.items():
        values[ticker] = pd.Series(ticker_values, dtype="float64")
    return pd.concat(values, axis=1).sort_index() if values else pd.DataFrame()


def _all_dates(frames: dict[str, pd.DataFrame], universe: list[str]) -> list[Any]:
    dates = set()
    for ticker in universe:
        frame = frames.get(ticker)
        if _usable_frame(frame):
            dates.update(frame.index)
    return sorted(dates)


def _build_samples(
    frames: dict[str, pd.DataFrame],
    universe: list[str],
    config: DLFactorConfig,
    cutoff_date: Any,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    samples: list[np.ndarray] = []
    targets: list[float] = []
    meta: list[dict[str, Any]] = []
    for ticker in universe:
        frame = frames.get(ticker)
        if not _usable_frame(frame):
            continue
        features, target = _features_and_target(frame, config.horizon)
        clean = pd.concat([features, target.rename("target")], axis=1).dropna()
        if len(clean) < config.lookback + config.horizon + 5:
            continue
        feature_values = clean[FEATURE_COLUMNS].astype("float64").to_numpy()
        target_values = clean["target"].astype("float64").to_numpy()
        index_values = list(clean.index)
        for end in range(config.lookback - 1, len(clean) - config.horizon):
            date = index_values[end]
            window = feature_values[end - config.lookback + 1 : end + 1]
            target_value = target_values[end]
            if not np.isfinite(window).all() or not np.isfinite(target_value):
                continue
            samples.append(window.astype("float32"))
            targets.append(float(target_value))
            meta.append({"ticker": ticker, "date": date, "train_cutoff": cutoff_date})
    if not samples:
        return np.empty((0, config.lookback, len(FEATURE_COLUMNS)), dtype="float32"), np.array([], dtype="float32"), []
    return np.stack(samples), np.array(targets, dtype="float32"), meta


def _features_and_target(frame: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.Series]:
    data = frame.copy()
    close = data["Close"].astype("float64")
    high = data["High"].astype("float64")
    low = data["Low"].astype("float64")
    volume = data["Volume"].astype("float64") if "Volume" in data.columns else pd.Series(0.0, index=data.index)
    returns = close.pct_change()
    downside = returns.where(returns < 0, 0.0)
    volume_mean = volume.rolling(20).mean()
    volume_std = volume.rolling(20).std().replace(0, np.nan)
    features = pd.DataFrame(
        {
            "ret_1": returns,
            "ret_5": close / close.shift(5) - 1.0,
            "ret_21": close / close.shift(21) - 1.0,
            "realized_vol_10": returns.rolling(10).std(),
            "downside_vol_10": downside.rolling(10).std(),
            "range_pct": (high - low) / close.replace(0, np.nan),
            "volume_z_20": (volume - volume_mean) / volume_std,
        },
        index=data.index,
    ).replace([np.inf, -np.inf], np.nan)
    target = close.shift(-horizon) / close - 1.0
    return features, target


def _usable_frame(frame: pd.DataFrame | None) -> bool:
    if frame is None or frame.empty:
        return False
    return all(column in frame.columns for column in ("High", "Low", "Close"))
