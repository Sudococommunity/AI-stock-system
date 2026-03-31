import math
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from stock_recommender.config import CONFIG
from stock_recommender.features.technical_indicators import MODEL_FEATURE_COLS, N_MODEL_FEATURES, compute_all

try:
    import talib
except ImportError:
    talib = None


N_FEATURES = N_MODEL_FEATURES


def _to_ohlcv_tensor(records: List[Dict], device: torch.device) -> torch.Tensor:
    if not records:
        return torch.empty((0, 5), dtype=torch.float32, device=device)
    data = [
        [
            float(row["open"]),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
            float(row["volume"]),
        ]
        for row in records
    ]
    return torch.tensor(data, dtype=torch.float32, device=device)


def _rolling_stat(x: torch.Tensor, window: int, kind: str) -> torch.Tensor:
    out = torch.full_like(x, torch.nan)
    if x.numel() < window:
        return out
    unfolded = x.unfold(0, window, 1)
    if kind == "mean":
        vals = unfolded.mean(dim=-1)
    elif kind == "std":
        vals = unfolded.std(dim=-1, unbiased=True)
    elif kind == "min":
        vals = unfolded.min(dim=-1).values
    elif kind == "max":
        vals = unfolded.max(dim=-1).values
    elif kind == "sum":
        vals = unfolded.sum(dim=-1)
    else:
        raise ValueError(f"Unsupported rolling stat: {kind}")
    out[window - 1 :] = vals
    return out


@torch.jit.script
def _ema_kernel(x: torch.Tensor, alpha: float, min_periods: int) -> torch.Tensor:
    """TorchScript-compiled EMA kernel — eliminates Python interpreter overhead per element."""
    out = torch.empty_like(x)
    out[0] = x[0]
    for i in range(1, x.shape[0]):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    if min_periods > 1:
        out[: min_periods - 1] = float("nan")
    return out


def _ema(x: torch.Tensor, period: int, alpha: float = None, min_periods: int = 1) -> torch.Tensor:
    if x.numel() == 0:
        return x.clone()
    a = alpha if alpha is not None else (2.0 / (period + 1.0))
    return _ema_kernel(x, a, min_periods)


def _pct_change(x: torch.Tensor, period: int) -> torch.Tensor:
    out = torch.full_like(x, torch.nan)
    if x.numel() <= period:
        return out
    base = x[:-period]
    out[period:] = (x[period:] - base) / torch.where(base == 0, torch.nan, base)
    return out


def _safe_div(numer: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
    return numer / torch.where(denom == 0, torch.nan, denom)


def _shift(x: torch.Tensor, periods: int = 1) -> torch.Tensor:
    out = torch.full_like(x, torch.nan)
    if periods <= 0:
        out[: x.numel() + periods] = x[-periods:]
    elif periods < x.numel():
        out[periods:] = x[:-periods]
    return out


@torch.jit.script
def _forward_fill_2d(x: torch.Tensor) -> torch.Tensor:
    """TorchScript-compiled forward-fill for 2-D tensors (time × features)."""
    if x.numel() == 0:
        return x
    out = x.clone()
    for i in range(1, out.size(0)):
        mask = torch.isnan(out[i])
        if mask.any():
            out[i] = torch.where(mask, out[i - 1], out[i])
    return out


def _talib_or_numpy(name: str, fallback: np.ndarray, *args, **kwargs) -> np.ndarray:
    if talib is None:
        return fallback
    fn = getattr(talib, name)
    return fn(*args, **kwargs)


def compute_feature_matrix_tensor(records: List[Dict], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if not records:
        return (
            torch.empty((0, N_FEATURES), dtype=torch.float32, device=device),
            torch.empty((0,), dtype=torch.float32, device=device),
        )

    df = pd.DataFrame(records)
    enriched = compute_all(df)
    feature_df = enriched[MODEL_FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.ffill().dropna()
    if feature_df.empty:
        return (
            torch.empty((0, N_FEATURES), dtype=torch.float32, device=device),
            torch.empty((0,), dtype=torch.float32, device=device),
        )

    close = enriched.loc[feature_df.index, "close"].astype(np.float32).to_numpy()
    features = torch.tensor(feature_df.to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
    close_t = torch.tensor(close, dtype=torch.float32, device=device)
    return features, close_t


def build_training_windows_tensor(
    records: List[Dict],
    seq_len: int,
    horizon: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    precomputed_features: torch.Tensor = None,
    precomputed_close: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if precomputed_features is not None and precomputed_close is not None:
        features = precomputed_features.to(device)
        close = precomputed_close.to(device)
    else:
        features, close = compute_feature_matrix_tensor(records, device=device)
    if features.size(0) <= seq_len + horizon:
        empty_x = torch.empty((0, seq_len, N_FEATURES), dtype=torch.float32)
        empty_y = torch.empty((0, 3), dtype=torch.float32)
        empty_f = torch.empty((0, N_FEATURES), dtype=torch.float32)
        return empty_x, empty_y, empty_f

    norm = torch.clamp((features - mean) / std, -5.0, 5.0)
    sample_count = norm.size(0) - horizon - seq_len
    windows = norm.unfold(0, seq_len, 1).permute(0, 2, 1)[:sample_count].contiguous()

    base_close = close[seq_len : seq_len + sample_count]
    ret_1d = (close[seq_len + 1 : seq_len + sample_count + 1] - base_close) / base_close
    ret_hd = (close[seq_len + horizon : seq_len + horizon + sample_count] - base_close) / base_close
    direction = torch.where(ret_1d > 0.005, 1.0, torch.where(ret_1d < -0.005, -1.0, 0.0))
    y = torch.stack([ret_1d, ret_hd, direction], dim=1).to(torch.float32)
    return windows.cpu(), y.cpu(), features.cpu()


def normalizer_state_from_features(features: torch.Tensor) -> Dict:
    if features.numel() == 0:
        zeros = np.zeros((N_FEATURES,), dtype=np.float64)
        return {"n": zeros.tolist(), "mean": zeros.tolist(), "M2": zeros.tolist()}

    feature_count = features.size(0)
    mean = features.mean(dim=0)
    if feature_count > 1:
        var = features.var(dim=0, unbiased=True)
        m2 = var * (feature_count - 1)
    else:
        m2 = torch.zeros_like(mean)

    n = torch.full_like(mean, float(feature_count))
    return {
        "n": n.cpu().numpy().tolist(),
        "mean": mean.cpu().numpy().tolist(),
        "M2": m2.cpu().numpy().tolist(),
    }


def _feature_cache_path(cache_dir: str, stock_id: int, records: List[Dict]) -> str:
    first = records[0]["date"] if records else "empty"
    latest = records[-1]["date"] if records else "empty"
    count = len(records)
    first_close = f"{float(records[0]['close']):.4f}" if records else "0"
    last_close = f"{float(records[-1]['close']):.4f}" if records else "0"
    filename = f"stock_{stock_id}_f{N_FEATURES}_{count}_{first}_{latest}_{first_close}_{last_close}.pt".replace(":", "-")
    return os.path.join(cache_dir, filename)


def load_or_compute_feature_cache(
    cache_dir: str,
    stock_id: int,
    records: List[Dict],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    os.makedirs(cache_dir, exist_ok=True)
    path = _feature_cache_path(cache_dir, stock_id, records)
    if os.path.exists(path):
        payload = torch.load(path, map_location="cpu")
        return payload["features"], payload["close"]

    features, close = compute_feature_matrix_tensor(records, device=device)
    payload = {
        "features": features.cpu(),
        "close": close.cpu(),
    }
    torch.save(payload, path)
    return payload["features"], payload["close"]
