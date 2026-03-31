"""
FeaturePipeline — assembles raw OHLCV data into model-ready tensors.
Handles normalization (rolling z-score) so the model always sees
standardized inputs regardless of price scale or market regime.
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from collections import deque

from stock_recommender.features.technical_indicators import compute_all, MODEL_FEATURE_COLS, N_MODEL_FEATURES
from stock_recommender.config import CONFIG


class RollingNormalizer:
    """
    Maintains per-feature running mean and variance using Welford's algorithm.
    Allows streaming normalization without storing the full history.
    """

    def __init__(self, n_features: int, eps: float = 1e-8):
        self.n = np.zeros(n_features)
        self.mean = np.zeros(n_features)
        self.M2 = np.zeros(n_features)      # sum of squared deviations
        self.eps = eps

    def update(self, x: np.ndarray) -> None:
        """Update statistics with a new observation (shape: [n_features])."""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def update_batch(self, X: np.ndarray) -> None:
        """Update statistics with a batch of observations (shape: [T, n_features])."""
        for row in X:
            self.update(row)

    @property
    def var(self) -> np.ndarray:
        return np.where(self.n > 1, self.M2 / (self.n - 1), 1.0)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var) + self.eps

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Z-score normalize. Clips to [-5, 5] to handle outliers."""
        return np.clip((x - self.mean) / self.std, -5.0, 5.0)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Update with batch then normalize it."""
        self.update_batch(X)
        return np.clip((X - self.mean) / self.std, -5.0, 5.0)

    def state_dict(self) -> Dict:
        return {"n": self.n.tolist(), "mean": self.mean.tolist(), "M2": self.M2.tolist()}

    def load_state_dict(self, d: Dict) -> None:
        self.n = np.array(d["n"])
        self.mean = np.array(d["mean"])
        self.M2 = np.array(d["M2"])


class FeaturePipeline:
    """
    Converts an OHLCV DataFrame into a normalized feature matrix suitable
    for the time-series Transformer and the Stock Tower.

    The pipeline is stateful: the normalizer accumulates statistics across
    all stocks it processes, giving consistent scaling across the universe.
    """

    def __init__(self, seq_len: int = CONFIG.model.seq_len):
        self.seq_len = seq_len
        self.normalizer = RollingNormalizer(N_MODEL_FEATURES)
        self.feature_cols = MODEL_FEATURE_COLS
        self.n_features = N_MODEL_FEATURES

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, ohlcv_df: pd.DataFrame) -> "FeaturePipeline":
        """Compute indicators on training data and fit the normalizer."""
        features = self._compute_features(ohlcv_df)
        self.normalizer.update_batch(features)
        return self

    def transform(self, ohlcv_df: pd.DataFrame) -> np.ndarray:
        """
        Returns normalized feature matrix of shape (T, n_features).
        T = number of valid (non-NaN) timesteps in the input.
        """
        features = self._compute_features(ohlcv_df)
        return self.normalizer.transform(features)

    def fit_transform(self, ohlcv_df: pd.DataFrame) -> np.ndarray:
        features = self._compute_features(ohlcv_df)
        return self.normalizer.fit_transform(features)

    def get_sequence(
        self, ohlcv_df: pd.DataFrame, end_idx: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        Extract a fixed-length sequence ending at `end_idx` (exclusive).
        Returns shape (seq_len, n_features) or None if not enough data.
        """
        features = self._compute_features(ohlcv_df)
        if end_idx is None:
            end_idx = len(features)
        if end_idx < self.seq_len:
            return None
        seq = features[end_idx - self.seq_len : end_idx]
        return self.normalizer.transform(seq)

    def get_latest_sequence(self, ohlcv_df: pd.DataFrame) -> Optional[np.ndarray]:
        """Most recent seq_len timesteps, normalized. Used for live inference."""
        return self.get_sequence(ohlcv_df)

    def get_latest_snapshot(self, ohlcv_df: pd.DataFrame) -> Optional[np.ndarray]:
        """Single most-recent feature vector (shape: [n_features]). Used for stock tower."""
        features = self._compute_features(ohlcv_df)
        if len(features) == 0:
            return None
        return self.normalizer.transform(features[-1:]).squeeze(0)

    def make_training_windows(
        self, ohlcv_df: pd.DataFrame, horizon: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Slide a window over the price history to create supervised training samples.

        Returns:
            X: (N, seq_len, n_features)
            y: (N, 3) — [return_1d, return_5d, direction_1d]
               direction: +1 up, 0 flat (< 0.5%), -1 down
        """
        all_features = self._compute_features(ohlcv_df)
        close = ohlcv_df["close"].values
        norm_features = self.normalizer.fit_transform(all_features)

        X_list, y_list = [], []
        for i in range(self.seq_len, len(norm_features) - horizon):
            seq = norm_features[i - self.seq_len : i]

            ret_1d = (close[i + 1] - close[i]) / close[i]
            ret_5d = (close[i + horizon] - close[i]) / close[i]
            direction = 1.0 if ret_1d > 0.005 else (-1.0 if ret_1d < -0.005 else 0.0)

            X_list.append(seq)
            y_list.append([ret_1d, ret_5d, direction])

        if not X_list:
            return np.empty((0, self.seq_len, self.n_features)), np.empty((0, 3))
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _compute_features(self, ohlcv_df: pd.DataFrame) -> np.ndarray:
        """Run technical indicators and extract model columns, dropping NaNs."""
        enriched = compute_all(ohlcv_df)
        feature_df = enriched[self.feature_cols].replace([np.inf, -np.inf], np.nan)
        # Forward-fill then drop remaining NaN rows at the start
        feature_df = feature_df.ffill().dropna()
        return feature_df.values.astype(np.float32)
