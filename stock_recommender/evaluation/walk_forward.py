from dataclasses import asdict, dataclass
from typing import Dict, List, Optional
import logging

import numpy as np
import pandas as pd
import torch

from stock_recommender.config import CONFIG
from stock_recommender.data.database import DatabaseManager
from stock_recommender.features.feature_pipeline import FeaturePipeline
from stock_recommender.features.tensor_preprocessing import (
    load_or_compute_feature_cache,
    normalizer_state_from_features,
)
from stock_recommender.models.time_series import StockTransformer
from stock_recommender.evaluation.scoring import ForecastScore, summarize_forecast_scores


logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    summary: ForecastScore
    records: List[Dict]

    def to_dict(self) -> Dict:
        return {
            "summary": asdict(self.summary),
            "records": self.records,
        }


class WalkForwardEvaluator:
    """
    Time-based evaluator for next-N-day forecasting.
    This is the correct evaluation mode for market models; no random split leakage.
    """

    def __init__(
        self,
        transformer: StockTransformer,
        feature_pipeline: FeaturePipeline,
        db: DatabaseManager,
        device: Optional[str] = None,
        normalizer_state: Optional[Dict] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.transformer = transformer.to(self.device)
        self.pipeline = feature_pipeline
        self.db = db
        # When provided, this normalizer state (fitted on training stocks only)
        # is used instead of re-fitting on evaluation data, preventing temporal leakage.
        self._normalizer_state = normalizer_state

    def evaluate(
        self,
        stock_ids: Optional[List[int]] = None,
        horizon_days: Optional[int] = None,
        step_days: Optional[int] = None,
        max_windows_per_stock: Optional[int] = None,
    ) -> WalkForwardResult:
        stock_ids = stock_ids or self.db.get_all_stock_ids()
        horizon_days = horizon_days or CONFIG.runtime.prediction_horizon_days
        step_days = step_days or CONFIG.runtime.walk_forward_step_days

        if self._normalizer_state is not None:
            # Use caller-supplied normalizer (e.g. fitted on training stocks only).
            # This avoids temporal leakage when called from population_trainer.
            self.pipeline.normalizer.load_state_dict(self._normalizer_state)
            logger.info("[WalkForward] Using pre-fitted normalizer (no temporal leakage).")
        else:
            # Standalone mode: fit on the first 80% of each stock's history so
            # the normalizer has never seen the evaluation period.
            logger.info("[WalkForward] Fitting normalizer on first 80%% of history (%s stocks)...", len(stock_ids))
            feature_batches: List[torch.Tensor] = []
            for idx, sid in enumerate(stock_ids, start=1):
                raw = self.db.get_price_history(sid, limit=10_000)
                if len(raw) < CONFIG.model.seq_len + horizon_days + 5:
                    continue
                # Only use the first 80% of history for normalizer fitting
                cutoff = max(CONFIG.model.seq_len + horizon_days + 5, int(len(raw) * 0.80))
                train_raw = raw[:cutoff]
                features, _ = load_or_compute_feature_cache(
                    CONFIG.data.tensor_cache_dir,
                    sid,
                    train_raw,
                    device=self.device,
                )
                if features.numel() > 0:
                    feature_batches.append(features)
                if idx % 100 == 0 or idx == len(stock_ids):
                    logger.info("[WalkForward] Normalizer progress: %s/%s stocks", idx, len(stock_ids))

            if feature_batches:
                stacked = torch.cat(feature_batches, dim=0)
                self.pipeline.normalizer.load_state_dict(normalizer_state_from_features(stacked))

        records: List[Dict] = []
        logger.info("[WalkForward] Evaluating %s stocks...", len(stock_ids))
        for idx, sid in enumerate(stock_ids, start=1):
            records.extend(
                self._evaluate_stock(
                    sid,
                    horizon_days=horizon_days,
                    step_days=step_days,
                    max_windows=max_windows_per_stock,
                )
            )
            if idx % 100 == 0 or idx == len(stock_ids):
                logger.info("[WalkForward] Evaluation progress: %s/%s stocks", idx, len(stock_ids))

        return WalkForwardResult(summary=summarize_forecast_scores(records), records=records)

    def _evaluate_stock(
        self,
        stock_id: int,
        horizon_days: int,
        step_days: int,
        max_windows: Optional[int] = None,
    ) -> List[Dict]:
        raw = self.db.get_price_history(stock_id, limit=5000)
        if len(raw) < CONFIG.model.seq_len + horizon_days + 5:
            return []

        info = self.db.get_stock_info(stock_id) or {}
        features, _ = load_or_compute_feature_cache(
            CONFIG.data.tensor_cache_dir,
            stock_id,
            raw,
            device=self.device,
        )
        df = pd.DataFrame(raw)
        if len(features) < CONFIG.model.seq_len + horizon_days:
            return []

        offset = len(df) - len(features)
        close = df["close"].to_numpy(dtype=float)
        records: List[Dict] = []
        self.transformer.eval()

        last_end = len(features) - horizon_days
        for end_idx in range(CONFIG.model.seq_len, last_end + 1, step_days):
            seq = features[end_idx - CONFIG.model.seq_len : end_idx].cpu().numpy()
            seq = self.pipeline.normalizer.transform(seq)

            actual_idx = offset + end_idx - 1
            future_idx = actual_idx + horizon_days
            if future_idx >= len(close):
                continue

            x = torch.tensor(seq, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                pred = self.transformer.predict(x)

            predicted_return = float(pred["ret_5d_forecast"][0])
            p_up = float(pred["direction_probs"][0][2])
            actual_return = float((close[future_idx] - close[actual_idx]) / close[actual_idx])

            records.append(
                {
                    "stock_id": stock_id,
                    "ticker": info.get("ticker", str(stock_id)),
                    "cutoff_date": str(df.iloc[actual_idx]["date"]),
                    "target_date": str(df.iloc[future_idx]["date"]),
                    "predicted_return": predicted_return,
                    "actual_return": actual_return,
                    "p_up": p_up,
                }
            )
            if max_windows is not None and len(records) >= max_windows:
                break

        return records
