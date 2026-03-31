import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stock_recommender.config import CONFIG
from stock_recommender.evaluation.scoring import score_prediction_record, summarize_forecast_scores
from stock_recommender.evaluation.walk_forward import WalkForwardEvaluator
from stock_recommender.features.feature_pipeline import FeaturePipeline
from tests.postgres_test_utils import create_test_db_manager, reset_database


class DummyTransformer(torch.nn.Module):
    def eval(self):
        return self

    def predict(self, x):
        batch = x.shape[0]
        return {
            "ret_1d_forecast": np.full(batch, 0.01, dtype=np.float32),
            "ret_5d_forecast": np.full(batch, 0.02, dtype=np.float32),
            "direction_probs": np.tile(np.array([[0.2, 0.2, 0.6]], dtype=np.float32), (batch, 1)),
            "predicted_direction": np.ones(batch, dtype=np.int64),
        }


def make_price_history(days: int = 320):
    dates = pd.bdate_range("2024-01-01", periods=days)
    rows = []
    price = 100.0
    for i, date in enumerate(dates):
        growth = 1.0 + 0.001 + 0.01 * np.sin(i / 4.0) + 0.005 * np.cos(i / 9.0)
        growth = max(growth, 0.96)
        open_price = price
        close_price = price * growth
        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "open": float(open_price),
                "high": float(max(open_price, close_price) * 1.01),
                "low": float(min(open_price, close_price) * 0.99),
                "close": float(close_price),
                "volume": float(1_000_000 + 5000 * i),
            }
        )
        price = close_price
    return rows


class WalkForwardTests(unittest.TestCase):
    def setUp(self):
        self.test_root = Path("D:/AI/hackathon/.tmp_test_data") / str(uuid.uuid4())
        self.test_root.mkdir(parents=True, exist_ok=True)
        self.db = create_test_db_manager()
        self.pipeline = FeaturePipeline()
        self.stock_id = self.db.upsert_stock("NSETEST", "NSE Test", "Technology")
        self.db.insert_price_batch(self.stock_id, make_price_history())

    def tearDown(self):
        reset_database(self.db)
        self.db.close()
        shutil.rmtree(self.test_root, ignore_errors=True)

    def test_scoring_rewards_correct_direction(self):
        good = score_prediction_record({"actual_return": 0.03, "predicted_return": 0.02, "p_up": 0.8})
        bad = score_prediction_record({"actual_return": -0.03, "predicted_return": 0.02, "p_up": 0.8})
        self.assertGreater(good, bad)

    def test_walk_forward_generates_records(self):
        evaluator = WalkForwardEvaluator(
            transformer=DummyTransformer(),
            feature_pipeline=self.pipeline,
            db=self.db,
        )
        result = evaluator.evaluate(
            stock_ids=[self.stock_id],
            horizon_days=4,
            step_days=20,
            max_windows_per_stock=3,
        )
        self.assertGreaterEqual(result.summary.sample_count, 1)
        self.assertIn(result.summary.grade, {"A", "B", "C", "D", "F"})
        self.assertEqual(len(result.records), result.summary.sample_count)
        self.assertEqual(result.records[0]["ticker"], "NSETEST")

    def test_summary_is_bounded(self):
        summary = summarize_forecast_scores(
            [
                {"actual_return": 0.01, "predicted_return": 0.02, "p_up": 0.7},
                {"actual_return": -0.02, "predicted_return": -0.01, "p_up": 0.3},
            ]
        )
        self.assertGreaterEqual(summary.reward_score, 0.0)
        self.assertLessEqual(summary.reward_score, 1.0)
        self.assertGreaterEqual(summary.direction_accuracy, 0.0)
        self.assertLessEqual(summary.direction_accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
