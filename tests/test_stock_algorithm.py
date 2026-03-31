import unittest
from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd
import torch

from stock_recommender.config import CONFIG
from stock_recommender.features.feature_pipeline import FeaturePipeline
from stock_recommender.features.technical_indicators import N_MODEL_FEATURES
from stock_recommender.market.regime import MarketRegimeSnapshot
from stock_recommender.models.two_tower import CandidateIndex
from stock_recommender.recommendation.engine import RecommendationEngine
from stock_recommender.risk.risk_metrics import RiskProfile, compute_full_risk_profile
from tests.postgres_test_utils import create_test_db_manager, reset_database


class DummyTwoTower(torch.nn.Module):
    def eval(self):
        return self

    def encode_user(self, user_ids, profile_features, history_embeds=None):
        batch = user_ids.shape[0]
        out = torch.zeros((batch, CONFIG.model.embed_dim), dtype=torch.float32, device=user_ids.device)
        out[:, 0] = 1.0
        return out


class DummyRanker(torch.nn.Module):
    def eval(self):
        return self

    def forward(self, user_emb, stock_emb, risk_features, forecast_features):
        return (0.6 * forecast_features[:, 1:2]) + (0.4 * forecast_features[:, 2:3])


class DummyTransformer(torch.nn.Module):
    def eval(self):
        return self

    def predict(self, x):
        batch = x.shape[0]
        up = np.full(batch, 0.7, dtype=np.float32)
        down = np.full(batch, 0.1, dtype=np.float32)
        flat = np.full(batch, 0.2, dtype=np.float32)
        return {
            "ret_1d_forecast": np.full(batch, 0.01, dtype=np.float32),
            "ret_5d_forecast": np.full(batch, 0.03, dtype=np.float32),
            "direction_probs": np.stack([down, flat, up], axis=1),
            "predicted_direction": np.ones(batch, dtype=np.int64),
        }


def make_price_history(days: int = 320, start_price: float = 100.0, drift: float = 0.002):
    dates = pd.bdate_range("2024-01-01", periods=days)
    rows = []
    price = start_price
    for i, date in enumerate(dates):
        growth = 1.0 + drift + 0.012 * np.sin(i / 5) + 0.006 * np.cos(i / 11)
        growth = max(growth, 0.96)
        open_price = price
        close_price = price * growth
        high_price = max(open_price, close_price) * 1.01
        low_price = min(open_price, close_price) * 0.99
        volume = 1_000_000 + i * 1000
        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "open": float(open_price),
                "high": float(high_price),
                "low": float(low_price),
                "close": float(close_price),
                "volume": float(volume),
            }
        )
        price = close_price
    return rows


class RecommendationEngineTests(unittest.TestCase):
    def setUp(self):
        self.test_root = Path("D:/AI/hackathon/.tmp_test_data") / str(uuid.uuid4())
        self.test_root.mkdir(parents=True, exist_ok=True)
        self.db = create_test_db_manager()
        self.pipeline = FeaturePipeline()

        self.user_id = self.db.create_user("tester", risk_tolerance="moderate")
        self.stock_1 = self.db.upsert_stock("AAA", "Alpha")
        self.stock_2 = self.db.upsert_stock("BBB", "Beta")

        self.db.insert_price_batch(self.stock_1, make_price_history(start_price=100.0, drift=0.002))
        self.db.insert_price_batch(self.stock_2, make_price_history(start_price=50.0, drift=0.0015))

        for stock_id in (self.stock_1, self.stock_2):
            raw = self.db.get_price_history(stock_id, limit=500)
            self.pipeline.fit(pd.DataFrame(raw))

        self.db.save_user_embedding(self.user_id, [1.0] + [0.0] * (CONFIG.model.embed_dim - 1))
        self.db.save_stock_embedding(self.stock_1, [1.0] + [0.0] * (CONFIG.model.embed_dim - 1))
        self.db.save_stock_embedding(self.stock_2, [0.9] + [0.1] + [0.0] * (CONFIG.model.embed_dim - 2))

        self.index = CandidateIndex()
        self.index.build(
            np.array([self.stock_1, self.stock_2]),
            np.array(
                [
                    [1.0] + [0.0] * (CONFIG.model.embed_dim - 1),
                    [0.9, 0.1] + [0.0] * (CONFIG.model.embed_dim - 2),
                ],
                dtype=np.float32,
            ),
        )

        self.engine = RecommendationEngine(
            two_tower=DummyTwoTower(),
            ranker=DummyRanker(),
            transformer=DummyTransformer(),
            candidate_index=self.index,
            feature_pipeline=self.pipeline,
            db=self.db,
        )
        self.neutral_market = MarketRegimeSnapshot(
            as_of_date="2025-01-01",
            universe_size=2,
            advancing_count=1,
            declining_count=1,
            advance_decline_ratio=1.0,
            pct_above_50dma=0.5,
            pct_above_200dma=0.5,
            median_return_20d=0.0,
            median_return_60d=0.0,
            median_volatility_20d=0.2,
            temperature="neutral",
            regime="sideways",
            breadth_label="mixed",
        )

    def tearDown(self):
        reset_database(self.db)
        self.db.close()
        shutil.rmtree(self.test_root, ignore_errors=True)

    def test_feature_count_matches_config(self):
        self.assertEqual(CONFIG.model.n_tech_features, N_MODEL_FEATURES)

    def test_risk_profile_scores_are_bounded(self):
        returns = np.array([0.01, -0.02, 0.015, -0.005, 0.008, 0.012, -0.01], dtype=float)
        profile = compute_full_risk_profile(returns)
        self.assertTrue(np.isfinite(profile.sharpe_ratio))
        self.assertGreaterEqual(profile.risk_score, 0.0)
        self.assertLessEqual(profile.risk_score, 100.0)
        self.assertGreaterEqual(profile.opportunity_score, 0.0)
        self.assertLessEqual(profile.opportunity_score, 100.0)

    def test_entry_signal_thresholds_include_strong_sell(self):
        strong_buy = self.engine._compute_entry_signal(
            {"ret_5d": 0.03, "direction_probs": [0.1, 0.1, 0.8]},
            RiskProfile(sharpe_ratio=0.8),
            self.neutral_market,
        )
        strong_sell = self.engine._compute_entry_signal(
            {"ret_5d": -0.04, "direction_probs": [0.8, 0.05, 0.15]},
            RiskProfile(sharpe_ratio=-0.8),
            self.neutral_market,
        )
        hold = self.engine._compute_entry_signal(
            {"ret_5d": 0.0, "direction_probs": [0.2, 0.6, 0.2]},
            RiskProfile(sharpe_ratio=0.1),
            self.neutral_market,
        )

        self.assertEqual(strong_buy, "strong buy")
        self.assertEqual(strong_sell, "strong sell")
        self.assertEqual(hold, "hold")

    def test_recommendations_exclude_interacted_stocks(self):
        self.db.log_event(self.user_id, self.stock_1, "watchlist_add", 1.0, 100.0)

        recs = self.engine.get_recommendations(
            self.user_id,
            k=5,
            exclude_interacted=True,
            user_risk_tolerance="moderate",
        )

        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0].stock_id, self.stock_2)
        self.assertEqual(recs[0].ticker, "BBB")
        self.assertIn(recs[0].market_temperature, {"hot", "neutral", "cold"})
        self.assertTrue(recs[0].market_note)

    def test_full_analysis_returns_expected_sections(self):
        analysis = self.engine.get_full_analysis(self.user_id, self.stock_1)

        self.assertIsNotNone(analysis)
        self.assertEqual(analysis.ticker, "AAA")
        self.assertIn("FORECAST", analysis.narrative)
        self.assertIn("RISK METRICS", analysis.narrative)
        self.assertGreaterEqual(analysis.position_sizing_pct, 2.0)
        self.assertLessEqual(analysis.position_sizing_pct, 15.0)


if __name__ == "__main__":
    unittest.main()
