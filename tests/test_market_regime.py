import shutil
import uuid
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stock_recommender.config import CONFIG
from stock_recommender.features.feature_pipeline import FeaturePipeline
from stock_recommender.market.regime import MarketRegimeAnalyzer
from stock_recommender.models.two_tower import CandidateIndex
from stock_recommender.recommendation.engine import RecommendationEngine
from tests.postgres_test_utils import create_test_db_manager, reset_database


class ConstantTwoTower(torch.nn.Module):
    def eval(self):
        return self

    def encode_user(self, user_ids, profile_features, history_embeds=None):
        batch = user_ids.shape[0]
        out = torch.zeros((batch, CONFIG.model.embed_dim), dtype=torch.float32, device=user_ids.device)
        out[:, 0] = 1.0
        return out


class ConstantRanker(torch.nn.Module):
    def eval(self):
        return self

    def forward(self, user_emb, stock_emb, risk_features, forecast_features):
        batch = user_emb.shape[0]
        return torch.zeros((batch, 1), dtype=torch.float32, device=user_emb.device)


class StaticTransformer(torch.nn.Module):
    def eval(self):
        return self

    def predict(self, x):
        batch = x.shape[0]
        up = np.full(batch, 0.60, dtype=np.float32)
        down = np.full(batch, 0.15, dtype=np.float32)
        flat = np.full(batch, 0.25, dtype=np.float32)
        return {
            "ret_1d_forecast": np.full(batch, 0.004, dtype=np.float32),
            "ret_5d_forecast": np.full(batch, 0.015, dtype=np.float32),
            "direction_probs": np.stack([down, flat, up], axis=1),
            "predicted_direction": np.ones(batch, dtype=np.int64),
        }


def make_price_history(
    days: int = 260,
    start_price: float = 100.0,
    drift: float = 0.001,
    vol: float = 0.01,
):
    dates = pd.bdate_range("2024-01-01", periods=days)
    rows = []
    price = start_price
    for i, date in enumerate(dates):
        periodic = np.sin(i / 9.0) * vol * 0.35
        growth = 1.0 + drift + periodic
        growth = max(growth, 0.90)
        open_price = price
        close_price = price * growth
        high_price = max(open_price, close_price) * (1.0 + vol * 0.5)
        low_price = min(open_price, close_price) * (1.0 - vol * 0.5)
        volume = 1_000_000 + i * 5_000
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


class MarketRegimeTests(unittest.TestCase):
    def setUp(self):
        self.test_root = Path("D:/AI/Hackathon/.tmp_test_data") / str(uuid.uuid4())
        self.test_root.mkdir(parents=True, exist_ok=True)
        self.db = create_test_db_manager()

    def tearDown(self):
        reset_database(self.db)
        self.db.close()
        shutil.rmtree(self.test_root, ignore_errors=True)

    def _seed_universe(self, drifts, sectors):
        for idx, (drift, sector) in enumerate(zip(drifts, sectors), start=1):
            stock_id = self.db.upsert_stock(f"STK{idx}", f"Stock {idx}", sector=sector, market_cap=1_000_000_000 + idx)
            self.db.insert_price_batch(stock_id, make_price_history(drift=drift))

    def test_hot_market_detection_uses_breadth_and_trend(self):
        self._seed_universe(
            drifts=[0.0040, 0.0038, 0.0042, 0.0036, 0.0035, 0.0039, 0.0041, 0.0037],
            sectors=["Technology", "Financials", "Industrials", "Auto", "Energy", "Healthcare", "Utilities", "Consumer"],
        )

        snapshot = MarketRegimeAnalyzer(self.db).analyze_market()

        self.assertEqual(snapshot.temperature, "hot")
        self.assertEqual(snapshot.regime, "bull")
        self.assertGreater(snapshot.advance_decline_ratio, 1.0)
        self.assertGreater(snapshot.pct_above_200dma, 0.75)
        self.assertTrue(snapshot.leading_sectors)

    def test_cold_market_detection_uses_vix_and_breadth(self):
        self._seed_universe(
            drifts=[-0.0035, -0.0040, -0.0038, -0.0036, -0.0041, -0.0039, -0.0037, -0.0042],
            sectors=["Technology", "Financials", "Industrials", "Auto", "Energy", "Healthcare", "Utilities", "Consumer"],
        )
        vix_id = self.db.upsert_stock("INDIAVIX", "India VIX", sector="Index")
        self.db.insert_price_batch(vix_id, make_price_history(start_price=24.0, drift=0.0005, vol=0.005))

        snapshot = MarketRegimeAnalyzer(self.db).analyze_market()

        self.assertEqual(snapshot.temperature, "cold")
        self.assertEqual(snapshot.regime, "bear")
        self.assertIsNotNone(snapshot.india_vix)
        self.assertGreater(snapshot.india_vix, 20.0)
        self.assertEqual(snapshot.vix_regime, "fear")
        self.assertLess(snapshot.pct_above_200dma, 0.30)

    def test_cold_market_reranks_toward_defensive_stock(self):
        user_id = self.db.create_user("cold_user", risk_tolerance="moderate")
        safe_id = self.db.upsert_stock("SAFE", "Defensive", sector="Healthcare", market_cap=5_000_000_000)
        risky_id = self.db.upsert_stock("RISK", "Risky", sector="Technology", market_cap=4_500_000_000)

        self.db.insert_price_batch(safe_id, make_price_history(start_price=120.0, drift=0.0008, vol=0.004))
        self.db.insert_price_batch(risky_id, make_price_history(start_price=120.0, drift=-0.0010, vol=0.025))

        self._seed_universe(
            drifts=[-0.0035, -0.0038, -0.0040, -0.0036, -0.0041, -0.0037],
            sectors=["Financials", "Industrials", "Auto", "Energy", "Real Estate", "Materials"],
        )
        vix_id = self.db.upsert_stock("INDIAVIX", "India VIX", sector="Index")
        self.db.insert_price_batch(vix_id, make_price_history(start_price=25.0, drift=0.0004, vol=0.004))

        pipeline = FeaturePipeline()
        for stock_id in self.db.get_all_stock_ids():
            raw = self.db.get_price_history(stock_id, limit=500)
            if len(raw) >= 220:
                pipeline.fit(pd.DataFrame(raw))

        self.db.save_user_embedding(user_id, [1.0] + [0.0] * (CONFIG.model.embed_dim - 1))
        self.db.save_stock_embedding(risky_id, [1.0] + [0.0] * (CONFIG.model.embed_dim - 1))
        self.db.save_stock_embedding(safe_id, [0.97, 0.03] + [0.0] * (CONFIG.model.embed_dim - 2))

        index = CandidateIndex()
        index.build(
            np.array([risky_id, safe_id]),
            np.array(
                [
                    [1.0] + [0.0] * (CONFIG.model.embed_dim - 1),
                    [0.97, 0.03] + [0.0] * (CONFIG.model.embed_dim - 2),
                ],
                dtype=np.float32,
            ),
        )

        engine = RecommendationEngine(
            two_tower=ConstantTwoTower(),
            ranker=ConstantRanker(),
            transformer=StaticTransformer(),
            candidate_index=index,
            feature_pipeline=pipeline,
            db=self.db,
        )

        original_epsilon = CONFIG.model.exploration_epsilon
        CONFIG.model.exploration_epsilon = 0.0
        try:
            recs = engine.get_recommendations(user_id, k=2, exclude_interacted=False)
        finally:
            CONFIG.model.exploration_epsilon = original_epsilon

        self.assertEqual(recs[0].ticker, "SAFE")
        self.assertEqual(recs[0].market_temperature, "cold")
        self.assertIn("Cold market backdrop", recs[0].market_note)


if __name__ == "__main__":
    unittest.main()
