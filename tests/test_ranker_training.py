import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from stock_recommender.config import CONFIG
from stock_recommender.features.feature_pipeline import FeaturePipeline
from stock_recommender.learning.trainer import Trainer
from stock_recommender.models.time_series import StockTransformer
from stock_recommender.models.two_tower import RankingModel, TwoTowerModel
from tests.postgres_test_utils import create_test_db_manager, reset_database


def make_price_history(days: int = 320, start_price: float = 100.0, drift: float = 0.002):
    dates = pd.bdate_range("2024-01-01", periods=days)
    rows = []
    price = start_price
    for i, date in enumerate(dates):
        growth = 1.0 + drift + 0.01 * np.sin(i / 7) + 0.005 * np.cos(i / 13)
        growth = max(growth, 0.96)
        open_price = price
        close_price = price * growth
        high_price = max(open_price, close_price) * 1.01
        low_price = min(open_price, close_price) * 0.99
        volume = 1_000_000 + i * 750
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


class RankerTrainingTests(unittest.TestCase):
    def setUp(self):
        self.test_root = Path("D:/AI/hackathon/.tmp_test_data") / str(uuid.uuid4())
        self.test_root.mkdir(parents=True, exist_ok=True)
        self.db = create_test_db_manager()
        self.pipeline = FeaturePipeline()

        self.old_checkpoint_dir = CONFIG.data.checkpoint_dir
        CONFIG.data.checkpoint_dir = str(self.test_root / "checkpoints")

        self.user_id = self.db.create_user("ranker_user", risk_tolerance="moderate")
        self.stock_positive = self.db.upsert_stock("POS", "Positive")
        self.stock_negative = self.db.upsert_stock("NEG", "Negative")

        self.db.insert_price_batch(self.stock_positive, make_price_history(start_price=120.0, drift=0.0025))
        self.db.insert_price_batch(self.stock_negative, make_price_history(start_price=80.0, drift=0.0005))

        for stock_id in (self.stock_positive, self.stock_negative):
            raw = self.db.get_price_history(stock_id, limit=500)
            self.pipeline.fit(pd.DataFrame(raw))

        self.db.log_event(self.user_id, self.stock_positive, "watchlist_add", 1.0, 120.0)
        self.db.log_event(self.user_id, self.stock_positive, "view_long", 45.0, 121.0)
        self.db.log_event(self.user_id, self.stock_negative, "watchlist_remove", -1.0, 80.0)

        self.transformer = StockTransformer()
        self.two_tower = TwoTowerModel()
        self.ranker = RankingModel()
        self.trainer = Trainer(
            transformer=self.transformer,
            two_tower=self.two_tower,
            ranker=self.ranker,
            feature_pipeline=self.pipeline,
            db=self.db,
            device="cpu",
        )

    def tearDown(self):
        CONFIG.data.checkpoint_dir = self.old_checkpoint_dir
        reset_database(self.db)
        self.db.close()
        shutil.rmtree(self.test_root, ignore_errors=True)

    def test_ranker_pairs_and_training_are_non_empty(self):
        pairs = self.trainer._build_ranker_training_pairs()

        self.assertGreater(len(pairs), 0)
        self.assertEqual(pairs[0]["user_emb"].shape[0], CONFIG.model.embed_dim)
        self.assertEqual(pairs[0]["pos_stock_emb"].shape[0], CONFIG.model.embed_dim)
        self.assertEqual(pairs[0]["pos_risk_feats"].shape[0], 8)
        self.assertEqual(pairs[0]["pos_fc_feats"].shape[0], 3)

        history = self.trainer.train_ranker(n_epochs=1)
        self.assertEqual(len(history["train_loss"]), 1)


if __name__ == "__main__":
    unittest.main()
