import unittest

import numpy as np
import pandas as pd

from stock_recommender.features.feature_pipeline import FeaturePipeline
from stock_recommender.features.technical_indicators import MODEL_FEATURE_COLS, N_MODEL_FEATURES, compute_all


def make_history(days: int = 320, start_price: float = 100.0, drift: float = 0.0015):
    dates = pd.bdate_range("2024-01-01", periods=days)
    rows = []
    price = start_price
    bench = 1000.0
    sector = 900.0
    for i, date in enumerate(dates):
        growth = max(1.0 + drift + 0.01 * np.sin(i / 7.0), 0.94)
        bench_growth = max(1.0 + 0.0008 + 0.006 * np.cos(i / 11.0), 0.96)
        sector_growth = max(1.0 + 0.0010 + 0.004 * np.sin(i / 13.0), 0.96)
        open_price = price
        close_price = price * growth
        high_price = max(open_price, close_price) * 1.01
        low_price = min(open_price, close_price) * 0.99
        volume = 1_000_000 + i * 1_500
        bench *= bench_growth
        sector *= sector_growth
        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "open": float(open_price),
                "high": float(high_price),
                "low": float(low_price),
                "close": float(close_price),
                "volume": float(volume),
                "benchmark_close": float(bench),
                "sector_benchmark_close": float(sector),
                "delivery_pct": 48.0 + (i % 7),
                "put_call_ratio": 0.95 + 0.01 * np.sin(i / 10.0),
            }
        )
        price = close_price
    return pd.DataFrame(rows)


class TechnicalIndicatorsTests(unittest.TestCase):
    def test_feature_count_matches_model_columns(self):
        self.assertEqual(N_MODEL_FEATURES, len(MODEL_FEATURE_COLS))

    def test_compute_all_exposes_goal_indicators(self):
        df = make_history()
        enriched = compute_all(df)

        expected = {
            "psar_gap",
            "ichimoku_tenkan_gap",
            "ichimoku_senkou_b_gap",
            "supertrend_gap",
            "keltner_width",
            "donchian_pct",
            "elder_bull_power",
            "chaikin_osc",
            "force_index_13",
            "classic_pivot_gap",
            "fib_618_gap",
            "rank_52w",
            "relative_strength_20d",
            "sector_relative_strength_20d",
            "obv_slope_10",
            "delivery_pct_feature",
            "put_call_ratio_feature",
        }
        self.assertTrue(expected.issubset(set(enriched.columns)))

        latest = enriched.iloc[-1]
        self.assertTrue(np.isfinite(latest["psar_gap"]))
        self.assertTrue(np.isfinite(latest["supertrend_gap"]))
        self.assertTrue(np.isfinite(latest["rank_52w"]))
        self.assertNotEqual(float(latest["relative_strength_20d"]), 0.0)
        self.assertNotEqual(float(latest["delivery_pct_feature"]), 0.0)

    def test_feature_pipeline_snapshot_uses_expanded_feature_set(self):
        df = make_history()
        pipeline = FeaturePipeline()
        pipeline.fit(df)
        snapshot = pipeline.get_latest_snapshot(df)

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.shape[0], N_MODEL_FEATURES)


if __name__ == "__main__":
    unittest.main()
