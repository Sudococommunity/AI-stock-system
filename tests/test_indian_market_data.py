import json
import shutil
import unittest
import uuid
from pathlib import Path

import stock_recommender.data.indian_market_data as india_data
from stock_recommender.data.indian_market_data import (
    ingest_indian_market_dataset,
    ingest_indian_market_history,
    normalize_rapidapi_corporate_actions,
    sync_corporate_actions_from_rapidapi,
)
from tests.postgres_test_utils import create_test_db_manager, reset_database


CSV_TEXT = """Date,Open,High,Low,Close,Volume
2024-01-01,100,101,99,100.5,100000
2024-01-02,100.5,102,100,101.5,120000
2024-01-03,101.5,103,101,102.0,150000
"""


class IndianMarketDataTests(unittest.TestCase):
    def setUp(self):
        self.test_root = Path("D:/AI/hackathon/.tmp_test_data") / str(uuid.uuid4())
        self.test_root.mkdir(parents=True, exist_ok=True)
        self.db = create_test_db_manager()

    def tearDown(self):
        reset_database(self.db)
        self.db.close()
        shutil.rmtree(self.test_root, ignore_errors=True)

    def test_csv_ingestion_imports_stock_history(self):
        data_dir = self.test_root / "csv"
        data_dir.mkdir()
        (data_dir / "RELIANCE.csv").write_text(CSV_TEXT, encoding="utf-8")

        metadata_path = self.test_root / "metadata.json"
        metadata_path.write_text(
            json.dumps({"RELIANCE": {"name": "Reliance Industries", "sector": "Energy"}}),
            encoding="utf-8",
        )

        result = ingest_indian_market_history(
            db=self.db,
            data_dir=str(data_dir),
            metadata_path=str(metadata_path),
            market_prefix="NSE",
        )

        self.assertEqual(result.imported_stocks, 1)
        stock_id = self.db.get_stock_id("NSE:RELIANCE")
        self.assertIsNotNone(stock_id)
        history = self.db.get_price_history(stock_id, limit=10)
        self.assertEqual(len(history), 3)

    def test_internet_or_csv_falls_back_to_csv(self):
        data_dir = self.test_root / "csv"
        data_dir.mkdir()
        (data_dir / "TCS.csv").write_text(CSV_TEXT, encoding="utf-8")

        original = india_data.ingest_from_yfinance
        india_data.ingest_from_yfinance = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("no internet"))
        try:
            result = ingest_indian_market_dataset(
                db=self.db,
                source="internet_or_csv",
                data_dir=str(data_dir),
                symbols=["TCS.NS"],
                market_prefix="NSE",
            )
        finally:
            india_data.ingest_from_yfinance = original

        self.assertEqual(result.imported_stocks, 1)
        self.assertIsNotNone(self.db.get_stock_id("NSE:TCS"))

    def test_normalize_rapidapi_actions(self):
        payload = {
            "data": [
                {
                    "date": "2024-01-01",
                    "type": "dividend",
                    "purpose": "Interim Dividend",
                    "details": "Rs 10 per share",
                }
            ]
        }
        actions = normalize_rapidapi_corporate_actions(payload)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["action_type"], "dividend")
        self.assertEqual(actions[0]["title"], "Interim Dividend")

    def test_sync_corporate_actions_persists_rows(self):
        stock_id = self.db.upsert_stock("NSE:INFY", "infosys", "Technology")

        original = india_data.fetch_rapidapi_corporate_actions
        india_data.fetch_rapidapi_corporate_actions = lambda *args, **kwargs: {
            "data": [
                {"date": "2024-01-01", "type": "dividend", "purpose": "Interim Dividend"},
                {"date": "2024-02-01", "type": "split", "purpose": "Stock Split"},
            ]
        }
        try:
            result = sync_corporate_actions_from_rapidapi(
                db=self.db,
                stock_names=["infosys"],
                rapidapi_key="dummy",
            )
        finally:
            india_data.fetch_rapidapi_corporate_actions = original

        self.assertEqual(result.synced_stocks, 1)
        self.assertEqual(result.inserted_actions, 2)
        actions = self.db.get_corporate_actions(stock_id, limit=10)
        self.assertEqual(len(actions), 2)


if __name__ == "__main__":
    unittest.main()
