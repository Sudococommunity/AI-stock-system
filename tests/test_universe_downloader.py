import json
import shutil
import unittest
import uuid
from pathlib import Path

from stock_recommender.data.universe_downloader import (
    load_universe_from_csv,
    to_yfinance_symbol,
    write_universe_metadata,
)


class UniverseDownloaderTests(unittest.TestCase):
    def setUp(self):
        self.test_root = Path("D:/AI/hackathon/.tmp_test_data") / str(uuid.uuid4())
        self.test_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_root, ignore_errors=True)

    def test_to_yfinance_symbol(self):
        self.assertEqual(to_yfinance_symbol("RELIANCE", "NSE"), "RELIANCE.NS")
        self.assertEqual(to_yfinance_symbol("500325", "BSE"), "500325.BO")

    def test_load_universe_from_csv(self):
        path = self.test_root / "symbols.csv"
        path.write_text(
            "Symbol,Company Name,Sector\nRELIANCE,Reliance Industries,Energy\nTCS,Tata Consultancy Services,Technology\n",
            encoding="utf-8",
        )
        symbols = load_universe_from_csv(str(path), exchange="NSE")
        self.assertEqual(len(symbols), 2)
        self.assertEqual(symbols[0].ticker, "NSE:RELIANCE")
        self.assertEqual(symbols[0].provider_symbol, "RELIANCE.NS")

    def test_write_universe_metadata(self):
        path = self.test_root / "symbols.csv"
        path.write_text(
            "Symbol,Company Name,Sector\nINFY,Infosys,Technology\n",
            encoding="utf-8",
        )
        symbols = load_universe_from_csv(str(path), exchange="NSE")
        metadata_path = self.test_root / "metadata.json"
        write_universe_metadata(str(metadata_path), symbols)
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertIn("INFY", payload)
        self.assertEqual(payload["INFY"]["file"], "NSE_INFY.csv")


if __name__ == "__main__":
    unittest.main()
