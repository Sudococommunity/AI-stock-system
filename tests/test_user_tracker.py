import shutil
import unittest
import uuid
from pathlib import Path

from stock_recommender.data.user_tracker import UserTracker
from tests.postgres_test_utils import create_test_db_manager, reset_database


class UserTrackerTests(unittest.TestCase):
    def setUp(self):
        self.test_root = Path("D:/AI/hackathon/.tmp_test_data") / str(uuid.uuid4())
        self.test_root.mkdir(parents=True, exist_ok=True)
        self.db = create_test_db_manager()
        self.tracker = UserTracker(self.db)

    def tearDown(self):
        reset_database(self.db)
        self.db.close()
        shutil.rmtree(self.test_root, ignore_errors=True)

    def test_sector_diversity_uses_interacted_stock_sectors(self):
        user_id = self.db.create_user("sector_user")
        bank_stock = self.db.upsert_stock("BANK", "Bank Stock", sector="Financials")
        pharma_stock = self.db.upsert_stock("PHARMA", "Pharma Stock", sector="Healthcare")

        self.db.log_event(user_id, bank_stock, "watchlist_add", 1.0, 100.0)
        self.db.log_event(user_id, pharma_stock, "view_long", 45.0, 200.0)

        features = self.tracker.get_profile_features(user_id)
        self.assertAlmostEqual(features[6], 1.0, places=6)

    def test_sector_diversity_reflects_partial_coverage(self):
        user_id = self.db.create_user("focused_user")
        focused_stock = self.db.upsert_stock("IT1", "IT Stock", sector="Technology")
        self.db.upsert_stock("AUTO1", "Auto Stock", sector="Automotive")

        self.db.log_event(user_id, focused_stock, "watchlist_add", 1.0, 150.0)

        features = self.tracker.get_profile_features(user_id)
        self.assertAlmostEqual(features[6], 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
