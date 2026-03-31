import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np

from stock_recommender.data.synthetic_bot_data import generate_synthetic_users, seed_database
from tests.postgres_test_utils import create_test_db_manager, reset_database


class SyntheticBotDataTests(unittest.TestCase):
    def setUp(self):
        self.test_root = Path("D:/AI/hackathon/.tmp_test_data") / str(uuid.uuid4())
        self.test_root.mkdir(parents=True, exist_ok=True)
        self.db = create_test_db_manager()
        np.random.seed(7)

    def tearDown(self):
        reset_database(self.db)
        self.db.close()
        shutil.rmtree(self.test_root, ignore_errors=True)

    def test_generate_synthetic_users_has_personas_and_activity_variation(self):
        users = generate_synthetic_users(64)

        personas = {user["bot_persona"] for user in users}
        activity_scales = [float(user["activity_scale"]) for user in users]

        self.assertGreaterEqual(len(personas), 4)
        self.assertGreater(max(activity_scales), min(activity_scales) * 2.0)

    def test_seed_database_spreads_events_and_creates_learnable_mixed_behavior(self):
        result = seed_database(self.db, n_users=32, n_days=260)
        self.assertGreater(result["n_events"], 0)

        with self.db.connection() as conn:
            with self.db._cur(conn) as cur:
                cur.execute(
                    """
                    SELECT user_id, COUNT(*) AS n_events,
                           SUM(CASE WHEN reward > 0.02 THEN 1 ELSE 0 END) AS pos_events,
                           SUM(CASE WHEN reward < -0.02 THEN 1 ELSE 0 END) AS neg_events
                    FROM user_events
                    GROUP BY user_id
                    ORDER BY user_id
                    """
                )
                per_user = cur.fetchall()
                cur.execute(
                    """
                    SELECT COUNT(DISTINCT ue.user_id) AS n_users
                    FROM user_events ue
                    JOIN users u ON u.user_id = ue.user_id
                    JOIN stocks s ON s.stock_id = ue.stock_id
                    WHERE ue.reward > 0.02
                      AND (
                        u.preferred_sectors = '[]'
                        OR POSITION(s.sector IN u.preferred_sectors) = 0
                      )
                    """
                )
                exploratory_users = cur.fetchone()
                cur.execute(
                    """
                    SELECT COUNT(*) AS n_users FROM (
                        SELECT user_id, stock_id
                        FROM user_events
                        GROUP BY user_id, stock_id
                        HAVING SUM(CASE WHEN event_type='watchlist_add' THEN 1 ELSE 0 END) > 0
                           AND SUM(CASE WHEN event_type='watchlist_remove' THEN 1 ELSE 0 END) > 0
                    ) AS contradictions
                    """
                )
                contradictory_users = cur.fetchone()
                cur.execute("SELECT MIN(timestamp) AS min_ts, MAX(timestamp) AS max_ts FROM user_events")
                timestamps = cur.fetchone()
                cur.execute(
                    """
                    SELECT stock_id,
                           TO_CHAR(TO_TIMESTAMP(timestamp), 'YYYY-MM-DD') AS event_day,
                           COUNT(*) AS herd_size
                    FROM user_events
                    GROUP BY stock_id, event_day
                    HAVING COUNT(*) >= 3
                    ORDER BY herd_size DESC
                    LIMIT 5
                    """
                )
                herd_rows = cur.fetchall()
                cur.execute(
                    """
                    SELECT stock_id,
                           TO_CHAR(TO_TIMESTAMP(timestamp), 'YYYY-MM-DD') AS event_day,
                           SUM(CASE WHEN event_type IN ('trade_buy','watchlist_add','view_long') THEN 1 ELSE 0 END) AS bullish,
                           SUM(CASE WHEN event_type IN ('trade_sell','watchlist_remove','view_short') THEN 1 ELSE 0 END) AS bearish
                    FROM user_events
                    GROUP BY stock_id, event_day
                    HAVING SUM(CASE WHEN event_type IN ('trade_buy','watchlist_add','view_long') THEN 1 ELSE 0 END) >= 2
                       AND SUM(CASE WHEN event_type IN ('trade_sell','watchlist_remove','view_short') THEN 1 ELSE 0 END) >= 2
                    ORDER BY (SUM(CASE WHEN event_type IN ('trade_buy','watchlist_add','view_long') THEN 1 ELSE 0 END)
                           + SUM(CASE WHEN event_type IN ('trade_sell','watchlist_remove','view_short') THEN 1 ELSE 0 END)) DESC
                    LIMIT 5
                    """
                )
                contra_rows = cur.fetchall()

        counts = [int(row["n_events"]) for row in per_user]
        self.assertEqual(len(per_user), 32)
        self.assertTrue(all(int(row["pos_events"]) > 0 for row in per_user))
        self.assertTrue(all(int(row["neg_events"]) > 0 for row in per_user))
        self.assertGreater(max(counts), min(counts) * 4.0)
        self.assertGreaterEqual(int(exploratory_users["n_users"]), 8)
        self.assertGreaterEqual(int(contradictory_users["n_users"]), 8)
        self.assertGreater(float(timestamps["max_ts"]) - float(timestamps["min_ts"]), 100 * 86400)
        self.assertGreaterEqual(len(herd_rows), 1)
        self.assertGreaterEqual(len(contra_rows), 1)

    def test_different_random_seeds_change_user_interaction_structure(self):
        def build_snapshot(seed: int):
            root = self.test_root / str(uuid.uuid4())
            root.mkdir(parents=True, exist_ok=True)
            np.random.seed(seed)
            db = create_test_db_manager()
            seed_database(db, n_users=16, n_days=180)

            with db.connection() as conn:
                with db._cur(conn) as cur:
                    cur.execute("SELECT username, preferred_sectors FROM users ORDER BY user_id")
                    users = [(row["username"], row["preferred_sectors"]) for row in cur.fetchall()]
                    cur.execute(
                        """
                        SELECT user_id, event_type, COUNT(*) AS n
                        FROM user_events
                        GROUP BY user_id, event_type
                        ORDER BY user_id, event_type
                        """
                    )
                    events = [
                        (int(row["user_id"]), row["event_type"], int(row["n"]))
                        for row in cur.fetchall()
                    ]
            reset_database(db)
            db.close()
            shutil.rmtree(root, ignore_errors=True)
            return users, events

        users_a, events_a = build_snapshot(11)
        users_b, events_b = build_snapshot(19)

        self.assertNotEqual(users_a, users_b)
        self.assertNotEqual(events_a, events_b)


if __name__ == "__main__":
    unittest.main()
