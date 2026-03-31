import os

from stock_recommender.config import CONFIG
from stock_recommender.data.database import DatabaseManager


TEST_DB_URL = (
    os.getenv("STOCK_RECOMMENDER_TEST_DB_URL")
    or os.getenv("STOCK_RECOMMENDER_DB_URL")
    or CONFIG.data.db_url
)


def reset_database(db: DatabaseManager) -> None:
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                TRUNCATE TABLE
                    recommendation_log,
                    user_embeddings,
                    stock_embeddings,
                    user_events,
                    price_history,
                    corporate_actions,
                    fno_snapshots,
                    model_checkpoints,
                    training_metrics,
                    users,
                    stocks
                RESTART IDENTITY CASCADE
                """
            )


def create_test_db_manager() -> DatabaseManager:
    db = DatabaseManager(TEST_DB_URL)
    reset_database(db)
    return db
