import shutil
import unittest
import uuid
from pathlib import Path

from stock_recommender.learning.population_trainer import PopulationTrainer
from tests.postgres_test_utils import create_test_db_manager, reset_database


class PopulationTrainerTests(unittest.TestCase):
    def setUp(self):
        self.test_root = Path("D:/AI/hackathon/.tmp_test_data") / str(uuid.uuid4())
        self.test_root.mkdir(parents=True, exist_ok=True)
        self.db = create_test_db_manager()

    def tearDown(self):
        reset_database(self.db)
        self.db.close()
        shutil.rmtree(self.test_root, ignore_errors=True)

    def test_build_candidate_specs(self):
        trainer = PopulationTrainer(db=self.db)
        specs = trainer.build_candidate_specs(population_size=4, epochs=2)
        self.assertEqual(len(specs), 4)
        self.assertEqual(specs[0].epochs, 2)
        self.assertEqual(specs[0].seed, 1000)
        self.assertNotEqual(specs[0].learning_rate_multiplier, specs[1].learning_rate_multiplier)


if __name__ == "__main__":
    unittest.main()
