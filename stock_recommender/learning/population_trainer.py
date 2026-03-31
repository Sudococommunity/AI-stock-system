import json
import logging
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from stock_recommender.config import CONFIG
from stock_recommender.data.database import DatabaseManager
from stock_recommender.evaluation.walk_forward import WalkForwardEvaluator
from stock_recommender.features.feature_pipeline import FeaturePipeline
from stock_recommender.learning.trainer import Trainer, build_time_series_dataset
from stock_recommender.models.time_series import StockTransformer
from stock_recommender.models.two_tower import RankingModel, TwoTowerModel


logger = logging.getLogger(__name__)


def _resolve_device(device: Optional[str] = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class CandidateSpec:
    name: str
    seed: int
    learning_rate_multiplier: float
    epochs: int


@dataclass
class CandidateOutcome:
    name: str
    checkpoint_path: str
    seed: int
    learning_rate_multiplier: float
    epochs: int
    reward_score: float
    direction_accuracy: float
    mean_abs_error: float
    grade: str
    sample_count: int


@dataclass
class PopulationTrainingResult:
    leaderboard: List[CandidateOutcome]
    survivors: List[CandidateOutcome]

    def to_dict(self) -> Dict:
        return {
            "leaderboard": [asdict(x) for x in self.leaderboard],
            "survivors": [asdict(x) for x in self.survivors],
        }


class PopulationTrainer:
    """
    Train several forecasting candidates, score them on walk-forward evaluation,
    and keep the strongest survivors. The recommendation system remains untouched.
    """

    def __init__(self, db: DatabaseManager, device: Optional[str] = None):
        self.db = db
        self.device = _resolve_device(device)

    def build_candidate_specs(
        self,
        population_size: Optional[int] = None,
        epochs: Optional[int] = None,
    ) -> List[CandidateSpec]:
        population_size = population_size or CONFIG.runtime.population_size
        epochs = epochs or min(CONFIG.training.max_epochs, 3)
        lr_cycle = [0.75, 1.0, 1.25, 1.5]

        specs: List[CandidateSpec] = []
        for idx in range(population_size):
            specs.append(
                CandidateSpec(
                    name=f"candidate_{idx+1:02d}",
                    seed=1000 + idx,
                    learning_rate_multiplier=lr_cycle[idx % len(lr_cycle)],
                    epochs=epochs,
                )
            )
        return specs

    def train_population(
        self,
        stock_ids: Optional[List[int]] = None,
        population_size: Optional[int] = None,
        survivor_count: Optional[int] = None,
        epochs: Optional[int] = None,
        max_windows_per_stock: Optional[int] = None,
        eval_fraction: float = 0.20,
    ) -> PopulationTrainingResult:
        all_ids = stock_ids or self.db.get_all_stock_ids()
        survivor_count = survivor_count or CONFIG.runtime.top_k_checkpoints
        specs = self.build_candidate_specs(population_size=population_size, epochs=epochs)

        # ── Out-of-sample stock split ─────────────────────────────────────────
        # Hold out the last `eval_fraction` of stocks (by list order, which is
        # alphabetical ticker order — a reasonable proxy for an independent set).
        # Candidates are trained on train_ids and scored on eval_ids so the
        # tournament leaderboard reflects generalisation, not memorisation.
        n_eval = max(1, int(len(all_ids) * eval_fraction))
        if len(all_ids) < 4:
            # Too few stocks to split — fall back to using all for both (tiny test runs)
            train_ids = all_ids
            eval_ids = all_ids
            logger.warning(
                "[PopulationTrainer] Only %d stocks — using same set for train and eval.", len(all_ids)
            )
        else:
            train_ids = all_ids[:-n_eval]
            eval_ids = all_ids[-n_eval:]

        logger.info(
            "[PopulationTrainer] %d candidates | train_stocks=%d eval_stocks=%d",
            len(specs),
            len(train_ids),
            len(eval_ids),
        )

        base_pipeline = FeaturePipeline()
        X, y = build_time_series_dataset(self.db, base_pipeline, train_ids)
        base_normalizer_state = base_pipeline.normalizer.state_dict()

        outcomes: List[CandidateOutcome] = []
        for idx, spec in enumerate(specs, start=1):
            logger.info(
                "[PopulationTrainer] Candidate %s/%s: %s (seed=%s lr_x=%.2f epochs=%s)",
                idx,
                len(specs),
                spec.name,
                spec.seed,
                spec.learning_rate_multiplier,
                spec.epochs,
            )
            outcome = self._train_and_score_candidate(
                spec=spec,
                train_stock_ids=train_ids,
                eval_stock_ids=eval_ids,
                max_windows_per_stock=max_windows_per_stock,
                prebuilt_X=X,
                prebuilt_y=y,
                base_normalizer_state=base_normalizer_state,
            )
            logger.info(
                "[PopulationTrainer] Finished %s with grade=%s reward=%.4f acc=%.2f%% mae=%.4f",
                outcome.name,
                outcome.grade,
                outcome.reward_score,
                outcome.direction_accuracy * 100.0,
                outcome.mean_abs_error,
            )
            outcomes.append(outcome)

        outcomes.sort(
            key=lambda x: (x.reward_score, x.direction_accuracy, -x.mean_abs_error),
            reverse=True,
        )
        survivors = outcomes[:survivor_count]
        return PopulationTrainingResult(leaderboard=outcomes, survivors=survivors)

    def save_report(self, result: PopulationTrainingResult, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

    def _train_and_score_candidate(
        self,
        spec: CandidateSpec,
        train_stock_ids: List[int],
        eval_stock_ids: List[int],
        max_windows_per_stock: Optional[int],
        prebuilt_X: np.ndarray,
        prebuilt_y: np.ndarray,
        base_normalizer_state: Dict,
    ) -> CandidateOutcome:
        _set_seeds(spec.seed)
        pipeline = FeaturePipeline()
        pipeline.normalizer.load_state_dict(base_normalizer_state)
        transformer = StockTransformer()
        two_tower = TwoTowerModel()
        ranker = RankingModel()
        trainer = Trainer(
            transformer=transformer,
            two_tower=two_tower,
            ranker=ranker,
            feature_pipeline=pipeline,
            db=self.db,
            device=self.device,
        )

        for group in trainer.ts_optimizer.param_groups:
            group["lr"] *= spec.learning_rate_multiplier

        before = set(_list_transformer_checkpoints(CONFIG.data.checkpoint_dir))
        trainer.train_transformer_prebuilt(prebuilt_X, prebuilt_y, n_epochs=spec.epochs)
        after = set(_list_transformer_checkpoints(CONFIG.data.checkpoint_dir))
        new_paths = sorted(after - before)
        checkpoint_path = new_paths[-1] if new_paths else _latest_file(list(after))

        # Pass the training normalizer state so the evaluator does NOT re-fit
        # on future data (which would be temporal leakage).
        evaluator = WalkForwardEvaluator(
            transformer=transformer,
            feature_pipeline=FeaturePipeline(),
            db=self.db,
            device=self.device,
            normalizer_state=base_normalizer_state,
        )
        summary = evaluator.evaluate(
            stock_ids=eval_stock_ids,   # out-of-sample stocks only
            horizon_days=CONFIG.runtime.prediction_horizon_days,
            step_days=CONFIG.runtime.walk_forward_step_days,
            max_windows_per_stock=max_windows_per_stock,
        ).summary

        return CandidateOutcome(
            name=spec.name,
            checkpoint_path=checkpoint_path,
            seed=spec.seed,
            learning_rate_multiplier=spec.learning_rate_multiplier,
            epochs=spec.epochs,
            reward_score=summary.reward_score,
            direction_accuracy=summary.direction_accuracy,
            mean_abs_error=summary.mean_abs_error,
            grade=summary.grade,
            sample_count=summary.sample_count,
        )


def _list_transformer_checkpoints(checkpoint_dir: str) -> List[str]:
    if not os.path.isdir(checkpoint_dir):
        return []
    return [
        os.path.join(checkpoint_dir, name)
        for name in os.listdir(checkpoint_dir)
        if name.startswith("transformer_") and name.endswith(".pt")
    ]


def _latest_file(paths: List[str]) -> str:
    if not paths:
        return ""
    return max(paths, key=os.path.getmtime)


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
