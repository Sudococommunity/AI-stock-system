import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from stock_recommender.config import CONFIG
from stock_recommender.evaluation.walk_forward import WalkForwardEvaluator
from stock_recommender.models.checkpoint_utils import can_load_model, load_model_state
from stock_recommender.models.time_series import StockTransformer


@dataclass
class TournamentEntry:
    checkpoint_path: str
    reward_score: float
    direction_accuracy: float
    mean_abs_error: float
    grade: str
    sample_count: int


@dataclass
class TournamentResult:
    leaderboard: List[TournamentEntry]
    survivors: List[TournamentEntry]

    def to_dict(self) -> Dict:
        return {
            "leaderboard": [asdict(x) for x in self.leaderboard],
            "survivors": [asdict(x) for x in self.survivors],
        }


class ModelTournament:
    """
    Grades multiple forecasting checkpoints on the same walk-forward task
    and keeps the strongest versions alive.
    """

    def __init__(self, evaluator: WalkForwardEvaluator, device: Optional[str] = None):
        self.evaluator = evaluator
        self.device = device or str(evaluator.device)

    def run(
        self,
        checkpoint_paths: List[str],
        survivor_count: Optional[int] = None,
        stock_ids: Optional[List[int]] = None,
    ) -> TournamentResult:
        entries: List[TournamentEntry] = []
        survivor_count = survivor_count or CONFIG.runtime.top_k_checkpoints

        for checkpoint_path in checkpoint_paths:
            model = StockTransformer().to(self.device)
            if not can_load_model(model, checkpoint_path, map_location=self.device):
                continue
            load_model_state(model, checkpoint_path, map_location=self.device)
            self.evaluator.transformer = model
            result = self.evaluator.evaluate(stock_ids=stock_ids)
            summary = result.summary
            entries.append(
                TournamentEntry(
                    checkpoint_path=checkpoint_path,
                    reward_score=summary.reward_score,
                    direction_accuracy=summary.direction_accuracy,
                    mean_abs_error=summary.mean_abs_error,
                    grade=summary.grade,
                    sample_count=summary.sample_count,
                )
            )

        entries.sort(key=lambda x: (x.reward_score, x.direction_accuracy, -x.mean_abs_error), reverse=True)
        survivors = entries[:survivor_count]
        return TournamentResult(leaderboard=entries, survivors=survivors)

    def save_report(self, result: TournamentResult, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
