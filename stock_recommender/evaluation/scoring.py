from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np


@dataclass
class ForecastScore:
    direction_accuracy: float
    mean_abs_error: float
    mean_confidence: float
    reward_score: float
    grade: str
    sample_count: int


def _grade_from_reward(reward_score: float) -> str:
    if reward_score >= 0.80:
        return "A"
    if reward_score >= 0.60:
        return "B"
    if reward_score >= 0.40:
        return "C"
    if reward_score >= 0.20:
        return "D"
    return "F"


def score_prediction_record(record: Dict) -> float:
    """
    Reward a prediction using 4-day realized movement.
    Strong correct calls are rewarded more; confident wrong calls are punished.
    """
    actual_ret = float(record.get("actual_return", 0.0))
    pred_ret = float(record.get("predicted_return", 0.0))
    p_up = float(record.get("p_up", 0.5))

    actual_dir = 1 if actual_ret > 0.0 else (-1 if actual_ret < 0.0 else 0)
    pred_dir = 1 if pred_ret > 0.0 else (-1 if pred_ret < 0.0 else 0)
    confidence = abs(p_up - 0.5) * 2.0

    if pred_dir == actual_dir:
        reward = 1.0 + min(abs(actual_ret), 0.10) * (1.0 + confidence)
    else:
        reward = -1.0 - confidence - min(abs(pred_ret - actual_ret), 0.10)

    return float(reward)


def summarize_forecast_scores(records: Iterable[Dict]) -> ForecastScore:
    records = list(records)
    if not records:
        return ForecastScore(0.0, 0.0, 0.0, -1.0, "F", 0)

    actual = np.array([float(r.get("actual_return", 0.0)) for r in records], dtype=float)
    predicted = np.array([float(r.get("predicted_return", 0.0)) for r in records], dtype=float)
    p_up = np.array([float(r.get("p_up", 0.5)) for r in records], dtype=float)

    actual_dir = np.sign(actual)
    pred_dir = np.sign(predicted)
    direction_accuracy = float(np.mean(actual_dir == pred_dir))
    mae = float(np.mean(np.abs(predicted - actual)))
    mean_confidence = float(np.mean(np.abs(p_up - 0.5) * 2.0))
    rewards = [score_prediction_record(r) for r in records]

    # Map average reward from roughly [-2, +2] into [0, 1] for easier grading.
    normalized_reward = float(np.clip((np.mean(rewards) + 2.0) / 4.0, 0.0, 1.0))
    return ForecastScore(
        direction_accuracy=direction_accuracy,
        mean_abs_error=mae,
        mean_confidence=mean_confidence,
        reward_score=normalized_reward,
        grade=_grade_from_reward(normalized_reward),
        sample_count=len(records),
    )

