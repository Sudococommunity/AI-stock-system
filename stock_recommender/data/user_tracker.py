"""
UserTracker — records user interactions and computes behavioral feature vectors.

The tracker is the bridge between raw user actions and the ML models:
  • Every interaction is logged to SQLite (for replay + reward attribution)
  • A compact profile feature vector is computed from interaction history
  • Signals are weighted by recency (exponential decay) and strength

Signal weights:
  trade_buy / trade_sell  = 1.0  (strongest — user put money on the line)
  rate                    = 0.8  (explicit rating)
  watchlist_add           = 0.6  (intent signal)
  view (long)             = 0.3  (interest)
  view (short)            = 0.1  (weak signal)
  watchlist_remove        = -0.4 (negative)
"""
import time
import numpy as np
import json
from typing import Dict, List, Optional, Tuple

from stock_recommender.data.database import DatabaseManager
from stock_recommender.config import CONFIG


# Event type → base signal strength
EVENT_WEIGHTS: Dict[str, float] = {
    "trade_buy":        1.0,
    "trade_sell":       1.0,
    "rate":             0.8,
    "watchlist_add":    0.6,
    "view_long":        0.3,   # view > 30 seconds
    "view_short":       0.1,   # view < 30 seconds
    "watchlist_remove": -0.4,
}

# Recency decay: half-life of 30 days
DECAY_HALF_LIFE_SECONDS = 30 * 86400


def recency_weight(timestamp: float, now: float = None) -> float:
    """Exponential decay weight based on event age."""
    if now is None:
        now = time.time()
    age = max(now - timestamp, 0)
    return float(np.exp(-age * np.log(2) / DECAY_HALF_LIFE_SECONDS))


# Risk tolerance → scalar (for model features)
RISK_TOLERANCE_MAP = {"conservative": 0.0, "moderate": 0.5, "aggressive": 1.0}
CAPITAL_MAP = {"small": 0.0, "medium": 0.5, "large": 1.0}
HORIZON_MAP = {"short": 0.0, "medium": 0.5, "long": 1.0}


class UserTracker:
    """
    Tracks user behavior and produces feature vectors consumed by the User Tower.
    """

    def __init__(self, db: DatabaseManager):
        self.db = db

    # ── Event logging ─────────────────────────────────────────────────────────

    def log_view(
        self, user_id: int, stock_id: int, duration_seconds: float, price: float = 0.0
    ) -> int:
        event_type = "view_long" if duration_seconds > 30 else "view_short"
        return self.db.log_event(user_id, stock_id, event_type, duration_seconds, price)

    def log_watchlist_add(self, user_id: int, stock_id: int, price: float = 0.0) -> int:
        return self.db.log_event(user_id, stock_id, "watchlist_add", 1.0, price)

    def log_watchlist_remove(self, user_id: int, stock_id: int, price: float = 0.0) -> int:
        return self.db.log_event(user_id, stock_id, "watchlist_remove", -1.0, price)

    def log_rating(self, user_id: int, stock_id: int, rating: float, price: float = 0.0) -> int:
        """Rating in [-1, 1] where +1 = very positive, -1 = very negative."""
        return self.db.log_event(user_id, stock_id, "rate", rating, price)

    def log_trade(
        self,
        user_id: int,
        stock_id: int,
        action: str,   # "buy" or "sell"
        quantity: float,
        price: float,
    ) -> int:
        event_type = f"trade_{action}"
        return self.db.log_event(user_id, stock_id, event_type, quantity, price)

    # ── Reward attribution ────────────────────────────────────────────────────

    def attribute_reward(self, event_id: int, current_price: float, event_price: float) -> float:
        """
        Compute delayed reward for an event once we know the price outcome.
        reward = sign(price_change) * clipped_return
        """
        if event_price <= 0:
            return 0.0
        ret = (current_price - event_price) / event_price
        reward = float(np.sign(ret) * min(abs(ret), 0.5))   # clip at ±50%
        self.db.update_event_reward(event_id, reward)
        return reward

    # ── Feature computation ───────────────────────────────────────────────────

    def get_profile_features(self, user_id: int) -> np.ndarray:
        """
        Compute a compact profile feature vector for the User Tower.

        Feature vector (length = CONFIG.model.n_user_profile_features = 10):
          [0] risk_tolerance_scalar
          [1] capital_range_scalar
          [2] investment_horizon_scalar
          [3] activity_level   (log-scaled event count)
          [4] avg_signal_strength (recency-weighted)
          [5] win_rate          (events where reward > 0)
          [6] sector_diversity  (# distinct sectors interacted with / total sectors)
          [7] watchlist_ratio   (watchlist adds / total events)
          [8] trade_ratio       (trades / total events)
          [9] recency           (time since last event, normalized)
        """
        user = self.db.get_user(user_id)
        if user is None:
            return np.zeros(CONFIG.model.n_user_profile_features, dtype=np.float32)

        events = self.db.get_user_events(user_id, limit=500)
        now = time.time()

        # Static profile features
        rt = RISK_TOLERANCE_MAP.get(user.get("risk_tolerance", "moderate"), 0.5)
        cap = CAPITAL_MAP.get(user.get("capital_range", "medium"), 0.5)
        hor = HORIZON_MAP.get(user.get("investment_horizon", "medium"), 0.5)

        if not events:
            return np.array([rt, cap, hor, 0, 0, 0.5, 0, 0, 0, 1], dtype=np.float32)

        n_events = len(events)
        activity = min(np.log1p(n_events) / np.log1p(500), 1.0)  # normalize to [0,1]

        signals = []
        rewards = []
        watchlist_count = 0
        trade_count = 0
        for e in events:
            w = recency_weight(e["timestamp"], now)
            base = EVENT_WEIGHTS.get(e["event_type"], 0.1)
            signals.append(w * abs(base))
            if e["reward"] is not None:
                rewards.append(e["reward"])
            if e["event_type"] == "watchlist_add":
                watchlist_count += 1
            if e["event_type"].startswith("trade_"):
                trade_count += 1

        avg_signal = float(np.mean(signals)) if signals else 0.0
        win_rate = float(np.mean([1 if r > 0 else 0 for r in rewards])) if rewards else 0.5
        sector_diversity = self._compute_sector_diversity(events)

        watchlist_ratio = watchlist_count / max(n_events, 1)
        trade_ratio = trade_count / max(n_events, 1)

        last_event_age = now - events[0]["timestamp"]
        recency = float(np.exp(-last_event_age / (7 * 86400)))  # 7-day half-life

        return np.array([
            rt, cap, hor, activity, avg_signal,
            win_rate, sector_diversity,
            watchlist_ratio, trade_ratio, recency,
        ], dtype=np.float32)

    def get_interaction_stock_ids(self, user_id: int, limit: int = 50) -> List[int]:
        """
        Return the stock IDs of the user's most recent positive interactions.
        Used to build the history embedding sequence for the User Tower.
        """
        events = self.db.get_user_events(
            user_id,
            event_types=["trade_buy", "watchlist_add", "rate", "view_long"],
            limit=limit,
        )
        return [e["stock_id"] for e in events]

    def get_weighted_stock_scores(self, user_id: int) -> Dict[int, float]:
        """
        Compute a recency-weighted preference score per stock for this user.
        Positive score = user likes this stock; negative = dislikes.
        Used for training data generation.
        """
        events = self.db.get_user_events(user_id, limit=500)
        now = time.time()
        scores: Dict[int, float] = {}

        for e in events:
            sid = e["stock_id"]
            w = recency_weight(e["timestamp"], now)
            base = EVENT_WEIGHTS.get(e["event_type"], 0.1)
            signal = w * base

            # If we have a reward, modulate the signal
            if e["reward"] is not None:
                signal *= (1 + e["reward"])

            scores[sid] = scores.get(sid, 0.0) + signal

        return scores

    def get_positive_negative_pairs(
        self, user_id: int, n_pairs: int = 32
    ) -> Tuple[List[int], List[int]]:
        """
        Sample positive (liked) and negative (disliked/ignored) stock pairs for training.
        Used by the online learner to generate training examples.
        """
        scores = self.get_weighted_stock_scores(user_id)
        if not scores:
            return [], []

        sorted_stocks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        positives = [s for s, sc in sorted_stocks if sc > 0.1]
        negatives = [s for s, sc in sorted_stocks if sc < -0.05]

        n = min(n_pairs, min(len(positives), len(negatives)))
        if n == 0:
            return [], []

        pos_sample = list(np.random.choice(positives, size=n, replace=len(positives) < n))
        neg_sample = list(np.random.choice(negatives, size=n, replace=len(negatives) < n))
        return pos_sample, neg_sample

    def get_all_interacted_stocks(self, user_id: int) -> List[int]:
        """All stock IDs the user has ever interacted with (for diversity filtering)."""
        events = self.db.get_user_events(user_id, limit=1000)
        return list({e["stock_id"] for e in events})

    def summarize(self, user_id: int) -> Dict:
        """Human-readable summary of the user's interaction history."""
        user = self.db.get_user(user_id)
        events = self.db.get_user_events(user_id, limit=1000)

        if not events:
            return {"user_id": user_id, "total_events": 0}

        by_type: Dict[str, int] = {}
        for e in events:
            by_type[e["event_type"]] = by_type.get(e["event_type"], 0) + 1

        rewards = [e["reward"] for e in events if e["reward"] is not None]

        return {
            "user_id": user_id,
            "username": user.get("username") if user else None,
            "risk_tolerance": user.get("risk_tolerance") if user else "unknown",
            "total_events": len(events),
            "events_by_type": by_type,
            "unique_stocks_interacted": len({e["stock_id"] for e in events}),
            "avg_reward": float(np.mean(rewards)) if rewards else None,
            "win_rate": float(np.mean([1 if r > 0 else 0 for r in rewards])) if rewards else None,
        }

    def _compute_sector_diversity(self, events: List[Dict]) -> float:
        """
        Fraction of active market sectors this user has interacted with.

        Uses unique stock sectors rather than raw event count so repeatedly
        viewing one bank stock does not look more diverse than touching one
        bank and one pharma stock.
        """
        interacted_stock_ids = {int(event["stock_id"]) for event in events}
        if not interacted_stock_ids:
            return 0.0

        interacted_sectors = set()
        for stock_id in interacted_stock_ids:
            stock = self.db.get_stock_info(stock_id)
            sector = (stock or {}).get("sector", "")
            if sector:
                interacted_sectors.add(sector)

        if not interacted_sectors:
            return 0.0

        total_sectors = {
            stock.get("sector", "")
            for stock in self.db.get_all_stocks()
            if stock.get("sector")
        }
        if not total_sectors:
            return 0.0

        return float(min(len(interacted_sectors) / len(total_sectors), 1.0))
