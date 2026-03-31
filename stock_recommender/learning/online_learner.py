"""
OnlineLearner — the self-training core of the system.

Reacts to two event streams:
  1. User interactions  → update user embeddings and two-tower model
  2. New market data    → update stock features, fine-tune time-series model

Self-training flow:
  • Every interaction is pushed to the ReplayBuffer
  • Every `online_update_every` interactions → micro-update (few gradient steps)
  • Every `full_retrain_every` interactions  → full retraining trigger
  • New market data daily → time-series model fine-tune + stock embedding refresh

The learner uses separate low learning rates for online updates to prevent
catastrophic forgetting of patterns learned during full training.
"""
import time
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Dict
import logging

from stock_recommender.config import CONFIG
from stock_recommender.data.replay_buffer import ReplayBuffer, ReplayEvent
from stock_recommender.data.database import DatabaseManager
from stock_recommender.models.two_tower import TwoTowerModel, InfoNCELoss, CandidateIndex
from stock_recommender.models.time_series import StockTransformer, TimeSeriesLoss
from stock_recommender.features.feature_pipeline import FeaturePipeline
from stock_recommender.data.user_tracker import UserTracker

logger = logging.getLogger(__name__)


class OnlineLearner:
    """
    Self-training engine. Runs continuously alongside the recommendation engine.

    Initialize once with trained model references.
    Call on_user_event() and on_new_market_data() as events arrive.
    The learner handles all micro-update scheduling internally.
    """

    def __init__(
        self,
        two_tower: TwoTowerModel,
        transformer: StockTransformer,
        candidate_index: CandidateIndex,
        feature_pipeline: FeaturePipeline,
        db: DatabaseManager,
        device: str = "cpu",
    ):
        self.two_tower = two_tower
        self.transformer = transformer
        self.candidate_index = candidate_index
        self.feature_pipeline = feature_pipeline
        self.db = db
        # Infer device from two_tower so online updates run on the same device as training.
        try:
            self.device = next(two_tower.parameters()).device
        except StopIteration:
            self.device = torch.device(device)

        self.replay = ReplayBuffer(capacity=CONFIG.training.replay_buffer_capacity)
        self.user_tracker = UserTracker(db)

        # Separate optimizers with lower LR for online updates
        self.tower_optimizer = torch.optim.Adam(
            two_tower.parameters(), lr=CONFIG.training.online_lr,
            weight_decay=CONFIG.training.weight_decay,
        )
        self.ts_optimizer = torch.optim.Adam(
            transformer.parameters(), lr=CONFIG.training.online_lr * 0.5,
            weight_decay=CONFIG.training.weight_decay,
        )

        self.infonce_loss = InfoNCELoss()
        self.ts_loss_fn = TimeSeriesLoss()

        # Counters and state
        self._interaction_count = 0
        self._micro_update_count = 0
        self._last_full_retrain_trigger = 0
        self._needs_index_rebuild = False
        self._online_losses: List[float] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def on_user_event(
        self,
        user_id: int,
        stock_id: int,
        user_features: np.ndarray,
        stock_features: np.ndarray,
        reward: float,
        is_positive: bool,
    ) -> None:
        """
        Called whenever a user interacts with a stock.

        Stores the event in the replay buffer and triggers a micro-update
        every `online_update_every` events.
        """
        event = ReplayEvent(
            user_id=user_id,
            stock_id=stock_id,
            user_features=user_features,
            stock_features=stock_features,
            reward=reward,
            is_positive=is_positive,
        )
        self.replay.push(event)
        self._interaction_count += 1

        # Micro-update: every N interactions
        if (self._interaction_count % CONFIG.training.online_update_every == 0
                and self.replay.is_ready):
            self._micro_update_towers()
            self._update_user_embedding(user_id, user_features)

        # Signal full retrain if threshold reached
        if self._interaction_count - self._last_full_retrain_trigger >= CONFIG.training.full_retrain_every:
            self._last_full_retrain_trigger = self._interaction_count
            logger.info(
                f"[OnlineLearner] Full retrain threshold reached at "
                f"{self._interaction_count} interactions."
            )

    def on_new_market_data(
        self,
        stock_id: int,
        feature_sequence: np.ndarray,   # (seq_len, n_features) normalized
        target_returns: np.ndarray,     # (2,) — [ret_1d, ret_5d]
        target_direction: int,          # 0=down, 1=flat, 2=up
    ) -> float:
        """
        Called when fresh OHLCV data arrives for a stock (typically daily).

        Fine-tunes the time-series transformer on the new data point,
        recomputes the stock embedding, and updates the candidate index.

        Returns the time-series loss for this update.
        """
        loss = self._micro_update_transformer(
            stock_id, feature_sequence, target_returns, target_direction
        )
        self._update_stock_embedding(stock_id, feature_sequence)
        return loss

    def get_stats(self) -> Dict:
        """Diagnostic summary of online learning state."""
        return {
            "total_interactions": self._interaction_count,
            "micro_updates": self._micro_update_count,
            "replay_buffer_size": self.replay.size,
            "replay_ready": self.replay.is_ready,
            "recent_reward_stats": self.replay.recent_reward_stats(),
            "avg_online_loss": float(np.mean(self._online_losses[-50:])) if self._online_losses else None,
        }

    # ── Micro-updates ─────────────────────────────────────────────────────────

    def _micro_update_towers(self) -> Optional[float]:
        """
        Single mini-batch gradient update on the two-tower model.
        Uses prioritized replay to focus on informative experiences.
        """
        batch_size = min(CONFIG.training.batch_size, self.replay.size)
        events, indices, is_weights = self.replay.sample(batch_size, strategy="prioritized")

        if not events:
            return None

        self.two_tower.train()
        self.tower_optimizer.zero_grad()

        # Build tensors from replay events
        user_ids = torch.tensor([e.user_id for e in events], dtype=torch.long, device=self.device)
        stock_ids = torch.tensor([e.stock_id for e in events], dtype=torch.long, device=self.device)
        user_feats = torch.tensor(
            np.stack([e.user_features for e in events]), dtype=torch.float32, device=self.device
        )
        stock_feats = torch.tensor(
            np.stack([e.stock_features for e in events]), dtype=torch.float32, device=self.device
        )
        rewards = torch.tensor([e.reward for e in events], dtype=torch.float32, device=self.device)
        is_w = torch.tensor(is_weights, dtype=torch.float32, device=self.device)

        # Forward pass: get logit matrix for contrastive loss
        # Encode once and reuse embeddings for both the contrastive and reward losses
        user_embs = self.two_tower.encode_user(user_ids, user_feats)
        stock_embs = self.two_tower.encode_stock(stock_ids, stock_feats)
        logits = torch.matmul(user_embs, stock_embs.T) / self.two_tower.temperature

        # InfoNCE with importance-sampling weighting
        labels = torch.arange(len(events), device=self.device)
        raw_loss = F.cross_entropy(logits, labels, reduction="none")
        weighted_loss = (raw_loss * is_w).mean()

        # Also add a reward-modulated term: push positive-reward pairs together
        # and negative-reward pairs apart in embedding space
        reward_sign = torch.sign(rewards)
        user_stock_scores = (user_embs * stock_embs).sum(-1)
        reward_loss = -F.logsigmoid(reward_sign * user_stock_scores).mean()

        total_loss = weighted_loss + 0.2 * reward_loss
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.two_tower.parameters(), max_norm=1.0)
        self.tower_optimizer.step()

        loss_val = float(total_loss.item())
        self._online_losses.append(loss_val)
        self._micro_update_count += 1

        # Update priorities based on per-sample loss
        with torch.no_grad():
            sample_losses = F.cross_entropy(logits, labels, reduction="none").cpu().numpy()
        self.replay.update_priorities(indices, sample_losses)

        logger.debug(f"[OnlineLearner] Micro-update #{self._micro_update_count}: loss={loss_val:.4f}")
        return loss_val

    def _micro_update_transformer(
        self,
        stock_id: int,
        feature_sequence: np.ndarray,
        target_returns: np.ndarray,
        target_direction: int,
    ) -> float:
        """Single gradient step on the transformer with one new market data point."""
        self.transformer.train()
        self.ts_optimizer.zero_grad()

        x = torch.tensor(feature_sequence, dtype=torch.float32, device=self.device).unsqueeze(0)
        target_ret = torch.tensor(
            target_returns, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        # target_direction is already in {0, 1, 2} (0=down, 1=flat, 2=up)
        dir_label = torch.tensor([target_direction], dtype=torch.long, device=self.device)

        price_fc, dir_logits, _ = self.transformer(x)
        loss, loss_dict = self.ts_loss_fn(price_fc, dir_logits, target_ret, dir_label)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), max_norm=1.0)
        self.ts_optimizer.step()

        logger.debug(
            f"[OnlineLearner] TS update stock {stock_id}: "
            f"ret_loss={loss_dict['return_loss']:.6f}, "
            f"dir_loss={loss_dict['direction_loss']:.4f}"
        )
        return float(loss.item())

    # ── Embedding updates ──────────────────────────────────────────────────────

    @torch.no_grad()
    def _update_user_embedding(self, user_id: int, user_features: np.ndarray) -> np.ndarray:
        """Recompute and persist user embedding after an interaction."""
        self.two_tower.eval()
        uid_t = torch.tensor([user_id], dtype=torch.long, device=self.device)
        feat_t = torch.tensor(user_features, dtype=torch.float32, device=self.device).unsqueeze(0)
        emb = self.two_tower.encode_user(uid_t, feat_t).squeeze(0).cpu().numpy()
        self.db.save_user_embedding(user_id, emb.tolist())
        return emb

    @torch.no_grad()
    def _update_stock_embedding(
        self, stock_id: int, feature_sequence: np.ndarray
    ) -> np.ndarray:
        """
        Recompute stock embedding by running the transformer + stock tower,
        then update the candidate index.
        """
        self.transformer.eval()
        self.two_tower.eval()

        x = torch.tensor(feature_sequence, dtype=torch.float32, device=self.device).unsqueeze(0)
        _, _, ts_emb = self.transformer(x)

        sid_t = torch.tensor([stock_id], dtype=torch.long, device=self.device)
        # Use last timestep features as stock feature snapshot
        snap = torch.tensor(
            feature_sequence[-1], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        stock_emb = self.two_tower.encode_stock(sid_t, snap, ts_emb).squeeze(0).cpu().numpy()

        self.db.save_stock_embedding(stock_id, stock_emb.tolist())
        self.candidate_index.update(stock_id, stock_emb)

        return stock_emb

    def rebuild_candidate_index(self) -> None:
        """
        Rebuild the full ANN index from all persisted stock embeddings.
        Call after a full retrain or when the index is stale.
        """
        all_embeddings = self.db.get_all_stock_embeddings()
        if not all_embeddings:
            logger.warning("[OnlineLearner] No stock embeddings to rebuild index from.")
            return

        stock_ids = np.array(list(all_embeddings.keys()))
        embeddings = np.array([all_embeddings[sid] for sid in stock_ids])
        self.candidate_index.build(stock_ids, embeddings)
        logger.info(f"[OnlineLearner] Rebuilt candidate index with {len(stock_ids)} stocks.")

    def attribute_delayed_rewards(self) -> int:
        """
        Resolve delayed rewards for past events where the outcome is now known.
        Called periodically (e.g., daily) to fill in reward values.
        Returns the number of events updated.
        """
        unresolved = self.db.get_unresolved_events(older_than_seconds=5 * 86400)
        updated = 0

        for event in unresolved:
            stock_id = event["stock_id"]
            event_price = event.get("price_at_event", 0.0)
            if event_price <= 0:
                continue

            # Get current price from DB
            history = self.db.get_price_history(stock_id, limit=1)
            if not history:
                continue

            current_price = history[-1]["close"]
            reward = self.user_tracker.attribute_reward(
                event["event_id"], current_price, event_price
            )
            updated += 1

        if updated > 0:
            logger.info(f"[OnlineLearner] Attributed delayed rewards to {updated} events.")
        return updated
