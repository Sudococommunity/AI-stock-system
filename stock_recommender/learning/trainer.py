"""
Trainer — full (batch) training pipeline for all three models.

Run this:
  1. At cold-start with synthetic data to initialize the models
  2. Periodically (e.g. weekly) to retrain from the full accumulated dataset

The trainer trains models in dependency order:
  1. StockTransformer  (time-series, no user data needed)
  2. TwoTowerModel     (needs stock embeddings from transformer)
  3. RankingModel      (needs embeddings from both towers)
"""
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
import logging
import pandas as pd

from stock_recommender.config import CONFIG
from stock_recommender.models.checkpoint_utils import load_model_state
from stock_recommender.models.time_series import StockTransformer, TimeSeriesLoss
from stock_recommender.models.two_tower import TwoTowerModel, InfoNCELoss, RankingModel
from stock_recommender.features.feature_pipeline import FeaturePipeline
from stock_recommender.features.tensor_preprocessing import (
    build_training_windows_tensor,
    compute_feature_matrix_tensor,
    load_or_compute_feature_cache,
    normalizer_state_from_features,
)
from stock_recommender.data.database import DatabaseManager
from stock_recommender.data.user_tracker import UserTracker
from stock_recommender.risk.risk_metrics import compute_full_risk_profile

logger = logging.getLogger(__name__)


def _resolve_device(device: Optional[str] = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        # Prefer the GPU with the most free memory (RTX 3090 at index 1 has 24 GB).
        best = max(range(torch.cuda.device_count()),
                   key=lambda i: torch.cuda.get_device_properties(i).total_memory)
        return torch.device(f"cuda:{best}")
    return torch.device("cpu")


def _dataloader_kwargs(device: torch.device) -> Dict:
    pin = device.type == "cuda"
    if os.name == "nt":
        # Windows: spawn-based multiprocessing is slow for small datasets;
        # keep num_workers=0 but enable pin_memory so host→GPU DMA is async.
        return {"num_workers": 0, "pin_memory": pin}

    cpu_count = os.cpu_count() or 1
    num_workers = min(4, max(0, cpu_count - 1))
    kwargs: Dict = {"num_workers": num_workers, "pin_memory": pin}
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4 if pin else 2   # deeper prefetch on GPU
    return kwargs


def build_time_series_dataset(
    db: DatabaseManager,
    feature_pipeline: FeaturePipeline,
    stock_ids: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    logger.info(f"[Trainer] Building time-series dataset for {len(stock_ids)} stocks...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Single pass: fetch raw data, compute/load features, cache both for the window-building step
    raw_cache: Dict[int, List[Dict]] = {}
    feature_batches = []
    for idx, sid in enumerate(stock_ids, start=1):
        raw = db.get_price_history(sid, limit=10_000)   # fetch full history (up to 30 years)
        if len(raw) < CONFIG.data.min_price_history_days:
            if idx % 100 == 0 or idx == len(stock_ids):
                logger.info(
                    "[Trainer] Dataset build progress: %s/%s stocks processed",
                    idx,
                    len(stock_ids),
                )
            continue

        features, _ = load_or_compute_feature_cache(
            CONFIG.data.tensor_cache_dir,
            sid,
            raw,
            device=device,
        )
        if features.numel() > 0:
            feature_batches.append(features)
            raw_cache[sid] = raw  # keep for window-building below (avoids second DB fetch)
        if idx % 100 == 0 or idx == len(stock_ids):
            logger.info(
                "[Trainer] Dataset build progress: %s/%s stocks processed",
                idx,
                len(stock_ids),
            )

    if not feature_batches:
        return (
            np.empty((0, CONFIG.model.seq_len, CONFIG.model.n_tech_features), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
        )

    all_features = torch.cat(feature_batches, dim=0)
    feature_pipeline.normalizer.load_state_dict(normalizer_state_from_features(all_features))
    mean = torch.tensor(feature_pipeline.normalizer.mean, dtype=torch.float32, device=device)
    std = torch.tensor(feature_pipeline.normalizer.std, dtype=torch.float32, device=device)

    X_all, y_all = [], []
    for sid in stock_ids:
        raw = raw_cache.get(sid)
        if raw is None:
            continue
        features, close = load_or_compute_feature_cache(
            CONFIG.data.tensor_cache_dir,
            sid,
            raw,
            device=device,
        )
        if features.numel() == 0:
            continue
        X_t, y_t, _ = build_training_windows_tensor(
            raw,
            seq_len=feature_pipeline.seq_len,
            horizon=CONFIG.runtime.prediction_horizon_days,
            mean=mean,
            std=std,
            device=device,
            precomputed_features=features,
            precomputed_close=close,
        )
        if X_t.numel() == 0:
            continue
        X_all.append(X_t.numpy())
        y_all.append(y_t.numpy())

    if not X_all:
        return (
            np.empty((0, CONFIG.model.seq_len, CONFIG.model.n_tech_features), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
        )

    X = np.concatenate(X_all).astype(np.float32, copy=False)
    y = np.concatenate(y_all).astype(np.float32, copy=False)
    logger.info(f"[Trainer] Time-series dataset: {len(X)} windows")
    return X, y


# ── Datasets ──────────────────────────────────────────────────────────────────

class TimeSeriesDataset(Dataset):
    """Sliding window dataset for the StockTransformer."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).to(torch.float32)
        self.y_returns = torch.from_numpy(y[:, :2]).to(torch.float32)
        # Direction: map -1,0,1 → 0,1,2
        self.y_direction = torch.from_numpy((y[:, 2] + 1).astype(np.int64))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_returns[idx], self.y_direction[idx]


class InteractionDataset(Dataset):
    """Dataset for two-tower training from user interaction history."""

    def __init__(
        self,
        user_ids: List[int],
        stock_ids: List[int],
        user_features: np.ndarray,
        stock_features: np.ndarray,
    ):
        self.user_ids = torch.tensor(user_ids, dtype=torch.long)
        self.stock_ids = torch.tensor(stock_ids, dtype=torch.long)
        self.user_features = torch.tensor(user_features, dtype=torch.float32)
        self.stock_features = torch.tensor(stock_features, dtype=torch.float32)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return (
            self.user_ids[idx],
            self.user_features[idx],
            self.stock_ids[idx],
            self.stock_features[idx],
        )


# ── Main Trainer ──────────────────────────────────────────────────────────────

class Trainer:
    """
    Orchestrates full training passes for all models.
    Saves checkpoints to CONFIG.data.checkpoint_dir after each epoch.
    """

    def __init__(
        self,
        transformer: StockTransformer,
        two_tower: TwoTowerModel,
        ranker: RankingModel,
        feature_pipeline: FeaturePipeline,
        db: DatabaseManager,
        device: Optional[str] = None,
    ):
        self.device = _resolve_device(device)
        self.use_amp = self.device.type == "cuda"

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True             # auto-tune cuDNN kernels for fixed-size inputs
            torch.set_float32_matmul_precision("high")        # enable TF32 tensor cores on Ampere+ GPUs

        self.transformer = transformer.to(self.device)
        self.two_tower = two_tower.to(self.device)
        self.ranker = ranker.to(self.device)

        # torch.compile() fuses and JIT-compiles the transformer for faster CUDA throughput.
        # Falls back silently on PyTorch < 2.0 or CPU.
        if self.device.type == "cuda" and hasattr(torch, "compile"):
            try:
                self.transformer = torch.compile(self.transformer)
                logger.info("[Trainer] torch.compile() applied to StockTransformer")
            except Exception as e:
                logger.warning("[Trainer] torch.compile() unavailable: %s", e)

        self.pipeline = feature_pipeline
        self.db = db
        self.ts_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.tower_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        os.makedirs(CONFIG.data.checkpoint_dir, exist_ok=True)
        logger.info("[Trainer] Using device: %s", self.device)

        self.ts_optimizer = torch.optim.AdamW(
            transformer.parameters(),
            lr=CONFIG.training.learning_rate,
            weight_decay=CONFIG.training.weight_decay,
        )
        self.tower_optimizer = torch.optim.AdamW(
            two_tower.parameters(),
            lr=CONFIG.training.learning_rate,
            weight_decay=CONFIG.training.weight_decay,
        )
        self.ranker_optimizer = torch.optim.AdamW(
            ranker.parameters(),
            lr=CONFIG.training.learning_rate,
            weight_decay=CONFIG.training.weight_decay,
        )

        self.ts_loss_fn = TimeSeriesLoss()
        self.infonce_loss = InfoNCELoss()

        self._step = 0

    # ── Time-series training ──────────────────────────────────────────────────

    def train_transformer(
        self, stock_ids: List[int], n_epochs: int = None, val_split: float = 0.1
    ) -> Dict[str, List[float]]:
        """
        Train the StockTransformer on sliding windows from all stocks' price history.
        """
        X, y = build_time_series_dataset(self.db, self.pipeline, stock_ids)
        return self.train_transformer_prebuilt(X, y, n_epochs=n_epochs, val_split=val_split)

    def train_transformer_prebuilt(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_epochs: int = None,
        val_split: float = 0.1,
    ) -> Dict[str, List[float]]:
        """
        Train the StockTransformer from prebuilt windows so preprocessing can be
        reused across multiple candidate models.
        """
        n_epochs = n_epochs or CONFIG.training.max_epochs

        if len(X) == 0 or len(y) == 0:
            logger.warning("[Trainer] No training data for transformer.")
            return {"train_loss": [], "val_loss": []}

        # Chronological train/val split — last val_split fraction of windows go to val.
        # Windows are built in time order per stock, so this avoids future-leakage into val.
        n_val = max(1, int(len(X) * val_split))
        X_train, y_train = X[:-n_val], y[:-n_val]
        X_val, y_val = X[-n_val:], y[-n_val:]

        train_ds = TimeSeriesDataset(X_train, y_train)
        val_ds = TimeSeriesDataset(X_val, y_val)
        dl_kwargs = _dataloader_kwargs(self.device)
        train_dl = DataLoader(train_ds, batch_size=CONFIG.training.batch_size, shuffle=True, **dl_kwargs)
        val_dl = DataLoader(val_ds, batch_size=CONFIG.training.batch_size, shuffle=False, **dl_kwargs)

        history = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        patience = 0

        for epoch in range(n_epochs):
            # Training
            self.transformer.train()
            train_losses = []
            for X_b, ret_b, dir_b in train_dl:
                X_b = X_b.to(self.device, non_blocking=self.use_amp)
                ret_b = ret_b.to(self.device, non_blocking=self.use_amp)
                dir_b = dir_b.to(self.device, non_blocking=self.use_amp)

                self.ts_optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    price_fc, dir_logits, _ = self.transformer(X_b)
                    loss, _ = self.ts_loss_fn(price_fc, dir_logits, ret_b, dir_b)
                self.ts_scaler.scale(loss).backward()
                self.ts_scaler.unscale_(self.ts_optimizer)
                torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 1.0)
                self.ts_scaler.step(self.ts_optimizer)
                self.ts_scaler.update()
                train_losses.append(loss.item())

            # Validation
            self.transformer.eval()
            val_losses = []
            with torch.no_grad():
                for X_b, ret_b, dir_b in val_dl:
                    X_b = X_b.to(self.device, non_blocking=self.use_amp)
                    ret_b = ret_b.to(self.device, non_blocking=self.use_amp)
                    dir_b = dir_b.to(self.device, non_blocking=self.use_amp)
                    with torch.amp.autocast("cuda", enabled=self.use_amp):
                        price_fc, dir_logits, _ = self.transformer(X_b)
                        loss, _ = self.ts_loss_fn(price_fc, dir_logits, ret_b, dir_b)
                    val_losses.append(loss.item())

            train_l = float(np.mean(train_losses))
            val_l = float(np.mean(val_losses))
            history["train_loss"].append(train_l)
            history["val_loss"].append(val_l)

            logger.info(f"[Transformer] Epoch {epoch+1}/{n_epochs} | train={train_l:.4f} val={val_l:.4f}")
            self.db.log_metric("transformer", "train_loss", train_l, epoch)
            self.db.log_metric("transformer", "val_loss", val_l, epoch)

            if val_l < best_val:
                best_val = val_l
                patience = 0
                self._save_checkpoint("transformer", self.transformer, epoch, {"val_loss": val_l})
            else:
                patience += 1
                if patience >= CONFIG.training.early_stop_patience:
                    logger.info(f"[Transformer] Early stopping at epoch {epoch+1}")
                    break

        return history

    # ── Two-tower training ────────────────────────────────────────────────────

    def train_towers(
        self,
        interaction_data: Optional[Dict] = None,
        n_epochs: int = None,
    ) -> Dict[str, List[float]]:
        """
        Train the two-tower model using user-stock interaction history.

        If interaction_data is None, it's built from the database.
        interaction_data format: {
            "user_ids": List[int],
            "stock_ids": List[int],
            "user_features": np.ndarray (N, n_user_features),
            "stock_features": np.ndarray (N, n_stock_features),
        }
        """
        n_epochs = n_epochs or CONFIG.training.max_epochs

        if interaction_data is None:
            interaction_data = self._build_interaction_data()

        if not interaction_data["user_ids"]:
            logger.warning("[Trainer] No interaction data for tower training.")
            return {"train_loss": []}

        ds = InteractionDataset(**interaction_data)
        dl = DataLoader(
            ds,
            batch_size=CONFIG.training.batch_size,
            shuffle=True,
            drop_last=True,
            **_dataloader_kwargs(self.device),
        )

        history = {"train_loss": []}
        best_loss = float("inf")

        for epoch in range(n_epochs):
            self.two_tower.train()
            epoch_losses = []

            for user_ids, user_feats, stock_ids, stock_feats in dl:
                user_ids = user_ids.to(self.device, non_blocking=self.use_amp)
                user_feats = user_feats.to(self.device, non_blocking=self.use_amp)
                stock_ids = stock_ids.to(self.device, non_blocking=self.use_amp)
                stock_feats = stock_feats.to(self.device, non_blocking=self.use_amp)

                self.tower_optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    logits, _ = self.two_tower(user_ids, user_feats, stock_ids, stock_feats)
                    loss = self.infonce_loss(logits)
                self.tower_scaler.scale(loss).backward()
                self.tower_scaler.unscale_(self.tower_optimizer)
                torch.nn.utils.clip_grad_norm_(self.two_tower.parameters(), 1.0)
                self.tower_scaler.step(self.tower_optimizer)
                self.tower_scaler.update()
                epoch_losses.append(loss.item())

            epoch_l = float(np.mean(epoch_losses))
            history["train_loss"].append(epoch_l)
            logger.info(f"[TwoTower] Epoch {epoch+1}/{n_epochs} | loss={epoch_l:.4f}")
            self.db.log_metric("two_tower", "train_loss", epoch_l, epoch)

            if epoch_l < best_loss:
                best_loss = epoch_l
                self._save_checkpoint("two_tower", self.two_tower, epoch, {"train_loss": epoch_l})

        return history

    # ── Ranker training ───────────────────────────────────────────────────────

    def train_ranker(self, n_epochs: int = None) -> Dict[str, List[float]]:
        """
        Train the ranking model using pairwise (positive, negative) stock examples.
        Requires the two-tower to be already trained (needs embeddings).
        """
        n_epochs = n_epochs or CONFIG.training.max_epochs

        training_pairs = self._build_ranker_training_pairs()
        if not training_pairs:
            logger.warning("[Trainer] No data for ranker training.")
            return {"train_loss": []}

        history = {"train_loss": []}
        best_loss = float("inf")

        for epoch in range(n_epochs):
            self.ranker.train()
            self.two_tower.eval()
            self.transformer.eval()
            epoch_losses = []

            np.random.shuffle(training_pairs)
            for i in range(0, len(training_pairs), CONFIG.training.batch_size):
                batch = training_pairs[i : i + CONFIG.training.batch_size]
                if not batch:
                    continue

                (
                    user_embs, pos_stock_embs, neg_stock_embs,
                    pos_risk_feats, neg_risk_feats,
                    pos_fc_feats, neg_fc_feats,
                ) = self._collate_ranker_batch(batch)

                self.ranker_optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    pos_scores = self.ranker(user_embs, pos_stock_embs, pos_risk_feats, pos_fc_feats)
                    neg_scores = self.ranker(user_embs, neg_stock_embs, neg_risk_feats, neg_fc_feats)
                    loss = self.ranker.pairwise_loss(pos_scores, neg_scores)
                self.tower_scaler.scale(loss).backward()
                self.tower_scaler.unscale_(self.ranker_optimizer)
                torch.nn.utils.clip_grad_norm_(self.ranker.parameters(), 1.0)
                self.tower_scaler.step(self.ranker_optimizer)
                self.tower_scaler.update()
                epoch_losses.append(loss.item())

            if not epoch_losses:
                continue

            epoch_l = float(np.mean(epoch_losses))
            history["train_loss"].append(epoch_l)
            logger.info(f"[Ranker] Epoch {epoch+1}/{n_epochs} | loss={epoch_l:.4f}")
            self.db.log_metric("ranker", "train_loss", epoch_l, epoch)

            if epoch_l < best_loss:
                best_loss = epoch_l
                self._save_checkpoint("ranker", self.ranker, epoch, {"train_loss": epoch_l})

        return history

    def train_all(self, stock_ids: Optional[List[int]] = None) -> Dict:
        """Train all models in dependency order."""
        if stock_ids is None:
            stock_ids = self.db.get_all_stock_ids()

        logger.info(f"[Trainer] Starting full training on {len(stock_ids)} stocks...")
        results = {}

        logger.info("[Trainer] Phase 1: Training StockTransformer...")
        results["transformer"] = self.train_transformer(stock_ids)

        # Two-tower and ranker require user interaction data (user_events rows).
        # Without it they cannot learn meaningful rankings — skip and warn clearly.
        user_ids = self._get_all_user_ids()
        if not user_ids:
            logger.warning(
                "[Trainer] No users found in database — skipping two-tower and ranker. "
                "Run 'python main.py seed' to generate synthetic users, "
                "or ingest real user events before calling train_all()."
            )
            results["two_tower"] = {"train_loss": []}
            results["ranker"] = {"train_loss": []}
            logger.info("[Trainer] Full training complete (transformer only).")
            return results

        logger.info("[Trainer] Phase 2: Training TwoTower (%d users)...", len(user_ids))
        results["two_tower"] = self.train_towers()

        logger.info("[Trainer] Phase 3: Training RankingModel...")
        results["ranker"] = self.train_ranker()

        logger.info("[Trainer] Full training complete.")
        return results

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def _save_checkpoint(self, model_type: str, model: nn.Module, epoch: int, metrics: Dict) -> str:
        version = f"v{int(time.time())}_{epoch}"
        path = os.path.join(CONFIG.data.checkpoint_dir, f"{model_type}_{version}.pt")
        torch.save(
            {
                "state_dict": model.state_dict(),
                "meta": {
                    "model_type": model_type,
                    "n_tech_features": CONFIG.model.n_tech_features,
                    "embed_dim": CONFIG.model.embed_dim,
                    "metrics": metrics,
                    "epoch": epoch,
                },
            },
            path,
        )
        self.db.log_metric(model_type, "checkpoint_saved", 1.0, epoch)
        logger.debug(f"[Trainer] Saved checkpoint: {path}")
        return path

    def load_checkpoint(self, model: nn.Module, path: str) -> None:
        load_model_state(model, path, map_location=self.device)
        logger.info(f"[Trainer] Loaded checkpoint: {path}")

    # ── Internal data builders ────────────────────────────────────────────────

    def _build_interaction_data(self) -> Dict:
        """Build two-tower training data from the interactions in the database."""
        user_tracker = UserTracker(self.db)
        user_ids_all, stock_ids_all = [], []
        user_feats_all, stock_feats_all = [], []

        all_users = self._get_all_user_ids()
        for uid in all_users:
            pos_stocks, neg_stocks = user_tracker.get_positive_negative_pairs(uid, n_pairs=20)
            if not pos_stocks:
                continue
            profile = user_tracker.get_profile_features(uid)
            for sid in pos_stocks + neg_stocks:
                raw = self.db.get_price_history(sid, limit=CONFIG.data.min_price_history_days)
                if len(raw) < 60:
                    continue
                import pandas as pd
                snap = self.pipeline.get_latest_snapshot(pd.DataFrame(raw))
                if snap is None:
                    continue
                user_ids_all.append(uid)
                stock_ids_all.append(sid)
                user_feats_all.append(profile)
                stock_feats_all.append(snap)

        if not user_ids_all:
            return {"user_ids": [], "stock_ids": [], "user_features": np.array([]), "stock_features": np.array([])}

        return {
            "user_ids": user_ids_all,
            "stock_ids": stock_ids_all,
            "user_features": np.array(user_feats_all),
            "stock_features": np.array(stock_feats_all),
        }

    def _build_ranker_training_pairs(self) -> List:
        """
        Build pairwise ranker examples from user interaction history.

        Each example contains a user embedding and one positive/negative stock
        context so the ranker can learn to score preferred stocks above avoided
        ones using the same feature blocks used at inference time.
        """
        user_tracker = UserTracker(self.db)
        user_cache: Dict[int, np.ndarray] = {}
        stock_cache: Dict[int, Optional[Dict[str, np.ndarray]]] = {}
        training_pairs: List[Dict[str, np.ndarray]] = []

        self.two_tower.eval()
        self.transformer.eval()

        for user_id in self._get_all_user_ids():
            pos_stocks, neg_stocks = user_tracker.get_positive_negative_pairs(user_id, n_pairs=32)
            if not pos_stocks or not neg_stocks:
                continue

            user_emb = self._get_ranker_user_embedding(user_id, user_tracker, user_cache)
            if user_emb is None:
                continue

            for pos_sid, neg_sid in zip(pos_stocks, neg_stocks):
                pos_sid = int(pos_sid)
                neg_sid = int(neg_sid)
                pos_ctx = self._get_ranker_stock_context(pos_sid, stock_cache)
                neg_ctx = self._get_ranker_stock_context(neg_sid, stock_cache)
                if pos_ctx is None or neg_ctx is None:
                    continue

                training_pairs.append(
                    {
                        "user_emb": user_emb,
                        "pos_stock_emb": pos_ctx["stock_emb"],
                        "neg_stock_emb": neg_ctx["stock_emb"],
                        "pos_risk_feats": pos_ctx["risk_feats"],
                        "neg_risk_feats": neg_ctx["risk_feats"],
                        "pos_fc_feats": pos_ctx["forecast_feats"],
                        "neg_fc_feats": neg_ctx["forecast_feats"],
                    }
                )

        logger.info("[Trainer] Built %s ranker training pairs", len(training_pairs))
        return training_pairs

    def _collate_ranker_batch(self, batch: List) -> tuple:
        """Stack precomputed pairwise ranker features onto the active device."""
        return (
            torch.tensor(
                np.stack([item["user_emb"] for item in batch]),
                dtype=torch.float32,
                device=self.device,
            ),
            torch.tensor(
                np.stack([item["pos_stock_emb"] for item in batch]),
                dtype=torch.float32,
                device=self.device,
            ),
            torch.tensor(
                np.stack([item["neg_stock_emb"] for item in batch]),
                dtype=torch.float32,
                device=self.device,
            ),
            torch.tensor(
                np.stack([item["pos_risk_feats"] for item in batch]),
                dtype=torch.float32,
                device=self.device,
            ),
            torch.tensor(
                np.stack([item["neg_risk_feats"] for item in batch]),
                dtype=torch.float32,
                device=self.device,
            ),
            torch.tensor(
                np.stack([item["pos_fc_feats"] for item in batch]),
                dtype=torch.float32,
                device=self.device,
            ),
            torch.tensor(
                np.stack([item["neg_fc_feats"] for item in batch]),
                dtype=torch.float32,
                device=self.device,
            ),
        )

    def _get_all_user_ids(self) -> List[int]:
        with self.db.connection() as conn:
            with self.db._cur(conn) as cur:
                cur.execute("SELECT user_id FROM users")
                return [r["user_id"] for r in cur.fetchall()]

    def _get_ranker_user_embedding(
        self,
        user_id: int,
        user_tracker: UserTracker,
        cache: Dict[int, np.ndarray],
    ) -> Optional[np.ndarray]:
        cached = cache.get(user_id)
        if cached is not None:
            return cached

        profile = user_tracker.get_profile_features(user_id)
        uid_t = torch.tensor([user_id], dtype=torch.long, device=self.device)
        feat_t = torch.tensor(profile, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.use_amp):
            emb = self.two_tower.encode_user(uid_t, feat_t).squeeze(0).float().cpu().numpy()

        emb = emb.astype(np.float32, copy=False)
        cache[user_id] = emb
        return emb

    def _get_ranker_stock_context(
        self,
        stock_id: int,
        cache: Dict[int, Optional[Dict[str, np.ndarray]]],
    ) -> Optional[Dict[str, np.ndarray]]:
        if stock_id in cache:
            return cache[stock_id]

        raw = self.db.get_price_history(stock_id, limit=500)
        if len(raw) < CONFIG.data.min_price_history_days:
            cache[stock_id] = None
            return None

        df = pd.DataFrame(raw)
        seq = self.pipeline.get_latest_sequence(df)
        snap = self.pipeline.get_latest_snapshot(df)
        if seq is None or snap is None:
            cache[stock_id] = None
            return None

        x = torch.tensor(seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        sid_t = torch.tensor([stock_id], dtype=torch.long, device=self.device)
        snap_t = torch.tensor(snap, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.use_amp):
            price_fc, dir_logits, ts_emb = self.transformer(x)
            direction_probs = torch.softmax(dir_logits, dim=-1)
            stock_emb = self.two_tower.encode_stock(sid_t, snap_t, ts_emb).squeeze(0).float().cpu().numpy()

        returns = df["close"].pct_change().dropna().values[-252:]
        risk_feats = self._risk_features_from_returns(returns)
        forecast_feats = np.array(
            [
                float(price_fc[0, 0].item()),
                float(price_fc[0, 1].item()),
                float(direction_probs[0, 2].item()),
            ],
            dtype=np.float32,
        )

        context = {
            "stock_emb": stock_emb.astype(np.float32, copy=False),
            "risk_feats": risk_feats,
            "forecast_feats": forecast_feats,
        }
        cache[stock_id] = context
        return context

    def _risk_features_from_returns(self, returns: np.ndarray) -> np.ndarray:
        if len(returns) <= 30:
            return np.zeros(8, dtype=np.float32)

        rp = compute_full_risk_profile(returns)
        return np.array(
            [
                rp.sharpe_ratio / 3.0,
                rp.sortino_ratio / 3.0,
                abs(rp.max_drawdown),
                rp.annualized_volatility,
                rp.var_95,
                rp.beta / 2.0,
                rp.win_rate,
                rp.risk_score / 100.0,
            ],
            dtype=np.float32,
        )
