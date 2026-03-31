"""
Two-Tower (Dual Encoder) Model — YouTube DNN-style recommendation architecture.

Architecture:
  • User Tower  : [user_id_emb, interaction_history_GRU, profile_MLP] → user_vector
  • Stock Tower : [stock_id_emb, feature_MLP, ts_embedding] → stock_vector
  • Score       : cosine similarity between user_vector and stock_vector
  • Training    : InfoNCE / in-batch contrastive loss (same as SimCLR / CLIP)

At inference:
  1. Pre-compute all stock embeddings → build ANN index (FAISS or numpy fallback)
  2. Encode the query user → top-K approximate nearest neighbors
  3. Top-K candidates go to the ranking model for fine-grained scoring

The two towers share the same embedding dimension so their outputs are
directly comparable via dot product.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Dict

from stock_recommender.config import CONFIG


# ── Building blocks ───────────────────────────────────────────────────────────

def _mlp(in_dim: int, hidden_dims: List[int], dropout: float) -> nn.Sequential:
    """Build a MLP with LayerNorm after each hidden layer."""
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
        prev = h
    return nn.Sequential(*layers)


# ── User Tower ────────────────────────────────────────────────────────────────

class UserTower(nn.Module):
    """
    Encodes user state into a dense embedding vector.

    Inputs (all optional — gracefully handles cold-start users):
      • user_id          : (B,) long — learnable ID embedding
      • profile_features : (B, n_profile) — risk tolerance, capital, preferred sectors, etc.
      • history_embeds   : (B, H, embed_dim) — sequence of stock embeddings the user
                           has interacted with (most recent last), encoded by a GRU

    Output: L2-normalized embedding of shape (B, embed_dim)
    """

    def __init__(
        self,
        n_users: int = CONFIG.model.max_users,
        n_profile_features: int = CONFIG.model.n_user_profile_features,
        embed_dim: int = CONFIG.model.embed_dim,
        hidden_dims: List[int] = None,
        dropout: float = CONFIG.model.tower_dropout,
    ):
        super().__init__()
        hidden_dims = hidden_dims or CONFIG.model.user_tower_hidden

        # Learnable user ID embedding (random init for cold-start, refined online)
        self.user_embed = nn.Embedding(n_users + 1, embed_dim, padding_idx=0)

        # GRU over sequence of stock embeddings the user interacted with
        self.history_gru = nn.GRU(
            embed_dim, embed_dim, num_layers=1, batch_first=True
        )

        # MLP for explicit profile features (risk tolerance, capital, sectors)
        self.profile_mlp = _mlp(n_profile_features, [128, embed_dim], dropout)

        # Fusion: concatenate [id_emb, history_emb, profile_emb] → project to embed_dim
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embed.weight, std=0.01)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        user_ids: torch.Tensor,                         # (B,)
        profile_features: torch.Tensor,                 # (B, n_profile)
        history_embeds: Optional[torch.Tensor] = None,  # (B, H, embed_dim)
    ) -> torch.Tensor:
        """Returns L2-normalized user embedding (B, embed_dim)."""
        id_emb = self.user_embed(user_ids)              # (B, embed_dim)

        if history_embeds is not None and history_embeds.size(1) > 0:
            _, h_n = self.history_gru(history_embeds)  # h_n: (1, B, embed_dim)
            hist_emb = h_n.squeeze(0)                  # (B, embed_dim)
        else:
            hist_emb = torch.zeros_like(id_emb)

        profile_emb = self.profile_mlp(profile_features)  # (B, embed_dim)

        fused = self.fusion(torch.cat([id_emb, hist_emb, profile_emb], dim=-1))
        return F.normalize(fused, dim=-1)


# ── Stock Tower ───────────────────────────────────────────────────────────────

class StockTower(nn.Module):
    """
    Encodes stock state into the same embedding space as the user tower.

    Inputs:
      • stock_ids       : (B,) long — learnable ID embedding
      • stock_features  : (B, n_features) — normalized technical indicators snapshot
      • ts_embedding    : (B, embed_dim) — output from StockTransformer (optional)

    Output: L2-normalized embedding of shape (B, embed_dim)
    """

    def __init__(
        self,
        n_stocks: int = CONFIG.model.max_stocks,
        n_stock_features: int = CONFIG.model.n_tech_features,
        embed_dim: int = CONFIG.model.embed_dim,
        hidden_dims: List[int] = None,
        dropout: float = CONFIG.model.tower_dropout,
    ):
        super().__init__()
        hidden_dims = hidden_dims or CONFIG.model.stock_tower_hidden

        # Learnable stock ID embedding
        self.stock_embed = nn.Embedding(n_stocks + 1, embed_dim, padding_idx=0)

        # MLP over current technical feature snapshot
        self.feature_mlp = _mlp(n_stock_features, hidden_dims[:-1], dropout)
        self.feature_out = nn.Linear(hidden_dims[-2] if len(hidden_dims) > 1 else n_stock_features, embed_dim)

        # Fusion: [stock_id_emb + feature_emb + ts_emb] → embed_dim
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.stock_embed.weight, std=0.01)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        stock_ids: torch.Tensor,                        # (B,)
        stock_features: torch.Tensor,                   # (B, n_features)
        ts_embedding: Optional[torch.Tensor] = None,    # (B, embed_dim)
    ) -> torch.Tensor:
        """Returns L2-normalized stock embedding (B, embed_dim)."""
        id_emb = self.stock_embed(stock_ids)            # (B, embed_dim)
        feat_emb = F.normalize(
            self.feature_out(self.feature_mlp(stock_features)), dim=-1
        )                                               # (B, embed_dim)

        if ts_embedding is not None:
            ts_emb = ts_embedding
        else:
            ts_emb = torch.zeros_like(id_emb)

        fused = self.fusion(torch.cat([id_emb, feat_emb, ts_emb], dim=-1))
        return F.normalize(fused, dim=-1)


# ── Two-Tower Model ───────────────────────────────────────────────────────────

class TwoTowerModel(nn.Module):
    """
    Full two-tower model combining both towers.
    Handles training (contrastive loss) and inference (similarity scoring).
    """

    def __init__(
        self,
        temperature: float = CONFIG.model.temperature,
        **tower_kwargs,
    ):
        super().__init__()
        self.temperature = temperature
        self.user_tower = UserTower()
        self.stock_tower = StockTower()

    def encode_user(self, user_ids, profile_features, history_embeds=None):
        return self.user_tower(user_ids, profile_features, history_embeds)

    def encode_stock(self, stock_ids, stock_features, ts_embedding=None):
        return self.stock_tower(stock_ids, stock_features, ts_embedding)

    def similarity(self, user_emb: torch.Tensor, stock_emb: torch.Tensor) -> torch.Tensor:
        """Cosine similarity (both already L2-normalized) scaled by temperature."""
        return (user_emb * stock_emb).sum(dim=-1) / self.temperature

    def forward(
        self,
        user_ids: torch.Tensor,
        profile_features: torch.Tensor,
        stock_ids: torch.Tensor,
        stock_features: torch.Tensor,
        history_embeds: Optional[torch.Tensor] = None,
        ts_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute logit matrix for a batch of (user, positive_stock) pairs.
        Uses in-batch negatives: every other stock in the batch is a negative.

        Returns:
            logits      : (B, B) similarity matrix
            user_embs   : (B, embed_dim) — needed for the ranking model
        """
        user_embs = self.encode_user(user_ids, profile_features, history_embeds)
        stock_embs = self.encode_stock(stock_ids, stock_features, ts_embeddings)

        # (B, embed_dim) × (B, embed_dim)^T → (B, B)
        logits = torch.matmul(user_embs, stock_embs.T) / self.temperature
        return logits, user_embs

    @torch.no_grad()
    def score_candidates(
        self,
        user_emb: torch.Tensor,          # (embed_dim,) or (1, embed_dim)
        stock_embs: torch.Tensor,        # (N, embed_dim)
    ) -> torch.Tensor:
        """Score all N candidates for one user. Returns (N,) similarity scores."""
        self.eval()
        if user_emb.dim() == 1:
            user_emb = user_emb.unsqueeze(0)
        scores = torch.matmul(user_emb, stock_embs.T).squeeze(0) / self.temperature
        return scores


# ── InfoNCE Loss ──────────────────────────────────────────────────────────────

class InfoNCELoss(nn.Module):
    """
    In-batch contrastive loss (InfoNCE / NT-Xent).
    Treats the i-th stock as the positive for the i-th user,
    and all other stocks in the batch as negatives.

    This is exactly what YouTube DNN's softmax loss does at scale.
    """

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, B) — diagonal entries are the positive pairs.
        Returns scalar loss.
        """
        batch_size = logits.size(0)
        labels = torch.arange(batch_size, device=logits.device)
        # Symmetric: both user→stock and stock→user directions
        loss_u2s = F.cross_entropy(logits, labels)
        loss_s2u = F.cross_entropy(logits.T, labels)
        return (loss_u2s + loss_s2u) / 2


# ── Ranking Model ─────────────────────────────────────────────────────────────

class RankingModel(nn.Module):
    """
    Stage-2 ranking model (wide-and-deep style).

    Takes the richer concatenated feature vector:
    [user_emb, stock_emb, cross_features, risk_features, ts_forecast]
    and outputs a scalar relevance score.

    Trained with pairwise ranking loss on (preferred_stock, ignored_stock) pairs.
    """

    def __init__(
        self,
        embed_dim: int = CONFIG.model.embed_dim,
        n_risk_features: int = 8,        # Sharpe, VaR, beta, volatility, etc.
        n_forecast_features: int = 3,    # ret_1d, ret_5d, direction_prob
        hidden_dims: List[int] = None,
        dropout: float = CONFIG.model.ranker_dropout,
    ):
        super().__init__()
        hidden_dims = hidden_dims or CONFIG.model.ranker_hidden

        # Cross features: element-wise product of user and stock embeddings
        n_cross = embed_dim

        in_dim = embed_dim + embed_dim + n_cross + n_risk_features + n_forecast_features
        self.deep = _mlp(in_dim, hidden_dims, dropout)
        self.output = nn.Linear(hidden_dims[-1], 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        user_emb: torch.Tensor,          # (B, embed_dim)
        stock_emb: torch.Tensor,         # (B, embed_dim)
        risk_features: torch.Tensor,     # (B, n_risk_features)
        forecast_features: torch.Tensor, # (B, n_forecast_features)
    ) -> torch.Tensor:
        """Returns scalar scores (B, 1)."""
        cross = user_emb * stock_emb    # element-wise product (interaction features)
        x = torch.cat([user_emb, stock_emb, cross, risk_features, forecast_features], dim=-1)
        x = self.deep(x)
        return self.output(x)           # (B, 1)

    def pairwise_loss(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
    ) -> torch.Tensor:
        """BPR (Bayesian Personalized Ranking) loss — positive should score higher."""
        return -F.logsigmoid(pos_scores - neg_scores).mean()


# ── Candidate Index ────────────────────────────────────────────────────────────
# Priority order for retrieval backend:
#   1. FAISS GPU / CPU  (if faiss package installed)
#   2. PyTorch CUDA     (cuBLAS torch.mv on RTX — sub-ms for 10 k stocks)
#   3. NumPy CPU        (always available fallback)

try:
    import faiss as _faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _faiss = None
    _FAISS_AVAILABLE = False

_CUDA_AVAILABLE = torch.cuda.is_available()


class CandidateIndex:
    """
    ANN index over stock embeddings.  Three execution backends, chosen
    automatically at build() time:

    ┌─────────────────────────┬──────────────────────────────────────┐
    │ Backend                 │ When used                            │
    ├─────────────────────────┼──────────────────────────────────────┤
    │ FAISS (GPU or CPU)      │ faiss package installed              │
    │ PyTorch CUDA (cuBLAS)   │ CUDA available, no faiss             │
    │ NumPy CPU               │ always-available fallback            │
    └─────────────────────────┴──────────────────────────────────────┘

    All backends L2-normalise embeddings so inner-product == cosine sim.
    Embeddings are kept in float32 numpy as source-of-truth; the GPU
    tensor mirror is rebuilt lazily when stale.
    """

    def __init__(self, embed_dim: int = CONFIG.model.embed_dim):
        self.embed_dim = embed_dim
        self.stock_ids: Optional[np.ndarray] = None      # (N,) int64
        self.embeddings: Optional[np.ndarray] = None     # (N, D) float32, L2-normed
        # FAISS path
        self._faiss_index = None
        # PyTorch-CUDA path
        self._cuda_emb: Optional[torch.Tensor] = None    # (N, D) float32 on GPU
        self._gpu_device = torch.device("cuda:0") if _CUDA_AVAILABLE else None
        self._stale: bool = False

    # ── index management ──────────────────────────────────────────────────────

    def build(self, stock_ids: np.ndarray, embeddings: np.ndarray) -> None:
        """Build / rebuild the full index from scratch."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        self.embeddings = (embeddings / norms).astype(np.float32)
        self.stock_ids = np.asarray(stock_ids, dtype=np.int64)
        self._stale = False
        self._rebuild_backends()

    def _rebuild_backends(self) -> None:
        """Rebuild whichever backend(s) are available."""
        if self.embeddings is None:
            return
        # FAISS (highest priority)
        if _FAISS_AVAILABLE:
            index = _faiss.IndexFlatIP(self.embed_dim)
            index.add(self.embeddings)
            self._faiss_index = index
            return
        # PyTorch CUDA (second priority)
        if _CUDA_AVAILABLE and self._gpu_device is not None:
            self._cuda_emb = torch.from_numpy(self.embeddings).to(self._gpu_device)

    def update(self, stock_id: int, embedding: np.ndarray) -> None:
        """Update (or insert) a single stock's embedding; GPU mirror rebuilt lazily."""
        if self.stock_ids is None:
            self.build(np.array([stock_id], dtype=np.int64), embedding.reshape(1, -1))
            return

        norm_emb = (embedding / (np.linalg.norm(embedding) + 1e-8)).astype(np.float32)
        idx = np.where(self.stock_ids == stock_id)[0]
        if len(idx) > 0:
            self.embeddings[idx[0]] = norm_emb
        else:
            self.stock_ids = np.append(self.stock_ids, np.int64(stock_id))
            self.embeddings = np.vstack([self.embeddings, norm_emb.reshape(1, -1)])
        self._stale = True   # rebuilt lazily on next retrieve()

    # ── retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, user_embedding: np.ndarray, k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve top-k stocks by cosine similarity.

        Returns:
            top_stock_ids : (k,)  int64
            top_scores    : (k,)  float32, cosine similarities in [-1, 1]
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        if self._stale:
            self._rebuild_backends()
            self._stale = False

        k = min(k, len(self.stock_ids))
        user_norm = (user_embedding / (np.linalg.norm(user_embedding) + 1e-8)).astype(np.float32)

        # ── Backend 1: FAISS ──────────────────────────────────────────────────
        if self._faiss_index is not None:
            scores, raw_ids = self._faiss_index.search(user_norm.reshape(1, -1), k)
            valid = raw_ids[0] >= 0
            return self.stock_ids[raw_ids[0][valid]], scores[0][valid]

        # ── Backend 2: PyTorch CUDA (cuBLAS matmul) ───────────────────────────
        if self._cuda_emb is not None and self._gpu_device is not None:
            u = torch.from_numpy(user_norm).to(self._gpu_device)
            scores_gpu = self._cuda_emb @ u                  # (N,)  cuBLAS
            top_k = torch.topk(scores_gpu, k)
            top_scores = top_k.values.cpu().numpy()
            top_idx = top_k.indices.cpu().numpy()
            return self.stock_ids[top_idx], top_scores

        # ── Backend 3: NumPy CPU fallback ─────────────────────────────────────
        scores = self.embeddings @ user_norm
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return self.stock_ids[top_idx], scores[top_idx]

    @property
    def n_stocks(self) -> int:
        return len(self.stock_ids) if self.stock_ids is not None else 0
