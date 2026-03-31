"""
StockTransformer — encoder-only Transformer for stock price prediction.

Architecture mirrors the "sequence model" used in modern time-series research:
  • Positional encoding preserves temporal order
  • Multi-head self-attention captures long-range dependencies across the window
  • Three output heads: return regression, direction classification, embedding
  • The embedding head output feeds into the Stock Tower (two-tower model)

The model sees only technical features — never raw prices — so it generalizes
across stocks of very different price scales.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from stock_recommender.config import CONFIG


# ── Positional Encoding ───────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding from "Attention Is All You Need".
    Encodes position as a fixed pattern — no learned parameters.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ── Stock Transformer ─────────────────────────────────────────────────────────

class StockTransformer(nn.Module):
    """
    Encoder-only Transformer for stock feature sequences.

    Input:  (batch, seq_len, n_features)
    Outputs:
        price_forecast   : (batch, 2)  — predicted 1-day and 5-day returns
        direction_logits : (batch, 3)  — logits for [down, flat, up]
        sequence_emb     : (batch, embed_dim)  — CLS-like aggregate embedding
                           fed into the Stock Tower of the two-tower model
    """

    def __init__(
        self,
        n_features: int = CONFIG.model.n_tech_features,
        d_model: int = CONFIG.model.transformer_dim,
        nhead: int = CONFIG.model.transformer_heads,
        num_layers: int = CONFIG.model.transformer_layers,
        dropout: float = CONFIG.model.transformer_dropout,
        embed_dim: int = CONFIG.model.embed_dim,
    ):
        super().__init__()
        self.d_model = d_model
        self.embed_dim = embed_dim

        # Project raw features into transformer dimension
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
        )

        # Learnable CLS token prepended to every sequence
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ── Output heads ──────────────────────────────────────────────────────
        # Return regression: predict 1-day and 5-day forward returns
        self.price_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

        # Direction classification: down / flat / up
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3),
        )

        # Embedding head: projects CLS token into the shared embedding space
        self.embed_head = nn.Sequential(
            nn.Linear(d_model, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, n_features)
            src_key_padding_mask: (batch, seq_len+1) — True positions are masked

        Returns:
            price_forecast   : (batch, 2)
            direction_logits : (batch, 3)
            sequence_emb     : (batch, embed_dim)
        """
        batch_size = x.size(0)

        # Project and add positional encoding
        x = self.input_proj(x)           # (B, T, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)   # (B, T+1, d_model)
        x = self.pos_enc(x)

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        cls_out = x[:, 0]    # CLS token representation (B, d_model)
        last_out = x[:, -1]  # Last timestep (most recent) (B, d_model)

        price_forecast = self.price_head(last_out)
        direction_logits = self.direction_head(cls_out)
        sequence_emb = F.normalize(self.embed_head(cls_out), dim=-1)

        return price_forecast, direction_logits, sequence_emb

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict:
        """
        Convenience method for inference. Returns a dict with readable keys.
        x: (batch, seq_len, n_features)
        """
        self.eval()
        price_fc, dir_logits, emb = self.forward(x)
        dir_probs = F.softmax(dir_logits, dim=-1)
        return {
            "ret_1d_forecast": price_fc[:, 0].cpu().numpy(),
            "ret_5d_forecast": price_fc[:, 1].cpu().numpy(),
            "direction_probs": dir_probs.cpu().numpy(),   # [P(down), P(flat), P(up)]
            "predicted_direction": dir_logits.argmax(-1).cpu().numpy() - 1,  # -1,0,1
            "sequence_embedding": emb.cpu().numpy(),
        }


# ── Time Series Loss ──────────────────────────────────────────────────────────

class TimeSeriesLoss(nn.Module):
    """
    Multi-task loss for the StockTransformer:
      • MSE on return predictions (regression task)
      • Cross-entropy on direction (classification task)
    Weighted sum — direction classification helps the model learn trend structure.
    """

    def __init__(self, return_weight: float = 0.6, direction_weight: float = 0.4):
        super().__init__()
        self.return_weight = return_weight
        self.direction_weight = direction_weight
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        price_forecast: torch.Tensor,   # (B, 2) — predicted [ret_1d, ret_5d]
        direction_logits: torch.Tensor, # (B, 3)
        target_returns: torch.Tensor,   # (B, 2) — true [ret_1d, ret_5d]
        target_direction: torch.Tensor, # (B,) — long int: 0=down, 1=flat, 2=up
    ) -> Tuple[torch.Tensor, dict]:

        return_loss = self.mse(price_forecast, target_returns)
        direction_loss = self.ce(direction_logits, target_direction)
        total = self.return_weight * return_loss + self.direction_weight * direction_loss

        return total, {
            "total": total.item(),
            "return_loss": return_loss.item(),
            "direction_loss": direction_loss.item(),
        }


# ── LSTM Baseline (alternative/comparison model) ──────────────────────────────

class StockLSTM(nn.Module):
    """
    LSTM-based alternative to the Transformer.
    Faster to train, useful as a sanity-check baseline.
    Same interface as StockTransformer.
    """

    def __init__(
        self,
        n_features: int = CONFIG.model.n_tech_features,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        embed_dim: int = CONFIG.model.embed_dim,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.price_head = nn.Linear(hidden_dim, 2)
        self.direction_head = nn.Linear(hidden_dim, 3)
        self.embed_head = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out, (hn, _) = self.lstm(x)
        last = out[:, -1]
        price_forecast = self.price_head(last)
        direction_logits = self.direction_head(last)
        sequence_emb = F.normalize(self.embed_head(last), dim=-1)
        return price_forecast, direction_logits, sequence_emb
