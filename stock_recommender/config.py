"""
Central configuration for the stock recommendation system.
All hyperparameters live here — change once, propagate everywhere.
"""
from dataclasses import asdict, dataclass, field, is_dataclass
import json
from typing import Any, Dict, List


@dataclass
class ModelConfig:
    # ── Universe sizes ────────────────────────────────────────────────────────
    max_stocks: int = 10_000       # grows as we index new tickers
    max_users: int = 100_000

    # ── Embedding space ───────────────────────────────────────────────────────
    embed_dim: int = 128           # MUST be equal for user and stock towers

    # ── Tower MLP hidden layers ───────────────────────────────────────────────
    user_tower_hidden: List[int] = field(default_factory=lambda: [512, 256, 128])
    stock_tower_hidden: List[int] = field(default_factory=lambda: [512, 256, 128])
    tower_dropout: float = 0.2

    # ── Time-series Transformer ───────────────────────────────────────────────
    seq_len: int = 60              # 60 trading days (~3 months)
    n_tech_features: int = 140     # output dim of FeaturePipeline (matches N_MODEL_FEATURES)
    transformer_dim: int = 128
    transformer_heads: int = 8
    transformer_layers: int = 4
    transformer_dropout: float = 0.1

    # ── User features ─────────────────────────────────────────────────────────
    n_user_profile_features: int = 10

    # ── Ranking model ─────────────────────────────────────────────────────────
    ranker_hidden: List[int] = field(default_factory=lambda: [256, 128, 64])
    ranker_dropout: float = 0.2

    # ── Contrastive loss temperature ──────────────────────────────────────────
    temperature: float = 0.07

    # ── Recommendation pipeline ───────────────────────────────────────────────
    candidate_pool_size: int = 200     # stage-1 retrieval size
    final_k: int = 10                  # stage-2 ranked output size
    exploration_epsilon: float = 0.10  # ε-greedy exploration rate


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    max_epochs: int = 50
    early_stop_patience: int = 5

    # Online / incremental learning
    online_lr: float = 5e-5
    replay_buffer_capacity: int = 50_000
    replay_min_fill: int = 256        # minimum events before first replay update
    online_update_every: int = 32     # events between micro-updates
    full_retrain_every: int = 10_000  # events before full retraining is triggered


@dataclass
class RiskConfig:
    risk_free_rate: float = 0.05         # annual, used in Sharpe / Sortino / Alpha
    var_confidence: float = 0.95
    trading_days_per_year: int = 252


@dataclass
class DataConfig:
    db_url: str = "postgresql://postgres:postgres@localhost:5432/stock_recommender"
    checkpoint_dir: str = "checkpoints/"
    tensor_cache_dir: str = "tensor_cache/"
    min_price_history_days: int = 200   # needed to compute all indicators reliably
    feature_lookback_days: int = 200


@dataclass
class RuntimeConfig:
    artifacts_dir: str = "artifacts/"
    reports_dir: str = "artifacts/reports/"
    tournament_dir: str = "artifacts/tournaments/"
    market_name: str = "demo"
    prediction_horizon_days: int = 5   # must match training horizon (ret_5d_forecast)
    walk_forward_step_days: int = 5
    top_k_checkpoints: int = 3
    population_size: int = 5


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    data: DataConfig = field(default_factory=DataConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


# Global singleton — import this everywhere
CONFIG = Config()


def _deep_update_dataclass(obj: Any, updates: Dict[str, Any]) -> None:
    for key, value in updates.items():
        if not hasattr(obj, key):
            raise KeyError(f"Unknown config key: {key}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _deep_update_dataclass(current, value)
        else:
            setattr(obj, key, value)


def load_config_overrides(path: str) -> Config:
    """
    Load JSON config overrides into the global CONFIG singleton.
    This keeps the code portable across machines without hard-coded paths.
    """
    with open(path, "r", encoding="utf-8") as f:
        overrides = json.load(f)
    if not isinstance(overrides, dict):
        raise ValueError("Config override file must contain a JSON object")
    _deep_update_dataclass(CONFIG, overrides)
    return CONFIG


def config_to_dict() -> Dict[str, Any]:
    return asdict(CONFIG)
