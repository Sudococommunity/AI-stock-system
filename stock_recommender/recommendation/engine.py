"""
RecommendationEngine — the main public interface of the system.

Two-stage pipeline (same as YouTube DNN):
  Stage 1 — Candidate Retrieval: ANN search over stock embeddings (fast, ~1ms)
  Stage 2 — Ranking: score each candidate with rich features (slower, ~50ms)
  Post-processing: diversity, exploration, risk adjustment, explanation

Usage:
    engine = RecommendationEngine(models, db, pipeline)
    recs = engine.get_recommendations(user_id=42, k=10)
    analysis = engine.get_full_analysis(user_id=42, stock_id=101)
"""
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import logging
import pandas as pd

from stock_recommender.config import CONFIG
from stock_recommender.models.two_tower import TwoTowerModel, RankingModel, CandidateIndex
from stock_recommender.models.time_series import StockTransformer
from stock_recommender.features.feature_pipeline import FeaturePipeline
from stock_recommender.market.regime import MarketRegimeAnalyzer, MarketRegimeSnapshot
from stock_recommender.risk.risk_metrics import compute_full_risk_profile, RiskProfile
from stock_recommender.data.database import DatabaseManager
from stock_recommender.data.user_tracker import UserTracker
from stock_recommender.data.indian_market_data import NIFTY50_TICKER, SENSEX_TICKER

logger = logging.getLogger(__name__)


@dataclass
class StockRecommendation:
    """A single stock recommendation with full context."""
    stock_id: int
    ticker: str
    rank: int
    relevance_score: float              # from ranking model
    retrieval_score: float              # from candidate generation
    predicted_return_1d: float
    predicted_return_5d: float
    direction_probs: List[float]        # [P(down), P(flat), P(up)]
    risk_profile: Optional[RiskProfile]
    risk_score: float                   # 0–100
    opportunity_score: float            # 0–100
    risk_to_reward: float
    entry_signal: str                   # "strong buy" / "buy" / "hold" / "sell" / "strong sell"
    market_temperature: str = "neutral"
    market_regime: str = "sideways"
    market_note: str = ""
    key_signals: List[str] = field(default_factory=list)
    explanation: str = ""


@dataclass
class FullAnalysis:
    """Complete analysis of a single stock for a specific user."""
    stock_id: int
    ticker: str
    risk_profile: RiskProfile
    technical_snapshot: Dict
    price_forecast: Dict
    recommendation: StockRecommendation
    position_sizing_pct: float          # suggested portfolio allocation %
    stop_loss_pct: float                # suggested stop loss level
    take_profit_pct: float              # suggested take-profit level
    narrative: str                      # human-readable summary


class RecommendationEngine:
    """
    Main entry point for generating and explaining stock recommendations.
    """

    def __init__(
        self,
        two_tower: TwoTowerModel,
        ranker: RankingModel,
        transformer: StockTransformer,
        candidate_index: CandidateIndex,
        feature_pipeline: FeaturePipeline,
        db: DatabaseManager,
        device: str = "cpu",
    ):
        self.two_tower = two_tower
        self.ranker = ranker
        self.transformer = transformer
        self.candidate_index = candidate_index
        self.pipeline = feature_pipeline
        self.db = db
        # Infer device from transformer so the engine stays on whatever device the
        # Trainer moved the model to (e.g. CUDA after training).
        try:
            self.device = next(transformer.parameters()).device
        except StopIteration:
            self.device = torch.device(device)
        self.user_tracker = UserTracker(db)
        self.market_regime_analyzer = MarketRegimeAnalyzer(db)

        # Pre-load Nifty50 benchmark returns for beta/alpha calculation.
        # Stored as INDEX:NIFTY50 by ingest_benchmark_indices().
        self._nifty_returns: Optional[np.ndarray] = self._load_benchmark_returns(NIFTY50_TICKER)

        # Put models in eval mode
        for m in [two_tower, ranker, transformer]:
            m.eval()

    # ── Primary API ───────────────────────────────────────────────────────────

    def get_recommendations(
        self,
        user_id: int,
        k: int = CONFIG.model.final_k,
        exclude_interacted: bool = True,
        user_risk_tolerance: Optional[str] = None,
    ) -> List[StockRecommendation]:
        """
        Full two-stage recommendation pipeline.

        Returns the top-k stocks personalized for this user, with risk and
        forecast context for each recommendation.
        """
        # ── Stage 1: Candidate retrieval ──────────────────────────────────────
        market_state = self.market_regime_analyzer.analyze_market()
        user_emb = self._get_user_embedding(user_id)
        candidate_ids, retrieval_scores = self.candidate_index.retrieve(
            user_emb, k=CONFIG.model.candidate_pool_size
        )

        if len(candidate_ids) == 0:
            logger.warning(f"[Engine] No candidates found for user {user_id}. Index may be empty.")
            return []

        # Filter out stocks the user has already interacted with
        interacted = set()
        if exclude_interacted:
            interacted = set(self.user_tracker.get_all_interacted_stocks(user_id))
            mask = np.array([sid not in interacted for sid in candidate_ids])
            candidate_ids = candidate_ids[mask]
            retrieval_scores = retrieval_scores[mask]

        if len(candidate_ids) == 0:
            return []

        # ── Exploration: ε-greedy — occasionally inject random candidates ─────
        candidate_ids, retrieval_scores = self._apply_exploration(
            user_id, candidate_ids, retrieval_scores
        )
        if exclude_interacted and interacted:
            mask = np.array([sid not in interacted for sid in candidate_ids])
            candidate_ids = candidate_ids[mask]
            retrieval_scores = retrieval_scores[mask]

        if len(candidate_ids) == 0:
            return []

        # ── Stage 2: Ranking ──────────────────────────────────────────────────
        ranked = self._rank_candidates(
            user_id, user_emb, candidate_ids, retrieval_scores, market_state
        )

        # ── Post-processing: diversity + risk adjustment ───────────────────────
        final = self._apply_diversity(ranked, k * 2)
        final = self._apply_risk_adjustment(
            final,
            user_risk_tolerance or self._get_user_risk_tolerance(user_id),
            market_state,
        )
        final = final[:k]

        # ── Build full recommendation objects ─────────────────────────────────
        results = []
        for rank_i, item in enumerate(final):
            rec = self._build_recommendation(
                stock_id=item["stock_id"],
                rank=rank_i + 1,
                relevance_score=item["relevance_score"],
                retrieval_score=item["retrieval_score"],
                market_state=market_state,
            )
            if rec:
                results.append(rec)

        # Log what was shown (for later reward attribution)
        self._log_impressions(user_id, results)
        return results

    def get_full_analysis(self, user_id: int, stock_id: int) -> Optional[FullAnalysis]:
        """
        Compute a complete investment analysis for a specific stock + user pair.
        Includes risk metrics, price forecast, entry/exit levels, and narrative.
        """
        info = self.db.get_stock_info(stock_id)
        if info is None:
            return None
        market_state = self.market_regime_analyzer.analyze_market()

        ticker = info["ticker"]

        # Historical data
        raw = self.db.get_price_history(stock_id, limit=500)
        if len(raw) < 60:
            logger.warning(f"[Engine] Not enough price history for stock {stock_id}")
            return None

        df = pd.DataFrame(raw)
        returns = df["close"].pct_change().dropna().values

        # Risk profile — use actual Nifty50 as benchmark for beta/alpha
        benchmark = self._align_benchmark_returns(returns)
        risk_profile = compute_full_risk_profile(returns, benchmark_returns=benchmark)

        # Price forecast
        seq = self.pipeline.get_latest_sequence(df)
        forecast = {}
        if seq is not None:
            x = torch.tensor(seq, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                forecast = self.transformer.predict(x)
                forecast = {
                    "ret_1d": float(forecast["ret_1d_forecast"][0]),
                    "ret_5d": float(forecast["ret_5d_forecast"][0]),
                    "direction_probs": forecast["direction_probs"][0].tolist(),
                    "predicted_direction": int(forecast["predicted_direction"][0]),
                }

        # Technical snapshot
        snapshot = self.pipeline.get_latest_snapshot(df)
        technical = self._build_technical_summary(df, snapshot)

        # Build recommendation object for this stock
        user_emb = self._get_user_embedding(user_id)
        stock_emb = self._get_stock_embedding(stock_id)
        rec = self._build_recommendation(
            stock_id=stock_id, rank=1,
            relevance_score=float(np.dot(user_emb, stock_emb)) if stock_emb is not None else 0.0,
            retrieval_score=0.0,
            market_state=market_state,
        )
        if rec is None:
            return None

        # Position sizing: Kelly-fraction approximation
        win_rate = risk_profile.win_rate
        avg_win = abs(risk_profile.avg_win)
        avg_loss = abs(risk_profile.avg_loss)
        kelly = (win_rate / max(avg_loss, 1e-6) - (1 - win_rate) / max(avg_win, 1e-6))
        position_pct = float(np.clip(kelly * 0.25, 0.02, 0.15))  # Quarter-Kelly, capped at 15%

        # Stop loss / take profit based on ATR
        atr_pct = technical.get("atr_pct", 0.02)
        stop_loss_pct = float(np.clip(atr_pct * 2, 0.02, 0.15))
        take_profit_pct = float(stop_loss_pct * max(rec.risk_to_reward, 1.5))

        narrative = self._build_narrative(ticker, rec, risk_profile, forecast, technical)

        return FullAnalysis(
            stock_id=stock_id,
            ticker=ticker,
            risk_profile=risk_profile,
            technical_snapshot=technical,
            price_forecast=forecast,
            recommendation=rec,
            position_sizing_pct=position_pct * 100,
            stop_loss_pct=stop_loss_pct * 100,
            take_profit_pct=take_profit_pct * 100,
            narrative=narrative,
        )

    def get_similar_stocks(self, stock_id: int, k: int = 10) -> List[Dict]:
        """Find stocks similar to the given one in the embedding space."""
        stock_emb = self._get_stock_embedding(stock_id)
        if stock_emb is None:
            return []

        candidate_ids, scores = self.candidate_index.retrieve(stock_emb, k=k + 1)
        results = []
        for sid, score in zip(candidate_ids, scores):
            if sid == stock_id:
                continue
            info = self.db.get_stock_info(int(sid))
            if info:
                results.append({"stock_id": int(sid), "ticker": info["ticker"], "similarity": float(score)})

        return results[:k]

    def get_market_regime(self) -> MarketRegimeSnapshot:
        return self.market_regime_analyzer.analyze_market()

    # ── Internal helpers ──────────────────────────────────────────────────────

    @torch.no_grad()
    def _get_user_embedding(self, user_id: int) -> np.ndarray:
        """Get user embedding from cache or recompute from model."""
        cached = self.db.get_user_embedding(user_id)
        if cached:
            return np.array(cached, dtype=np.float32)

        # Cold-start: compute from profile
        profile = self.user_tracker.get_profile_features(user_id)
        uid_t = torch.tensor([user_id], dtype=torch.long, device=self.device)
        feat_t = torch.tensor(profile, dtype=torch.float32, device=self.device).unsqueeze(0)
        emb = self.two_tower.encode_user(uid_t, feat_t).squeeze(0).cpu().numpy()
        self.db.save_user_embedding(user_id, emb.tolist())
        return emb

    @torch.no_grad()
    def _get_stock_embedding(self, stock_id: int) -> Optional[np.ndarray]:
        """Get stock embedding from cache or recompute."""
        cached = self.db.get_stock_embedding(stock_id)
        if cached:
            return np.array(cached, dtype=np.float32)

        raw = self.db.get_price_history(stock_id, limit=500)
        if len(raw) < CONFIG.data.min_price_history_days:
            return None

        df = pd.DataFrame(raw)
        seq = self.pipeline.get_latest_sequence(df)
        snap = self.pipeline.get_latest_snapshot(df)
        if seq is None or snap is None:
            return None

        x = torch.tensor(seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        _, _, ts_emb = self.transformer(x)

        sid_t = torch.tensor([stock_id], dtype=torch.long, device=self.device)
        feat_t = torch.tensor(snap, dtype=torch.float32, device=self.device).unsqueeze(0)
        emb = self.two_tower.encode_stock(sid_t, feat_t, ts_emb).squeeze(0).cpu().numpy()
        self.db.save_stock_embedding(stock_id, emb.tolist())
        return emb

    def _rank_candidates(
        self,
        user_id: int,
        user_emb: np.ndarray,
        candidate_ids: np.ndarray,
        retrieval_scores: np.ndarray,
        market_state: MarketRegimeSnapshot,
    ) -> List[Dict]:
        """Score each candidate with the ranking model + risk adjustment."""
        results = []
        user_emb_t = torch.tensor(user_emb, dtype=torch.float32, device=self.device).unsqueeze(0)

        for sid, ret_score in zip(candidate_ids, retrieval_scores):
            sid = int(sid)
            stock_emb = self._get_stock_embedding(sid)
            if stock_emb is None:
                continue

            # Get price forecast
            raw = self.db.get_price_history(sid, limit=500)
            if len(raw) < CONFIG.data.min_price_history_days:
                continue

            df = pd.DataFrame(raw)
            seq = self.pipeline.get_latest_sequence(df)
            if seq is None:
                continue

            returns = df["close"].pct_change().dropna().values[-252:]  # 1-year window

            # Compute risk features — use actual Nifty50 as benchmark
            bench = self._align_benchmark_returns(returns)
            risk_profile = compute_full_risk_profile(returns, benchmark_returns=bench) if len(returns) > 30 else None
            risk_feats = self._risk_profile_to_tensor(risk_profile)
            info = self.db.get_stock_info(sid) or {}

            # Compute forecast features
            x = torch.tensor(seq, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                pred = self.transformer.predict(x)

            fc_feats = torch.tensor(
                [pred["ret_1d_forecast"][0], pred["ret_5d_forecast"][0],
                 pred["direction_probs"][0][2]],  # P(up)
                dtype=torch.float32, device=self.device,
            ).unsqueeze(0)

            stock_emb_t = torch.tensor(stock_emb, dtype=torch.float32, device=self.device).unsqueeze(0)

            with torch.no_grad():
                relevance = self.ranker(user_emb_t, stock_emb_t, risk_feats, fc_feats)

            market_bonus = self._compute_market_bonus(
                stock_info=info,
                df=df,
                risk_profile=risk_profile,
                forecast=pred,
                market_state=market_state,
            )
            blended_score = 0.7 * float(relevance.item()) + 0.3 * float(ret_score) + market_bonus

            results.append({
                "stock_id": sid,
                "relevance_score": float(relevance.item()),
                "retrieval_score": float(ret_score),
                "risk_profile": risk_profile,
                "forecast": pred,
                "market_bonus": float(market_bonus),
                "stock_info": info,
                "blended_score": blended_score,
            })

        results.sort(key=lambda x: x["blended_score"], reverse=True)
        return results

    def _apply_exploration(
        self,
        user_id: int,
        candidate_ids: np.ndarray,
        scores: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        ε-greedy exploration: with probability ε, replace some candidates
        with random stocks from the full universe to discover new patterns.
        """
        if np.random.random() > CONFIG.model.exploration_epsilon:
            return candidate_ids, scores

        all_stock_ids = np.array(self.db.get_all_stock_ids())
        if len(all_stock_ids) == 0:
            return candidate_ids, scores

        # Replace ~10% of candidates with random stocks
        n_explore = max(1, len(candidate_ids) // 10)
        explore_ids = np.random.choice(all_stock_ids, size=n_explore, replace=False)
        explore_scores = np.full(n_explore, scores.min() * 0.8)  # lower score so they rarely rank 1st

        combined_ids = np.concatenate([candidate_ids[:-n_explore], explore_ids])
        combined_scores = np.concatenate([scores[:-n_explore], explore_scores])
        return combined_ids, combined_scores

    def _apply_diversity(self, ranked: List[Dict], k: int) -> List[Dict]:
        """
        Maximal Marginal Relevance (MMR) — balance relevance and diversity.
        Prevents recommending 5 similar tech stocks when 2 would be better.
        """
        if not ranked:
            return []

        selected = [ranked[0]]
        remaining = ranked[1:]

        while len(selected) < k and remaining:
            best_score = -float("inf")
            best_item = None

            for item in remaining:
                relevance = item["blended_score"]
                # Penalize similarity to already-selected items
                sid = item["stock_id"]
                emb = self._get_stock_embedding(sid)
                if emb is None:
                    mmr_score = relevance * 0.5
                else:
                    sims = []
                    for sel in selected:
                        sel_emb = self._get_stock_embedding(sel["stock_id"])
                        if sel_emb is not None:
                            sims.append(float(np.dot(emb, sel_emb)))
                    max_sim = max(sims) if sims else 0.0
                    mmr_score = 0.7 * relevance - 0.3 * max_sim

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_item = item

            if best_item:
                selected.append(best_item)
                remaining.remove(best_item)

        return selected

    def _apply_risk_adjustment(
        self,
        ranked: List[Dict],
        user_risk_tolerance: str,
        market_state: MarketRegimeSnapshot,
    ) -> List[Dict]:
        """
        Re-sort after penalizing stocks that are too risky for the user's tolerance.
        conservative → prefer low risk_score
        aggressive   → minimal penalty
        """
        penalty_map = {"conservative": 0.5, "moderate": 0.2, "aggressive": 0.05}
        penalty = penalty_map.get(user_risk_tolerance, 0.2)
        regime_multiplier = {"cold": 1.5, "neutral": 1.0, "hot": 0.8}.get(market_state.temperature, 1.0)

        for item in ranked:
            rp = item.get("risk_profile")
            risk_score = rp.risk_score if rp else 50.0
            item["adjusted_score"] = item["blended_score"] - (penalty * regime_multiplier) * (risk_score / 100)

        ranked.sort(key=lambda x: x["adjusted_score"], reverse=True)
        return ranked

    def _build_recommendation(
        self,
        stock_id: int,
        rank: int,
        relevance_score: float,
        retrieval_score: float,
        market_state: MarketRegimeSnapshot,
    ) -> Optional[StockRecommendation]:
        info = self.db.get_stock_info(stock_id)
        if info is None:
            return None

        raw = self.db.get_price_history(stock_id, limit=300)
        if len(raw) < 60:
            return None

        df = pd.DataFrame(raw)
        returns = df["close"].pct_change().dropna().values

        ret_window = returns[-252:] if len(returns) > 252 else returns
        bench = self._align_benchmark_returns(ret_window)
        risk_profile = compute_full_risk_profile(ret_window, benchmark_returns=bench)

        # Price forecast
        seq = self.pipeline.get_latest_sequence(df)
        forecast = {"ret_1d": 0.0, "ret_5d": 0.0, "direction_probs": [0.33, 0.33, 0.34]}
        if seq is not None:
            x = torch.tensor(seq, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                pred = self.transformer.predict(x)
            forecast = {
                "ret_1d": float(pred["ret_1d_forecast"][0]),
                "ret_5d": float(pred["ret_5d_forecast"][0]),
                "direction_probs": pred["direction_probs"][0].tolist(),
            }

        # Entry signal from forecasts and RSI
        entry_signal = self._compute_entry_signal(forecast, risk_profile, market_state)
        key_signals = self._compute_key_signals(df, forecast, market_state, info)
        market_note = self._build_market_note(info, market_state)

        return StockRecommendation(
            stock_id=stock_id,
            ticker=info["ticker"],
            rank=rank,
            relevance_score=relevance_score,
            retrieval_score=retrieval_score,
            predicted_return_1d=forecast["ret_1d"],
            predicted_return_5d=forecast["ret_5d"],
            direction_probs=forecast["direction_probs"],
            risk_profile=risk_profile,
            risk_score=risk_profile.risk_score,
            opportunity_score=risk_profile.opportunity_score,
            risk_to_reward=risk_profile.risk_to_reward,
            entry_signal=entry_signal,
            market_temperature=market_state.temperature,
            market_regime=market_state.regime,
            market_note=market_note,
            key_signals=key_signals,
            explanation=self._build_short_explanation(
                info["ticker"], entry_signal, key_signals, risk_profile, market_state
            ),
        )

    def _compute_entry_signal(
        self,
        forecast: Dict,
        risk_profile: RiskProfile,
        market_state: MarketRegimeSnapshot,
    ) -> str:
        """Combine forecast and risk into a single entry signal."""
        ret_5d = forecast.get("ret_5d", 0.0)
        p_up = forecast.get("direction_probs", [0.33, 0.33, 0.34])[2]
        sharpe = risk_profile.sharpe_ratio if risk_profile else 0.0
        is_cold = market_state.temperature == "cold"
        is_hot = market_state.temperature == "hot"

        if p_up > (0.70 if is_cold else 0.65) and ret_5d > (0.025 if is_cold else 0.02) and sharpe > 0.5:
            return "strong buy"
        elif p_up > (0.62 if is_cold else 0.55) and ret_5d > (0.015 if is_cold else 0.01):
            return "buy"
        elif p_up < (0.22 if is_hot else 0.25) and ret_5d < -0.03:
            return "strong sell"
        elif p_up < (0.30 if is_hot else 0.35) and ret_5d < -0.02:
            return "sell"
        else:
            return "hold"

    def _compute_key_signals(
        self,
        df: pd.DataFrame,
        forecast: Dict,
        market_state: MarketRegimeSnapshot,
        stock_info: Dict,
    ) -> List[str]:
        """Extract the top 3 most important technical signals as human-readable strings."""
        from stock_recommender.features.technical_indicators import compute_all
        signals = [self._build_market_note(stock_info, market_state)]

        try:
            enriched = compute_all(df)
            latest = enriched.iloc[-1]

            rsi_val = latest.get("rsi_14", 50)
            if rsi_val < 30:
                signals.append(f"RSI oversold ({rsi_val:.0f}) — potential bounce")
            elif rsi_val > 70:
                signals.append(f"RSI overbought ({rsi_val:.0f}) — caution")

            macd_hist = latest.get("macd_hist", 0)
            if macd_hist > 0:
                signals.append("MACD bullish crossover")
            elif macd_hist < 0:
                signals.append("MACD bearish pressure")

            bb_pct = latest.get("bb_pct", 0.5)
            if bb_pct < 0.1:
                signals.append("Price near lower Bollinger Band — oversold zone")
            elif bb_pct > 0.9:
                signals.append("Price near upper Bollinger Band — resistance zone")

            adx_val = latest.get("adx", 0)
            if adx_val > 25:
                di_diff = latest.get("di_diff", 0)
                signals.append(
                    f"Strong trend (ADX {adx_val:.0f}) — {'bullish' if di_diff > 0 else 'bearish'}"
                )

            p_up = forecast.get("direction_probs", [0.33, 0.33, 0.34])[2]
            signals.append(f"Model predicts {p_up*100:.0f}% probability of upward move")

        except Exception:
            signals.append("Technical analysis unavailable")

        return signals[:4]

    def _build_short_explanation(
        self,
        ticker: str,
        entry_signal: str,
        key_signals: List[str],
        risk_profile: RiskProfile,
        market_state: MarketRegimeSnapshot,
    ) -> str:
        rr = risk_profile.risk_to_reward if risk_profile else 0
        sharpe = risk_profile.sharpe_ratio if risk_profile else 0
        top_signal = key_signals[0] if key_signals else "mixed signals"
        return (
            f"{ticker}: {entry_signal.upper()} — {top_signal}. "
            f"Risk/Reward: {rr:.2f}x | Sharpe: {sharpe:.2f} | "
            f"Market: {market_state.temperature.upper()} {market_state.regime.upper()}"
        )

    def _build_technical_summary(self, df: pd.DataFrame, snapshot: Optional[np.ndarray]) -> Dict:
        """Extract human-readable technical metrics from the latest bar."""
        from stock_recommender.features.technical_indicators import compute_all, MODEL_FEATURE_COLS
        try:
            enriched = compute_all(df)
            latest = enriched.iloc[-1]
            return {
                "rsi_14": float(latest.get("rsi_14", np.nan)),
                "macd_hist": float(latest.get("macd_hist", np.nan)),
                "bb_pct": float(latest.get("bb_pct", np.nan)),
                "adx": float(latest.get("adx", np.nan)),
                "atr_pct": float(latest.get("atr_pct", np.nan)),
                "rel_volume": float(latest.get("rel_volume", np.nan)),
                "ret_1d": float(latest.get("ret_1d", np.nan)),
                "ret_5d": float(latest.get("ret_5d", np.nan)),
                "hvol_20": float(latest.get("hvol_20", np.nan)),
                "close_vs_sma50": float(latest.get("close_vs_sma50", np.nan)),
            }
        except Exception:
            return {}

    def _build_narrative(
        self,
        ticker: str,
        rec: StockRecommendation,
        risk: RiskProfile,
        forecast: Dict,
        technical: Dict,
    ) -> str:
        p_up = forecast.get("direction_probs", [0.33, 0.33, 0.34])[2] * 100
        ret_5d = forecast.get("ret_5d", 0) * 100

        lines = [
            f"=== {ticker} Full Investment Analysis ===",
            f"Signal: {rec.entry_signal.upper()}",
            f"",
            f"MARKET REGIME",
            f"  Temperature           : {rec.market_temperature.upper()}",
            f"  Trend                 : {rec.market_regime.upper()}",
            f"  Context               : {rec.market_note}",
            f"",
            f"FORECAST",
            f"  1-day expected return : {rec.predicted_return_1d*100:+.2f}%",
            f"  5-day expected return : {rec.predicted_return_5d*100:+.2f}%",
            f"  Probability of gain   : {p_up:.0f}%",
            f"",
            f"RISK METRICS",
            f"  Sharpe Ratio          : {risk.sharpe_ratio:.2f}",
            f"  Sortino Ratio         : {risk.sortino_ratio:.2f}",
            f"  Max Drawdown          : {risk.max_drawdown*100:.1f}%",
            f"  VaR (95%)             : {risk.var_95*100:.2f}%",
            f"  CVaR (95%)            : {risk.cvar_95*100:.2f}%",
            f"  Annualized Volatility : {risk.annualized_volatility*100:.1f}%",
            f"  Beta                  : {risk.beta:.2f}",
            f"",
            f"PERFORMANCE",
            f"  Win Rate              : {risk.win_rate*100:.1f}%",
            f"  Risk/Reward Ratio     : {rec.risk_to_reward:.2f}x",
            f"  Risk Score            : {risk.risk_score:.0f}/100",
            f"  Opportunity Score     : {risk.opportunity_score:.0f}/100",
            f"",
            f"KEY SIGNALS",
        ]
        for s in rec.key_signals:
            lines.append(f"  • {s}")

        return "\n".join(lines)

    def _risk_profile_to_tensor(self, rp: Optional[RiskProfile]) -> torch.Tensor:
        """Convert a RiskProfile to a fixed-size feature tensor for the ranker."""
        if rp is None:
            return torch.zeros(1, 8, device=self.device)
        feats = [
            rp.sharpe_ratio / 3.0,           # normalize ~[-1, 3]
            rp.sortino_ratio / 3.0,
            abs(rp.max_drawdown),
            rp.annualized_volatility,
            rp.var_95,
            rp.beta / 2.0,
            rp.win_rate,
            rp.risk_score / 100.0,
        ]
        return torch.tensor([feats], dtype=torch.float32, device=self.device)

    def _compute_market_bonus(
        self,
        stock_info: Dict,
        df: pd.DataFrame,
        risk_profile: Optional[RiskProfile],
        forecast: Dict,
        market_state: MarketRegimeSnapshot,
    ) -> float:
        if df.empty:
            return 0.0

        sector = str(stock_info.get("sector") or "").strip()
        risk_score = risk_profile.risk_score if risk_profile else 50.0
        latest_close = float(df["close"].iloc[-1])
        sma_200 = float(df["close"].rolling(200).mean().iloc[-1]) if len(df) >= 200 else latest_close
        above_200 = latest_close >= sma_200
        p_up = float(forecast["direction_probs"][0][2])
        ret_5d = float(forecast["ret_5d_forecast"][0])

        bonus = 0.0
        if sector and sector in market_state.leading_sectors:
            bonus += 0.04
        if sector and sector in market_state.lagging_sectors:
            bonus -= 0.04

        if market_state.temperature == "cold":
            bonus -= 0.18 * (risk_score / 100.0)
            if not above_200:
                bonus -= 0.08
            if sector and self._is_defensive_sector(sector):
                bonus += 0.05
            if p_up > 0.70 and ret_5d > 0.02:
                bonus += 0.02
        elif market_state.temperature == "hot":
            bonus += 0.10 * ((100.0 - risk_score) / 100.0)
            if above_200:
                bonus += 0.03
            if sector and sector in market_state.leading_sectors:
                bonus += 0.03
            if p_up > 0.60 and ret_5d > 0.01:
                bonus += 0.02
        else:
            bonus -= 0.05 * (risk_score / 100.0)
            if above_200:
                bonus += 0.02
        return float(bonus)

    def _build_market_note(self, stock_info: Dict, market_state: MarketRegimeSnapshot) -> str:
        sector = str(stock_info.get("sector") or "").strip()
        sector_note = ""
        if sector and sector in market_state.leading_sectors:
            sector_note = f"; {sector} is leading"
        elif sector and sector in market_state.lagging_sectors:
            sector_note = f"; {sector} is lagging"

        if market_state.temperature == "cold":
            return (
                f"Cold market backdrop: breadth is {market_state.breadth_label}, "
                f"{market_state.pct_above_200dma*100:.0f}% of stocks are above 200 DMA"
                f"{sector_note}"
            )
        if market_state.temperature == "hot":
            return (
                f"Hot market backdrop: breadth is {market_state.breadth_label}, "
                f"A/D ratio is {market_state.advance_decline_ratio:.2f}"
                f"{sector_note}"
            )
        return f"Neutral market backdrop with {market_state.sector_rotation} sector rotation{sector_note}"

    def _is_defensive_sector(self, sector: str) -> bool:
        normalized = sector.lower()
        return any(tag in normalized for tag in (
            "consumer staples",
            "fmcg",
            "healthcare",
            "pharma",
            "pharmaceutical",
            "utilities",
        ))

    def _get_user_risk_tolerance(self, user_id: int) -> str:
        user = self.db.get_user(user_id)
        return user.get("risk_tolerance", "moderate") if user else "moderate"

    def _log_impressions(self, user_id: int, recommendations: List[StockRecommendation]) -> None:
        for rec in recommendations:
            self.db.log_recommendation(
                user_id, rec.stock_id, rec.rank, rec.relevance_score
            )

    # ── Benchmark returns helpers ─────────────────────────────────────────────

    def _load_benchmark_returns(self, ticker: str) -> Optional[np.ndarray]:
        """Load full Nifty50 (or Sensex) daily returns from DB into memory."""
        info = self.db.get_stock_info_by_ticker(ticker)
        if info is None:
            return None
        history = self.db.get_price_history(int(info["stock_id"]), limit=3000)
        if len(history) < 30:
            return None
        closes = np.array([float(r["close"]) for r in history], dtype=np.float64)
        returns = np.diff(closes) / closes[:-1]
        return returns.astype(np.float32)

    def _align_benchmark_returns(self, stock_returns: np.ndarray) -> Optional[np.ndarray]:
        """
        Tail-align Nifty50 returns to match the length of stock_returns.
        Returns None if Nifty50 data isn't loaded or is too short.
        """
        if self._nifty_returns is None or len(self._nifty_returns) < len(stock_returns):
            return None
        return self._nifty_returns[-len(stock_returns):]
