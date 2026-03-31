from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from stock_recommender.data.database import DatabaseManager


DEFENSIVE_SECTORS = {
    "consumer staples",
    "fmcg",
    "healthcare",
    "pharma",
    "pharmaceuticals",
    "utilities",
}

CYCLICAL_SECTORS = {
    "automotive",
    "auto",
    "banking",
    "financials",
    "finance",
    "industrials",
    "infrastructure",
    "it",
    "technology",
    "materials",
    "metals",
    "realty",
    "real estate",
    "capital goods",
    "energy",
}

SPECIAL_TICKER_TOKENS = ("VIX", "INDIAVIX", "ADVANCE", "DECLINE")


@dataclass
class MarketRegimeSnapshot:
    as_of_date: Optional[str]
    universe_size: int
    advancing_count: int
    declining_count: int
    advance_decline_ratio: float
    pct_above_50dma: float
    pct_above_200dma: float
    median_return_20d: float
    median_return_60d: float
    median_volatility_20d: float
    temperature: str
    regime: str
    breadth_label: str
    india_vix: Optional[float] = None
    vix_regime: str = "unknown"
    institutional_flow_signal: str = "neutral"
    institutional_flow_source: str = "proxy_large_cap_participation"
    leading_sectors: List[str] = field(default_factory=list)
    lagging_sectors: List[str] = field(default_factory=list)
    sector_rotation: str = "balanced"
    score: float = 0.0
    summary: str = ""


class MarketRegimeAnalyzer:
    """
    Compute a live market state from the current DB universe.

    The signal is based on real, currently-ingested market data rather than the
    synthetic bull/bear labels used during synthetic data generation.
    """

    def __init__(self, db: DatabaseManager):
        self.db = db

    def analyze_market(self, lookback_days: int = 260) -> MarketRegimeSnapshot:
        universe = self._collect_universe_rows(lookback_days=lookback_days)
        if not universe:
            return MarketRegimeSnapshot(
                as_of_date=None,
                universe_size=0,
                advancing_count=0,
                declining_count=0,
                advance_decline_ratio=1.0,
                pct_above_50dma=0.0,
                pct_above_200dma=0.0,
                median_return_20d=0.0,
                median_return_60d=0.0,
                median_volatility_20d=0.0,
                temperature="neutral",
                regime="sideways",
                breadth_label="unknown",
                summary="Market regime unavailable: insufficient universe history.",
            )

        as_of_date = max(row["date"] for row in universe)
        advancing = sum(1 for row in universe if row["ret_1d"] > 0)
        declining = sum(1 for row in universe if row["ret_1d"] < 0)
        universe_size = len(universe)
        ad_ratio = advancing / max(declining, 1)
        pct_above_50 = float(np.mean([row["above_50dma"] for row in universe]))
        pct_above_200 = float(np.mean([row["above_200dma"] for row in universe]))
        median_ret_20 = float(np.median([row["ret_20d"] for row in universe]))
        median_ret_60 = float(np.median([row["ret_60d"] for row in universe]))
        median_vol_20 = float(np.median([row["volatility_20d"] for row in universe]))

        breadth_score = np.clip((ad_ratio - 1.0) / 0.8, -1.0, 1.0) * 0.35
        breadth_score += np.clip((pct_above_200 - 0.5) / 0.3, -1.0, 1.0) * 0.35
        breadth_score += np.clip((pct_above_50 - 0.5) / 0.3, -1.0, 1.0) * 0.30

        trend_score = np.clip(median_ret_20 / 0.08, -1.0, 1.0) * 0.55
        trend_score += np.clip(median_ret_60 / 0.18, -1.0, 1.0) * 0.45

        sector_scores = self._compute_sector_scores(universe)
        leading_sectors = [item["sector"] for item in sector_scores[:3]]
        lagging_sectors = [item["sector"] for item in sector_scores[-3:]] if sector_scores else []
        sector_rotation = self._classify_sector_rotation(leading_sectors)
        sector_score = self._sector_score(sector_rotation, sector_scores)

        india_vix = self._read_india_vix()
        vix_regime, vix_score = self._classify_vix(india_vix)

        flow_signal, flow_score, flow_source = self._compute_institutional_flow_signal(universe)

        score = float(
            np.clip(
                0.35 * breadth_score
                + 0.35 * trend_score
                + 0.15 * sector_score
                + 0.15 * flow_score
                + vix_score,
                -1.0,
                1.0,
            )
        )

        if score >= 0.35:
            temperature = "hot"
        elif score <= -0.35:
            temperature = "cold"
        else:
            temperature = "neutral"

        if median_ret_60 > 0.05 and pct_above_200 > 0.60:
            regime = "bull"
        elif median_ret_60 < -0.05 and pct_above_200 < 0.40:
            regime = "bear"
        else:
            regime = "sideways"

        breadth_label = self._classify_breadth_label(ad_ratio, pct_above_200)
        summary = (
            f"{temperature.upper()} {regime.upper()} market: "
            f"A/D {ad_ratio:.2f}, {pct_above_200*100:.0f}% above 200 DMA, "
            f"20d median return {median_ret_20*100:+.1f}%."
        )
        if india_vix is not None:
            summary += f" India VIX {india_vix:.1f} ({vix_regime})."
        summary += f" Rotation: {sector_rotation}. Flow: {flow_signal}."

        return MarketRegimeSnapshot(
            as_of_date=as_of_date,
            universe_size=universe_size,
            advancing_count=advancing,
            declining_count=declining,
            advance_decline_ratio=float(ad_ratio),
            pct_above_50dma=pct_above_50,
            pct_above_200dma=pct_above_200,
            median_return_20d=median_ret_20,
            median_return_60d=median_ret_60,
            median_volatility_20d=median_vol_20,
            temperature=temperature,
            regime=regime,
            breadth_label=breadth_label,
            india_vix=india_vix,
            vix_regime=vix_regime,
            institutional_flow_signal=flow_signal,
            institutional_flow_source=flow_source,
            leading_sectors=leading_sectors,
            lagging_sectors=lagging_sectors,
            sector_rotation=sector_rotation,
            score=score,
            summary=summary,
        )

    def _collect_universe_rows(self, lookback_days: int) -> List[Dict]:
        rows: List[Dict] = []
        for stock in self.db.get_all_stocks():
            ticker = str(stock.get("ticker", "") or "").upper()
            if any(token in ticker for token in SPECIAL_TICKER_TOKENS):
                continue

            history = self.db.get_price_history(int(stock["stock_id"]), limit=lookback_days)
            if len(history) < 220:
                continue

            df = pd.DataFrame(history)
            close = df["close"].astype(float)
            sma_50 = close.rolling(50).mean().iloc[-1]
            sma_200 = close.rolling(200).mean().iloc[-1]
            if not np.isfinite(sma_50) or not np.isfinite(sma_200):
                continue

            returns = close.pct_change().dropna()
            if len(returns) < 60:
                continue

            latest_close = float(close.iloc[-1])
            latest_volume = float(df["volume"].iloc[-1])
            rows.append(
                {
                    "stock_id": int(stock["stock_id"]),
                    "ticker": ticker,
                    "sector": str(stock.get("sector") or "").strip() or "Unknown",
                    "market_cap": float(stock.get("market_cap") or 0.0),
                    "liquidity_proxy": latest_close * max(latest_volume, 0.0),
                    "date": str(df["date"].iloc[-1]),
                    "ret_1d": float(returns.iloc[-1]),
                    "ret_20d": float(close.iloc[-1] / close.iloc[-21] - 1.0),
                    "ret_60d": float(close.iloc[-1] / close.iloc[-61] - 1.0),
                    "volatility_20d": float(returns.tail(20).std(ddof=0) * np.sqrt(252)),
                    "above_50dma": bool(latest_close > float(sma_50)),
                    "above_200dma": bool(latest_close > float(sma_200)),
                }
            )
        return rows

    def _compute_sector_scores(self, universe: List[Dict]) -> List[Dict]:
        by_sector: Dict[str, List[Dict]] = {}
        for row in universe:
            by_sector.setdefault(row["sector"], []).append(row)

        scores: List[Dict] = []
        for sector, members in by_sector.items():
            mean_ret_20 = float(np.mean([item["ret_20d"] for item in members]))
            pct_above_200 = float(np.mean([item["above_200dma"] for item in members]))
            score = mean_ret_20 + 0.05 * (pct_above_200 - 0.5)
            scores.append(
                {
                    "sector": sector,
                    "score": score,
                    "ret_20d": mean_ret_20,
                    "pct_above_200dma": pct_above_200,
                }
            )
        scores.sort(key=lambda item: item["score"], reverse=True)
        return scores

    def _classify_sector_rotation(self, leading_sectors: List[str]) -> str:
        if not leading_sectors:
            return "balanced"
        normalized = [sector.lower() for sector in leading_sectors]
        defensive = sum(any(tag in sector for tag in DEFENSIVE_SECTORS) for sector in normalized)
        cyclical = sum(any(tag in sector for tag in CYCLICAL_SECTORS) for sector in normalized)
        if cyclical >= 2:
            return "risk_on"
        if defensive >= 2:
            return "risk_off"
        return "balanced"

    def _sector_score(self, sector_rotation: str, sector_scores: List[Dict]) -> float:
        if not sector_scores:
            return 0.0
        spread = sector_scores[0]["score"] - sector_scores[-1]["score"]
        spread_score = float(np.clip(spread / 0.10, -1.0, 1.0)) * 0.4
        rotation_bonus = {
            "risk_on": 0.25,
            "balanced": 0.0,
            "risk_off": -0.25,
        }.get(sector_rotation, 0.0)
        return float(np.clip(spread_score + rotation_bonus, -1.0, 1.0))

    def _read_india_vix(self) -> Optional[float]:
        for stock in self.db.get_all_stocks():
            ticker = str(stock.get("ticker", "") or "").upper()
            if "VIX" not in ticker:
                continue
            history = self.db.get_price_history(int(stock["stock_id"]), limit=5)
            if history:
                return float(history[-1]["close"])
        return None

    def _classify_vix(self, india_vix: Optional[float]) -> tuple[str, float]:
        if india_vix is None:
            return "unknown", 0.0
        if india_vix > 20.0:
            return "fear", -0.20
        if india_vix < 14.0:
            return "calm", 0.10
        return "normal", 0.0

    def _compute_institutional_flow_signal(self, universe: List[Dict]) -> tuple[str, float, str]:
        if len(universe) < 10:
            return "neutral", 0.0, "proxy_large_cap_participation"

        ranked = sorted(
            universe,
            key=lambda row: (row["market_cap"] if row["market_cap"] > 0 else row["liquidity_proxy"]),
            reverse=True,
        )
        split = max(len(ranked) // 5, 3)
        leaders = ranked[:split]
        rest = ranked[split:] or ranked

        leader_ret = float(np.mean([row["ret_20d"] for row in leaders]))
        rest_ret = float(np.mean([row["ret_20d"] for row in rest]))
        leader_breadth = float(np.mean([row["above_200dma"] for row in leaders]))
        rest_breadth = float(np.mean([row["above_200dma"] for row in rest]))
        score = np.clip((leader_ret - rest_ret) / 0.06, -1.0, 1.0) * 0.6
        score += np.clip((leader_breadth - rest_breadth) / 0.30, -1.0, 1.0) * 0.4
        score = float(np.clip(score, -1.0, 1.0))

        if score >= 0.25:
            label = "buying"
        elif score <= -0.25:
            label = "selling"
        else:
            label = "neutral"
        return label, score, "proxy_large_cap_participation"

    def _classify_breadth_label(self, ad_ratio: float, pct_above_200: float) -> str:
        if ad_ratio >= 1.4 and pct_above_200 >= 0.65:
            return "strong"
        if ad_ratio <= 0.8 and pct_above_200 <= 0.40:
            return "weak"
        return "mixed"
