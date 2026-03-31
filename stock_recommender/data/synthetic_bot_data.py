"""
Synthetic bot/user data generator for recommendation training.

This module keeps the synthetic market generator but replaces the old clean
user seeding with noisier behavior:
- random explorers with weak or no sector preference
- contrarians that act around crashes
- momentum chasers that pile into pumps
- herd followers that cluster in the same stock/time windows
- heavy-tailed activity so a few users dominate event volume
- events timestamped across the full price history
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from stock_recommender.data.database import DatabaseManager


REGIMES = {
    "bull": {"mu": 0.0008, "sigma": 0.012},
    "bear": {"mu": -0.0006, "sigma": 0.022},
    "sideways": {"mu": 0.0001, "sigma": 0.015},
}

TRANSITION_MATRIX = np.array(
    [
        [0.97, 0.01, 0.02],
        [0.02, 0.95, 0.03],
        [0.04, 0.03, 0.93],
    ]
)

REGIME_NAMES = ["bull", "bear", "sideways"]

SYNTHETIC_STOCKS = [
    ("AAPL-SYN", "Apple Synthetic", "Technology", 150.0, 1.1, 0.006),
    ("MSFT-SYN", "Microsoft Synthetic", "Technology", 280.0, 0.9, 0.005),
    ("GOOGL-SYN", "Google Synthetic", "Technology", 120.0, 1.0, 0.007),
    ("AMZN-SYN", "Amazon Synthetic", "Consumer", 130.0, 1.2, 0.008),
    ("JPM-SYN", "JPMorgan Synthetic", "Finance", 145.0, 1.3, 0.009),
    ("GS-SYN", "Goldman Sachs Synthetic", "Finance", 340.0, 1.4, 0.010),
    ("JNJ-SYN", "J&J Synthetic", "Healthcare", 160.0, 0.6, 0.004),
    ("PFE-SYN", "Pfizer Synthetic", "Healthcare", 35.0, 0.7, 0.005),
    ("XOM-SYN", "Exxon Synthetic", "Energy", 80.0, 1.0, 0.010),
    ("CVX-SYN", "Chevron Synthetic", "Energy", 150.0, 0.9, 0.009),
    ("TSLA-SYN", "Tesla Synthetic", "Automotive", 220.0, 1.8, 0.018),
    ("NVDA-SYN", "Nvidia Synthetic", "Technology", 450.0, 1.6, 0.015),
    ("AMD-SYN", "AMD Synthetic", "Technology", 110.0, 1.5, 0.014),
    ("META-SYN", "Meta Synthetic", "Technology", 290.0, 1.1, 0.008),
    ("NFLX-SYN", "Netflix Synthetic", "Media", 350.0, 1.2, 0.012),
    ("DIS-SYN", "Disney Synthetic", "Media", 90.0, 1.0, 0.009),
    ("BA-SYN", "Boeing Synthetic", "Industrial", 195.0, 1.3, 0.011),
    ("CAT-SYN", "Caterpillar Synthetic", "Industrial", 220.0, 1.1, 0.008),
    ("WMT-SYN", "Walmart Synthetic", "Retail", 60.0, 0.5, 0.004),
    ("COST-SYN", "Costco Synthetic", "Retail", 560.0, 0.6, 0.005),
]


def generate_regime_sequence(n_days: int, initial_regime: int = 0, rng: Optional[np.random.Generator] = None) -> List[int]:
    rng = rng or np.random.default_rng()
    regimes = [initial_regime]
    for _ in range(n_days - 1):
        current = regimes[-1]
        regimes.append(int(rng.choice(3, p=TRANSITION_MATRIX[current])))
    return regimes


def generate_ohlcv(
    n_days: int = 500,
    initial_price: float = 100.0,
    beta: float = 1.0,
    idiosyncratic_vol: float = 0.005,
    seed: Optional[int] = None,
    initial_regime: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    regime_seq = generate_regime_sequence(n_days, initial_regime, rng=rng)
    close_prices = [initial_price]

    for i in range(1, n_days):
        regime = REGIME_NAMES[regime_seq[i]]
        mu = REGIMES[regime]["mu"]
        sigma = REGIMES[regime]["sigma"]
        market_shock = rng.normal(mu, sigma)
        idio_shock = rng.normal(0, idiosyncratic_vol)
        daily_return = float(np.clip(beta * market_shock + idio_shock, -0.15, 0.15))

        new_close = close_prices[-1] * (1 + daily_return)
        close_prices.append(max(new_close, 0.01))

    close = np.array(close_prices)
    intraday_range = np.abs(rng.normal(0, 0.008, n_days))
    opens = np.array([close[0]] + [close[i - 1] * (1 + rng.normal(0, 0.003)) for i in range(1, n_days)])
    highs = np.maximum(opens, close) * (1 + intraday_range)
    lows = np.minimum(opens, close) * (1 - intraday_range)

    base_volume = 1_000_000
    volume = base_volume * np.exp(rng.normal(0, 0.5, n_days))
    price_changes = np.abs(np.r_[0.0, np.diff(close) / np.maximum(close[:-1], 1e-6)])
    volume = volume * (1 + 3 * price_changes)

    end_date = pd.offsets.BDay().rollback(pd.Timestamp(datetime.today().date()))
    dates = pd.bdate_range(end=end_date, periods=n_days)

    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "open": np.round(opens, 2),
            "high": np.round(highs, 2),
            "low": np.round(lows, 2),
            "close": np.round(close, 2),
            "volume": np.round(volume).astype(int),
        }
    )


def generate_synthetic_users(n: int = 20) -> List[Dict[str, Any]]:
    risk_levels = ["conservative", "moderate", "aggressive"]
    capitals = ["small", "medium", "large"]
    horizons = ["short", "medium", "long"]
    sector_pool = sorted({sector for _, _, sector, _, _, _ in SYNTHETIC_STOCKS})
    persona_pool = [
        ("sector_focused", 0.28),
        ("random_explorer", 0.18),
        ("contrarian", 0.18),
        ("momentum_chaser", 0.16),
        ("herd_follower", 0.14),
        ("chaotic_mix", 0.06),
    ]
    persona_names = [name for name, _ in persona_pool]
    persona_weights = [weight for _, weight in persona_pool]

    users: List[Dict[str, Any]] = []
    for i in range(n):
        persona = str(np.random.choice(persona_names, p=persona_weights))
        n_sectors = int(np.random.randint(0, 4))
        sectors = list(np.random.choice(sector_pool, size=n_sectors, replace=False)) if n_sectors else []
        if persona == "sector_focused" and not sectors:
            sectors = [str(np.random.choice(sector_pool))]

        users.append(
            {
                "username": f"user_{i + 1:03d}",
                "risk_tolerance": str(np.random.choice(risk_levels, p=[0.3, 0.5, 0.2])),
                "capital_range": str(np.random.choice(capitals, p=[0.4, 0.4, 0.2])),
                "investment_horizon": str(np.random.choice(horizons, p=[0.25, 0.5, 0.25])),
                "preferred_sectors": sectors,
                "bot_persona": persona,
                "activity_scale": float(np.random.lognormal(mean=0.0, sigma=1.1)),
            }
        )

    return users


def _event_value(event_type: str) -> float:
    if event_type == "view_long":
        return float(np.random.uniform(30, 300))
    if event_type == "view_short":
        return float(np.random.uniform(1, 29))
    if event_type == "rate":
        return float(np.random.uniform(-1.0, 1.0))
    if event_type in {"trade_buy", "trade_sell"}:
        return float(np.random.randint(5, 100))
    if event_type == "watchlist_add":
        return 1.0
    if event_type == "watchlist_remove":
        return -1.0
    return 0.0


def _history_with_timestamp(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for row in history:
        dt = datetime.strptime(str(row["date"]), "%Y-%m-%d")
        enriched.append({**row, "dt": dt, "ts": dt.timestamp()})
    return enriched


def _compute_stock_market_signals(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    closes = np.array([float(row["close"]) for row in history], dtype=np.float64)
    shock_indices = {"crash": [], "pump": [], "trend_up": [], "trend_down": []}

    for idx in range(5, len(history) - 6):
        trailing_5d = float((closes[idx] / max(closes[idx - 5], 1e-6)) - 1.0)
        forward_5d = float((closes[idx + 5] / max(closes[idx], 1e-6)) - 1.0)
        if trailing_5d <= -0.10:
            shock_indices["crash"].append(idx)
        if trailing_5d >= 0.12:
            shock_indices["pump"].append(idx)
        if forward_5d >= 0.05:
            shock_indices["trend_up"].append(idx)
        if forward_5d <= -0.05:
            shock_indices["trend_down"].append(idx)

    return {"rows": history, "shock_indices": shock_indices}


def _sample_activity_count(activity_scale: float, n_stocks: int) -> int:
    raw = 1.0 + np.random.pareto(1.45)
    scaled = raw * max(activity_scale, 0.15)
    max_events = max(20, n_stocks * 20)
    return int(np.clip(round(2 + scaled * 7), 2, max_events))


def _pick_ticker_for_user(
    user: Dict[str, Any],
    stock_meta: Dict[str, Dict[str, Any]],
    available_tickers: List[str],
) -> str:
    preferred_sectors = set(user.get("preferred_sectors", []))
    persona = str(user.get("bot_persona", "chaotic_mix"))
    preferred = [ticker for ticker in available_tickers if stock_meta[ticker]["sector"] in preferred_sectors]

    if persona == "random_explorer":
        weights = np.ones(len(available_tickers), dtype=np.float64)
    elif persona == "sector_focused":
        weights = np.array([3.5 if ticker in preferred else 0.8 for ticker in available_tickers], dtype=np.float64)
    elif persona == "contrarian":
        weights = np.array(
            [1.6 if stock_meta[ticker]["sector"] in preferred_sectors else 1.0 for ticker in available_tickers],
            dtype=np.float64,
        )
    elif persona == "momentum_chaser":
        weights = np.array([2.0 if ticker in preferred else 1.1 for ticker in available_tickers], dtype=np.float64)
    elif persona == "herd_follower":
        weights = np.array([1.8 if ticker in preferred else 0.9 for ticker in available_tickers], dtype=np.float64)
    else:
        weights = np.array([2.2 if ticker in preferred else 1.4 for ticker in available_tickers], dtype=np.float64)

    weights = weights / weights.sum()
    return str(np.random.choice(available_tickers, p=weights))


def _pick_event_index(
    user: Dict[str, Any],
    stock_signal: Dict[str, Any],
    herd_targets: Optional[List[int]],
) -> int:
    rows = stock_signal["rows"]
    persona = str(user.get("bot_persona", "chaotic_mix"))
    shocks = stock_signal["shock_indices"]

    if persona == "contrarian":
        candidates = shocks["crash"] or shocks["pump"]
        if candidates:
            return int(np.random.choice(candidates))
    elif persona == "momentum_chaser":
        candidates = shocks["pump"] or shocks["trend_up"]
        if candidates:
            return int(np.random.choice(candidates))
    elif persona == "herd_follower" and herd_targets and np.random.rand() < 0.65:
        return int(np.random.choice(herd_targets))
    elif persona == "sector_focused" and shocks["trend_up"] and np.random.rand() < 0.4:
        return int(np.random.choice(shocks["trend_up"]))

    valid = np.arange(5, len(rows) - 5)
    weights = np.linspace(0.6, 1.6, len(valid), dtype=np.float64)
    weights = weights / weights.sum()
    return int(np.random.choice(valid, p=weights))


def _pick_event_type(user: Dict[str, Any], forward_ret: float) -> str:
    persona = str(user.get("bot_persona", "chaotic_mix"))

    if persona == "contrarian":
        if forward_ret > 0.03:
            choices, probs = ["trade_buy", "watchlist_add", "view_long"], [0.45, 0.30, 0.25]
        else:
            choices, probs = ["watchlist_remove", "view_short", "trade_sell"], [0.45, 0.35, 0.20]
    elif persona == "momentum_chaser":
        if forward_ret > 0:
            choices, probs = ["trade_buy", "view_long", "watchlist_add", "rate"], [0.38, 0.22, 0.24, 0.16]
        else:
            choices, probs = ["view_short", "watchlist_remove", "trade_sell"], [0.45, 0.35, 0.20]
    elif persona == "random_explorer":
        choices, probs = ["view_short", "view_long", "watchlist_add", "watchlist_remove", "rate"], [0.32, 0.28, 0.16, 0.12, 0.12]
    elif persona == "herd_follower":
        if forward_ret >= 0:
            choices, probs = ["watchlist_add", "trade_buy", "view_long"], [0.38, 0.34, 0.28]
        else:
            choices, probs = ["watchlist_remove", "view_short", "trade_sell"], [0.34, 0.42, 0.24]
    elif persona == "sector_focused":
        if forward_ret >= 0:
            choices, probs = ["view_long", "watchlist_add", "trade_buy", "rate"], [0.34, 0.26, 0.22, 0.18]
        else:
            choices, probs = ["view_short", "watchlist_remove"], [0.65, 0.35]
    else:
        if forward_ret > 0.04:
            choices, probs = ["trade_buy", "watchlist_add", "view_long", "rate"], [0.30, 0.25, 0.25, 0.20]
        elif forward_ret < -0.04:
            choices, probs = ["watchlist_remove", "view_short", "trade_sell"], [0.35, 0.40, 0.25]
        else:
            choices, probs = ["view_short", "view_long", "watchlist_add", "watchlist_remove"], [0.28, 0.32, 0.20, 0.20]

    return str(np.random.choice(choices, p=probs))


def _event_reward_for_type(event_type: str, forward_ret: float) -> float:
    positive_types = {"trade_buy", "watchlist_add", "view_long", "rate"}
    direction = 1.0 if event_type in positive_types else -1.0
    signal = float(np.clip(direction * forward_ret, -0.5, 0.5))
    bias = 0.03 if direction > 0 else -0.03
    return float(np.clip(signal + bias, -0.5, 0.5))


def _emit_event(
    db: DatabaseManager,
    user_id: int,
    stock_id: int,
    event_type: str,
    price: float,
    timestamp: float,
    reward: float,
) -> int:
    event_id = db.log_event(
        user_id,
        stock_id,
        event_type,
        _event_value(event_type),
        price,
        timestamp=timestamp,
    )
    db.update_event_reward(event_id, reward)
    return event_id


def seed_database(db: DatabaseManager, n_users: int = 20, n_days: int = 500) -> Dict[str, Any]:
    """Populate the database with noisy synthetic market and user interaction data."""
    print(f"[SyntheticData] Seeding database: {len(SYNTHETIC_STOCKS)} stocks, {n_users} users, {n_days} days each")

    stock_id_map: Dict[str, int] = {}
    stock_meta: Dict[str, Dict[str, Any]] = {}
    stock_signals: Dict[str, Dict[str, Any]] = {}

    for i, (ticker, name, sector, init_price, beta, idio_vol) in enumerate(SYNTHETIC_STOCKS):
        stock_id = db.upsert_stock(ticker, name, sector)
        stock_id_map[ticker] = stock_id
        stock_meta[ticker] = {"stock_id": stock_id, "sector": sector}

        history = db.get_price_history(stock_id, limit=n_days + 10)
        if not history:
            df = generate_ohlcv(
                n_days=n_days,
                initial_price=init_price,
                beta=beta,
                idiosyncratic_vol=idio_vol,
                seed=i * 42,
                initial_regime=i % 3,
            )
            history = df.to_dict("records")
            db.insert_price_batch(stock_id, history)
            print(f"  [+] {ticker}: inserted {n_days} days")

        stock_signals[ticker] = _compute_stock_market_signals(_history_with_timestamp(history))

    synthetic_users = generate_synthetic_users(n_users)
    user_id_map: Dict[str, int] = {}
    for user in synthetic_users:
        user_id = db.create_user(
            username=user["username"],
            risk_tolerance=user["risk_tolerance"],
            capital_range=user["capital_range"],
            investment_horizon=user["investment_horizon"],
            preferred_sectors=user["preferred_sectors"],
        )
        user_id_map[user["username"]] = user_id

    herd_windows: Dict[str, List[int]] = {}
    for ticker, signal in stock_signals.items():
        source = signal["shock_indices"]["pump"] or signal["shock_indices"]["trend_up"]
        if source:
            center = int(np.random.choice(source))
            herd_windows[ticker] = list(range(max(5, center - 2), min(len(signal["rows"]) - 6, center + 3)))
        else:
            herd_windows[ticker] = []

    users_by_persona: Dict[str, List[Dict[str, Any]]] = {}
    for user in synthetic_users:
        users_by_persona.setdefault(str(user["bot_persona"]), []).append(user)

    herd_campaigns: Dict[str, Dict[str, Any]] = {}
    candidate_herd_tickers = [ticker for ticker, windows in herd_windows.items() if windows]
    n_campaigns = min(3, len(candidate_herd_tickers))
    if n_campaigns:
        chosen_herd_tickers = list(np.random.choice(candidate_herd_tickers, size=n_campaigns, replace=False))
        crowd_users = (
            users_by_persona.get("herd_follower", [])
            + users_by_persona.get("momentum_chaser", [])
            + users_by_persona.get("chaotic_mix", [])
        )
        contrarian_users = users_by_persona.get("contrarian", [])
        for ticker in chosen_herd_tickers:
            herd_members: List[str] = []
            contrarian_members: List[str] = []
            if crowd_users:
                crowd_size = min(len(crowd_users), max(3, int(np.random.randint(3, 7))))
                herd_members = [user["username"] for user in np.random.choice(crowd_users, size=crowd_size, replace=False)]
            if contrarian_users:
                contra_size = min(len(contrarian_users), max(2, int(np.random.randint(2, 5))))
                contrarian_members = [user["username"] for user in np.random.choice(contrarian_users, size=contra_size, replace=False)]
            herd_campaigns[ticker] = {
                "indices": herd_windows[ticker],
                "herd_members": set(herd_members),
                "contrarian_members": set(contrarian_members),
            }

    event_count = 0
    all_tickers = list(stock_id_map.keys())

    for username, user_id in user_id_map.items():
        user = next(item for item in synthetic_users if item["username"] == username)
        persona = str(user.get("bot_persona", "chaotic_mix"))
        target_events = _sample_activity_count(float(user.get("activity_scale", 1.0)), len(all_tickers))

        herd_ticker = ""
        if persona == "herd_follower":
            herd_ticker = str(np.random.choice(all_tickers))
        elif persona == "momentum_chaser" and np.random.rand() < 0.4:
            herd_ticker = str(np.random.choice(all_tickers))

        visited_tickers = set()
        emitted_events = 0

        for ticker, campaign in herd_campaigns.items():
            if username not in campaign["herd_members"] and username not in campaign["contrarian_members"]:
                continue
            base_idx = int(np.random.choice(campaign["indices"]))
            row = stock_signals[ticker]["rows"][base_idx]
            price = float(row["close"])
            base_ts = float(row["ts"] + np.random.uniform(9 * 3600, 15 * 3600))

            if username in campaign["herd_members"]:
                event_type = str(np.random.choice(["watchlist_add", "trade_buy", "view_long"], p=[0.35, 0.4, 0.25]))
                reward = max(_event_reward_for_type(event_type, 0.08), 0.08)
                _emit_event(db, user_id, stock_id_map[ticker], event_type, price, base_ts, reward)
                emitted_events += 1
                event_count += 1
                visited_tickers.add(ticker)

            if username in campaign["contrarian_members"]:
                event_type = str(np.random.choice(["watchlist_remove", "trade_sell", "view_short"], p=[0.35, 0.4, 0.25]))
                reward = min(_event_reward_for_type(event_type, 0.08), -0.08)
                _emit_event(db, user_id, stock_id_map[ticker], event_type, price, base_ts + np.random.uniform(300, 5400), reward)
                emitted_events += 1
                event_count += 1
                visited_tickers.add(ticker)

        while emitted_events < target_events:
            ticker = herd_ticker if herd_ticker and np.random.rand() < 0.45 else _pick_ticker_for_user(user, stock_meta, all_tickers)
            signal = stock_signals[ticker]
            idx = _pick_event_index(user, signal, herd_windows.get(ticker))
            idx = int(np.clip(idx, 5, len(signal["rows"]) - 6))

            row = signal["rows"][idx]
            future_row = signal["rows"][idx + 5]
            price = float(row["close"])
            future_price = float(future_row["close"])
            forward_ret = float(np.clip((future_price - price) / max(price, 1e-6), -0.5, 0.5))
            event_type = _pick_event_type(user, forward_ret)

            if user.get("preferred_sectors") and stock_meta[ticker]["sector"] not in user["preferred_sectors"] and np.random.rand() < 0.35:
                event_type = str(np.random.choice(["view_long", "watchlist_add", "view_short", "watchlist_remove"]))
            if stock_meta[ticker]["sector"] in user.get("preferred_sectors", []) and np.random.rand() < 0.20:
                event_type = str(np.random.choice(["watchlist_remove", "view_short", "trade_sell", event_type]))

            timestamp = float(row["ts"] + np.random.uniform(8 * 3600, 17 * 3600))
            reward = _event_reward_for_type(event_type, forward_ret)
            _emit_event(db, user_id, stock_id_map[ticker], event_type, price, timestamp, reward)

            visited_tickers.add(ticker)
            emitted_events += 1
            event_count += 1

        preferred_sectors = set(user.get("preferred_sectors", []))
        off_sector_tickers = [ticker for ticker in all_tickers if stock_meta[ticker]["sector"] not in preferred_sectors]
        if off_sector_tickers and (persona == "random_explorer" or preferred_sectors):
            exploratory_events = 2 if persona == "random_explorer" else 1
            for ticker in list(np.random.choice(off_sector_tickers, size=min(exploratory_events, len(off_sector_tickers)), replace=False)):
                rows = stock_signals[ticker]["rows"]
                idx = int(np.random.randint(10, len(rows) - 10))
                row = rows[idx]
                future_row = rows[idx + 5]
                forward_ret = float(np.clip((float(future_row["close"]) - float(row["close"])) / max(float(row["close"]), 1e-6), -0.5, 0.5))
                event_type = str(np.random.choice(["watchlist_add", "trade_buy", "view_long"], p=[0.35, 0.3, 0.35]))
                reward = max(_event_reward_for_type(event_type, forward_ret), 0.04 if persona == "random_explorer" else -0.02)
                _emit_event(
                    db,
                    user_id,
                    stock_id_map[ticker],
                    event_type,
                    float(row["close"]),
                    float(row["ts"] + np.random.uniform(10 * 3600, 16 * 3600)),
                    reward,
                )
                visited_tickers.add(ticker)
                event_count += 1

        contradiction_candidates = list(visited_tickers) or all_tickers
        contradiction_count = 2 if persona in {"chaotic_mix", "random_explorer"} else 1
        for ticker in list(np.random.choice(contradiction_candidates, size=min(contradiction_count, len(contradiction_candidates)), replace=False)):
            rows = stock_signals[ticker]["rows"]
            idx = int(np.random.randint(15, len(rows) - 12))
            row = rows[idx]
            later_row = rows[idx + 7]
            first_type = "watchlist_add"
            second_type = "watchlist_remove"
            if np.random.rand() < 0.35:
                first_type, second_type = second_type, first_type
            _emit_event(
                db,
                user_id,
                stock_id_map[ticker],
                first_type,
                float(row["close"]),
                float(row["ts"] + np.random.uniform(9 * 3600, 11 * 3600)),
                0.06 if first_type == "watchlist_add" else -0.06,
            )
            _emit_event(
                db,
                user_id,
                stock_id_map[ticker],
                second_type,
                float(later_row["close"]),
                float(later_row["ts"] + np.random.uniform(13 * 3600, 16 * 3600)),
                -0.06 if second_type == "watchlist_remove" else 0.06,
            )
            event_count += 2

        events = db.get_user_events(user_id, limit=max(target_events + 4, 32))
        has_positive = any(event["reward"] is not None and event["reward"] > 0.02 for event in events)
        has_negative = any(event["reward"] is not None and event["reward"] < -0.02 for event in events)
        if not visited_tickers:
            visited_tickers = set(all_tickers)

        for direction in ("positive", "negative"):
            if direction == "positive" and has_positive:
                continue
            if direction == "negative" and has_negative:
                continue

            best_ticker = None
            best_idx = None
            best_score = -math.inf
            for ticker in visited_tickers:
                rows = stock_signals[ticker]["rows"]
                for idx in range(5, len(rows) - 6, 15):
                    price = float(rows[idx]["close"])
                    future_price = float(rows[idx + 5]["close"])
                    forward_ret = float(np.clip((future_price - price) / max(price, 1e-6), -0.5, 0.5))
                    score = forward_ret if direction == "positive" else -forward_ret
                    if score > best_score:
                        best_score = score
                        best_ticker = ticker
                        best_idx = idx

            if best_ticker is None or best_idx is None:
                continue

            row = stock_signals[best_ticker]["rows"][best_idx]
            future_row = stock_signals[best_ticker]["rows"][best_idx + 5]
            forward_ret = float(
                np.clip(
                    (float(future_row["close"]) - float(row["close"])) / max(float(row["close"]), 1e-6),
                    -0.5,
                    0.5,
                )
            )
            event_type = "trade_buy" if direction == "positive" else "watchlist_remove"
            event_id = db.log_event(
                user_id,
                stock_id_map[best_ticker],
                event_type,
                _event_value(event_type),
                float(row["close"]),
                timestamp=float(row["ts"] + 12 * 3600),
            )
            forced_reward = _event_reward_for_type(event_type, forward_ret)
            if direction == "positive":
                forced_reward = max(forced_reward, 0.08)
            else:
                forced_reward = min(forced_reward, -0.08)
            db.update_event_reward(event_id, forced_reward)
            event_count += 1

    print(f"[SyntheticData] Complete: {len(stock_id_map)} stocks, {len(user_id_map)} users, {event_count} events")
    return {
        "n_stocks": len(stock_id_map),
        "n_users": len(user_id_map),
        "n_events": event_count,
        "stock_ids": stock_id_map,
        "user_ids": user_id_map,
    }
