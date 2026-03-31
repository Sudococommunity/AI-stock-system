"""
Synthetic data generator — creates realistic stock price data so the
entire pipeline can be exercised without a live market data feed.

Uses Geometric Brownian Motion (GBM) with regime switching:
  • Bull regime  : positive drift, low volatility
  • Bear regime  : negative drift, high volatility
  • Sideways     : near-zero drift, medium volatility

Regime transitions are modeled as a Hidden Markov Model (simple 3-state).
"""
import time
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from datetime import datetime, timedelta

from stock_recommender.data.database import DatabaseManager


# ── Market regimes ────────────────────────────────────────────────────────────

REGIMES = {
    "bull":    {"mu": 0.0008,  "sigma": 0.012},   # ~20% annual return, 19% vol
    "bear":    {"mu": -0.0006, "sigma": 0.022},   # ~-15% annual, 35% vol
    "sideways":{"mu": 0.0001,  "sigma": 0.015},   # ~2.5% annual, 24% vol
}

# Transition matrix: P(next_regime | current_regime)
# Rows: current, Cols: [bull, bear, sideways]
TRANSITION_MATRIX = np.array([
    [0.97, 0.01, 0.02],   # from bull
    [0.02, 0.95, 0.03],   # from bear
    [0.04, 0.03, 0.93],   # from sideways
])

REGIME_NAMES = ["bull", "bear", "sideways"]


def generate_regime_sequence(n_days: int, initial_regime: int = 0) -> List[int]:
    """Sample regime indices using the Markov transition matrix."""
    regimes = [initial_regime]
    for _ in range(n_days - 1):
        current = regimes[-1]
        next_r = np.random.choice(3, p=TRANSITION_MATRIX[current])
        regimes.append(int(next_r))
    return regimes


def generate_ohlcv(
    n_days: int = 500,
    initial_price: float = 100.0,
    beta: float = 1.0,                   # sensitivity to market (1.0 = market-like)
    idiosyncratic_vol: float = 0.005,    # additional company-specific noise
    seed: Optional[int] = None,
    initial_regime: int = 0,
) -> pd.DataFrame:
    """
    Generate a realistic OHLCV DataFrame using regime-switching GBM.

    The model:
      log(S_{t+1}/S_t) = regime_mu + beta*market_shock + idiosyncratic_shock
    """
    if seed is not None:
        np.random.seed(seed)

    regime_seq = generate_regime_sequence(n_days, initial_regime)

    close_prices = [initial_price]
    market_returns = []  # common factor (all stocks share this)

    for i in range(1, n_days):
        r = REGIME_NAMES[regime_seq[i]]
        mu = REGIMES[r]["mu"]
        sigma = REGIMES[r]["sigma"]

        # Market (systematic) component
        market_shock = np.random.normal(mu, sigma)
        # Idiosyncratic (company-specific) component
        idio_shock = np.random.normal(0, idiosyncratic_vol)

        daily_return = beta * market_shock + idio_shock
        # Clip to realistic range
        daily_return = np.clip(daily_return, -0.15, 0.15)

        new_close = close_prices[-1] * (1 + daily_return)
        close_prices.append(max(new_close, 0.01))
        market_returns.append(market_shock)

    close = np.array(close_prices)

    # Generate OHLV from close
    # High = close * (1 + |noise|), Low = close * (1 - |noise|)
    # Volume: log-normal with some mean-reversion
    intraday_range = np.abs(np.random.normal(0, 0.008, n_days))
    opens = np.array([close[0]] + [close[i-1] * (1 + np.random.normal(0, 0.003)) for i in range(1, n_days)])
    highs = np.maximum(opens, close) * (1 + intraday_range)
    lows = np.minimum(opens, close) * (1 - intraday_range)

    # Volume: 1M base shares, 3x variation
    base_volume = 1_000_000
    volume = base_volume * np.exp(np.random.normal(0, 0.5, n_days))
    # High volume on big moves (volume-price correlation)
    price_changes = np.abs(np.diff(close, prepend=close[0]) / close)
    volume = volume * (1 + 3 * price_changes)

    # Build date index (business days)
    end_date = datetime.today()
    dates = pd.bdate_range(end=end_date, periods=n_days)

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open": np.round(opens, 2),
        "high": np.round(highs, 2),
        "low": np.round(lows, 2),
        "close": np.round(close, 2),
        "volume": np.round(volume).astype(int),
    })

    return df


# ── Sector definitions ────────────────────────────────────────────────────────

SYNTHETIC_STOCKS = [
    # (ticker, name, sector, initial_price, beta, idio_vol)
    ("AAPL-SYN",  "Apple Synthetic",        "Technology",    150.0, 1.1,  0.006),
    ("MSFT-SYN",  "Microsoft Synthetic",    "Technology",    280.0, 0.9,  0.005),
    ("GOOGL-SYN", "Google Synthetic",       "Technology",    120.0, 1.0,  0.007),
    ("AMZN-SYN",  "Amazon Synthetic",       "Consumer",      130.0, 1.2,  0.008),
    ("JPM-SYN",   "JPMorgan Synthetic",     "Finance",        145.0, 1.3,  0.009),
    ("GS-SYN",    "Goldman Sachs Synthetic","Finance",        340.0, 1.4,  0.010),
    ("JNJ-SYN",   "J&J Synthetic",          "Healthcare",     160.0, 0.6,  0.004),
    ("PFE-SYN",   "Pfizer Synthetic",       "Healthcare",      35.0, 0.7,  0.005),
    ("XOM-SYN",   "Exxon Synthetic",        "Energy",          80.0, 1.0,  0.010),
    ("CVX-SYN",   "Chevron Synthetic",      "Energy",         150.0, 0.9,  0.009),
    ("TSLA-SYN",  "Tesla Synthetic",        "Automotive",     220.0, 1.8,  0.018),
    ("NVDA-SYN",  "Nvidia Synthetic",       "Technology",     450.0, 1.6,  0.015),
    ("AMD-SYN",   "AMD Synthetic",          "Technology",     110.0, 1.5,  0.014),
    ("META-SYN",  "Meta Synthetic",         "Technology",     290.0, 1.1,  0.008),
    ("NFLX-SYN",  "Netflix Synthetic",      "Media",          350.0, 1.2,  0.012),
    ("DIS-SYN",   "Disney Synthetic",       "Media",           90.0, 1.0,  0.009),
    ("BA-SYN",    "Boeing Synthetic",       "Industrial",     195.0, 1.3,  0.011),
    ("CAT-SYN",   "Caterpillar Synthetic",  "Industrial",     220.0, 1.1,  0.008),
    ("WMT-SYN",   "Walmart Synthetic",      "Retail",          60.0, 0.5,  0.004),
    ("COST-SYN",  "Costco Synthetic",       "Retail",         560.0, 0.6,  0.005),
]


def generate_synthetic_users(n: int = 20) -> List[Dict]:
    """
    Generate bot users with varied archetypes:
      • focused   (60%) — 1-2 preferred sectors, mostly stays in them
      • explorer  (25%) — no strong sector preference, wanders freely
      • contrarian(15%) — actively seeks out beaten-down / out-of-favour stocks
    Activity level is sampled from a Pareto distribution so most bots are
    low-activity and a few are hyperactive, matching real usage patterns.
    """
    risk_levels = ["conservative", "moderate", "aggressive"]
    capitals = ["small", "medium", "large"]
    horizons = ["short", "medium", "long"]
    sector_pool = ["Technology", "Finance", "Healthcare", "Energy", "Consumer", "Retail",
                   "Automotive", "Industrial", "Media", "Retail"]

    archetype_weights = [0.60, 0.25, 0.15]   # focused / explorer / contrarian

    users = []
    for i in range(n):
        archetype = np.random.choice(["focused", "explorer", "contrarian"], p=archetype_weights)
        if archetype == "focused":
            n_sectors = np.random.randint(1, 3)
            sectors = list(np.random.choice(sector_pool, size=n_sectors, replace=False))
        elif archetype == "explorer":
            sectors = []   # no preference — cross-sector noise drives everything
        else:  # contrarian
            n_sectors = np.random.randint(1, 3)
            sectors = list(np.random.choice(sector_pool, size=n_sectors, replace=False))

        # Pareto activity multiplier: shape=1.5 → median≈1, mean≈3, heavy tail to ~20
        activity_mult = float(np.clip((np.random.pareto(1.5) + 1.0), 1.0, 20.0))

        users.append({
            "username": f"user_{i+1:03d}",
            "risk_tolerance": np.random.choice(risk_levels, p=[0.3, 0.5, 0.2]),
            "capital_range": np.random.choice(capitals, p=[0.4, 0.4, 0.2]),
            "investment_horizon": np.random.choice(horizons, p=[0.25, 0.5, 0.25]),
            "preferred_sectors": sectors,
            "archetype": archetype,
            "activity_mult": activity_mult,
        })
    return users


# ── Interaction helpers ───────────────────────────────────────────────────────

# Positive events — signals that a user likes / follows a stock
_POS_TYPES   = ["view_long", "watchlist_add", "rate", "trade_buy", "view_short"]
_POS_WEIGHTS = [0.35, 0.25, 0.20, 0.10, 0.10]

# Negative events — explicit disinterest / rejection signals
# watchlist_remove has base weight -0.4 in EVENT_WEIGHTS so it drives negative scores
_NEG_TYPES   = ["watchlist_remove", "watchlist_remove", "view_short"]
_NEG_WEIGHTS = [0.50, 0.30, 0.20]   # extra remove weight ensures score < -0.05


def _event_value(etype: str) -> float:
    """Generate a realistic value for the given event type."""
    if etype == "view_long":
        return float(np.random.uniform(30, 300))
    if etype == "view_short":
        return float(np.random.uniform(1, 29))
    if etype == "rate":
        return float(np.random.uniform(0.5, 1.0))    # positive rating for liked stocks
    if etype == "trade_buy":
        return float(np.random.randint(5, 100))
    if etype == "watchlist_add":
        return 1.0
    if etype == "watchlist_remove":
        return -1.0
    return 0.0


def _forward_return(history: List[Dict]) -> float:
    """5-day forward return clipped to ±50%."""
    if len(history) < 6:
        return 0.0
    base = float(history[-6]["close"])
    return float(np.clip((float(history[-1]["close"]) - base) / base, -0.5, 0.5)) if base > 0 else 0.0


def _random_past_timestamp(n_days: int) -> float:
    """Return a Unix timestamp for a random moment within the past n_days."""
    offset_seconds = np.random.uniform(0, n_days * 86400)
    return time.time() - offset_seconds


def seed_database(db: DatabaseManager, n_users: int = 20, n_days: int = 500) -> Dict:
    """
    Populate the database with synthetic stocks, price histories, and users.
    Safe to call multiple times — uses INSERT OR IGNORE.

    Three fixes vs the original:
      1. Timestamps spread across the full n_days history (recency decay works).
      2. Cross-sector noise: focused bots occasionally wander; explorers are fully random;
         contrarians buy non-preferred and ignore preferred.
      3. Activity is Pareto-distributed per user (activity_mult from generate_synthetic_users).
    """
    print(f"[SyntheticData] Seeding database: {len(SYNTHETIC_STOCKS)} stocks, {n_users} users, {n_days} days each")

    # ── Stocks ────────────────────────────────────────────────────────────────
    stock_id_map: Dict[str, int] = {}
    # Build a per-ticker price snapshot keyed by ticker for fast reward lookup
    price_history_cache: Dict[str, List[Dict]] = {}

    for i, (ticker, name, sector, init_price, beta, idio_vol) in enumerate(SYNTHETIC_STOCKS):
        sid = db.upsert_stock(ticker, name, sector)
        stock_id_map[ticker] = sid

        if len(db.get_price_history(sid, limit=5)) > 0:
            price_history_cache[ticker] = db.get_price_history(sid, limit=10)
            continue  # already seeded

        df = generate_ohlcv(
            n_days=n_days,
            initial_price=init_price,
            beta=beta,
            idiosyncratic_vol=idio_vol,
            seed=i * 42,
            initial_regime=i % 3,
        )
        db.insert_price_batch(sid, df.to_dict("records"))
        price_history_cache[ticker] = db.get_price_history(sid, limit=10)
        print(f"  [+] {ticker}: inserted {n_days} days")

    all_tickers = [t for t, *_ in SYNTHETIC_STOCKS]

    # ── Users ─────────────────────────────────────────────────────────────────
    user_id_map: Dict[str, int] = {}
    synthetic_users = generate_synthetic_users(n_users)
    for u in synthetic_users:
        # create_user must not receive the extra archetype/activity_mult keys
        uid = db.create_user(
            username=u["username"],
            risk_tolerance=u["risk_tolerance"],
            capital_range=u["capital_range"],
            investment_horizon=u["investment_horizon"],
            preferred_sectors=u["preferred_sectors"],
        )
        user_id_map[u["username"]] = uid

    # ── Synthetic interactions ────────────────────────────────────────────────
    event_count = 0

    for username, uid in user_id_map.items():
        user = next(u for u in synthetic_users if u["username"] == username)
        archetype     = user["archetype"]
        activity_mult = user["activity_mult"]
        preferred     = set(user["preferred_sectors"])

        preferred_tickers = [t for t, _, sector, *_ in SYNTHETIC_STOCKS if sector in preferred]
        other_tickers     = [t for t in all_tickers if t not in preferred_tickers]

        # ── Build positive pool based on archetype ────────────────────────────
        if archetype == "explorer":
            # Fully random — no sector bias at all
            pos_pool = all_tickers[:]
            neg_pool = []   # explorers don't explicitly dislike anything
        elif archetype == "contrarian":
            # Contrarians buy what focused users avoid and vice-versa
            pos_pool = other_tickers if other_tickers else all_tickers[:]
            neg_pool = preferred_tickers if preferred_tickers else other_tickers
        else:  # focused
            # 80% preferred, 20% random cross-sector noise
            noise_count = max(1, int(len(other_tickers) * 0.20))
            noise = list(np.random.choice(other_tickers, size=noise_count, replace=False)) \
                    if other_tickers else []
            pos_pool = preferred_tickers + noise
            neg_pool = [t for t in other_tickers if t not in noise]

        if not pos_pool:
            pos_pool = all_tickers[:]

        # ── Positive interactions ─────────────────────────────────────────────
        base_n_pos = int(np.round(np.random.randint(3, 8) * activity_mult))
        n_pos = min(base_n_pos, len(pos_pool))
        pos_chosen = list(np.random.choice(pos_pool, size=n_pos, replace=False))

        for ticker in pos_chosen:
            sid     = stock_id_map[ticker]
            history = price_history_cache.get(ticker, [])
            price   = float(history[-1]["close"]) if history else 0.0
            fwd_ret = abs(_forward_return(history)) + np.random.uniform(0.01, 0.08)
            reward  = float(np.clip(fwd_ret, 0.02, 0.5))

            # Pareto event count per stock: 1–12, activity-scaled
            n_events = int(np.clip(np.round(np.random.pareto(2.0) + 1) * activity_mult, 1, 12))
            for _ in range(n_events):
                etype = np.random.choice(_POS_TYPES, p=_POS_WEIGHTS)
                ts    = _random_past_timestamp(n_days)
                event_id = db.log_event(uid, sid, etype, _event_value(etype), price, timestamp=ts)
                db.update_event_reward(event_id, reward)
                event_count += 1

            # Contrarian / inconsistent bots: 15% chance of adding then removing
            # the same stock (changed mind), producing a weak net-negative signal
            if archetype != "explorer" and np.random.random() < 0.15:
                ts_add = _random_past_timestamp(n_days // 2)
                ts_rem = ts_add - np.random.uniform(1, 7) * 86400   # removed before add in history
                # add first (older), remove later (newer) — timestamps are in the past
                add_id = db.log_event(uid, sid, "watchlist_add", 1.0, price, timestamp=ts_rem)
                rem_id = db.log_event(uid, sid, "watchlist_remove", -1.0, price, timestamp=ts_add)
                db.update_event_reward(add_id, reward * 0.3)
                db.update_event_reward(rem_id, -reward * 0.3)
                event_count += 2

        # ── Negative interactions ─────────────────────────────────────────────
        if not neg_pool:
            # explorers: randomly pick a few stocks to dislike
            neg_pool = list(np.random.choice(
                [t for t in all_tickers if t not in pos_chosen],
                size=min(3, len(all_tickers) - len(pos_chosen)),
                replace=False,
            )) if len(all_tickers) > len(pos_chosen) else []

        if neg_pool:
            base_n_neg = int(np.round(np.random.randint(3, 6) * min(activity_mult, 3.0)))
            n_neg = min(base_n_neg, len(neg_pool))
            neg_chosen = list(np.random.choice(neg_pool, size=n_neg, replace=False))

            for ticker in neg_chosen:
                sid     = stock_id_map[ticker]
                history = price_history_cache.get(ticker, [])
                price   = float(history[-1]["close"]) if history else 0.0
                fwd_ret = abs(_forward_return(history)) + np.random.uniform(0.01, 0.08)
                reward  = float(np.clip(-fwd_ret, -0.5, -0.02))

                n_events = int(np.clip(np.round(np.random.pareto(2.0) + 1), 1, 5))
                for _ in range(n_events):
                    etype = np.random.choice(_NEG_TYPES, p=_NEG_WEIGHTS)
                    ts    = _random_past_timestamp(n_days)
                    event_id = db.log_event(uid, sid, etype, _event_value(etype), price, timestamp=ts)
                    db.update_event_reward(event_id, reward)
                    event_count += 1

    print(f"[SyntheticData] Complete: {len(stock_id_map)} stocks, {len(user_id_map)} users, {event_count} events")
    return {
        "n_stocks": len(stock_id_map),
        "n_users": len(user_id_map),
        "n_events": event_count,
        "stock_ids": stock_id_map,
        "user_ids": user_id_map,
    }
