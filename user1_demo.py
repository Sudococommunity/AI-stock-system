"""
user1_demo.py — "Alex" (Aggressive Growth & Tech Investor)
===========================================================
Standalone demo showing how the recommendation algorithm learns from Alex's
behavior. Runs independently — just: python user1_demo.py

What this script does:
  1. Downloads real OHLCV market data via data_downloader.py (cached to disk)
  2. Trains the StockTransformer on that real historical data
  3. Shows Alex's INITIAL recommendations (before any interactions)
  4. Scrapes & displays recent news + sentiment for recommended stocks
  5. Replays Alex's interactions from user1/interactions.json
  6. Shows UPDATED recommendations (after the online learner adapts)
  7. Prints a side-by-side comparison proving the algorithm learned

Alex's profile: aggressive, short-term, loves tech/growth (NVDA, TSLA, AMD, META)
Expected learning: after interactions, tech stocks should rank much higher.
"""
import sys
import os
import json
import warnings
warnings.filterwarnings("ignore")

# Allow imports from parent project directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

# ── Tweak config for demo (more aggressive online learning) ──────────────────
from stock_recommender.config import CONFIG
CONFIG.data.db_path = "demo_shared.db"
CONFIG.data.checkpoint_dir = "demo_checkpoints/"
CONFIG.training.online_lr = 2e-3
CONFIG.training.replay_min_fill = 16
CONFIG.training.online_update_every = 4
CONFIG.model.final_k = 8

# ── Stock universe for this demo ──────────────────────────────────────────────
DEMO_TICKERS = [
    # Tech / Growth (Alex's favorites)
    "NVDA", "TSLA", "AMD",  "META", "AAPL", "GOOGL", "AMZN", "MSFT", "NFLX",
    # Defensive / Value (Alex dislikes these)
    "JNJ",  "KO",   "WMT",  "PG",   "JPM",  "VZ",    "XOM",  "PFE",
    # Extra for diversity
    "V",    "HD",
]

SECTOR_MAP = {
    "NVDA": "Technology", "TSLA": "Automotive",  "AMD":  "Technology",
    "META": "Technology", "AAPL": "Technology",  "GOOGL":"Technology",
    "AMZN": "Consumer",   "MSFT": "Technology",  "NFLX": "Media",
    "JNJ":  "Healthcare", "KO":   "Consumer",    "WMT":  "Retail",
    "PG":   "Consumer",   "JPM":  "Finance",     "VZ":   "Telecom",
    "XOM":  "Energy",     "PFE":  "Healthcare",  "V":    "Finance",
    "HD":   "Retail",
}

PROFILE_PATH      = os.path.join(os.path.dirname(__file__), "user1", "profile.json")
INTERACTIONS_PATH = os.path.join(os.path.dirname(__file__), "user1", "interactions.json")
DATA_CACHE_DIR    = os.path.join(os.path.dirname(__file__), "data_cache")


# ── Utilities ─────────────────────────────────────────────────────────────────

def print_header(title: str):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def print_section(title: str):
    print(f"\n--- {title} ---")


# ── Step 1: Download market data (separated from training) ────────────────────

def step_download_data(force_refresh: bool = False) -> dict:
    """
    Download real OHLCV data via data_downloader.py.
    Data is cached to DATA_CACHE_DIR — subsequent runs load from disk.
    Completely decoupled from the ML pipeline.
    """
    from data_downloader import MarketDataDownloader
    print_section("STEP 1 — Downloading historical market data (yfinance)")
    print(f"  Cache directory : {DATA_CACHE_DIR}")
    print(f"  Tickers         : {len(DEMO_TICKERS)} stocks")
    print(f"  Period          : 2 years of daily OHLCV")
    dl = MarketDataDownloader(cache_dir=DATA_CACHE_DIR, verbose=True)
    market_data = dl.get_ohlcv(DEMO_TICKERS, period="2y",
                                min_rows=260, force_refresh=force_refresh)
    print(f"\n  {len(market_data)}/{len(DEMO_TICKERS)} tickers ready for training.")
    return market_data


# ── Step 2: Initialize ML pipeline ───────────────────────────────────────────

def step_init_pipeline():
    """Initialize all ML components."""
    from stock_recommender.data.database import DatabaseManager
    from stock_recommender.features.feature_pipeline import FeaturePipeline
    from stock_recommender.models.time_series import StockTransformer
    from stock_recommender.models.two_tower import TwoTowerModel, RankingModel, CandidateIndex
    from stock_recommender.learning.online_learner import OnlineLearner
    from stock_recommender.recommendation.engine import RecommendationEngine

    os.makedirs(CONFIG.data.checkpoint_dir, exist_ok=True)

    db               = DatabaseManager(CONFIG.data.db_path)
    pipeline         = FeaturePipeline()
    transformer      = StockTransformer()
    two_tower        = TwoTowerModel()
    ranker           = RankingModel()
    candidate_index  = CandidateIndex()

    online_learner = OnlineLearner(
        two_tower=two_tower, transformer=transformer,
        candidate_index=candidate_index, feature_pipeline=pipeline, db=db,
    )
    engine = RecommendationEngine(
        two_tower=two_tower, ranker=ranker, transformer=transformer,
        candidate_index=candidate_index, feature_pipeline=pipeline, db=db,
    )
    return db, pipeline, transformer, two_tower, ranker, candidate_index, online_learner, engine


# ── Step 3: Store real OHLCV data into database ───────────────────────────────

def step_seed_database(db, market_data: dict) -> dict:
    """Store downloaded OHLCV into SQLite. Returns {ticker: stock_id}."""
    print_section("STEP 3 — Storing real OHLCV data in database")
    stock_id_map = {}
    for ticker, df in market_data.items():
        sid = db.upsert_stock(ticker, ticker, SECTOR_MAP.get(ticker, "Unknown"))
        db.insert_price_batch(sid, df.to_dict("records"))
        stock_id_map[ticker] = sid
    print(f"  {len(stock_id_map)} stocks stored (real yfinance data, no synthetics)")
    return stock_id_map


# ── Step 4: Train transformer on real historical data ─────────────────────────

def step_train_transformer(db, pipeline, transformer, two_tower, ranker,
                            stock_id_map: dict):
    """
    Train (or load) the StockTransformer.
    Training uses ONLY real yfinance OHLCV data that was stored in step 3.
    No synthetic data is used anywhere in this script.
    """
    from stock_recommender.learning.trainer import Trainer
    import torch

    checkpoint = os.path.join(CONFIG.data.checkpoint_dir, "transformer_user1.pt")

    if os.path.exists(checkpoint):
        print_section("STEP 4 — Loading transformer checkpoint (skip retraining)")
        transformer.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        print(f"  Loaded: {checkpoint}")
        print(f"  (Delete this file to force retrain on latest yfinance data)")
        return

    print_section("STEP 4 — Training StockTransformer on real yfinance data")
    print("  Using real OHLCV history — no synthetic data.")
    trainer = Trainer(transformer, two_tower, ranker, pipeline, db)
    CONFIG.training.max_epochs = 2
    CONFIG.training.early_stop_patience = 1

    stock_ids = list(stock_id_map.values())
    history = trainer.train_transformer(stock_ids, n_epochs=2)

    if history["train_loss"]:
        losses = history["train_loss"]
        print(f"  Epoch 1 loss : {losses[0]:.4f}")
        if len(losses) > 1:
            print(f"  Epoch 2 loss : {losses[-1]:.4f}  "
                  f"({'improved' if losses[-1] < losses[0] else 'stable'})")

    torch.save(transformer.state_dict(), checkpoint)
    print(f"  Checkpoint saved: {checkpoint}")


# ── Step 5: Build feature pipeline + ANN candidate index ─────────────────────

def step_build_index(db, pipeline, transformer, online_learner, stock_id_map: dict):
    """Fit feature pipeline on real data and build the nearest-neighbour index."""
    print_section("STEP 5 — Building feature pipeline and candidate index")
    built = 0
    for ticker, sid in stock_id_map.items():
        raw = db.get_price_history(sid, limit=600)
        if len(raw) < 260:
            continue
        df = pd.DataFrame(raw)
        pipeline.fit(df)
        seq = pipeline.get_latest_sequence(df)
        if seq is not None:
            online_learner._update_stock_embedding(sid, seq)
            built += 1
    online_learner.rebuild_candidate_index()
    n = online_learner.candidate_index.n_stocks
    print(f"  Index built: {n} stocks ({built} embeddings from real yfinance data)")


# ── Recommendations display ───────────────────────────────────────────────────

def get_recommendations(engine, user_id: int) -> list:
    return engine.get_recommendations(user_id, k=CONFIG.model.final_k,
                                      exclude_interacted=False)


def display_recommendations(recs: list, label: str):
    print(f"\n  {label}:")
    print(f"  {'#':<3} {'Ticker':<8} {'Signal':<12} {'P(up)':<7} "
          f"{'Sharpe':<8} {'VaR(95%)':<10} {'Risk':<6} {'Opp'}")
    print(f"  {'-'*3} {'-'*8} {'-'*12} {'-'*7} {'-'*8} {'-'*10} {'-'*6} {'-'*5}")
    for rec in recs:
        rp    = rec.risk_profile
        sharpe = f"{rp.sharpe_ratio:.2f}" if rp else "n/a"
        var    = f"{rp.var_95*100:.2f}%" if rp else "n/a"
        p_up   = f"{rec.direction_probs[2]*100:.0f}%"
        print(f"  {rec.rank:<3} {rec.ticker:<8} {rec.entry_signal:<12} {p_up:<7} "
              f"{sharpe:<8} {var:<10} "
              f"{rec.risk_score:.0f}/100  {rec.opportunity_score:.0f}/100")


# ── News display ──────────────────────────────────────────────────────────────

def step_show_news(recs: list, max_articles: int = 4):
    """
    Fetch and display recent news + sentiment for the top recommended tickers.
    Uses data_downloader.MarketDataDownloader — separate from the training pipeline.
    """
    from data_downloader import MarketDataDownloader

    tickers = [r.ticker for r in recs[:CONFIG.model.final_k]]
    print_section(f"NEWS & SENTIMENT for top {len(tickers)} recommended stocks")
    print("  Fetching recent news articles via yfinance...")

    dl   = MarketDataDownloader(cache_dir=DATA_CACHE_DIR, verbose=False)
    news = dl.get_news(tickers, max_per_ticker=max_articles, cache_max_age_hours=2)

    sentiment_map = {}
    for ticker in tickers:
        articles = news.get(ticker, [])
        dl.print_news_summary(ticker, articles, max_show=max_articles)
        agg = dl.aggregate_news_sentiment(articles)
        sentiment_map[ticker] = agg

    # Show how news sentiment aligns with ML recommendation
    print("\n  News-vs-Model alignment:")
    print(f"  {'Ticker':<8} {'Rank':<5} {'Model Signal':<14} {'News Sentiment':<14} {'Alignment'}")
    print(f"  {'-'*8} {'-'*5} {'-'*14} {'-'*14} {'-'*20}")
    for rec in recs[:CONFIG.model.final_k]:
        t   = rec.ticker
        agg = sentiment_map.get(t, {})
        ns  = agg.get("overall_label", "NO DATA")
        ms  = rec.entry_signal.upper()
        # Simple alignment check
        model_bull  = ms in ("BUY", "STRONG BUY")
        model_hold  = ms == "HOLD"
        news_bull   = ns == "BULLISH"
        news_bear   = ns == "BEARISH"
        if model_bull and news_bull:
            align = "[CONFIRMED] both bullish"
        elif model_bull and news_bear:
            align = "[CONFLICT]  model buy, news bearish"
        elif model_hold and news_bull:
            align = "[UPSIDE]    news ahead of model"
        elif model_hold and news_bear:
            align = "[CAUTION]   news bearish on hold"
        elif ns == "NO DATA":
            align = "[NO NEWS]   model signal only"
        else:
            align = "[NEUTRAL]   mixed/neutral signals"
        print(f"  {t:<8} #{rec.rank:<4} {ms:<14} {ns:<14} {align}")


# ── Interactions replay ───────────────────────────────────────────────────────

def step_replay_interactions(user_id: int, interactions: list, stock_id_map: dict,
                              db, pipeline, online_learner):
    """Replay all interactions; each one updates the online learner."""
    from stock_recommender.data.user_tracker import UserTracker
    tracker = UserTracker(db)

    print_section(f"Replaying {len(interactions)} interactions")
    update_count = 0

    for event in interactions:
        ticker = event["stock"]
        if ticker not in stock_id_map:
            continue

        sid    = stock_id_map[ticker]
        action = event["action"]
        note   = event.get("note", "")

        if action == "view_long":
            dur = event.get("duration_sec", 120)
            tracker.log_view(user_id, sid, dur)
            signal_str = f"view {dur}s"
        elif action == "view_short":
            dur = event.get("duration_sec", 15)
            tracker.log_view(user_id, sid, dur)
            signal_str = f"view {dur}s (brief)"
        elif action == "watchlist_add":
            tracker.log_watchlist_add(user_id, sid)
            signal_str = "watchlist ADD"
        elif action == "trade_buy":
            qty = event.get("quantity", 1)
            tracker.log_trade(user_id, sid, "buy", qty, 100.0)
            signal_str = f"BUY x{qty}"
        elif action == "rate":
            rating = event.get("rating", 0.5)
            tracker.log_rating(user_id, sid, rating)
            signal_str = f"rate {rating:.2f}"
        else:
            continue

        raw = db.get_price_history(sid, limit=600)
        if len(raw) >= 260:
            df   = pd.DataFrame(raw)
            snap = pipeline.get_latest_snapshot(df)
            if snap is not None:
                profile = tracker.get_profile_features(user_id)
                reward_map = {
                    "view_long": 0.3, "view_short": -0.1, "watchlist_add": 0.6,
                    "trade_buy": 1.0, "rate": event.get("rating", 0.5),
                }
                reward      = reward_map.get(action, 0.1)
                is_positive = action != "view_short"
                online_learner.on_user_event(
                    user_id=user_id, stock_id=sid,
                    user_features=profile, stock_features=snap,
                    reward=reward, is_positive=is_positive,
                )
                update_count += 1

        icon = "+" if action != "view_short" else "-"
        print(f"  [{icon}] {ticker:<6} | {signal_str:<20} | {note[:45]}")

    stats = online_learner.get_stats()
    print(f"\n  Online learner stats after interactions:")
    print(f"    Replay buffer : {stats['replay_buffer_size']} events")
    print(f"    Micro-updates : {stats['micro_updates']} gradient steps taken")
    rew = stats["recent_reward_stats"]
    print(f"    Avg reward    : {rew['mean_reward']:+.3f}  |  "
          f"Positive rate: {rew['positive_rate']*100:.0f}%")


def step_consolidate(online_learner, db, pipeline, stock_id_map, candidate_index):
    """Run extra gradient steps then rebuild the candidate index."""
    print_section("Consolidating learning (mini-batch gradient steps)")
    losses = []
    for _ in range(40):
        loss = online_learner._micro_update_towers()
        if loss is not None:
            losses.append(loss)
    if losses:
        print(f"  40 gradient steps  loss: {losses[0]:.4f} -> {losses[-1]:.4f} "
              f"({'improved' if losses[-1] < losses[0] else 'adapting'})")

    for ticker, sid in stock_id_map.items():
        raw = db.get_price_history(sid, limit=600)
        if len(raw) >= 260:
            df  = pd.DataFrame(raw)
            seq = pipeline.get_latest_sequence(df)
            if seq is not None:
                online_learner._update_stock_embedding(sid, seq)
    online_learner.rebuild_candidate_index()
    print(f"  Candidate index rebuilt with {candidate_index.n_stocks} stocks")


def compare_recommendations(before: list, after: list):
    print_section("How recommendations changed (algorithm learning effect)")

    before_rank = {r.ticker: r.rank for r in before}
    after_rank  = {r.ticker: r.rank for r in after}
    all_tickers = list(dict.fromkeys([r.ticker for r in before] + [r.ticker for r in after]))

    print(f"  {'Ticker':<8} {'Before':>8} {'After':>7}  Change")
    print(f"  {'-'*8} {'-'*8} {'-'*7}  {'-'*30}")

    for ticker in all_tickers:
        b = before_rank.get(ticker)
        a = after_rank.get(ticker)
        if b is None:
            print(f"  {ticker:<8} {'--':>8} #{a:<6}  ** NEW in recommendations")
        elif a is None:
            print(f"  {ticker:<8} #{b:<7} {'--':>7}  dropped out")
        else:
            diff = b - a
            note = "  (no change)" if diff == 0 else (
                f"  UP {diff} spots" if diff > 0 else f"  down {abs(diff)} spots")
            print(f"  {ticker:<8} #{b:<7} #{a:<6} {note}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print_header("USER 1 DEMO — Alex (Aggressive Tech Investor)")
    print("  Demonstrating personalized recommendation learning")
    print("  Training data: REAL yfinance OHLCV (no synthetics)")
    print("  News: scraped live from Yahoo Finance per recommendation")

    with open(PROFILE_PATH) as f:
        profile_data = json.load(f)
    with open(INTERACTIONS_PATH) as f:
        interactions = json.load(f)

    print(f"\n  User   : {profile_data['display_name']} ({profile_data['persona']})")
    print(f"  Style  : {profile_data['risk_tolerance']} risk, "
          f"{profile_data['investment_horizon']}-term horizon")
    print(f"  Loves  : {', '.join(profile_data['favorite_stocks'])}")
    print(f"  Avoids : {', '.join(profile_data['dislike_stocks'])}")

    # ── 1. Download market data (separate from training) ──────────────────────
    market_data = step_download_data()
    if len(market_data) < 5:
        print("\n[ERROR] Not enough data downloaded. Check your internet connection.")
        sys.exit(1)

    # ── 2. Init ML components ─────────────────────────────────────────────────
    print_section("STEP 2 — Initializing ML pipeline")
    (db, pipeline, transformer, two_tower, ranker,
     candidate_index, online_learner, engine) = step_init_pipeline()

    # ── 3. Store real OHLCV in database ──────────────────────────────────────
    stock_id_map = step_seed_database(db, market_data)

    # ── Create user ───────────────────────────────────────────────────────────
    user_id = db.create_user(
        username=profile_data["username"],
        risk_tolerance=profile_data["risk_tolerance"],
        capital_range=profile_data["capital_range"],
        investment_horizon=profile_data["investment_horizon"],
        preferred_sectors=profile_data["preferred_sectors"],
    )
    print(f"\n  User '{profile_data['display_name']}' registered with ID {user_id}")

    # ── 4. Train transformer on real yfinance data ────────────────────────────
    step_train_transformer(db, pipeline, transformer, two_tower, ranker, stock_id_map)

    # ── 5. Build feature pipeline + candidate index ───────────────────────────
    step_build_index(db, pipeline, transformer, online_learner, stock_id_map)

    if candidate_index.n_stocks == 0:
        print("\n[ERROR] No stocks indexed. Exiting.")
        sys.exit(1)

    # ── 6. Initial recommendations (before any user interactions) ─────────────
    print_header("BEFORE LEARNING — Initial Recommendations for Alex")
    print("  (Algorithm has NO knowledge of Alex's preferences yet)")
    recs_before = get_recommendations(engine, user_id)
    display_recommendations(recs_before, "Initial top recommendations")

    # ── 7. News & sentiment for initial recommendations ───────────────────────
    step_show_news(recs_before, max_articles=4)

    input("\n  Press ENTER to start replaying Alex's interactions...")

    # ── 8. Replay interactions — online learning phase ────────────────────────
    print_header("ALEX'S INTERACTIONS — Learning Phase")
    step_replay_interactions(user_id, interactions, stock_id_map,
                             db, pipeline, online_learner)

    # ── 9. Consolidation: extra gradient steps + index rebuild ─────────────────
    step_consolidate(online_learner, db, pipeline, stock_id_map, candidate_index)

    # ── 10. Updated recommendations after learning ────────────────────────────
    print_header("AFTER LEARNING — Updated Recommendations for Alex")
    print("  (Algorithm has adapted to Alex's tech/growth preference)")

    from stock_recommender.data.user_tracker import UserTracker
    tracker      = UserTracker(db)
    profile_feats = tracker.get_profile_features(user_id)
    online_learner._update_user_embedding(user_id, profile_feats)

    recs_after = get_recommendations(engine, user_id)
    display_recommendations(recs_after, "Updated top recommendations")

    # ── 11. News for updated recommendations ──────────────────────────────────
    step_show_news(recs_after, max_articles=3)

    # ── 12. Learning delta comparison ─────────────────────────────────────────
    print_header("LEARNING SUMMARY")
    compare_recommendations(recs_before, recs_after)

    tech = {"NVDA", "TSLA", "AMD", "META", "AAPL", "GOOGL", "AMZN", "MSFT", "NFLX"}
    before_tech = [r.ticker for r in recs_before if r.ticker in tech]
    after_tech  = [r.ticker for r in recs_after  if r.ticker in tech]
    print(f"\n  Tech stocks in top-{CONFIG.model.final_k} BEFORE : "
          f"{len(before_tech)} — {before_tech}")
    print(f"  Tech stocks in top-{CONFIG.model.final_k} AFTER  : "
          f"{len(after_tech)} — {after_tech}")

    if len(after_tech) >= len(before_tech):
        print("\n  RESULT: Algorithm successfully learned Alex's tech preference.")
    else:
        print("\n  RESULT: Learning is happening — "
              "run user2_demo.py next to see contrast.")

    # ── 13. Full risk analysis for top pick ───────────────────────────────────
    if recs_after:
        top = recs_after[0]
        print_header(f"FULL RISK ANALYSIS — Alex's Top Pick: {top.ticker}")
        analysis = engine.get_full_analysis(user_id, top.stock_id)
        if analysis:
            print(analysis.narrative)
            print(f"\n  POSITION SIZING RECOMMENDATION")
            print(f"    Suggested allocation : {analysis.position_sizing_pct:.1f}% of portfolio")
            print(f"    Stop loss level      : -{analysis.stop_loss_pct:.1f}% from entry")
            print(f"    Take profit target   : +{analysis.take_profit_pct:.1f}% from entry")

    print_header("DEMO COMPLETE")
    print("  Run python user2_demo.py to see Sam (conservative investor)")
    print("  and compare how differently the algorithm adapts to each user.")
    print()


if __name__ == "__main__":
    main()
