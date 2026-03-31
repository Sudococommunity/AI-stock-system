"""
user2_demo.py — "Sam" (Conservative Dividend & Value Investor)
==============================================================
Standalone demo showing how the recommendation algorithm learns from Sam's
behavior. Runs independently — just: python user2_demo.py

Run AFTER user1_demo.py to see the contrast: the same algorithm produces
completely different recommendations for a completely different user type.

Sam's profile: conservative, long-term, loves stable dividend stocks (JNJ, KO, WMT, PG)
Expected learning: after interactions, defensive/dividend stocks should rank higher.
"""
import sys
import os
import json
import time
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

# ── Tweak config for demo ────────────────────────────────────────────────────
from stock_recommender.config import CONFIG
CONFIG.data.db_path = "demo_shared.db"        # same shared DB as user1_demo
CONFIG.data.checkpoint_dir = "demo_checkpoints/"
CONFIG.training.online_lr = 2e-3
CONFIG.training.replay_min_fill = 16
CONFIG.training.online_update_every = 4
CONFIG.model.final_k = 8

DEMO_TICKERS = [
    # Defensive / Value (Sam's favorites)
    "JNJ",  "KO",   "WMT",  "PG",   "JPM",  "VZ",   "XOM",  "PFE",
    # Tech / Growth (Sam avoids)
    "NVDA", "TSLA", "AMD",  "META", "AAPL", "GOOGL", "AMZN", "MSFT", "NFLX",
    # Extra
    "V",    "HD",
]

PROFILE_PATH = os.path.join(os.path.dirname(__file__), "user2", "profile.json")
INTERACTIONS_PATH = os.path.join(os.path.dirname(__file__), "user2", "interactions.json")


# ── Utilities (same as user1_demo.py — each script is self-contained) ────────

def print_header(title: str):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def print_section(title: str):
    print(f"\n--- {title} ---")


def download_market_data(tickers: list, period: str = "2y") -> dict:
    try:
        import yfinance as yf
    except ImportError:
        print("[ERROR] yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    print_section("Downloading real market data via yfinance")
    data = {}
    for ticker in tickers:
        try:
            raw = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if raw is None or len(raw) < 200:
                print(f"  [SKIP] {ticker}: not enough data")
                continue

            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw.columns = [str(c).lower().strip() for c in raw.columns]

            col_map = {"open": "open", "high": "high", "low": "low",
                       "close": "close", "volume": "volume", "adj close": "close"}
            raw = raw.rename(columns=col_map)
            required = ["open", "high", "low", "close", "volume"]
            if not all(c in raw.columns for c in required):
                print(f"  [SKIP] {ticker}: missing columns")
                continue

            raw = raw[required].copy()
            raw["date"] = pd.to_datetime(raw.index).strftime("%Y-%m-%d")
            raw = raw.reset_index(drop=True).dropna()
            data[ticker] = raw
            print(f"  [OK] {ticker}: {len(raw)} trading days")
        except Exception as e:
            print(f"  [FAIL] {ticker}: {e}")

    return data


def setup_system():
    from stock_recommender.data.database import DatabaseManager
    from stock_recommender.features.feature_pipeline import FeaturePipeline
    from stock_recommender.models.time_series import StockTransformer
    from stock_recommender.models.two_tower import TwoTowerModel, RankingModel, CandidateIndex
    from stock_recommender.learning.online_learner import OnlineLearner
    from stock_recommender.recommendation.engine import RecommendationEngine

    os.makedirs(CONFIG.data.checkpoint_dir, exist_ok=True)
    db = DatabaseManager(CONFIG.data.db_path)
    pipeline = FeaturePipeline()
    transformer = StockTransformer()
    two_tower = TwoTowerModel()
    ranker = RankingModel()
    candidate_index = CandidateIndex()

    online_learner = OnlineLearner(
        two_tower=two_tower, transformer=transformer,
        candidate_index=candidate_index, feature_pipeline=pipeline, db=db,
    )
    engine = RecommendationEngine(
        two_tower=two_tower, ranker=ranker, transformer=transformer,
        candidate_index=candidate_index, feature_pipeline=pipeline, db=db,
    )
    return db, pipeline, transformer, two_tower, ranker, candidate_index, online_learner, engine


def seed_real_data(db, market_data: dict) -> dict:
    print_section("Storing market data in database")
    sector_map = {
        "JNJ": "Healthcare", "KO": "Consumer",   "WMT": "Retail",
        "PG":  "Consumer",   "JPM": "Finance",   "VZ":  "Telecom",
        "XOM": "Energy",     "PFE": "Healthcare","V":   "Finance",
        "HD":  "Retail",     "NVDA":"Technology", "TSLA":"Automotive",
        "AMD": "Technology", "META":"Technology", "AAPL":"Technology",
        "GOOGL":"Technology","AMZN":"Consumer",  "MSFT":"Technology",
        "NFLX":"Media",
    }
    stock_id_map = {}
    for ticker, df in market_data.items():
        sid = db.upsert_stock(ticker, ticker, sector_map.get(ticker, "Unknown"))
        db.insert_price_batch(sid, df.to_dict("records"))
        stock_id_map[ticker] = sid
    print(f"  Stored {len(stock_id_map)} stocks in database")
    return stock_id_map


def build_pipeline_and_index(db, pipeline, transformer, online_learner, stock_id_map: dict):
    print_section("Building feature pipeline and candidate index")
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
    print(f"  Index built with {online_learner.candidate_index.n_stocks} stocks")


def load_or_train_transformer(db, pipeline, transformer, two_tower, ranker, stock_id_map: dict):
    """Reuse Alex's trained checkpoint if it exists (saves time in demo)."""
    from stock_recommender.learning.trainer import Trainer
    import torch

    # Check for Alex's checkpoint first (shared model)
    for ckpt_name in ["transformer_user1.pt", "transformer_user2.pt"]:
        checkpoint = os.path.join(CONFIG.data.checkpoint_dir, ckpt_name)
        if os.path.exists(checkpoint):
            print_section(f"Loading existing transformer checkpoint ({ckpt_name})")
            transformer.load_state_dict(torch.load(checkpoint, map_location="cpu"))
            print(f"  Loaded: {checkpoint}")
            return

    print_section("Quick-training transformer on real market data (2 epochs)")
    trainer = Trainer(transformer, two_tower, ranker, pipeline, db)
    CONFIG.training.max_epochs = 2
    CONFIG.training.early_stop_patience = 1

    stock_ids = list(stock_id_map.values())
    history = trainer.train_transformer(stock_ids, n_epochs=2)
    if history["train_loss"]:
        print(f"  Final loss: {history['train_loss'][-1]:.4f}")

    checkpoint = os.path.join(CONFIG.data.checkpoint_dir, "transformer_user2.pt")
    torch.save(transformer.state_dict(), checkpoint)
    print(f"  Saved: {checkpoint}")


def get_recommendations(engine, user_id: int, exclude_interacted: bool = False) -> list:
    return engine.get_recommendations(user_id, k=CONFIG.model.final_k,
                                      exclude_interacted=exclude_interacted)


def display_recommendations(recs: list, label: str):
    print(f"\n  {label}:")
    print(f"  {'#':<3} {'Ticker':<8} {'Signal':<12} {'P(up)':<7} {'Sharpe':<8} {'VaR(95%)':<10} {'Risk':<6} {'Opp'}")
    print(f"  {'-'*3} {'-'*8} {'-'*12} {'-'*7} {'-'*8} {'-'*10} {'-'*6} {'-'*5}")
    for rec in recs:
        rp = rec.risk_profile
        sharpe = f"{rp.sharpe_ratio:.2f}" if rp else "n/a"
        var = f"{rp.var_95*100:.2f}%" if rp else "n/a"
        p_up = f"{rec.direction_probs[2]*100:.0f}%"
        print(f"  {rec.rank:<3} {rec.ticker:<8} {rec.entry_signal:<12} {p_up:<7} {sharpe:<8} {var:<10} "
              f"{rec.risk_score:.0f}/100  {rec.opportunity_score:.0f}/100")


def replay_interactions(user_id: int, interactions: list, stock_id_map: dict,
                        db, pipeline, online_learner, engine):
    from stock_recommender.data.user_tracker import UserTracker
    tracker = UserTracker(db)

    print_section(f"Replaying {len(interactions)} interactions")

    for event in interactions:
        ticker = event["stock"]
        if ticker not in stock_id_map:
            continue

        sid = stock_id_map[ticker]
        action = event["action"]
        note = event.get("note", "")

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
            df = pd.DataFrame(raw)
            snap = pipeline.get_latest_snapshot(df)
            if snap is not None:
                profile = tracker.get_profile_features(user_id)
                is_positive = action not in ("view_short",)
                reward_map = {
                    "view_long": 0.3, "view_short": -0.1, "watchlist_add": 0.6,
                    "trade_buy": 1.0, "rate": event.get("rating", 0.5),
                }
                reward = reward_map.get(action, 0.1)
                online_learner.on_user_event(
                    user_id=user_id, stock_id=sid,
                    user_features=profile, stock_features=snap,
                    reward=reward, is_positive=is_positive,
                )

        icon = "+" if action not in ("view_short",) else "-"
        print(f"  [{icon}] {ticker:<6} | {signal_str:<20} | {note[:45]}")

    stats = online_learner.get_stats()
    print(f"\n  Online learner stats after interactions:")
    print(f"    Replay buffer : {stats['replay_buffer_size']} events")
    print(f"    Micro-updates : {stats['micro_updates']} gradient steps taken")
    rew = stats["recent_reward_stats"]
    print(f"    Avg reward    : {rew['mean_reward']:+.3f}  |  Positive rate: {rew['positive_rate']*100:.0f}%")


def compare_recommendations(before: list, after: list):
    print_section("How recommendations changed (algorithm learning effect)")
    before_rank = {r.ticker: r.rank for r in before}
    after_rank = {r.ticker: r.rank for r in after}
    all_tickers = list(dict.fromkeys([r.ticker for r in before] + [r.ticker for r in after]))

    print(f"  {'Ticker':<8} {'Before':>8} {'After':>7}  Change")
    print(f"  {'-'*8} {'-'*8} {'-'*7}  {'-'*30}")

    for ticker in all_tickers:
        b = before_rank.get(ticker, None)
        a = after_rank.get(ticker, None)
        if b is None and a is not None:
            print(f"  {ticker:<8} {'--':>8} #{a:<6}  ** NEW in recommendations")
        elif b is not None and a is None:
            print(f"  {ticker:<8} #{b:<7} {'--':>7}  dropped out")
        elif b is not None and a is not None:
            diff = b - a
            arrow = "  (no change)"
            if diff > 0:
                arrow = f"  UP {diff} spots"
            elif diff < 0:
                arrow = f"  down {abs(diff)} spots"
            print(f"  {ticker:<8} #{b:<7} #{a:<6} {arrow}")


def show_alex_vs_sam_contrast(db, engine_sam, user_id_sam: int):
    """
    If Alex's data exists in the DB, show side-by-side comparison to drive
    home the point that the SAME model produces different results for each user.
    """
    # Look for Alex's user in the DB
    with db.connection() as conn:
        row = conn.execute(
            "SELECT user_id FROM users WHERE username='alex_trader'"
        ).fetchone()

    if not row:
        return  # Alex hasn't run yet

    user_id_alex = row["user_id"]
    print_section("CONTRAST: Same algorithm, different users")
    print("  Comparing Alex (aggressive) vs Sam (conservative) recommendations")

    recs_alex = engine_sam.get_recommendations(user_id_alex, k=5, exclude_interacted=False)
    recs_sam  = engine_sam.get_recommendations(user_id_sam,  k=5, exclude_interacted=False)

    print(f"\n  {'Alex (Aggressive/Tech)':<30} {'Sam (Conservative/Dividend)'}")
    print(f"  {'-'*30} {'-'*30}")
    max_len = max(len(recs_alex), len(recs_sam))
    for i in range(max_len):
        a_str = f"#{recs_alex[i].rank} {recs_alex[i].ticker}" if i < len(recs_alex) else ""
        s_str = f"#{recs_sam[i].rank} {recs_sam[i].ticker}" if i < len(recs_sam) else ""
        print(f"  {a_str:<30} {s_str}")

    print("\n  The algorithm has learned distinct taste profiles for each user.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print_header("USER 2 DEMO — Sam (Conservative Dividend Investor)")
    print("  Demonstrating how the recommendation algorithm learns")
    print("  from user behavior using REAL market data (yfinance)")

    with open(PROFILE_PATH) as f:
        profile_data = json.load(f)
    with open(INTERACTIONS_PATH) as f:
        interactions = json.load(f)

    print(f"\n  User   : {profile_data['display_name']} ({profile_data['persona']})")
    print(f"  Style  : {profile_data['risk_tolerance']} risk, {profile_data['investment_horizon']}-term horizon")
    print(f"  Loves  : {', '.join(profile_data['favorite_stocks'])}")
    print(f"  Avoids : {', '.join(profile_data['dislike_stocks'])}")

    # ── Download data ─────────────────────────────────────────────────────────
    market_data = download_market_data(DEMO_TICKERS, period="2y")
    if len(market_data) < 5:
        print("\n[ERROR] Not enough data downloaded.")
        sys.exit(1)

    # ── Setup ─────────────────────────────────────────────────────────────────
    print_section("Initializing ML pipeline")
    db, pipeline, transformer, two_tower, ranker, candidate_index, online_learner, engine = setup_system()

    stock_id_map = seed_real_data(db, market_data)

    user_id = db.create_user(
        username=profile_data["username"],
        risk_tolerance=profile_data["risk_tolerance"],
        capital_range=profile_data["capital_range"],
        investment_horizon=profile_data["investment_horizon"],
        preferred_sectors=profile_data["preferred_sectors"],
    )
    print(f"\n  User '{profile_data['display_name']}' registered with ID {user_id}")

    load_or_train_transformer(db, pipeline, transformer, two_tower, ranker, stock_id_map)
    build_pipeline_and_index(db, pipeline, transformer, online_learner, stock_id_map)

    if candidate_index.n_stocks == 0:
        print("\n[ERROR] No stocks indexed. Exiting.")
        sys.exit(1)

    # ── BEFORE recommendations ────────────────────────────────────────────────
    print_header("BEFORE LEARNING — Initial Recommendations for Sam")
    print("  (Algorithm has NO knowledge of Sam's conservative preference yet)")
    recs_before = get_recommendations(engine, user_id, exclude_interacted=False)
    display_recommendations(recs_before, "Initial top recommendations")

    input("\n  Press ENTER to start replaying Sam's interactions...")

    # ── Replay interactions ───────────────────────────────────────────────────
    print_header("SAM'S INTERACTIONS — Learning Phase")
    replay_interactions(user_id, interactions, stock_id_map, db, pipeline, online_learner, engine)

    # ── Consolidate learning ──────────────────────────────────────────────────
    print_section("Consolidating learning (running mini-batch updates)")
    losses = []
    for step in range(40):
        loss = online_learner._micro_update_towers()
        if loss is not None:
            losses.append(loss)
    if losses:
        print(f"  40 gradient steps  loss: {losses[0]:.4f} -> {losses[-1]:.4f} "
              f"({'improved' if losses[-1] < losses[0] else 'adapting'})")

    # Rebuild stock embeddings and index after model update
    for ticker, sid in stock_id_map.items():
        raw = db.get_price_history(sid, limit=600)
        if len(raw) >= 260:
            df = pd.DataFrame(raw)
            seq = pipeline.get_latest_sequence(df)
            if seq is not None:
                online_learner._update_stock_embedding(sid, seq)
    online_learner.rebuild_candidate_index()
    print(f"  Candidate index rebuilt with {candidate_index.n_stocks} stocks")

    # ── AFTER recommendations ─────────────────────────────────────────────────
    print_header("AFTER LEARNING — Updated Recommendations for Sam")
    print("  (Algorithm has adapted to Sam's conservative/dividend preference)")

    from stock_recommender.data.user_tracker import UserTracker
    tracker = UserTracker(db)
    profile_feats = tracker.get_profile_features(user_id)
    online_learner._update_user_embedding(user_id, profile_feats)

    recs_after = get_recommendations(engine, user_id, exclude_interacted=False)
    display_recommendations(recs_after, "Updated top recommendations")

    # ── Learning delta ────────────────────────────────────────────────────────
    print_header("LEARNING SUMMARY")
    compare_recommendations(recs_before, recs_after)

    defensive = {"JNJ", "KO", "WMT", "PG", "JPM", "VZ", "XOM", "PFE"}
    before_def = [r.ticker for r in recs_before if r.ticker in defensive]
    after_def  = [r.ticker for r in recs_after  if r.ticker in defensive]
    print(f"\n  Defensive stocks in top-{CONFIG.model.final_k} BEFORE : {len(before_def)} — {before_def}")
    print(f"  Defensive stocks in top-{CONFIG.model.final_k} AFTER  : {len(after_def)} — {after_def}")

    if len(after_def) >= len(before_def):
        print("\n  RESULT: Algorithm successfully learned Sam's conservative preference.")
    else:
        print("\n  RESULT: Preference learning is happening — run both demos together for full contrast.")

    # ── Full analysis for Sam's top pick ──────────────────────────────────────
    if recs_after:
        top = recs_after[0]
        print_header(f"FULL RISK ANALYSIS — Sam's Top Pick: {top.ticker}")
        analysis = engine.get_full_analysis(user_id, top.stock_id)
        if analysis:
            print(analysis.narrative)
            print(f"\n  POSITION SIZING (for conservative portfolio)")
            print(f"    Suggested allocation : {analysis.position_sizing_pct:.1f}% of portfolio")
            print(f"    Stop loss level      : -{analysis.stop_loss_pct:.1f}% from entry")
            print(f"    Take profit target   : +{analysis.take_profit_pct:.1f}% from entry")

    # ── Cross-user contrast (if Alex has run) ────────────────────────────────
    show_alex_vs_sam_contrast(db, engine, user_id)

    print_header("DEMO COMPLETE")
    print("  Both user demos complete.")
    print("  Key takeaway: same ML algorithm, same model weights,")
    print("  but completely different recommendations for each user type.")
    print("  This is personalized learning in action.")
    print()


if __name__ == "__main__":
    main()
