"""
main.py — entry point and demo for the Stock Recommendation System.

Run modes:
  python main.py seed      — populate database with synthetic data
  python main.py train     — train all models on synthetic data
  python main.py recommend — get recommendations for a demo user
  python main.py analyze   — full analysis of a specific stock
  python main.py demo      — run the full pipeline end-to-end

The self-training loop (OnlineLearner) is demonstrated by simulating
user interactions and showing how the model refines itself.
"""
import sys
import os
import argparse
import logging
import json
import numpy as np
import torch
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def setup_system(db_url: str = None, config_path: str = None):
    """Initialize all system components and return them."""
    from stock_recommender.config import CONFIG, load_config_overrides
    from stock_recommender.data.database import DatabaseManager
    from stock_recommender.features.feature_pipeline import FeaturePipeline
    from stock_recommender.models.time_series import StockTransformer
    from stock_recommender.models.two_tower import TwoTowerModel, RankingModel, CandidateIndex
    from stock_recommender.learning.online_learner import OnlineLearner
    from stock_recommender.recommendation.engine import RecommendationEngine

    if config_path:
        load_config_overrides(config_path)

    if db_url:
        CONFIG.data.db_url = db_url
    os.makedirs(CONFIG.data.checkpoint_dir, exist_ok=True)
    os.makedirs(CONFIG.runtime.artifacts_dir, exist_ok=True)
    os.makedirs(CONFIG.runtime.reports_dir, exist_ok=True)
    os.makedirs(CONFIG.runtime.tournament_dir, exist_ok=True)

    db = DatabaseManager(db_url)
    pipeline = FeaturePipeline()
    transformer = StockTransformer()
    two_tower = TwoTowerModel()
    ranker = RankingModel()
    candidate_index = CandidateIndex()

    online_learner = OnlineLearner(
        two_tower=two_tower,
        transformer=transformer,
        candidate_index=candidate_index,
        feature_pipeline=pipeline,
        db=db,
    )

    engine = RecommendationEngine(
        two_tower=two_tower,
        ranker=ranker,
        transformer=transformer,
        candidate_index=candidate_index,
        feature_pipeline=pipeline,
        db=db,
    )

    return {
        "db": db,
        "pipeline": pipeline,
        "transformer": transformer,
        "two_tower": two_tower,
        "ranker": ranker,
        "candidate_index": candidate_index,
        "online_learner": online_learner,
        "engine": engine,
    }


def _limit_stock_ids(db, max_stocks: int = None):
    stock_ids = db.get_all_stock_ids()
    if max_stocks is not None and max_stocks > 0:
        return stock_ids[:max_stocks]
    return stock_ids


def _resolve_transformer_checkpoint(explicit_path: str = None):
    from stock_recommender.config import CONFIG
    from stock_recommender.models.checkpoint_utils import can_load_model
    from stock_recommender.models.time_series import StockTransformer

    if explicit_path:
        return explicit_path

    checkpoint_dir = CONFIG.data.checkpoint_dir
    if not os.path.isdir(checkpoint_dir):
        return None

    checkpoints = sorted(
        os.path.join(checkpoint_dir, name)
        for name in os.listdir(checkpoint_dir)
        if name.startswith("transformer_") and name.endswith(".pt")
    )
    if not checkpoints:
        return None

    probe_model = StockTransformer()
    for checkpoint_path in reversed(checkpoints):
        try:
            if can_load_model(probe_model, checkpoint_path, map_location="cpu"):
                return checkpoint_path
        except Exception:
            continue
    return None


def cmd_seed(args):
    """Populate database with synthetic market data."""
    from stock_recommender.data.database import DatabaseManager
    from stock_recommender.data.synthetic_bot_data import seed_database
    from stock_recommender.config import CONFIG, load_config_overrides

    db_url = getattr(args, "db", None)
    if getattr(args, "config", None):
        load_config_overrides(args.config)
    if db_url:
        CONFIG.data.db_url = db_url
    db = DatabaseManager(db_url)

    result = seed_database(db, n_users=20, n_days=500)
    print(f"\n[OK] Database seeded:")
    print(f"  Stocks  : {result['n_stocks']}")
    print(f"  Users   : {result['n_users']}")
    print(f"  Events  : {result['n_events']}")


def cmd_train(args):
    """Train all models on the seeded database."""
    components = setup_system(getattr(args, "db", None), getattr(args, "config", None))
    db = components["db"]
    pipeline = components["pipeline"]
    transformer = components["transformer"]
    two_tower = components["two_tower"]
    ranker = components["ranker"]

    from stock_recommender.learning.trainer import Trainer
    from stock_recommender.config import CONFIG

    os.makedirs(CONFIG.data.checkpoint_dir, exist_ok=True)
    trainer = Trainer(transformer, two_tower, ranker, pipeline, db)

    stock_ids = _limit_stock_ids(db, getattr(args, "max_stocks", None))
    if not stock_ids:
        print("No stocks in database. Run 'python main.py seed' first.")
        return

    print(f"\nTraining on {len(stock_ids)} stocks...")
    print("Phase 1: StockTransformer (time-series)...")

    # Fit the feature pipeline normalizer
    for sid in stock_ids:
        raw = db.get_price_history(sid, limit=500)
        if len(raw) >= 60:
            df = pd.DataFrame(raw)
            pipeline.fit(df)

    # Train transformer (a few epochs for demo speed)
    from stock_recommender.config import CONFIG
    CONFIG.training.max_epochs = 3
    CONFIG.training.early_stop_patience = 2

    history = trainer.train_transformer(stock_ids, n_epochs=3)
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final val loss  : {history['val_loss'][-1]:.4f}")

    print("Phase 2: Building candidate index from stock embeddings...")
    online_learner = components["online_learner"]

    # Compute and index all stock embeddings (need 500 days: 200 for SMA warmup + seq)
    for sid in stock_ids:
        raw = db.get_price_history(sid, limit=500)
        if len(raw) < 260:
            continue
        df = pd.DataFrame(raw)
        seq = pipeline.get_latest_sequence(df)
        if seq is None:
            continue
        online_learner._update_stock_embedding(sid, seq)

    online_learner.rebuild_candidate_index()
    print(f"  Index built with {components['candidate_index'].n_stocks} stocks")

    print("Phase 3: TwoTower model (2 epochs)...")
    tower_history = trainer.train_towers(n_epochs=2)
    if tower_history["train_loss"]:
        print(f"  Final tower loss: {tower_history['train_loss'][-1]:.4f}")
    else:
        print("  (No interaction data yet — towers initialized with random weights)")

    print("\n[OK] Training complete. Models saved to checkpoints/")


def cmd_recommend(args):
    """Generate recommendations for a demo user."""
    components = setup_system(getattr(args, "db", None), getattr(args, "config", None))
    db = components["db"]
    engine = components["engine"]
    online_learner = components["online_learner"]
    pipeline = components["pipeline"]

    stock_ids = db.get_all_stock_ids()
    if not stock_ids:
        print("No stocks in database. Run 'python main.py seed' first.")
        return

    # Fit pipeline normalizer
    for sid in stock_ids:
        raw = db.get_price_history(sid, limit=500)
        if len(raw) >= 260:
            pipeline.fit(pd.DataFrame(raw))

    # Rebuild candidate index
    for sid in stock_ids:
        raw = db.get_price_history(sid, limit=500)
        if len(raw) < 260:
            continue
        seq = pipeline.get_latest_sequence(pd.DataFrame(raw))
        if seq is not None:
            online_learner._update_stock_embedding(sid, seq)

    online_learner.rebuild_candidate_index()

    # Get the first user
    with db.connection() as conn:
        with db._cur(conn) as cur:
            cur.execute("SELECT user_id, username, risk_tolerance FROM users LIMIT 1")
            row = cur.fetchone()

    if not row:
        print("No users found. Run 'python main.py seed' first.")
        return

    user_id = row["user_id"]
    print(f"\nGenerating recommendations for user: {row['username']} (risk: {row['risk_tolerance']})")
    print("-" * 60)
    market_state = engine.get_market_regime()
    print(
        "Market State: "
        f"{market_state.temperature.upper()} / {market_state.regime.upper()} | "
        f"A/D {market_state.advance_decline_ratio:.2f} | "
        f"Above 200 DMA {market_state.pct_above_200dma*100:.0f}%"
    )
    if market_state.india_vix is not None:
        print(f"India VIX : {market_state.india_vix:.1f} ({market_state.vix_regime})")
    print(f"Rotation  : {market_state.sector_rotation} | Flow: {market_state.institutional_flow_signal}")
    print(f"Summary   : {market_state.summary}")

    recs = engine.get_recommendations(user_id, k=5, exclude_interacted=False)

    if not recs:
        print("No recommendations generated. The index may be empty.")
        print("Tip: Make sure you ran 'python main.py train' first.")
        return

    for rec in recs:
        signal_icon = {
            "strong buy": "++", "buy": "+", "hold": "=",
            "sell": "-", "strong sell": "--",
        }.get(rec.entry_signal, "?")

        print(f"\n#{rec.rank} {rec.ticker} [{signal_icon} {rec.entry_signal.upper()}]")
        print(f"  Forecast  : 1d {rec.predicted_return_1d*100:+.2f}%  |  5d {rec.predicted_return_5d*100:+.2f}%")
        p_up = rec.direction_probs[2] * 100
        print(f"  P(up)     : {p_up:.0f}%  |  Risk Score: {rec.risk_score:.0f}/100  |  Opp Score: {rec.opportunity_score:.0f}/100")
        print(f"  Market    : {rec.market_temperature.upper()} {rec.market_regime.upper()}  |  {rec.market_note}")

        if rec.risk_profile:
            rp = rec.risk_profile
            print(f"  Sharpe    : {rp.sharpe_ratio:.2f}  |  VaR(95%): {rp.var_95*100:.2f}%  |  MaxDD: {rp.max_drawdown*100:.1f}%")

        print(f"  R/R Ratio : {rec.risk_to_reward:.2f}x")
        for sig in rec.key_signals[:2]:
            print(f"  Signal    : {sig}")


def cmd_analyze(args):
    """Full analysis of a specific stock."""
    components = setup_system(getattr(args, "db", None), getattr(args, "config", None))
    db = components["db"]
    engine = components["engine"]
    online_learner = components["online_learner"]
    pipeline = components["pipeline"]

    ticker = getattr(args, "ticker", "AAPL-SYN")
    stock_id = db.get_stock_id(ticker)

    if stock_id is None:
        print(f"Stock '{ticker}' not found. Run 'python main.py seed' first.")
        print("Available tickers: AAPL-SYN, MSFT-SYN, TSLA-SYN, JPM-SYN, ...")
        return

    # Fit pipeline on all stocks
    stock_ids = db.get_all_stock_ids()
    for sid in stock_ids:
        raw = db.get_price_history(sid, limit=500)
        if len(raw) >= 260:
            pipeline.fit(pd.DataFrame(raw))

    # Build stock embeddings
    for sid in stock_ids:
        raw = db.get_price_history(sid, limit=500)
        if len(raw) < 260:
            continue
        seq = pipeline.get_latest_sequence(pd.DataFrame(raw))
        if seq is not None:
            online_learner._update_stock_embedding(sid, seq)

    online_learner.rebuild_candidate_index()

    # Get first user
    with db.connection() as conn:
        with db._cur(conn) as cur:
            cur.execute("SELECT user_id FROM users LIMIT 1")
            row = cur.fetchone()

    if not row:
        print("No users found.")
        return

    user_id = row["user_id"]
    analysis = engine.get_full_analysis(user_id, stock_id)

    if analysis is None:
        print(f"Could not compute analysis for {ticker}.")
        return

    print()
    print(analysis.narrative)
    print()
    print("POSITION SIZING")
    print(f"  Suggested allocation : {analysis.position_sizing_pct:.1f}% of portfolio")
    print(f"  Stop loss            : -{analysis.stop_loss_pct:.1f}% from entry")
    print(f"  Take profit          : +{analysis.take_profit_pct:.1f}% from entry")


def cmd_demo_online_learning(args):
    """
    Demonstrate the self-training loop by simulating user interactions
    and showing how the recommendation scores evolve.
    """
    components = setup_system(getattr(args, "db", None), getattr(args, "config", None))
    db = components["db"]
    online_learner = components["online_learner"]
    pipeline = components["pipeline"]

    stock_ids = db.get_all_stock_ids()
    if not stock_ids:
        print("Run 'python main.py seed' first.")
        return

    # Fit pipeline
    for sid in stock_ids:
        raw = db.get_price_history(sid, limit=300)
        if len(raw) >= 60:
            pipeline.fit(pd.DataFrame(raw))

    # Build initial index
    print("Building initial stock embeddings...")
    for sid in stock_ids:
        raw = db.get_price_history(sid, limit=500)
        if len(raw) < 260:
            continue
        df_s = pd.DataFrame(raw)
        pipeline.fit(df_s)
        seq = pipeline.get_latest_sequence(df_s)
        if seq is not None:
            online_learner._update_stock_embedding(sid, seq)
    online_learner.rebuild_candidate_index()
    print(f"  Index: {online_learner.candidate_index.n_stocks} stocks\n")

    # Simulate interactions
    print("Simulating user interactions and online learning...")
    print("=" * 60)

    with db.connection() as conn:
        with db._cur(conn) as cur:
            cur.execute("SELECT user_id FROM users LIMIT 1")
            row = cur.fetchone()

    if not row:
        print("No users found.")
        return

    user_id = row["user_id"]

    from stock_recommender.data.user_tracker import UserTracker
    tracker = UserTracker(db)

    n_interactions = 320   # enough to trigger micro-updates (replay_min_fill=256, update_every=32)
    successful = 0

    # Pre-fetch stock data to avoid repeated DB calls
    stock_data = {}
    for sid in stock_ids:
        raw = db.get_price_history(sid, limit=500)
        if len(raw) >= 260:
            stock_data[sid] = pd.DataFrame(raw)

    if not stock_data:
        print("No valid stock data found. Run 'python main.py seed' first.")
        return

    available_sids = list(stock_data.keys())

    for i in range(n_interactions):
        sid = int(np.random.choice(available_sids))
        df = stock_data[sid]

        snap = pipeline.get_latest_snapshot(df)
        if snap is None:
            continue

        profile = tracker.get_profile_features(user_id)
        reward = np.random.choice([-1, 0, 1], p=[0.2, 0.4, 0.4])
        is_pos = bool(reward > 0)

        online_learner.on_user_event(
            user_id=user_id,
            stock_id=sid,
            user_features=profile,
            stock_features=snap,
            reward=float(reward),
            is_positive=is_pos,
        )
        successful += 1

        # Simulate new market data arriving every 10 interactions
        if i % 10 == 0:
            try:
                seq = pipeline.get_latest_sequence(df)
                returns = df["close"].pct_change().dropna().values
                target_ret = float(returns[-1]) if len(returns) > 0 else 0.0
                target_ret_5d = float(returns[-5:].mean()) if len(returns) > 5 else 0.0
                # direction: 0=down, 1=flat, 2=up  (correct for CrossEntropyLoss)
                direction = 1 if abs(target_ret) < 0.005 else (2 if target_ret > 0 else 0)
                if seq is not None:
                    online_learner.on_new_market_data(
                        stock_id=sid,
                        feature_sequence=seq,
                        target_returns=np.array([target_ret, target_ret_5d]),
                        target_direction=direction,
                    )
            except Exception as e:
                pass  # market data update is best-effort

        if (i + 1) % 64 == 0:
            stats = online_learner.get_stats()
            print(f"  After {i+1:3d} steps ({successful} successful):")
            print(f"    Replay buffer       : {stats['replay_buffer_size']} events")
            print(f"    Micro-updates done  : {stats['micro_updates']}")
            rew_stats = stats["recent_reward_stats"]
            print(f"    Avg reward          : {rew_stats['mean_reward']:+.3f}  |  Win rate: {rew_stats['positive_rate']*100:.0f}%")
            if stats["avg_online_loss"]:
                print(f"    Online loss (avg)   : {stats['avg_online_loss']:.4f}")
            print()

    print("=" * 60)
    print("Self-training demonstration complete.")
    print("The models have been refined on the simulated interaction data.")
    print("In production, this loop runs continuously as users interact with the system.")


def cmd_demo(args):
    """Run the complete demo: seed → train → recommend → analyze → online learning."""
    print("=" * 60)
    print("STOCK RECOMMENDATION SYSTEM — FULL DEMO")
    print("=" * 60)

    print("\n[1/5] Seeding database with synthetic market data...")
    cmd_seed(args)

    print("\n[2/5] Training models...")
    cmd_train(args)

    print("\n[3/5] Generating personalized recommendations...")
    cmd_recommend(args)

    print("\n[4/5] Full analysis for TSLA-SYN...")
    args.ticker = "TSLA-SYN"
    cmd_analyze(args)

    print("\n[5/5] Demonstrating online self-learning...")
    cmd_demo_online_learning(args)

    print("\n" + "=" * 60)
    print("Demo complete. The system is now ready to:")
    print("  • Recommend stocks personalized to each user's risk profile")
    print("  • Provide full risk analysis (Sharpe, VaR, CVaR, Beta, etc.)")
    print("  • Continuously refine itself from user interactions")
    print("  • Adapt to new market data as it arrives daily")
    print("=" * 60)


def cmd_backtest4(args):
    """Run 4-day walk-forward forecasting evaluation."""
    components = setup_system(getattr(args, "db", None), getattr(args, "config", None))
    from stock_recommender.config import CONFIG
    from stock_recommender.models.checkpoint_utils import load_model_state
    from stock_recommender.evaluation.walk_forward import WalkForwardEvaluator

    checkpoint_path = _resolve_transformer_checkpoint(getattr(args, "checkpoint", None))
    if checkpoint_path:
        load_model_state(components["transformer"], checkpoint_path, map_location="cpu")

    evaluator = WalkForwardEvaluator(
        transformer=components["transformer"],
        feature_pipeline=components["pipeline"],
        db=components["db"],
    )
    result = evaluator.evaluate(
        stock_ids=_limit_stock_ids(components["db"], getattr(args, "max_stocks", None)),
        horizon_days=getattr(args, "horizon", None) or CONFIG.runtime.prediction_horizon_days,
        step_days=getattr(args, "step_days", None) or CONFIG.runtime.walk_forward_step_days,
        max_windows_per_stock=getattr(args, "max_windows", None),
    )

    report_path = os.path.join(CONFIG.runtime.reports_dir, "walk_forward_4day.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)

    summary = result.summary
    print("\n4-DAY WALK-FORWARD REPORT")
    print("-" * 60)
    print(f"Samples              : {summary.sample_count}")
    print(f"Direction accuracy   : {summary.direction_accuracy*100:.1f}%")
    print(f"Mean abs error       : {summary.mean_abs_error:.4f}")
    print(f"Mean confidence      : {summary.mean_confidence:.4f}")
    print(f"Reward score         : {summary.reward_score:.4f}")
    print(f"Grade                : {summary.grade}")
    if checkpoint_path:
        print(f"Checkpoint           : {checkpoint_path}")
    print(f"Saved report         : {report_path}")


def cmd_tournament(args):
    """Grade multiple transformer checkpoints and keep the best survivors."""
    components = setup_system(getattr(args, "db", None), getattr(args, "config", None))
    from stock_recommender.config import CONFIG
    from stock_recommender.evaluation.tournament import ModelTournament
    from stock_recommender.evaluation.walk_forward import WalkForwardEvaluator

    checkpoint_dir = CONFIG.data.checkpoint_dir
    checkpoints = sorted(
        [
            os.path.join(checkpoint_dir, name)
            for name in os.listdir(checkpoint_dir)
            if name.startswith("transformer_") and name.endswith(".pt")
        ]
    )
    if not checkpoints:
        print("No transformer checkpoints found.")
        return

    evaluator = WalkForwardEvaluator(
        transformer=components["transformer"],
        feature_pipeline=components["pipeline"],
        db=components["db"],
    )
    tournament = ModelTournament(evaluator)
    result = tournament.run(
        checkpoint_paths=checkpoints,
        survivor_count=getattr(args, "top_k", None) or CONFIG.runtime.top_k_checkpoints,
        stock_ids=_limit_stock_ids(components["db"], getattr(args, "max_stocks", None)),
    )

    report_path = os.path.join(CONFIG.runtime.tournament_dir, "transformer_tournament.json")
    tournament.save_report(result, report_path)

    print("\nMODEL TOURNAMENT")
    print("-" * 60)
    for idx, entry in enumerate(result.leaderboard, start=1):
        print(
            f"#{idx} {os.path.basename(entry.checkpoint_path)} | "
            f"grade={entry.grade} reward={entry.reward_score:.4f} "
            f"acc={entry.direction_accuracy*100:.1f}% mae={entry.mean_abs_error:.4f}"
        )

    print("\nSURVIVORS")
    for entry in result.survivors:
        print(f"  {os.path.basename(entry.checkpoint_path)}")
    print(f"\nSaved report: {report_path}")


def cmd_export_config(args):
    """Write the current runtime config to a JSON file for another machine."""
    from stock_recommender.config import config_to_dict, load_config_overrides

    if getattr(args, "config", None):
        load_config_overrides(args.config)

    path = getattr(args, "output", "train_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config_to_dict(), f, indent=2)
    print(f"Exported config to {path}")


def cmd_ingest_india(args):
    """Ingest Indian market history from internet first, then CSV if needed."""
    from stock_recommender.config import CONFIG, load_config_overrides
    from stock_recommender.data.database import DatabaseManager
    from stock_recommender.data.indian_market_data import ingest_indian_market_dataset

    if getattr(args, "config", None):
        load_config_overrides(args.config)

    db_url = getattr(args, "db", None)
    if db_url:
        CONFIG.data.db_url = db_url
    db = DatabaseManager(db_url)

    symbols = None
    raw_symbols = getattr(args, "symbols", None)
    if raw_symbols:
        symbols = [s.strip() for s in raw_symbols.split(",") if s.strip()]

    result = ingest_indian_market_dataset(
        db=db,
        source=getattr(args, "source", "internet_or_csv"),
        data_dir=getattr(args, "data_dir", None),
        metadata_path=getattr(args, "metadata", None),
        symbols=symbols,
        start=getattr(args, "start", None),
        end=getattr(args, "end", None),
        market_prefix=getattr(args, "market_prefix", "NSE"),
    )

    print("\nINDIAN MARKET INGESTION")
    print("-" * 60)
    print(f"Imported stocks      : {result.imported_stocks}")
    print(f"Imported price rows  : {result.imported_rows}")
    print(f"Skipped items        : {len(result.skipped_files)}")
    if result.skipped_files:
        for item in result.skipped_files[:10]:
            print(f"  {item}")


def cmd_train_population(args):
    """Train multiple forecasting candidates and keep the best versions."""
    components = setup_system(getattr(args, "db", None), getattr(args, "config", None))
    from stock_recommender.config import CONFIG
    from stock_recommender.learning.population_trainer import PopulationTrainer

    population_trainer = PopulationTrainer(db=components["db"])
    result = population_trainer.train_population(
        stock_ids=_limit_stock_ids(components["db"], getattr(args, "max_stocks", None)),
        population_size=getattr(args, "population_size", None) or CONFIG.runtime.population_size,
        survivor_count=getattr(args, "top_k", None) or CONFIG.runtime.top_k_checkpoints,
        epochs=getattr(args, "population_epochs", None),
        max_windows_per_stock=getattr(args, "max_windows", None),
    )

    report_path = os.path.join(CONFIG.runtime.tournament_dir, "population_training.json")
    population_trainer.save_report(result, report_path)

    print("\nPOPULATION TRAINING")
    print("-" * 60)
    for idx, entry in enumerate(result.leaderboard, start=1):
        print(
            f"#{idx} {entry.name} | grade={entry.grade} reward={entry.reward_score:.4f} "
            f"acc={entry.direction_accuracy*100:.1f}% mae={entry.mean_abs_error:.4f}"
        )
    print("\nSURVIVORS")
    for entry in result.survivors:
        print(f"  {entry.name} -> {entry.checkpoint_path}")
    print(f"\nSaved report: {report_path}")


def cmd_sync_actions(args):
    """Fetch corporate actions from RapidAPI and store them locally."""
    from stock_recommender.config import CONFIG, load_config_overrides
    from stock_recommender.data.database import DatabaseManager
    from stock_recommender.data.indian_market_data import sync_corporate_actions_from_rapidapi

    if getattr(args, "config", None):
        load_config_overrides(args.config)

    db_url = getattr(args, "db", None)
    if db_url:
        CONFIG.data.db_url = db_url
    db = DatabaseManager(db_url)

    stock_names = None
    raw_names = getattr(args, "stock_names", None)
    if raw_names:
        stock_names = [s.strip() for s in raw_names.split(",") if s.strip()]

    result = sync_corporate_actions_from_rapidapi(
        db=db,
        stock_names=stock_names,
        rapidapi_key=os.getenv("RAPIDAPI_KEY"),
    )

    print("\nCORPORATE ACTION SYNC")
    print("-" * 60)
    print(f"Synced stocks        : {result.synced_stocks}")
    print(f"Inserted actions     : {result.inserted_actions}")
    print(f"Skipped symbols      : {len(result.skipped_symbols)}")
    if result.skipped_symbols:
        for item in result.skipped_symbols[:10]:
            print(f"  {item}")


def main():
    parser = argparse.ArgumentParser(description="Stock Recommendation System")
    parser.add_argument("command", choices=["seed", "train", "recommend", "analyze", "online", "demo", "backtest4", "tournament", "export_config", "ingest_india", "train_population", "sync_actions"],
                        help="Command to run")
    parser.add_argument(
        "--db",
        default=None,
        help=(
            "PostgreSQL DSN, e.g. postgresql://user:pass@localhost:5432/stock_recommender. "
            "Defaults to STOCK_RECOMMENDER_DB_URL env var or CONFIG.data.db_url."
        ),
    )
    parser.add_argument("--ticker", default="AAPL-SYN", help="Ticker for analyze command")
    parser.add_argument("--config", default=None, help="JSON config override path")
    parser.add_argument("--horizon", type=int, default=None, help="Prediction horizon in trading days")
    parser.add_argument("--step-days", type=int, default=None, help="Walk-forward step size")
    parser.add_argument("--max-windows", type=int, default=None, help="Limit windows per stock for faster tests")
    parser.add_argument("--top-k", type=int, default=None, help="Number of tournament survivors")
    parser.add_argument("--output", default="train_config.json", help="Output path for export_config")
    parser.add_argument("--source", default="internet_or_csv", help="Data source: internet, csv, internet_or_csv")
    parser.add_argument("--data-dir", default=None, help="CSV folder for historical data fallback")
    parser.add_argument("--metadata", default=None, help="Optional metadata CSV/JSON")
    parser.add_argument("--symbols", default=None, help="Comma-separated internet symbols like RELIANCE.NS,TCS.NS")
    parser.add_argument("--start", default=None, help="Historical start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="Historical end date YYYY-MM-DD")
    parser.add_argument("--market-prefix", default="NSE", help="Ticker prefix for DB entries")
    parser.add_argument("--population-size", type=int, default=None, help="Number of candidate models to train")
    parser.add_argument("--population-epochs", type=int, default=None, help="Epochs per candidate model")
    parser.add_argument("--stock-names", default=None, help="Comma-separated stock names/tickers for corporate action sync")
    parser.add_argument("--max-stocks", type=int, default=None, help="Limit the number of stocks used by training/evaluation commands")
    parser.add_argument("--checkpoint", default=None, help="Transformer checkpoint path for backtest/evaluation commands")

    args = parser.parse_args()

    commands = {
        "seed":      cmd_seed,
        "train":     cmd_train,
        "recommend": cmd_recommend,
        "analyze":   cmd_analyze,
        "online":    cmd_demo_online_learning,
        "demo":      cmd_demo,
        "backtest4": cmd_backtest4,
        "tournament": cmd_tournament,
        "export_config": cmd_export_config,
        "ingest_india": cmd_ingest_india,
        "train_population": cmd_train_population,
        "sync_actions": cmd_sync_actions,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
