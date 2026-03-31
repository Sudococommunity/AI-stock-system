# Architecture Overview

This document explains how the repository is structured from a systems and data-flow perspective.

## System Goals

The system is designed to support:

- short-term stock movement forecasting
- personalized stock recommendation
- Indian-market historical ingestion
- market regime awareness
- future extension into stock-tip content and video generation

## High-Level Architecture

The project has five major layers:

1. Data ingestion and storage
2. Feature engineering
3. Machine learning models
4. Evaluation and selection
5. Recommendation and explanation

## Layer 1: Data Ingestion and Storage

### Main modules

- [`stock_recommender/data/database.py`](/D:/AI/hackathon/stock_recommender/data/database.py)
- [`stock_recommender/data/indian_market_data.py`](/D:/AI/hackathon/stock_recommender/data/indian_market_data.py)
- [`stock_recommender/data/fno_data.py`](/D:/AI/hackathon/stock_recommender/data/fno_data.py)
- [`stock_recommender/data/synthetic_data.py`](/D:/AI/hackathon/stock_recommender/data/synthetic_data.py)
- [`stock_recommender/data/synthetic_bot_data.py`](/D:/AI/hackathon/stock_recommender/data/synthetic_bot_data.py)

### Responsibilities

- store stock universe and OHLCV history in PostgreSQL
- store user profiles and user events
- store learned embeddings and checkpoints
- ingest Indian equity history from CSV or Yahoo Finance
- ingest benchmark indices and India VIX
- store initial F&O snapshots
- generate synthetic users and synthetic interaction histories for demo mode

### Database shape

The database stores:

- `stocks`
- `price_history`
- `users`
- `user_events`
- `user_embeddings`
- `stock_embeddings`
- `recommendation_log`
- `corporate_actions`
- `model_checkpoints`
- `training_metrics`
- `fno_snapshots`

## Layer 2: Feature Engineering

### Main modules

- [`stock_recommender/features/technical_indicators.py`](/D:/AI/hackathon/stock_recommender/features/technical_indicators.py)
- [`stock_recommender/features/feature_pipeline.py`](/D:/AI/hackathon/stock_recommender/features/feature_pipeline.py)
- [`stock_recommender/features/tensor_preprocessing.py`](/D:/AI/hackathon/stock_recommender/features/tensor_preprocessing.py)

### Responsibilities

- convert OHLCV history into dense technical features
- normalize features across the market universe
- build sliding training windows for the transformer
- produce latest sequence and latest snapshot representations for inference

### Output forms

The feature layer feeds two different model styles:

- sequence tensors for the forecasting model
- single-timestep snapshots for the stock tower and ranker context

## Layer 3: Machine Learning Models

### Forecasting model

- [`stock_recommender/models/time_series.py`](/D:/AI/hackathon/stock_recommender/models/time_series.py)

Main model:

- `StockTransformer`

Outputs:

- 1-day return forecast
- multi-day return forecast
- direction logits
- learned sequence embedding

This is the main predictive model for short-term market movement.

### Recommendation models

- [`stock_recommender/models/two_tower.py`](/D:/AI/hackathon/stock_recommender/models/two_tower.py)

Main models:

- `UserTower`
- `StockTower`
- `TwoTowerModel`
- `RankingModel`
- `CandidateIndex`

Responsibilities:

- encode users into embedding vectors
- encode stocks into embedding vectors
- retrieve candidate stocks quickly
- rank candidates using richer context

### Risk layer

- [`stock_recommender/risk/risk_metrics.py`](/D:/AI/hackathon/stock_recommender/risk/risk_metrics.py)

This computes:

- Sharpe
- Sortino
- volatility
- VaR / CVaR
- beta
- risk score
- opportunity score

## Layer 4: Training and Evaluation

### Main modules

- [`stock_recommender/learning/trainer.py`](/D:/AI/hackathon/stock_recommender/learning/trainer.py)
- [`stock_recommender/learning/population_trainer.py`](/D:/AI/hackathon/stock_recommender/learning/population_trainer.py)
- [`stock_recommender/learning/online_learner.py`](/D:/AI/hackathon/stock_recommender/learning/online_learner.py)
- [`stock_recommender/evaluation/walk_forward.py`](/D:/AI/hackathon/stock_recommender/evaluation/walk_forward.py)
- [`stock_recommender/evaluation/tournament.py`](/D:/AI/hackathon/stock_recommender/evaluation/tournament.py)
- [`stock_recommender/evaluation/scoring.py`](/D:/AI/hackathon/stock_recommender/evaluation/scoring.py)

### Training modes

There are two distinct training modes.

#### 1. Full recommender training

This includes:

- transformer training
- two-tower training
- ranking-model training

Used mainly for the synthetic/demo workflow.

#### 2. Population forecasting training

This is the main Indian-market training workflow.

It:

- creates multiple forecasting candidates
- trains them on a training subset of stocks
- evaluates them on a held-out stock subset
- keeps the best-performing checkpoints

### Evaluation style

Evaluation is based on:

- walk-forward forecasting
- out-of-sample candidate comparison
- reward-score based ranking

This is more realistic than evaluating on a random time split.

## Layer 5: Recommendation and Explanation

### Main modules

- [`stock_recommender/recommendation/engine.py`](/D:/AI/hackathon/stock_recommender/recommendation/engine.py)
- [`stock_recommender/market/regime.py`](/D:/AI/hackathon/stock_recommender/market/regime.py)
- [`stock_recommender/data/user_tracker.py`](/D:/AI/hackathon/stock_recommender/data/user_tracker.py)

### Recommendation flow

The recommendation engine works as follows:

1. get or compute user embedding
2. retrieve candidate stocks from the ANN index
3. compute stock embeddings
4. compute forecasts and risk context
5. rank candidates using the ranking model
6. apply diversity and risk adjustments
7. return recommendations plus explanation fields

### Explanation output

The engine can produce:

- recommendation lists
- per-stock full analysis
- key technical signals
- market regime notes
- position sizing and risk guidance

## End-to-End Data Flow

### Indian forecasting path

1. CSV history is stored in [`india_universe_data/`](/D:/AI/hackathon/india_universe_data)
2. ingestion loads OHLCV into PostgreSQL
3. feature pipeline computes technical features
4. transformer training creates checkpoints
5. walk-forward evaluation scores checkpoints
6. recommendation and analysis can consume model outputs

### Demo recommender path

1. synthetic stocks and users are created
2. synthetic user events are generated
3. recommender models are trained
4. recommendation engine generates candidate stocks for demo users

## Current Architectural Strengths

- clear module separation
- real forecasting model plus recommendation stack
- walk-forward evaluation exists
- GPU-aware training path exists
- Indian-market data ingestion exists

## Current Architectural Gaps

- PostgreSQL is external and not provisioned by the repo
- F&O is only partially integrated
- video/content generation layer is not implemented yet
- NSE archive automation is still experimental

## Future Extension: Stock-Tip Content Generation

The natural extension of the current architecture is:

1. trained recommendation engine generates structured recommendation payload
2. payload is converted into short scripts and subtitles
3. narration is recorded or synthesized
4. template-based or AI-assisted video renderer assembles the final stock-tip clip

The current repository already provides most of the forecasting and recommendation side of that flow.

