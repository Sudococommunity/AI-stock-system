# Stock Recommender and Market Forecasting Prototype

This repository is a prototype machine learning system for:

- short-horizon stock movement forecasting
- personalized stock recommendation
- Indian-market data ingestion and evaluation
- market regime analysis
- future extension into narrated stock-tip video generation

The codebase combines a forecasting model, a retrieval-and-ranking recommendation stack, data ingestion for Indian equities, and walk-forward evaluation utilities.

## What The Project Does

The repository currently supports two main workflows:

1. Demo recommender flow using synthetic users and interaction history
2. Forecast training on Indian historical market data using a transformer population-training pipeline

At a high level:

- Indian OHLCV history is ingested into PostgreSQL
- features are built from technical indicators
- a transformer predicts short-term returns and direction
- a two-tower model retrieves candidate stocks for a user
- a ranking model scores candidates using forecast and risk context
- walk-forward backtesting grades transformer checkpoints

## Repository Layout

### Core package

- [`stock_recommender/config.py`](/D:/AI/hackathon/stock_recommender/config.py): global configuration dataclasses
- [`stock_recommender/data`](/D:/AI/hackathon/stock_recommender/data): database layer, ingestion, synthetic data, F&O helpers
- [`stock_recommender/features`](/D:/AI/hackathon/stock_recommender/features): technical indicators and model input pipelines
- [`stock_recommender/models`](/D:/AI/hackathon/stock_recommender/models): forecasting, retrieval, and ranking models
- [`stock_recommender/learning`](/D:/AI/hackathon/stock_recommender/learning): training loops and population training
- [`stock_recommender/evaluation`](/D:/AI/hackathon/stock_recommender/evaluation): walk-forward scoring and tournaments
- [`stock_recommender/market`](/D:/AI/hackathon/stock_recommender/market): market regime analysis
- [`stock_recommender/recommendation`](/D:/AI/hackathon/stock_recommender/recommendation): recommendation engine
- [`stock_recommender/risk`](/D:/AI/hackathon/stock_recommender/risk): risk and portfolio metrics

### Entry points and scripts

- [`main.py`](/D:/AI/hackathon/main.py): main CLI for seeding, training, recommendation, analysis, ingestion, and evaluation
- [`scripts/train_india_local.ps1`](/D:/AI/hackathon/scripts/train_india_local.ps1): bounded local training wrapper
- [`scripts/rebuild_env.ps1`](/D:/AI/hackathon/scripts/rebuild_env.ps1): portable environment rebuild script
- [`scripts/download_nse_fo_archives.py`](/D:/AI/hackathon/scripts/download_nse_fo_archives.py): prototype NSE archive downloader

### Data and artifacts

- [`india_universe_data/`](/D:/AI/hackathon/india_universe_data): historical NSE CSV data used for training
- [`data/`](/D:/AI/hackathon/data): static market/universe reference data, not primary ML training data
- [`checkpoints/`](/D:/AI/hackathon/checkpoints): saved model checkpoints
- [`artifacts/`](/D:/AI/hackathon/artifacts): reports and tournament outputs
- [`tests/`](/D:/AI/hackathon/tests): unit tests and training-path tests

## Main ML Components

### Forecasting model

The primary forecasting model is:

- [`stock_recommender/models/time_series.py`](/D:/AI/hackathon/stock_recommender/models/time_series.py): `StockTransformer`

It predicts:

- next 1-day return
- next multi-day return
- movement direction probabilities

This is the main model for short-horizon price movement prediction.

### Recommendation models

The recommendation stack is implemented in:

- [`stock_recommender/models/two_tower.py`](/D:/AI/hackathon/stock_recommender/models/two_tower.py): `TwoTowerModel`, `RankingModel`, `CandidateIndex`

This gives the system:

- user embedding generation
- stock embedding generation
- approximate candidate retrieval
- pairwise ranking for final recommendations

### Training loops

Training is handled by:

- [`stock_recommender/learning/trainer.py`](/D:/AI/hackathon/stock_recommender/learning/trainer.py): full trainer for transformer, towers, and ranker
- [`stock_recommender/learning/population_trainer.py`](/D:/AI/hackathon/stock_recommender/learning/population_trainer.py): multi-candidate forecasting tournament

### Recommendation orchestration

End-user recommendations and analysis are handled in:

- [`stock_recommender/recommendation/engine.py`](/D:/AI/hackathon/stock_recommender/recommendation/engine.py): `RecommendationEngine`

This module combines:

- retrieval
- ranking
- market regime
- risk scoring
- narrative analysis output

### Market context

Market condition analysis is implemented in:

- [`stock_recommender/market/regime.py`](/D:/AI/hackathon/stock_recommender/market/regime.py): `MarketRegimeAnalyzer`

It derives:

- hot / neutral / cold market temperature
- bull / bear / sideways regime
- breadth
- VIX context
- sector leadership

## Data Model

### Primary training data

The real training base for forecasting is:

- [`india_universe_data/`](/D:/AI/hackathon/india_universe_data)

This contains per-symbol OHLCV CSV files and metadata for Indian equities.

### Database

The system now uses PostgreSQL only.

Database logic lives in:

- [`stock_recommender/data/database.py`](/D:/AI/hackathon/stock_recommender/data/database.py)

It stores:

- stocks
- price history
- users
- user events
- embeddings
- checkpoints
- training metrics
- corporate actions
- F&O snapshots

### F&O status

The repo includes initial F&O support in:

- [`stock_recommender/data/fno_data.py`](/D:/AI/hackathon/stock_recommender/data/fno_data.py)
- [`stock_recommender/features/technical_indicators.py`](/D:/AI/hackathon/stock_recommender/features/technical_indicators.py)

Current status:

- F&O ingestion/storage exists
- a few derivative-related features exist
- F&O is not yet deeply integrated into the full forecasting and recommendation training pipeline

## Environment Setup

### Rebuild the local environment

If `.venv` is stale or was created on another machine:

```powershell
.\scripts\rebuild_env.ps1
```

Optional:

```powershell
.\scripts\rebuild_env.ps1 -InstallNodeDeps
.\scripts\rebuild_env.ps1 -PythonExe py
.\scripts\rebuild_env.ps1 -TorchIndexUrl https://download.pytorch.org/whl/cu128
```

### Python dependencies

Main Python dependencies are declared in:

- [`requirements.txt`](/D:/AI/hackathon/requirements.txt)

Important packages:

- `torch`
- `pandas`
- `scikit-learn`
- `yfinance`
- `psycopg2-binary`
- `selenium`

### GPU

The training code supports CUDA and automatically uses GPU when available.

Relevant logic:

- [`stock_recommender/learning/trainer.py`](/D:/AI/hackathon/stock_recommender/learning/trainer.py)

This includes:

- CUDA device selection
- AMP
- optional `torch.compile`
- pinned-memory dataloaders

## PostgreSQL Setup

Set the DSN before running any command:

```powershell
$env:STOCK_RECOMMENDER_DB_URL = "postgresql://postgres:postgres@localhost:5432/stock_recommender"
```

Or pass `--db` directly to the CLI.

The project expects a running PostgreSQL instance. If PostgreSQL is not running, training and ingestion will fail on connection setup.

## Main Commands

### Seed synthetic demo data

```powershell
.\.venv\Scripts\python.exe main.py seed
```

### Full synthetic training demo

```powershell
.\.venv\Scripts\python.exe main.py train
```

### Generate recommendations

```powershell
.\.venv\Scripts\python.exe main.py recommend
```

### Analyze a stock

```powershell
.\.venv\Scripts\python.exe main.py analyze --ticker NSE:RELIANCE
```

### Ingest Indian market history from CSV

```powershell
.\.venv\Scripts\python.exe main.py ingest_india --config configs/india_local_training.json --source csv --data-dir india_universe_data --metadata india_universe_data/_metadata.json
```

### Train forecasting population

```powershell
.\.venv\Scripts\python.exe main.py train_population --config configs/india_local_training.json --population-size 3 --population-epochs 2 --max-stocks 200 --max-windows 3 --top-k 1
```

### Walk-forward evaluation

```powershell
.\.venv\Scripts\python.exe main.py backtest4 --config configs/india_local_training.json --max-stocks 200 --max-windows 3
```

### Local bounded training wrapper

```powershell
.\scripts\train_india_local.ps1 -MaxStocks 25 -PopulationSize 1 -PopulationEpochs 1 -MaxWindows 1
```

## Evaluation Strategy

The forecasting path uses walk-forward evaluation and tournament-style checkpoint grading.

Relevant modules:

- [`stock_recommender/evaluation/walk_forward.py`](/D:/AI/hackathon/stock_recommender/evaluation/walk_forward.py)
- [`stock_recommender/evaluation/scoring.py`](/D:/AI/hackathon/stock_recommender/evaluation/scoring.py)
- [`stock_recommender/evaluation/tournament.py`](/D:/AI/hackathon/stock_recommender/evaluation/tournament.py)

The population trainer:

- trains multiple candidate transformers
- evaluates them out-of-sample
- ranks them by reward score, direction accuracy, and MAE

## Testing

Tests live in:

- [`tests/`](/D:/AI/hackathon/tests)

The test suite covers:

- training-path logic
- walk-forward evaluation
- market regime analysis
- recommendation engine behavior
- Indian market ingestion

The tests were migrated to PostgreSQL-oriented setup helpers in:

- [`tests/postgres_test_utils.py`](/D:/AI/hackathon/tests/postgres_test_utils.py)

## Current Status

### Stronger areas

- forecasting transformer exists and is trainable
- recommendation retrieval and ranking stack exists
- Indian historical CSV ingestion exists
- walk-forward evaluation exists
- GPU training path exists

### Weaker areas

- F&O is not yet fully integrated into the end-to-end ML pipeline
- PostgreSQL must be running externally; the repo does not provision it
- some workflows are more prototype-grade than production-grade
- NSE archive automation is still experimental

## Recommended Prototype Demo Flow

For a prototype demo:

1. rebuild the environment
2. confirm GPU-enabled PyTorch
3. ingest Indian data into PostgreSQL
4. run a bounded `train_population`
5. run `backtest4`
6. run `analyze` or `recommend`

This gives:

- a trained checkpoint
- a measurable backtest result
- a recommendation or analysis output suitable for downstream script/video generation

## Vision Beyond This Repo

The long-term product direction described in project work is:

- stock movement prediction
- personalized recommendation
- market regime explanation
- narrative generation
- human-voice or AI-voice stock-tip video generation

The current repository already provides most of the forecasting and recommendation backbone needed for that, but the video-generation layer itself is not yet implemented inside this repo.

## Related Documents

- [`TRAINING.md`](/D:/AI/hackathon/TRAINING.md): practical training setup and local commands

