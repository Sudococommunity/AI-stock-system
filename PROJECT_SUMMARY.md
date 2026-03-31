# Project Summary

## One-Line Summary

This project is an AI-driven Indian stock forecasting and recommendation prototype designed to analyze market data, predict short-term movement, personalize stock picks, and serve as the intelligence layer for future stock-tip content or video generation.

## Problem

Retail stock content is often:

- generic
- not personalized
- disconnected from actual market data
- weak on risk context

This project aims to solve that by combining:

- market forecasting
- user-aware recommendation
- risk scoring
- narrative analysis

## What The Project Builds

The system ingests Indian stock market history, trains machine learning models to forecast movement, evaluates them using walk-forward testing, and produces structured stock analysis and recommendation outputs.

The long-term product direction is to use these outputs for narrated stock-tip content and AI-assisted stock video generation.

## Core Capabilities

- short-term stock movement forecasting
- personalized recommendation based on user preference and market context
- market regime detection using breadth, trend, and volatility proxies
- checkpointed ML training pipeline
- walk-forward evaluation and model comparison
- support for Indian equity historical data

## Technical Backbone

The project includes:

- a transformer-based forecasting model
- a two-tower retrieval model
- a ranking model for recommendation scoring
- a PostgreSQL-backed data layer
- technical-indicator feature engineering
- walk-forward evaluation and tournament-style candidate selection

## Main Files

- [`main.py`](/D:/AI/hackathon/main.py): command-line entry point
- [`stock_recommender/models/time_series.py`](/D:/AI/hackathon/stock_recommender/models/time_series.py): forecasting model
- [`stock_recommender/models/two_tower.py`](/D:/AI/hackathon/stock_recommender/models/two_tower.py): recommendation models
- [`stock_recommender/recommendation/engine.py`](/D:/AI/hackathon/stock_recommender/recommendation/engine.py): recommendation and analysis output
- [`stock_recommender/market/regime.py`](/D:/AI/hackathon/stock_recommender/market/regime.py): market context engine

## Current Status

### Working today

- historical data ingestion from Indian-market CSVs
- GPU-enabled Python environment
- transformer training path
- checkpoint management
- walk-forward evaluation
- recommendation and analysis logic

### Partially built

- F&O integration
- end-to-end production training workflow
- automated archive downloading

### Not yet implemented

- final video-generation layer
- polished creator workflow for stock-tip publishing

## Best Use Today

The repository is best used today as:

- a forecasting and recommendation prototype
- a technical foundation for stock-analysis automation
- a backend intelligence layer for future stock-tip narration or video systems

## Vision

The final product vision is not just a stock model. It is an AI content engine that can:

- analyze Indian stocks
- pick relevant names based on user profile and market conditions
- generate concise stock commentary
- feed those outputs into narrated stock-tip videos

In other words, this repo is the intelligence core of a future AI stock-tip creator system.

