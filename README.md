# Real-Time Crypto Sentiment-Price Analytics

M.Tech thesis project — real-time cryptocurrency analytics system fusing live market data
with sentiment analysis to detect lag relationships between sentiment and price movement,
and to generate trading signals.

## Folder Structure

- ingestion/market/     — Exchange WebSocket/ccxt client -> Kafka producer (raw-market-data)
- ingestion/sentiment/  — Reddit (PRAW) + News RSS collectors -> Kafka producer (raw-sentiment-data)
- streaming/            — Kafka consumers, PySpark Structured Streaming jobs (Week 7+)
- sentiment/            — FinBERT + VADER sentiment scoring engine
- analytics/lag_detection/  — Cross-correlation + Granger causality
- analytics/signals/        — Rule-based buy/hold/sell signal generation
- analytics/backtesting/    — Vectorized backtest engine + performance metrics
- api/                  — FastAPI service
- dashboard/            — Streamlit dashboard (talks to API only, never the DB directly)
- config/               — config.yaml loader, settings management
- tests/                — Unit + integration tests
- docs/                 — Architecture diagrams, decision log
- logs/                 — Runtime logs

## Design principle

Each module has a single responsibility. Ingestion never touches business logic,
analytics never touches Kafka/DB clients directly, and the dashboard never talks
to the database directly — only to the API.

## Status

Repo scaffolded — Week 1, Task 1 of the execution plan.