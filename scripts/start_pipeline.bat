@echo off
REM Starts the live pipeline: infrastructure, then consumers, producers and scorer.
REM Each Python process runs in its own titled window so failures are visible.
cd /d "%~dp0.."
call venv\Scripts\activate

echo Starting infrastructure...
docker compose up -d

echo Waiting for TimescaleDB to accept connections...
:wait_db
docker exec crypto-timescaledb pg_isready -U crypto_user -d crypto_analytics >nul 2>&1
if errorlevel 1 (
    timeout /t 2 /nobreak >nul
    goto wait_db
)
echo TimescaleDB ready. Giving Kafka a few more seconds...
timeout /t 10 /nobreak >nul

REM Consumers first, so nothing produced sits unread.
start "storage-consumers"   cmd /k python -m streaming.run_storage_consumers
start "market-ingestion"    cmd /k python -m ingestion.market.run_market_ingestion
start "sentiment-ingestion" cmd /k python -m ingestion.sentiment.run_sentiment_ingestion
start "sentiment-scoring"   cmd /k python -m sentiment.run_sentiment_scoring
start "signals"             cmd /k python -m signals.run_signals --every 300

echo Pipeline started. 5 windows should be open.