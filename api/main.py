"""
Read-only REST layer over TimescaleDB and reports/, consumed by the Streamlit dashboard.
Run from project root:   uvicorn api.main:app --reload --port 8000
Interactive docs:        http://localhost:8000/docs
"""
import csv
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import psycopg2
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from api import db

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger("api")

REPORTS = Path(__file__).resolve().parents[1] / "reports"
ES_DIR = REPORTS / "event_study"

# stage -> (SQL returning one column "latest", max acceptable lag in minutes)
FRESHNESS = {
    "market_ingestion":  ("SELECT MAX(time) AS latest FROM market_data", 5),
    "news_ingestion":    ("SELECT MAX(time) AS latest FROM raw_sentiment", 240),
    "sentiment_scoring": ("SELECT MAX(time) AS latest FROM sentiment_scores", 240),
    "signals":           ("SELECT MAX(bucket) AS latest FROM signals", 150),
}


# ---------- helpers ----------
def to_symbol(coin: str) -> str:
    """'sol', 'SOL/USDT', 'solusdt' -> 'SOLUSDT' (market_data / signals format)."""
    s = coin.upper().replace("/", "").replace("-", "").strip()
    s = s if s.endswith("USDT") else f"{s}USDT"
    if not s.isalnum():  # also blocks path tricks like '..\\' in file-based endpoints
        raise HTTPException(400, f"invalid coin: {coin!r}")
    return s


def to_pair(coin: str) -> str:
    """'btc', 'BTCUSDT', 'market' -> 'BTC/USDT' / 'MARKET' (sentiment_scores format)."""
    if coin.strip().upper() == "MARKET":
        return "MARKET"
    return f"{to_symbol(coin)[:-4]}/USDT"


def _num(v: str):
    if v in ("True", "False"):
        return v == "True"
    for cast in (int, float):
        try:
            return cast(v)
        except (TypeError, ValueError):
            pass
    return v


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise HTTPException(404, f"{path.name} not found - run the job that produces it")
    with path.open(newline="", encoding="utf-8") as f:
        return [{k: _num(v) for k, v in row.items()} for row in csv.DictReader(f)]


def read_json(path: Path):
    if not path.exists():
        raise HTTPException(404, f"{path.name} not found - run the job that produces it")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------- app ----------
@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        db.init_pool()
    except psycopg2.OperationalError as exc:
        logger.warning("DB not reachable at startup (%s); will retry on first request", exc)
    yield
    db.close_pool()


app = FastAPI(title="Crypto Signal Desk API", version="1.0.0", lifespan=lifespan)


@app.exception_handler(psycopg2.OperationalError)
async def db_unavailable(_request, exc):
    logger.error("Database unavailable: %s", exc)
    return JSONResponse(status_code=503, content={"detail": "database unavailable"})


@app.get("/health")
def health():
    now = datetime.now(timezone.utc)
    stages = []
    for stage, (sql, max_lag) in FRESHNESS.items():
        latest = db.fetch_all(sql)[0]["latest"]
        lag = None if latest is None else round((now - latest).total_seconds() / 60, 1)
        stages.append({"stage": stage, "latest": latest, "lag_minutes": lag,
                       "max_lag_minutes": max_lag, "ok": lag is not None and lag <= max_lag})
    return {"checked_at": now, "ok": all(s["ok"] for s in stages), "stages": stages}


@app.get("/signals/latest")
def signals_latest(freq: str = "1h"):
    rows = db.fetch_all("""
        SELECT DISTINCT ON (symbol) symbol, coin, bucket, valid_from, freq, horizon_bars,
               close, prob_up, action, model_version, n_train, created_at
        FROM signals
        WHERE freq = %s
        ORDER BY symbol, bucket DESC""", (freq,))
    if not rows:
        raise HTTPException(404, f"no signals for freq={freq}")
    return rows


@app.get("/signals/history")
def signals_history(coin: str, hours: int = Query(96, ge=1, le=1440), freq: str = "1h"):
    symbol = to_symbol(coin)
    rows = db.fetch_all("""
        SELECT bucket, close, prob_up, action, model_version
        FROM signals
        WHERE symbol = %s AND freq = %s AND bucket >= NOW() - %s * INTERVAL '1 hour'
        ORDER BY bucket""", (symbol, freq, hours))
    return {"symbol": symbol, "freq": freq, "hours": hours, "rows": rows}


@app.get("/news/recent")
def news_recent(limit: int = Query(20, ge=1, le=200), coin: Optional[str] = None):
    return db.fetch_all("""
        SELECT s.time, s.coin, s.source_name, s.finbert_label,
               s.finbert_compound, s.vader_compound, r.title, s.url
        FROM (SELECT * FROM sentiment_scores
              WHERE %(coin)s IS NULL OR coin = %(coin)s
              ORDER BY time DESC LIMIT %(limit)s) s
        LEFT JOIN raw_sentiment r ON r.url = s.url AND r.time = s.time
        ORDER BY s.time DESC""", {"coin": to_pair(coin) if coin else None, "limit": limit})


@app.get("/backtest/summary")
def backtest_summary():
    run_file = REPORTS / "backtest" / "run.json"
    return {"run": read_json(run_file) if run_file.exists() else None,
            "rows": read_csv(REPORTS / "backtest" / "summary.csv")}


@app.get("/backtest/tuning")
def backtest_tuning():
    d = REPORTS / "tuning"
    return {"run": read_json(d / "run.json"),
            "choose": read_csv(d / "choose.csv"),
            "confirm": read_csv(d / "confirm.csv")}


@app.get("/event-study")
def event_study_index():
    pairs = []
    for f in sorted(ES_DIR.glob("*_to_*_summary.json")):
        cause, rest = f.stem.removesuffix("_summary").split("_to_", 1)
        target, score = rest.split("_", 1)
        pairs.append({"cause": cause, "target": target, "score": score})
    return pairs


@app.get("/event-study/{cause}/{target}")
def event_study(cause: str, target: str, score: str = "finbert_compound"):
    if not score.replace("_", "").isalnum():
        raise HTTPException(400, f"invalid score: {score!r}")
    prefix = f"{to_symbol(cause)}_to_{to_symbol(target)}_{score}"
    out = {"summary": read_json(ES_DIR / f"{prefix}_summary.json")}
    for side in ("positive", "negative"):
        p = ES_DIR / f"{prefix}_{side}_curve.csv"
        out[f"{side}_curve"] = read_csv(p) if p.exists() else []
    return out

CONFIG_FILE = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


@app.get("/meta")
def meta():
    """Live signal settings, read from config.yaml so the UI never hardcodes them."""
    cfg = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8")).get("signal_model", {})
    return {"freq": cfg.get("freq", "1h"),
            "targets": cfg.get("targets", []),
            "input_coins": cfg.get("input_coins", []),
            "horizon_bars": cfg.get("horizon_bars"),
            "buy_threshold": cfg.get("buy_threshold"),
            "sell_threshold": cfg.get("sell_threshold"),
            "fee_bps": cfg.get("backtest", {}).get("fee_bps")}