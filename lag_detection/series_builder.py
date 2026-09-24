"""
Builds the aligned (sentiment, return) dataset for one coin at one bar size.

Alignment convention (state this in the thesis methodology chapter):
  * A bar labelled t covers [t, t + freq).
  * close_t   = close of the last 1-minute candle inside bar t.
  * return_t  = ln(close_t / close_{t-1})  -> price move DURING bar t.
  * sentiment_t = mean article score of articles PUBLISHED during bar t.
  * "sentiment leads by k bars" means sentiment_{t-k} explains return_t.

sentiment_scores.time MUST be the article publish time (not scored_at),
otherwise every lag estimate is shifted by the scoring delay.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import sql

from lag_detection.config import (
    ALLOWED_SENTIMENT_COLUMNS,
    CANDLES_PER_BAR,
    FREQ_TO_PANDAS,
    FREQ_TO_PG_INTERVAL,
    DBConfig,
    LagConfig,
)

logger = logging.getLogger(__name__)

_WINDOW_SQL = """
SELECT min(time), max(time) FROM market_data WHERE symbol = %(symbol)s;
"""

_MARKET_SQL = """
SELECT time_bucket(%(interval)s::interval, time) AS bucket,
       last(close, time)                         AS close,
       count(*)                                  AS n_candles
FROM market_data
WHERE symbol = %(symbol)s
  AND time >= %(start)s
  AND time <  %(end)s
GROUP BY bucket
ORDER BY bucket;
"""

# DISTINCT ON (url): the same story can arrive via RSS *and* CryptoCompare.
# Keep the earliest timestamp so duplicates cannot double-weight a bar.
_SENTIMENT_SQL = sql.SQL("""
WITH dedup AS (
    SELECT DISTINCT ON (url) time, {col}::double precision AS score
    FROM sentiment_scores
    WHERE coin = %(coin)s
      AND time >= %(start)s
      AND time <  %(end)s
      AND {col} IS NOT NULL
    ORDER BY url, time
)
SELECT time_bucket(%(interval)s::interval, time) AS bucket,
       avg(score)  AS sent_mean,
       sum(score)  AS sent_sum,
       count(*)    AS n_articles
FROM dedup
GROUP BY bucket
ORDER BY bucket;
""")


def coin_to_symbol(coin: str) -> str:
    """'BTC/USDT' (sentiment_scores.coin) -> 'BTCUSDT' (market_data.symbol)."""
    return coin.replace("/", "").upper()


def connect(db: DBConfig):
    return psycopg2.connect(
        host=db.host, port=db.port, dbname=db.dbname,
        user=db.user, password=db.password, connect_timeout=10,
    )


def _frame(rows, columns) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return df.set_index(columns[0])
    df[columns[0]] = pd.to_datetime(df[columns[0]], utc=True)
    for c in columns[1:]:
        df[c] = df[c].astype(float)  # NUMERIC arrives as Decimal
    return df.set_index(columns[0])


def fetch_market_bars(conn, coin: str, freq: str, start: datetime, end: datetime) -> pd.DataFrame:
    params = {"interval": FREQ_TO_PG_INTERVAL[freq], "symbol": coin_to_symbol(coin),
              "start": start, "end": end}
    with conn.cursor() as cur:
        cur.execute(_MARKET_SQL, params)
        rows = cur.fetchall()
    return _frame(rows, ["bucket", "close", "n_candles"])


def fetch_sentiment_bars(conn, coin: str, freq: str, start: datetime, end: datetime,
                         column: str) -> pd.DataFrame:
    if column not in ALLOWED_SENTIMENT_COLUMNS:
        raise ValueError(f"Unsupported sentiment column: {column}")
    query = _SENTIMENT_SQL.format(col=sql.Identifier(column))
    params = {"interval": FREQ_TO_PG_INTERVAL[freq], "coin": coin, "start": start, "end": end}
    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()
    return _frame(rows, ["bucket", "sent_mean", "sent_sum", "n_articles"])


def align_series(market: pd.DataFrame, sentiment: pd.DataFrame, freq: str,
                 fill_method: str = "zero") -> tuple[pd.DataFrame, dict]:
    """Pure function (no DB) — unit-testable. Returns (dataset, diagnostics)."""
    if market.empty:
        raise ValueError("No market bars in the requested range")

    full_index = pd.date_range(market.index.min(), market.index.max(), freq=FREQ_TO_PANDAS[freq])
    df = market.reindex(full_index)
    df.index.name = "bucket"
    df["n_candles"] = df["n_candles"].fillna(0)

    expected = CANDLES_PER_BAR[freq]
    # The final bar may still be forming — drop it if incomplete.
    if df["n_candles"].iloc[-1] < expected:
        df = df.iloc[:-1]
    partial_bars = int(((df["n_candles"] > 0) & (df["n_candles"] < expected)).sum())
    empty_bars = int((df["n_candles"] == 0).sum())
    df.loc[df["n_candles"] == 0, "close"] = np.nan

    df["log_return"] = np.log(df["close"]).diff()
    df["abs_return"] = df["log_return"].abs()

    if sentiment.empty:
        sentiment = pd.DataFrame(columns=["sent_mean", "sent_sum", "n_articles"], dtype=float)
    sent = sentiment.reindex(df.index)
    df["n_articles"] = sent["n_articles"].fillna(0).astype(int)
    df["sent_sum"] = sent["sent_sum"].fillna(0.0)
    if fill_method == "zero":      # no news = neutral mood
        df["sentiment"] = sent["sent_mean"].fillna(0.0)
    elif fill_method == "ffill":   # last known mood persists
        df["sentiment"] = sent["sent_mean"].ffill().fillna(0.0)
    else:
        raise ValueError(f"Unknown fill_method: {fill_method}")

    df = df.dropna(subset=["log_return"])
    diagnostics = {
        "n_bars": int(len(df)),
        "first_bar": df.index.min().isoformat() if len(df) else None,
        "last_bar": df.index.max().isoformat() if len(df) else None,
        "partial_price_bars": partial_bars,
        "empty_price_bars": empty_bars,
        "total_articles": int(df["n_articles"].sum()),
        "coverage": float((df["n_articles"] > 0).mean()) if len(df) else 0.0,
    }
    return df, diagnostics


def resolve_window(conn, coin: str, start: datetime | None, end: datetime | None
                   ) -> tuple[datetime, datetime, str]:
    """Analysis window = the span of available price data, unless pinned.

    Returns (start, end_exclusive, source) where source is 'auto', 'pinned' or 'partly pinned'.
    """
    with conn.cursor() as cur:
        cur.execute(_WINDOW_SQL, {"symbol": coin_to_symbol(coin)})
        data_start, data_end = cur.fetchone()
    if data_start is None:
        raise ValueError(f"No market_data rows for {coin} ({coin_to_symbol(coin)})")
    auto_start = data_start.astimezone(timezone.utc)
    auto_end = (data_end + timedelta(minutes=1)).astimezone(timezone.utc)  # exclusive
    w_start, w_end = start or auto_start, end or auto_end
    source = "auto" if not (start or end) else ("pinned" if (start and end) else "partly pinned")
    if w_end <= w_start:
        raise ValueError(f"Empty window for {coin}: {w_start} -> {w_end}")
    return w_start, w_end, source


def build_dataset(conn, cfg: LagConfig, coin: str, freq: str) -> tuple[pd.DataFrame, dict]:
    start, end, source = resolve_window(conn, coin, cfg.start, cfg.end)
    market = fetch_market_bars(conn, coin, freq, start, end)
    sentiment = fetch_sentiment_bars(conn, coin, freq, start, end, cfg.sentiment_column)
    df, diag = align_series(market, sentiment, freq, cfg.fill_method)
    diag.update({"coin": coin, "freq": freq, "sentiment_column": cfg.sentiment_column,
                 "fill_method": cfg.fill_method, "window_source": source,
                 "window_query_start": start.isoformat(), "window_query_end": end.isoformat()})
    logger.info("%s @ %s: %d bars, %d articles, coverage %.1f%%",
                coin, freq, diag["n_bars"], diag["total_articles"], 100 * diag["coverage"])
    if diag["coverage"] < cfg.min_coverage_warn:
        logger.warning("%s @ %s: only %.1f%% of bars contain news — results will be weak; "
                       "consider a coarser frequency", coin, freq, 100 * diag["coverage"])
    return df, diag