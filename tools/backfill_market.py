"""Backfill missing 1-minute candles into market_data from Binance's public
REST klines endpoint. Safe to re-run: duplicates are skipped via
ON CONFLICT (time, symbol) DO NOTHING.

This does NOT need an API key — /api/v3/klines is a public endpoint.

Run from project root:
    python -m tools.backfill_market --start 2026-08-12 --end 2026-09-22
    python -m tools.backfill_market --start 2026-08-12 --symbols BTCUSDT,ETHUSDT

Env vars (same as db_inspect.py): DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
"""
import argparse
import os
import random
import sys
import time as time_module
from datetime import datetime, timedelta, timezone

import psycopg2
import psycopg2.extras
import requests

BASE_URL = "https://api.binance.com/api/v3/klines"
INTERVAL = "1m"
MAX_CANDLES_PER_CALL = 1000  # Binance's hard limit per request
REQUEST_SLEEP_SECONDS = 0.3  # stay well under Binance's weight limit
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]  # falls back to this if --symbols omitted

CONN = dict(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "5432")),
    dbname=os.getenv("DB_NAME", "crypto_analytics"),
    user=os.getenv("DB_USER", "crypto_user"),
    password=os.getenv("DB_PASSWORD", "crypto_pass"),
)

INSERT_SQL = """
    INSERT INTO market_data
        (time, symbol, exchange, open, high, low, close, volume,
         kline_close_time, ingested_at)
    VALUES %s
    ON CONFLICT (time, symbol) DO NOTHING
"""


def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


MAX_ATTEMPTS = 6
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_SESSION = requests.Session()   # reuses the TCP/TLS connection across pages


class BinanceBanError(RuntimeError):
    """HTTP 418: Binance has banned this IP for ignoring rate limits. Never retry."""


def _backoff(attempt: int) -> float:
    """Exponential backoff with jitter: ~1s, 2s, 4s, 8s, 16s (capped at 60s)."""
    return min(60, 2 ** (attempt - 1)) + random.uniform(0, 1)


def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list:
    """One page (<=1000 candles) of raw Binance kline arrays.

    Retries transient failures here, in application code. urllib3's Retry
    (mounted on a Session) would NOT cover a timeout while the response BODY is
    being read -- which is exactly the failure seen on 2026-10-24.
    """
    params = {
        "symbol": symbol,
        "interval": INTERVAL,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": MAX_CANDLES_PER_CALL,
    }
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = _SESSION.get(BASE_URL, params=params, timeout=(5, 30))  # (connect, read)
            if resp.status_code == 418:
                raise BinanceBanError(f"Binance IP ban (HTTP 418): {resp.text[:200]}")
            if resp.status_code in RETRYABLE_STATUS and attempt < MAX_ATTEMPTS:
                wait = float(resp.headers.get("Retry-After", 0) or 0) or _backoff(attempt)
                print(f"    {symbol}: HTTP {resp.status_code}, retry {attempt}/{MAX_ATTEMPTS} in {wait:.1f}s")
                time_module.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as e:
            if attempt == MAX_ATTEMPTS:
                raise
            wait = _backoff(attempt)
            print(f"    {symbol}: {type(e).__name__}, retry {attempt}/{MAX_ATTEMPTS} in {wait:.1f}s")
            time_module.sleep(wait)
    raise RuntimeError(f"{symbol}: exhausted {MAX_ATTEMPTS} attempts")


def backfill_symbol(conn, symbol: str, start: datetime, end: datetime) -> int:
    """Pages through [start, end) for one symbol, inserts as it goes.
    Returns count of candles fetched (not all necessarily new, due to
    ON CONFLICT DO NOTHING)."""
    cursor_time = start
    total_fetched = 0
    ingested_at = datetime.now(timezone.utc)

    with conn.cursor() as cur:
        while cursor_time < end:
            window_end = min(end, cursor_time + timedelta(minutes=MAX_CANDLES_PER_CALL))
            klines = fetch_klines(symbol, to_ms(cursor_time), to_ms(window_end))

            if not klines:
                # No data in this window (symbol didn't exist yet, or a genuine
                # gap on Binance's side) - advance past it rather than looping forever.
                cursor_time = window_end
                time_module.sleep(REQUEST_SLEEP_SECONDS)
                continue

            rows = []
            for k in klines:
                open_time = datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc)
                close_time = datetime.fromtimestamp(k[6] / 1000, tz=timezone.utc)
                rows.append((
                    open_time,
                    symbol,
                    "binance",
                    float(k[1]),  # open
                    float(k[2]),  # high
                    float(k[3]),  # low
                    float(k[4]),  # close
                    float(k[5]),  # volume
                    close_time,
                    ingested_at,
                ))

            psycopg2.extras.execute_values(cur, INSERT_SQL, rows)
            conn.commit()
            total_fetched += len(rows)

            last_open_time = datetime.fromtimestamp(klines[-1][0] / 1000, tz=timezone.utc)
            print(f"  {symbol}: {last_open_time.isoformat()}  (+{len(rows)} candles, "
                  f"{total_fetched} so far)")

            # Advance to just after the last candle we received, so a partial
            # page (fewer than requested) doesn't cause an infinite loop.
            cursor_time = last_open_time + timedelta(minutes=1)
            time_module.sleep(REQUEST_SLEEP_SECONDS)

    return total_fetched


def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD (UTC, inclusive)")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD (UTC, exclusive). Default: now")
    parser.add_argument("--symbols", default=None,
                         help="Comma-separated, e.g. BTCUSDT,ETHUSDT. Default: built-in list")
    args = parser.parse_args()

    start = parse_date(args.start)
    end = parse_date(args.end) if args.end else datetime.now(timezone.utc)
    symbols = args.symbols.split(",") if args.symbols else DEFAULT_SYMBOLS

    print(f"Backfilling {symbols} from {start.date()} to {end.date()} (UTC)\n")

    conn = psycopg2.connect(**CONN)
    failed = []
    try:
        grand_total = 0
        for symbol in symbols:
            print(f"--- {symbol} ---")
            try:
                grand_total += backfill_symbol(conn, symbol, start, end)
            except BinanceBanError:
                raise   # every later symbol would hit the same ban
            except (requests.RequestException, RuntimeError) as e:
                # One symbol failing must not lose the others. Pages already
                # written are committed; a re-run skips them.
                conn.rollback()
                print(f"  {symbol} FAILED: {e}")
                failed.append(symbol)
        print(f"\nDone. {grand_total} candles fetched across {len(symbols)} symbol(s).")
        print("Note: ON CONFLICT DO NOTHING means re-running this is always safe -- "
              "it will only fill in what's still missing.")
    finally:
        conn.close()
    if failed:
        print(f"FAILED: {','.join(failed)}  -> re-run with --symbols {','.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()