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


def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list:
    """One page (<=1000 candles) of raw Binance kline arrays."""
    params = {
        "symbol": symbol,
        "interval": INTERVAL,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": MAX_CANDLES_PER_CALL,
    }
    resp = requests.get(BASE_URL, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


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
    try:
        grand_total = 0
        for symbol in symbols:
            print(f"--- {symbol} ---")
            grand_total += backfill_symbol(conn, symbol, start, end)
        print(f"\nDone. {grand_total} candles fetched across {len(symbols)} symbol(s).")
        print("Note: ON CONFLICT DO NOTHING means re-running this is always safe -- "
              "it will only fill in what's still missing.")
    except requests.HTTPError as e:
        print(f"\nBinance API error: {e}")
        print("Common causes: bad symbol name, or startTime after endTime.")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()