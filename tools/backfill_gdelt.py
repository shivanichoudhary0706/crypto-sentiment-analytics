"""Backfill raw_sentiment from GDELT's DOC 2.0 API for a given date range.

KNOWN RISK: this project has already documented GDELT returning HTTP 200 with
a plain-text "please limit requests" body (an IP-level quota block) instead of
JSON, on both campus WiFi and a mobile hotspot. This script detects that case
explicitly and will STOP EARLY (after --max-consecutive-blocks worth of days)
rather than grinding through the whole date range for nothing. Treat a 0-row
result as expected, not a bug -- rerun later from a different network if you
get one.

GDELT's DOC API: only reliably covers the last ~3 months, and caps at 250
articles per call, so this script pages one 3-hour window at a time per coin
(a full day can exceed 250 crypto-related articles at busy volume, so daily
chunks risked silently truncating -- 3-hour windows keep well under that cap).

Run from project root:
    python -m tools.backfill_gdelt --start 2026-08-12 --end 2026-09-22

Env vars (same as db_inspect.py / backfill_market.py):
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
"""
import argparse
import os
import random
import time as time_module
from datetime import datetime, timedelta, timezone

import psycopg2
import psycopg2.extras
import requests

BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
MAX_RECORDS = 250  # GDELT's hard cap per call
SECONDS_BETWEEN_REQUESTS = 5  # GDELT asks for ~1 request per 5 seconds
WINDOW_HOURS = 3  # size of each fetch window, replaces the old 1-day chunking
MAX_RETRIES_PER_CALL = 5
RETRY_MIN_SECONDS = 15
RETRY_MAX_SECONDS = 120  # each retry waits a random 15s-2min, not a fixed backoff curve
MAX_CONSECUTIVE_BLOCKS = 3  # stop a coin's run after this many blocked windows in a row

# Match these to your actual coin_keywords in config.yaml -- edit if they differ.
COIN_QUERIES = {
    "BTC/USDT": "bitcoin OR BTC OR btcusdt",
    "ETH/USDT": "ethereum OR ETH OR ethusdt",
}

CONN = dict(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "5432")),
    dbname=os.getenv("DB_NAME", "crypto_analytics"),
    user=os.getenv("DB_USER", "crypto_user"),
    password=os.getenv("DB_PASSWORD", "crypto_pass"),
)

INSERT_SQL = """
    INSERT INTO raw_sentiment
        (time, source_type, source_name, title, summary, url, coin_tags, ingested_at)
    VALUES %s
    ON CONFLICT (time, url) DO NOTHING
"""


class BlockedError(Exception):
    """Raised when GDELT's response looks like a quota block, not real data."""


def gdelt_datetime(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M%S")


def parse_seendate(seendate: str) -> datetime:
    # GDELT format: "20260815T163700Z"
    return datetime.strptime(seendate, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def fetch_window(query: str, window_start: datetime, window_end: datetime) -> list:
    """One window's articles for one coin query. Raises BlockedError if GDELT
    is rate-limiting this IP rather than returning real results."""
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": MAX_RECORDS,
        "sort": "datedesc",
        "startdatetime": gdelt_datetime(window_start),
        "enddatetime": gdelt_datetime(window_end),
    }

    last_error = None
    for attempt in range(MAX_RETRIES_PER_CALL):
        resp = requests.get(BASE_URL, params=params, timeout=20)

        if resp.status_code == 429:
            last_error = BlockedError(f"HTTP 429 on attempt {attempt + 1}")
        else:
            resp.raise_for_status()
            try:
                data = resp.json()
                return data.get("articles", [])
            except ValueError:
                # HTTP 200 but not valid JSON -- this is the documented
                # "Please limit requests..." plain-text block response.
                snippet = resp.text[:120].replace("\n", " ")
                last_error = BlockedError(f"non-JSON 200 response: {snippet!r}")

        # Random delay before retrying (15s-2min) rather than a fixed backoff
        # curve -- avoids retrying in a predictable pattern GDELT's quota
        # logic might key off of.
        sleep_for = random.uniform(RETRY_MIN_SECONDS, RETRY_MAX_SECONDS)
        print(f"    blocked ({last_error}); retrying in {sleep_for:.0f}s "
              f"[{attempt + 1}/{MAX_RETRIES_PER_CALL}]")
        time_module.sleep(sleep_for)

    raise last_error


def backfill_coin(conn, coin: str, query: str, start: datetime, end: datetime) -> tuple:
    """Returns (rows_inserted, consecutive_blocks_at_end)."""
    total_rows = 0
    consecutive_blocks = 0
    window = start
    ingested_at = datetime.now(timezone.utc)

    with conn.cursor() as cur:
        while window < end:
            window_end = min(end, window + timedelta(hours=WINDOW_HOURS))
            label = f"{window.isoformat()} -> {window_end.isoformat()}"
            try:
                articles = fetch_window(query, window, window_end)
                consecutive_blocks = 0
            except BlockedError as e:
                consecutive_blocks += 1
                print(f"  {coin} {label}: BLOCKED after retries ({e})")
                if consecutive_blocks >= MAX_CONSECUTIVE_BLOCKS:
                    print(f"  {consecutive_blocks} consecutive blocked windows -- "
                          f"stopping this coin early. This matches the documented "
                          f"IP-level GDELT quota issue; try a different network later.")
                    return total_rows, consecutive_blocks
                window = window_end
                time_module.sleep(SECONDS_BETWEEN_REQUESTS)
                continue

            rows = []
            for a in articles:
                try:
                    seen = parse_seendate(a["seendate"])
                except (KeyError, ValueError):
                    continue  # skip malformed entries rather than crash the run
                rows.append((
                    seen,
                    "gdelt",
                    a.get("domain", "unknown"),
                    a.get("title", ""),
                    None,  # GDELT's article-list mode has no summary/body field
                    a.get("url", ""),
                    [coin],
                    ingested_at,
                ))

            if rows:
                psycopg2.extras.execute_values(cur, INSERT_SQL, rows)
                conn.commit()
            total_rows += len(rows)
            print(f"  {coin} {label}: +{len(rows)} articles ({total_rows} so far)")

            if len(articles) >= MAX_RECORDS:
                print(f"    WARNING: hit the {MAX_RECORDS}-record cap in this window -- "
                      f"some articles in this window were likely missed. Consider a "
                      f"narrower WINDOW_HOURS if this happens often.")

            window = window_end
            time_module.sleep(SECONDS_BETWEEN_REQUESTS)

    return total_rows, consecutive_blocks


def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD (UTC, inclusive)")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD (UTC, exclusive). Default: now")
    parser.add_argument("--coins", default=None,
                         help="Comma-separated coin labels matching COIN_QUERIES keys, "
                              "e.g. 'BTC/USDT,ETH/USDT'. Default: all configured")
    args = parser.parse_args()

    start = parse_date(args.start)
    end = parse_date(args.end) if args.end else datetime.now(timezone.utc)
    coins = args.coins.split(",") if args.coins else list(COIN_QUERIES.keys())

    print(f"Backfilling GDELT sentiment for {coins} from {start.date()} to {end.date()} (UTC)")
    print("Note: this project has a documented IP-level GDELT block; a 0-row "
          "result is expected, not necessarily a bug.\n")

    conn = psycopg2.connect(**CONN)
    try:
        grand_total = 0
        for coin in coins:
            if coin not in COIN_QUERIES:
                print(f"Skipping unknown coin '{coin}' -- add it to COIN_QUERIES first.")
                continue
            print(f"--- {coin} ---")
            rows, blocks = backfill_coin(conn, coin, COIN_QUERIES[coin], start, end)
            grand_total += rows
            if blocks >= MAX_CONSECUTIVE_BLOCKS:
                print(f"  ({coin} stopped early due to blocking)")
        print(f"\nDone. {grand_total} articles inserted across {len(coins)} coin(s).")
        print("Safe to rerun -- ON CONFLICT (time, url) DO NOTHING skips anything already stored.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()