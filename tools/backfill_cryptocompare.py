"""Backfill raw_sentiment from CryptoCompare's (CCData) News API.

Supports rotating across MULTIPLE free-tier API keys: each key has its own
~100 calls/month quota, so pooling several keys multiplies your effective
budget. When one key gets rate-limited, the script automatically switches to
the next and keeps going -- it only stops once every key is exhausted.

Secrets: API keys are NEVER stored in this file. They are read from the
project's .env file (git-ignored), via one of:
    CRYPTOCOMPARE_API_KEYS=key1,key2,key3   (comma-separated, preferred)
    CRYPTOCOMPARE_API_KEY=key1              (single key, fallback)

Run from project root:
    python -m tools.backfill_cryptocompare --start 2026-08-12 --end 2026-09-22 --coins ETH/USDT
"""
import argparse
import os
import time as time_module
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras
import requests
from dotenv import load_dotenv

# Load .env BEFORE anything reads os.getenv (CONN below is built at import time).
load_dotenv()

BASE_URL = "https://min-api.cryptocompare.com/data/v2/news/"
SECONDS_BETWEEN_REQUESTS = 1.0
MAX_PAGES_PER_COIN = 500  # hard safety cap so a bug can't loop forever
SUMMARY_MAX_CHARS = 600  # store an excerpt, not the full republished article body

# CryptoCompare's news categories -- map to your coin labels.
COIN_CATEGORIES = {
    "BTC/USDT": "BTC",
    "ETH/USDT": "ETH",
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
    ON CONFLICT (time, url) DO UPDATE
    SET coin_tags = (
        SELECT ARRAY(
            SELECT DISTINCT unnest(raw_sentiment.coin_tags || EXCLUDED.coin_tags)
        )
    )
"""
# NOTE: the same article (same time+url) can legitimately be tagged for
# multiple coins -- e.g. CryptoCompare returns one "BTC and ETH both rallied"
# article under both the BTC and ETH category queries. DO UPDATE merges
# coin_tags arrays instead of silently dropping the second coin's tag.


class AllKeysExhaustedError(Exception):
    """Every configured API key is rate-limited or otherwise rejected."""


def mask_key(key: str) -> str:
    """Show only the first/last 4 chars of a key, safe for logs."""
    return key[:4] + "..." + key[-4:] if len(key) > 8 else "***"


class KeyPool:
    """Rotates across multiple API keys, skipping any that come back
    rate-limited, so one call to fetch_page can transparently keep going
    across several keys' quotas without the caller needing to know."""

    def __init__(self, keys: list):
        if not keys:
            raise ValueError("No API keys provided.")
        self.keys = keys
        self.index = 0
        self.exhausted = set()

    def current_key(self) -> str:
        return self.keys[self.index]

    def mark_exhausted(self, reason: str) -> None:
        key = self.keys[self.index]
        self.exhausted.add(key)
        print(f"    Key {self.index + 1}/{len(self.keys)} ({mask_key(key)}) exhausted: {reason}")
        self._advance()

    def _advance(self) -> None:
        for _ in range(len(self.keys)):
            self.index = (self.index + 1) % len(self.keys)
            if self.keys[self.index] not in self.exhausted:
                print(f"    Switching to key {self.index + 1}/{len(self.keys)}")
                return

    def all_exhausted(self) -> bool:
        return len(self.exhausted) >= len(self.keys)


def _is_error_payload(payload: dict) -> tuple:
    """Returns (is_error, message). Checks the actual error shape CryptoCompare
    uses -- {"Response": "Error", "Message": "...", "Type": 99, ...} -- which
    can arrive with HTTP 200 and a non-empty 'Data' key, so checking status
    code or 'Data' presence alone is not enough."""
    if payload.get("Response") == "Error":
        return True, payload.get("Message", "unknown error")
    if payload.get("Type") not in (100, None):
        return True, payload.get("Message", f"unexpected Type={payload.get('Type')}")
    return False, ""


def fetch_page(pool: KeyPool, category: str, before_ts: int | None) -> list:
    params = {"lang": "EN", "categories": category, "sortOrder": "latest"}
    if before_ts is not None:
        params["lTs"] = before_ts

    while not pool.all_exhausted():
        headers = {"authorization": f"Apikey {pool.current_key()}"}
        resp = requests.get(BASE_URL, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
        payload = resp.json()

        is_error, message = _is_error_payload(payload)
        if is_error:
            pool.mark_exhausted(message)
            time_module.sleep(SECONDS_BETWEEN_REQUESTS)
            continue

        return payload.get("Data", [])

    raise AllKeysExhaustedError(
        f"All {len(pool.keys)} API key(s) are rate-limited or rejected. "
        f"Wait for the monthly reset, add more keys, or upgrade a plan."
    )


def backfill_coin(conn, pool: KeyPool, coin: str, category: str,
                  start: datetime, end: datetime) -> int:
    total_rows = 0
    before_ts = int(end.timestamp())
    ingested_at = datetime.now(timezone.utc)
    start_ts = int(start.timestamp())

    with conn.cursor() as cur:
        for page in range(MAX_PAGES_PER_COIN):
            articles = fetch_page(pool, category, before_ts if page > 0 else None)
            if not articles:
                print(f"  {coin}: no more articles returned, stopping.")
                break

            rows = []
            oldest_ts_this_page = None
            for a in articles:
                published_on = a.get("published_on")
                if published_on is None:
                    continue
                if oldest_ts_this_page is None or published_on < oldest_ts_this_page:
                    oldest_ts_this_page = published_on
                if published_on < start_ts:
                    continue
                if published_on > before_ts:
                    continue

                pub_dt = datetime.fromtimestamp(published_on, tz=timezone.utc)
                body = (a.get("body") or "")[:SUMMARY_MAX_CHARS]
                source_name = (a.get("source_info") or {}).get("name") or a.get("source", "unknown")

                rows.append((
                    pub_dt,
                    "cryptocompare",
                    source_name,
                    a.get("title", ""),
                    body,
                    a.get("url", ""),
                    [coin],
                    ingested_at,
                ))

            if rows:
                psycopg2.extras.execute_values(cur, INSERT_SQL, rows)
                conn.commit()
            total_rows += len(rows)

            newest = datetime.fromtimestamp(articles[0]["published_on"], tz=timezone.utc)
            oldest = datetime.fromtimestamp(oldest_ts_this_page, tz=timezone.utc) if oldest_ts_this_page else None
            print(f"  {coin} page {page + 1}: {newest.date()} -> {oldest.date() if oldest else '?'}"
                  f"  (+{len(rows)} in range, {total_rows} total)")

            if oldest_ts_this_page is None or oldest_ts_this_page <= start_ts:
                print(f"  {coin}: reached {start.date()}, stopping.")
                break

            before_ts = oldest_ts_this_page - 1
            time_module.sleep(SECONDS_BETWEEN_REQUESTS)
        else:
            print(f"  {coin}: hit MAX_PAGES_PER_COIN safety cap ({MAX_PAGES_PER_COIN}); "
                  f"stopped without necessarily reaching {start.date()}.")

    return total_rows


def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def load_keys() -> list:
    """Read API keys from environment (populated from .env). Never from code."""
    multi = os.getenv("CRYPTOCOMPARE_API_KEYS", "")
    if multi.strip():
        return [k.strip() for k in multi.split(",") if k.strip()]
    single = os.getenv("CRYPTOCOMPARE_API_KEY", "")
    return [single.strip()] if single.strip() else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD (UTC, inclusive)")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD (UTC). Default: now")
    parser.add_argument("--coins", default=None,
                        help="Comma-separated coin labels matching COIN_CATEGORIES keys. "
                             "Default: all configured")
    args = parser.parse_args()

    keys = load_keys()
    if not keys:
        print("ERROR: no API key(s) found. Add CRYPTOCOMPARE_API_KEYS=key1,key2,... "
              "(or CRYPTOCOMPARE_API_KEY=key) to the .env file in the project root. "
              "Aborting -- an unauthenticated request would just look like an empty "
              "result, not a clear error.")
        return
    print(f"Loaded {len(keys)} API key(s) from environment for rotation.\n")
    pool = KeyPool(keys)

    start = parse_date(args.start)
    end = parse_date(args.end) if args.end else datetime.now(timezone.utc)
    coins = args.coins.split(",") if args.coins else list(COIN_CATEGORIES.keys())

    print(f"Backfilling CryptoCompare news for {coins} from {start.date()} to {end.date()} (UTC)\n")

    conn = psycopg2.connect(**CONN)
    try:
        grand_total = 0
        for coin in coins:
            if coin not in COIN_CATEGORIES:
                print(f"Skipping unknown coin '{coin}' -- add it to COIN_CATEGORIES first.")
                continue
            print(f"--- {coin} ---")
            try:
                grand_total += backfill_coin(conn, pool, coin, COIN_CATEGORIES[coin], start, end)
            except AllKeysExhaustedError as e:
                print(f"  STOPPING: {e}")
                break
        print(f"\nDone. {grand_total} articles inserted across {len(coins)} coin(s) attempted.")
        print("Safe to rerun -- coin_tags merge on conflict means nothing is lost or duplicated.")
    except requests.HTTPError as e:
        print(f"\nHTTP error: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()