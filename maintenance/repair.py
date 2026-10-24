"""
Repair steps for tools/health_check.py, run in dependency order:

    market backfill -> news backfill -> re-tag -> score -> features -> signals -> reports

Every step is idempotent (ON CONFLICT upserts, deterministic re-tagging and
scoring), so an interrupted run is fixed by simply running it again.
Each step returns a one-line summary for the final report.
"""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone

import requests
from psycopg2.extras import execute_batch

from config.logging_config import get_logger
from maintenance.checks import PROJECT_ROOT, ReportSpec, State
from maintenance.config import MaintenanceConfig
from maintenance.ranges import coalesce, recent_day_ranges

logger = get_logger("health_check")


def backfill_market(conn, st: State) -> str:
    from tools import backfill_market as bm

    total, failed, unreachable = 0, [], 0
    join = timedelta(minutes=bm.MAX_CANDLES_PER_CALL)   # one API page
    for symbol, gaps in st.market_gaps.items():
        if not gaps:
            continue
        if unreachable >= 2:
            # Two coins in a row could not reach Binance at all: the network is
            # down, so retrying every other coin would only waste minutes.
            failed.append(symbol)
            continue
        fetched_before = total
        for start, end in coalesce(gaps, join):
            logger.info("Market backfill %s: %s -> %s", symbol, f"{start:%Y-%m-%d %H:%M}", f"{end:%Y-%m-%d %H:%M}")
            try:
                total += bm.backfill_symbol(conn, symbol, start, end)
            except bm.BinanceBanError:
                raise   # every further call would extend the ban
            except (requests.RequestException, RuntimeError) as exc:
                conn.rollback()   # pages already written stay committed
                logger.error("Market backfill %s failed: %s", symbol, str(exc)[:200])
                failed.append(symbol)
                no_progress = total == fetched_before
                is_network = isinstance(exc, (requests.ConnectionError, requests.Timeout))
                unreachable = unreachable + 1 if (no_progress and is_network) else 0
                break
    summary = f"{total} candles fetched"
    if failed:
        summary += f", FAILED for {','.join(failed)}"
        if unreachable >= 2:
            summary += " (Binance unreachable - check internet, then re-run)"
    return summary


def backfill_news(conn, st: State, since: date | None = None) -> str:
    """since=None: every gap. Otherwise only the days on/after `since` (quota-friendly)."""
    from tools import backfill_cryptocompare as cc

    keys = cc.load_keys()
    if not keys:
        return "skipped: no CRYPTOCOMPARE_API_KEYS in .env"
    pool = cc.KeyPool(keys)
    total = 0
    for coin, runs in st.news_gaps.items():
        if since is not None:
            runs = recent_day_ranges(runs, since)
        if not runs:
            continue
        category = cc.COIN_CATEGORIES.get(coin)
        if category is None:
            logger.warning("No CryptoCompare category for %s - add it to COIN_CATEGORIES", coin)
            continue
        for first_day, day_after in sorted(runs, reverse=True):   # newest gaps first
            start = datetime(first_day.year, first_day.month, first_day.day, tzinfo=timezone.utc)
            end = datetime(day_after.year, day_after.month, day_after.day, tzinfo=timezone.utc)
            logger.info("News backfill %s: %s -> %s", coin, first_day, day_after - timedelta(days=1))
            try:
                total += cc.backfill_coin(conn, pool, coin, category, start, end)
            except cc.AllKeysExhaustedError as exc:
                return f"{total} articles inserted, then stopped: {exc}"
    return f"{total} articles inserted"


def retag(conn, st: State) -> str:
    """Write the current tagger's tags and drop the old scores of those articles,
    so the scorer re-scores them for the right coins. Update + delete commit
    together, so the live scorer never sees new tags with old scores."""
    if not st.retag:
        return "nothing to do"
    for i in range(0, len(st.retag), 500):
        batch = st.retag[i:i + 500]
        with conn.cursor() as cur:
            execute_batch(cur, "UPDATE raw_sentiment SET coin_tags = %s WHERE time = %s AND url = %s",
                          [(tags, t, url) for t, url, tags in batch])
            execute_batch(cur, "DELETE FROM sentiment_scores WHERE time = %s AND url = %s",
                          [(t, url) for t, url, _ in batch])
        conn.commit()
    return f"{len(st.retag)} articles re-tagged (old scores removed for re-scoring)"


def score(conn) -> str:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM raw_sentiment rs WHERE rs.coin_tags != '{}' AND NOT EXISTS "
                    "(SELECT 1 FROM sentiment_scores ss WHERE ss.url = rs.url AND ss.time = rs.time)")
        todo = cur.fetchone()[0]
    conn.commit()
    if not todo:
        return "nothing to do"
    # Imported only now: loading torch/transformers takes seconds.
    from sentiment import run_sentiment_scoring as scorer
    logger.info("Scoring %d articles with FinBERT + VADER ...", todo)
    done = written = batches = 0
    max_batches = todo // scorer.BATCH_SIZE + 5   # guard against a non-draining queue
    t0 = time.perf_counter()
    while batches < max_batches:
        articles = scorer._fetch_unscored_articles(conn, scorer.BATCH_SIZE)
        if not articles:
            break
        written += scorer._score_and_store(conn, articles)
        conn.commit()
        done += len(articles)
        batches += 1
        if batches % 10 == 0:
            logger.info("  scored %d/%d articles (%.0fs)", done, todo, time.perf_counter() - t0)
    return f"{done} articles scored -> {written} coin rows"


def _lock_features(conn, wait_s: int) -> bool:
    from features import store

    deadline = time.monotonic() + wait_s
    while True:
        if store.try_lock(conn):
            return True
        if time.monotonic() >= deadline:
            return False
        logger.info("Feature lock held by the live signal loop - waiting ...")
        time.sleep(5)


def rebuild_features(conn, st: State, cfg: MaintenanceConfig) -> str:
    from features import store
    from features.config import load_config
    from features.run_feature_job import run_once
    from features.transforms import feature_columns

    fcfg = load_config()
    added = store.ensure_schema(conn, fcfg.table, feature_columns(tuple(fcfg.news_streams)))
    full = st.features_full_refresh or bool(added)
    # Hold the job's own lock (advisory locks are re-entrant per session), so the
    # passes below cannot silently skip because the live loop is mid-pass.
    if not _lock_features(conn, cfg.feature_lock_wait_seconds):
        return f"skipped: feature lock still held after {cfg.feature_lock_wait_seconds}s"
    try:
        n = 0
        if full:
            n += run_once(conn, fcfg, full_refresh=True)
            what = "full rebuild"
        elif st.features_rebuild_from is not None:
            n += run_once(conn, fcfg, start=st.features_rebuild_from)
            what = f"rebuilt from {st.features_rebuild_from:%Y-%m-%d %H:%M}"
        else:
            n += run_once(conn, fcfg)
            what = "incremental pass"
        return f"{what}, {n} bars written"
    finally:
        store.unlock(conn)


def generate_signals(conn) -> str:
    from signals.config import load_config
    from signals.run_signals import run_once

    n = run_once(conn, load_config(), with_features=False)   # features were just refreshed
    return f"{n} signals written (model retrained on latest history)"


def run_reports(specs: list[ReportSpec]) -> str:
    results = []
    for spec in specs:
        logger.info("Running %s (python -m %s) ...", spec.name, spec.module)
        rc = subprocess.run([sys.executable, "-m", spec.module], cwd=PROJECT_ROOT).returncode
        results.append(f"{spec.name} {'OK' if rc == 0 else f'FAILED (exit {rc})'}")
    return "; ".join(results) if results else "nothing to do"


def run_tests() -> tuple[bool, str]:
    """Unit tests gate the repairs: broken code must not rewrite the data."""
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                          "--continue-on-collection-errors"],
                         cwd=PROJECT_ROOT, capture_output=True, text=True)
    lines = [l for l in out.stdout.strip().splitlines() if l.strip()]
    summary = lines[-1] if lines else (out.stderr.strip().splitlines() or ["no output"])[-1]
    if out.returncode != 0:
        print("\n".join(lines[-25:]))
    return out.returncode == 0, summary.strip("= ")
