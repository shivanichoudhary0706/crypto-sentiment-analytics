"""
Feature job: market_data + sentiment_scores -> features_bar.

Run from the project root:
    python -m features.run_feature_job                     # one incremental run
    python -m features.run_feature_job --full-refresh      # rebuild everything
    python -m features.run_feature_job --every 300         # incremental run every 5 min
    python -m features.run_feature_job --start 2026-06-01 --end 2026-09-27   # rebuild a window

Incremental mode re-computes the last `incremental_lookback_bars` bars of each
frequency (late candles, the bar that was still forming last time, freshly
scored articles). After a historical sentiment BACKFILL, run --full-refresh:
an incremental run will not revisit old bars.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime

import psycopg2

from config.logging_config import get_logger
from features import store
from features.config import FeatureConfig, load_config
from features.transforms import build_features
from lag_detection.config import to_utc

logger = get_logger(__name__)


@dataclass(frozen=True)
class RunPlan:
    mode: str                       # full | incremental | pinned
    read_from: datetime | None      # None = from the first row in the DB
    write_from: datetime | None     # None = write every bar computed
    read_end: datetime | None       # None = up to now (exclusive end)


def plan_run(cfg: FeatureConfig, watermarks: dict[str, datetime], full_refresh: bool,
             start: datetime | None = None, end: datetime | None = None) -> RunPlan:
    """Pure function: decide what to read and what to write."""
    if start is not None or end is not None:
        read_from = start - cfg.history_needed() if start else None
        return RunPlan("pinned", read_from, start, end)
    if full_refresh or any(f not in watermarks for f in cfg.frequencies):
        return RunPlan("full", None, None, None)
    write_from = min(watermarks[f] - cfg.lookback(f) for f in cfg.frequencies)
    return RunPlan("incremental", write_from - cfg.history_needed(), write_from, None)


def _connect():
    from config.settings import settings
    return psycopg2.connect(
        host=settings.timescaledb_host, port=settings.timescaledb_port,
        dbname=settings.timescaledb_db, user=settings.timescaledb_user,
        password=settings.timescaledb_password, connect_timeout=10,
    )


def run_once(conn, cfg: FeatureConfig, full_refresh: bool = False,
             start: datetime | None = None, end: datetime | None = None) -> int:
    """One feature-job pass. Returns rows written (0 if another job holds the lock)."""
    if not store.try_lock(conn):
        logger.warning("Another feature job is running (advisory lock held) — skipping this pass.")
        return 0
    run_id = None
    try:
        t0 = time.perf_counter()
        plan = plan_run(cfg, store.fetch_watermarks(conn, cfg.table), full_refresh, start, end)
        run_id = store.start_run(conn, plan.mode, plan.write_from)

        symbols = tuple(cfg.symbol_to_coin)
        candles = store.fetch_candles(conn, symbols, plan.read_from, plan.read_end)
        scores = store.fetch_scores(conn, cfg.coins, plan.read_from, plan.read_end)
        t_read = time.perf_counter()

        feats = build_features(candles, scores, cfg.frequencies, cfg.rolling_short_bars,
                               cfg.rolling_long_bars, cfg.symbol_to_coin)
        if plan.write_from is not None and not feats.empty:
            feats = feats[feats["bucket"] >= plan.write_from]
        t_build = time.perf_counter()

        n = store.upsert_features(conn, cfg.table, feats)
        t_write = time.perf_counter()

        summary = (f"mode={plan.mode} write_from={plan.write_from} candles={len(candles)} "
                   f"articles={len(scores)} rows={n} | read {t_read - t0:.1f}s, "
                   f"build {t_build - t_read:.1f}s, write {t_write - t_build:.1f}s")
        store.finish_run(conn, run_id, "success", n, summary)
        logger.info("Feature job OK: %s", summary)
        if not feats.empty:
            for freq, grp in feats.groupby("freq"):
                logger.info("  %s: %d bars, %s -> %s, %d incomplete, %d with news",
                            freq, len(grp), grp["bucket"].min(), grp["bucket"].max(),
                            int((~grp["is_complete"]).sum()), int((grp["n_articles"] > 0).sum()))
        return n
    except Exception as exc:
        # Best effort bookkeeping: if the connection itself died, these fail too and
        # the original exception (re-raised below) is what the caller must see.
        try:
            conn.rollback()
            if run_id is not None:
                store.finish_run(conn, run_id, "failed", None, f"{type(exc).__name__}: {exc}")
        except psycopg2.Error:
            logger.warning("Could not record the failed run (connection unusable).")
        raise
    finally:
        try:
            store.unlock(conn)
        except psycopg2.Error:
            pass   # a dead session releases its advisory lock automatically


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build bar features into features_bar.")
    p.add_argument("--full-refresh", action="store_true", help="Recompute every bar from scratch.")
    p.add_argument("--start", help="Rebuild from this date (UTC, ISO), e.g. 2026-06-01")
    p.add_argument("--end", help="Rebuild up to this date (UTC, exclusive)")
    p.add_argument("--every", type=int, metavar="SECONDS",
                   help="Keep running: one incremental pass every SECONDS (e.g. 300)")
    args = p.parse_args(argv)
    if args.every is not None and args.every < 30:
        p.error("--every must be at least 30 seconds")
    if args.every and (args.start or args.end):
        p.error("--every cannot be combined with --start/--end")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = load_config()
    start, end = to_utc(args.start), to_utc(args.end)

    conn = _connect()
    store.ensure_schema(conn, cfg.table)
    logger.info("Feature job: coins=%s freqs=%s table=%s", list(cfg.coins), list(cfg.frequencies), cfg.table)

    if not args.every:
        run_once(conn, cfg, args.full_refresh, start, end)
        conn.close()
        return

    # Loop mode: first pass honours --full-refresh, later passes are incremental.
    full, failures = args.full_refresh, 0
    try:
        while True:
            wait = args.every
            try:
                if conn is None or conn.closed:
                    conn = _connect()
                run_once(conn, cfg, full_refresh=full)
                full, failures = False, 0
            except (psycopg2.OperationalError, psycopg2.InterfaceError) as exc:
                failures += 1
                wait = min(args.every, 15 * 2 ** min(failures, 4))   # 30s, 60s, ... capped
                logger.error("DB error (%s) — reconnecting in %ds (failure %d)", exc, wait, failures)
                _safe_close(conn)
                conn = None
            except Exception:
                failures += 1
                logger.exception("Feature pass failed (%d in a row); retrying in %ds", failures, wait)
            time.sleep(wait)
    except KeyboardInterrupt:
        logger.info("Feature job stopped by user.")
    finally:
        _safe_close(conn)


def _safe_close(conn) -> None:
    try:
        if conn is not None and not conn.closed:
            conn.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()