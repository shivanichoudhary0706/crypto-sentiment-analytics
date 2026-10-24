"""
Health check + self-repair for the whole pipeline.

Run from the project root (venv active, Docker running):

    python -m tools.health_check                 # report, repair everything safe, report again
    python -m tools.health_check --check-only    # report only, change nothing
    python -m tools.health_check --with-news     # fill ALL news gaps (default: only the last
                                                 #   maintenance.auto_news_backfill_days days)
    python -m tools.health_check --with-reports  # also re-run lag / event study / backtest / tuning
    python -m tools.health_check --since 2026-08-01   # override maintenance.history_start

What it checks (each line shows how much work is pending):
    Infrastructure  Docker containers, Kafka topics, consumer lag
    Market data     missing 1-minute candles per coin since history_start; live stream
    News            days without articles per news stream; live RSS poller
    Tagging         articles whose stored coin tags differ from the current tagger
    Scoring         articles not yet scored by FinBERT + VADER; live scorer
    Features        feature bars missing or computed before their inputs changed
    Signals + model latest BUY/HOLD/SELL per coin, training history, live runner
    Reports         analysis outputs older than the data they were computed from

Repairs, in dependency order (each is idempotent - safe to re-run):
    unit tests -> market backfill -> news backfill -> re-tag -> score
               -> feature rebuild -> signals -> [reports]

It does NOT start the live processes (they need their own windows); when one
is down it prints the command. Exit code: 0 = nothing failing, 1 = something
still FAIL after repairs.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from datetime import datetime, timedelta, timezone

import psycopg2

from config.logging_config import get_logger
from lag_detection.config import to_utc
from maintenance import repair
from maintenance.checks import AUTO, FAIL, INFO, OK, WARN, Finding, State, check_features, inspect_all
from maintenance.config import load_config
from maintenance.ranges import recent_day_ranges

logger = get_logger("health_check")

AREA_ORDER = ("Code", "Infrastructure", "Database", "Market data", "News", "Tagging", "Scoring",
              "Features", "Signals + model", "Reports")
_LOCK_KEY = 7_042_026_002   # one repair run at a time (feature job uses ...001)
WIDTH = 100


# ------------------------------------------------------------------ printing (ASCII only: Windows cmd)
def print_report(findings: list[Finding], title: str) -> None:
    print("\n" + "=" * WIDTH)
    print(f" {title}  ({datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC)")
    print("=" * WIDTH)
    by_area: dict[str, list[Finding]] = {}
    for f in findings:
        by_area.setdefault(f.area, []).append(f)
    for area in sorted(by_area, key=lambda a: AREA_ORDER.index(a) if a in AREA_ORDER else 99):
        print(f"\n-- {area} --")
        for f in by_area[area]:
            print(f"  [{f.status:<4}] {f.check:<34} {f.detail}")
    counts = {s: sum(f.status == s for f in findings) for s in (OK, WARN, FAIL, INFO)}
    print(f"\n  Summary: {counts[OK]} OK, {counts[WARN]} WARN, {counts[FAIL]} FAIL, {counts[INFO]} INFO")


def print_pending(findings: list[Finding]) -> None:
    pending = [f for f in findings if f.pending or (f.status == FAIL and f.action)]
    print("\n" + "-" * WIDTH)
    print(" PENDING WORK")
    print("-" * WIDTH)
    if not pending:
        print("  Nothing pending.")
        return
    print(f"  {'What':<44} {'Amount':>18}   Fix")
    for f in pending:
        amount = f"{f.pending:,} {f.unit}" if f.pending else "-"
        print(f"  {(f.area + ': ' + f.check)[:44]:<44} {amount:>18}   {f.action}")


# ------------------------------------------------------------------ helpers
def _connect():
    from streaming.db import get_db_connection
    try:
        conn = get_db_connection()
    except psycopg2.OperationalError as exc:
        logger.error("Cannot connect to TimescaleDB: %s", str(exc).strip())
        return None
    with conn.cursor() as cur:
        cur.execute("SET TIME ZONE 'UTC'")   # day boundaries in UTC, like every other job
    conn.commit()
    return conn


def _step(results: list[tuple[str, str]], name: str, fn) -> bool:
    print(f"\n>>> {name}")
    t0 = time.perf_counter()
    try:
        summary = fn()
        ok = not summary.startswith("skipped") and "FAILED" not in summary
    except Exception as exc:
        logger.exception("%s failed", name)
        summary, ok = f"FAILED: {type(exc).__name__}: {exc}", False
    summary = f"{summary}  ({time.perf_counter() - t0:.0f}s)"
    print(f"    {summary}")
    results.append((name, summary))
    return ok


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pipeline health check and self-repair.")
    p.add_argument("--check-only", action="store_true", help="Report only; change nothing.")
    p.add_argument("--with-news", action="store_true",
                   help="Also back-fill days without news from CryptoCompare (spends API quota).")
    p.add_argument("--with-reports", action="store_true",
                   help="Also re-run lag analysis, event study, backtest and tuning when stale.")
    p.add_argument("--skip-tests", action="store_true", help="Do not run the unit tests first.")
    p.add_argument("--skip-market", action="store_true", help="Do not back-fill market candles.")
    p.add_argument("--force", action="store_true", help="Repair even if unit tests fail.")
    p.add_argument("--since", help="Override maintenance.history_start, e.g. 2026-08-01")
    return p.parse_args(argv)


# ------------------------------------------------------------------ main
def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config()
    if args.since:
        cfg = dataclasses.replace(cfg, history_start=to_utc(args.since))

    code_findings: list[Finding] = []
    tests_ok = True
    if not args.skip_tests:
        print("Running unit tests ...")
        tests_ok, summary = repair.run_tests()
        code_findings.append(Finding("Code", "Unit tests", OK if tests_ok else FAIL, summary,
                                     action="" if tests_ok else "fix the failing tests first"))

    conn = _connect()
    st = inspect_all(conn, cfg)
    st.findings = code_findings + st.findings
    print_report(st.findings, "HEALTH CHECK - BEFORE")
    print_pending(st.findings)

    if args.check_only or conn is None:
        if conn is None:
            print("\nNo database connection - nothing can be repaired. Start Docker and re-run.")
        else:
            conn.close()
        return 1 if st.worst == FAIL else 0
    if not tests_ok and not args.force:
        print("\nRepairs SKIPPED: unit tests are failing (use --force to repair anyway).")
        conn.close()
        return 1

    with conn.cursor() as cur:
        cur.execute("SELECT pg_try_advisory_lock(%s)", (_LOCK_KEY,))
        got_lock = cur.fetchone()[0]
    conn.commit()
    if not got_lock:
        print("\nAnother health_check repair is already running - exiting.")
        conn.close()
        return 1

    print("\n" + "=" * WIDTH + "\n REPAIRS\n" + "=" * WIDTH)
    results: list[tuple[str, str]] = []
    try:
        if args.skip_market:
            print("\n>>> Market backfill: skipped (--skip-market)")
        elif any(st.market_gaps.values()):
            _step(results, "Market backfill (Binance REST)", lambda: repair.backfill_market(conn, st))
        if args.with_news and any(st.news_gaps.values()):
            _step(results, "News backfill (CryptoCompare, all gaps)", lambda: repair.backfill_news(conn, st))
        elif cfg.auto_news_backfill_days and any(st.news_gaps.values()):
            recent = datetime.now(timezone.utc).date() - timedelta(days=cfg.auto_news_backfill_days)
            if any(recent_day_ranges(r, recent) for r in st.news_gaps.values()):
                _step(results, f"News backfill (CryptoCompare, last {cfg.auto_news_backfill_days} days)",
                      lambda: repair.backfill_news(conn, st, since=recent))
        if st.retag:
            _step(results, "Re-tag articles", lambda: repair.retag(conn, st))
        _step(results, "Score articles (FinBERT + VADER)", lambda: repair.score(conn))

        # Backfill, re-tagging and scoring change the inputs, so decide what to
        # rebuild only now.
        fresh = State()
        check_features(fresh, conn, cfg, datetime.now(timezone.utc))
        st.features_full_refresh = fresh.features_full_refresh
        st.features_rebuild_from = fresh.features_rebuild_from
        if _step(results, "Feature bars", lambda: repair.rebuild_features(conn, st, cfg)):
            _step(results, "Signals (retrain + BUY/HOLD/SELL)", lambda: repair.generate_signals(conn))
        else:
            print("    Signals skipped: features were not refreshed.")

        if args.with_reports:
            after = inspect_all(conn, cfg, infrastructure=False)
            if after.stale_reports:
                _step(results, "Analysis reports", lambda: repair.run_reports(after.stale_reports))
    finally:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_unlock(%s)", (_LOCK_KEY,))
            conn.commit()
        except psycopg2.Error:
            pass

    final = inspect_all(conn, cfg)
    final.findings = code_findings + final.findings
    conn.close()
    print_report(final.findings, "HEALTH CHECK - AFTER REPAIRS")
    print_pending(final.findings)
    print("\n" + "-" * WIDTH + "\n REPAIRS DONE\n" + "-" * WIDTH)
    for name, summary in results:
        print(f"  {name:<36} {summary}")
    leftover = [f for f in final.findings if f.pending and f.action == AUTO]
    if leftover:
        print("\n  Still pending after this run (its step failed or was skipped above,\n"
              "  or the source has no data for that period):")
        for f in leftover:
            print(f"    {f.area}: {f.check} - {f.detail}")
    return 1 if final.worst == FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
