"""
Week 7 entry point — offline event-study analysis over stored data.

    python -m lag_detection.run_event_study
    python -m lag_detection.run_event_study --cause BTC/USDT --target LTC/USDT
    python -m lag_detection.run_event_study --start 2026-07-01 --end 2026-09-22

cause_coins and price_targets come from the `event_study:` section of
config.yaml. Every (cause, target) pair configured is run, including
cause == target (own-coin reaction) and cause != target (spillover: does
a big coin's news move a coin it never mentioned). The date window is
auto-detected from price data unless pinned; the window actually used is
recorded in every <prefix>_summary.json, so any run can be reproduced.

Outputs (per cause x target pair) in reports/event_study/:
    <prefix>_events.csv          every clustered event (time, direction, magnitude)
    <prefix>_positive_curve.csv  hour-by-hour significance test, positive events
    <prefix>_negative_curve.csv  hour-by-hour significance test, negative events
    <prefix>_positive_exits.csv  per-event hindsight-optimal exit hour, positive
    <prefix>_negative_exits.csv  per-event hindsight-optimal exit hour, negative
    <prefix>_summary.json        onset hour + exit-hour distribution per direction
"""
from __future__ import annotations

import argparse
import dataclasses
import logging

from lag_detection.config import load_db_config, load_event_study_config, to_utc
from lag_detection.event_study import run_one
from lag_detection.series_builder import connect

logger = logging.getLogger("event_study")


def _print_summary(results: list[dict]) -> None:
    header = f"{'cause':<9}{'target':<9}{'rel':<9}{'events':>7} | {'pos onset':<10}{'neg onset':<10}"
    logger.info("=" * len(header))
    logger.info(header)
    logger.info("-" * len(header))
    for r in results:
        pos, neg = r.get("positive"), r.get("negative")
        pos_s = f"h={pos['onset_hour']}" if pos and pos["onset_hour"] else "n/a"
        neg_s = f"h={neg['onset_hour']}" if neg and neg["onset_hour"] else "n/a"
        logger.info(
            f"{r['cause_coin']:<9}{r['price_target']:<9}{r['relationship']:<9}"
            f"{r['n_events_total']:>7} | {pos_s:<10}{neg_s:<10}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sentiment EVENT -> price reaction study (Week 7)")
    parser.add_argument("--cause", action="append",
                        help="Repeatable; must be in config.yaml event_study.cause_coins. "
                             "Default: all configured cause_coins")
    parser.add_argument("--target", action="append",
                        help="Repeatable; must be in config.yaml event_study.price_targets. "
                             "Default: all configured price_targets")
    parser.add_argument("--start", help="Pin window start (ISO date/time, UTC)")
    parser.add_argument("--end", help="Pin window end, exclusive (ISO, UTC)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_event_study_config()
    if args.start or args.end:
        cfg = dataclasses.replace(cfg, start=to_utc(args.start), end=to_utc(args.end))
    db = load_db_config()

    causes = args.cause or list(cfg.cause_coins)
    targets = args.target or list(cfg.price_targets)
    bad_c = [c for c in causes if c not in cfg.cause_coins]
    bad_t = [t for t in targets if t not in cfg.price_targets]
    if bad_c or bad_t:
        raise ValueError(f"Not in config.yaml event_study: causes={bad_c} targets={bad_t}")

    conn = connect(db)
    results = []
    try:
        for cause in causes:
            for target in targets:
                logger.info("Running event study: %s -> %s", cause, target)
                r = run_one(conn, cfg, cause, target)
                if r:
                    results.append(r)
    finally:
        conn.close()

    if results:
        _print_summary(results)
    logger.info("Completed %d/%d (cause, target) pairs (rest skipped: too few events)",
               len(results), len(causes) * len(targets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())