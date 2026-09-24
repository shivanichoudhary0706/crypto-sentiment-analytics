"""
Week 6 entry point — offline lag analysis over stored data.

    python -m lag_detection.run_analysis
    python -m lag_detection.run_analysis --coin BTC/USDT --freq 1h
    python -m lag_detection.run_analysis --sentiment-column vader_compound --no-rolling
    python -m lag_detection.run_analysis --start 2026-08-12 --end 2026-09-22   # pin a thesis run

Coins come from the top-level `coins:` list in config.yaml. The date window is
auto-detected from market_data unless pinned; the window actually used is
recorded in every <prefix>_summary.json, so any run can be reproduced exactly.

Outputs (per coin x freq x sentiment column) in reports/lag/:
    <prefix>_dataset.csv        aligned bars (raw, before differencing)
    <prefix>_ccf.csv            cross-correlation for lags -K..K
    <prefix>_granger_fwd.csv    sentiment -> return, per lag (with FDR)
    <prefix>_granger_rev.csv    return -> sentiment, per lag (with FDR)
    <prefix>_rolling.csv        rolling-window lag scan
    <prefix>_summary.json       everything the thesis tables need
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from lag_detection.config import (
    ALLOWED_SENTIMENT_COLUMNS, ALLOWED_TARGETS, LagConfig, load_config, load_db_config, to_utc, validate,
)
from lag_detection.rolling import rolling_lag_scan
from lag_detection.series_builder import build_dataset, connect
from lag_detection.stats_tests import cross_correlation, granger_test, make_stationary, peak_lag

logger = logging.getLogger("lag_detection")
CAUSE = "sentiment"


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def analyse_one(conn, cfg: LagConfig, coin: str, freq: str) -> dict:
    effect, alpha = cfg.target, cfg.significance
    df, diag = build_dataset(conn, cfg, coin, freq)

    max_lag = cfg.max_lag
    n_valid = diag["n_valid_bars"]
    if n_valid < 5 * max_lag:
        max_lag = max(1, n_valid // 5)
        logger.warning("%s @ %s: only %d valid bars — max_lag reduced to %d", coin, freq, n_valid, max_lag)

    stat_df, stationarity = make_stationary(df, [CAUSE, effect], alpha)

    ccf = cross_correlation(stat_df[CAUSE], stat_df[effect], max_lag, alpha)
    g_fwd, g_fwd_table = granger_test(stat_df, CAUSE, effect, max_lag, alpha)
    g_rev, g_rev_table = granger_test(stat_df, effect, CAUSE, max_lag, alpha)

    rolling_table, rolling_summary = None, None
    rc = cfg.rolling
    if rc.enabled and n_valid >= rc.window_bars:
        rolling_table, rolling_summary = rolling_lag_scan(
            stat_df, CAUSE, effect, rc.window_bars, rc.step_bars, max_lag, alpha, rc.min_coverage)
    elif rc.enabled:
        logger.warning("%s @ %s: %d bars < rolling window %d — rolling scan skipped",
                       coin, freq, n_valid, rc.window_bars)

    prefix = f"{coin.replace('/', '')}_{freq}_{cfg.sentiment_column}"
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{prefix}_dataset.csv")
    ccf.to_csv(out / f"{prefix}_ccf.csv", index=False)
    g_fwd_table.to_csv(out / f"{prefix}_granger_fwd.csv", index=False)
    g_rev_table.to_csv(out / f"{prefix}_granger_rev.csv", index=False)
    if rolling_table is not None:
        rolling_table.to_csv(out / f"{prefix}_rolling.csv", index=False)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "coin": coin, "freq": freq, "target": effect, "sentiment_column": cfg.sentiment_column,
        "fill_method": cfg.fill_method, "alpha": alpha, "max_lag_bars": max_lag,
        "data": diag,
        "stationarity": stationarity,
        "ccf": {"sentiment_leads": peak_lag(ccf, "lead"), "price_leads": peak_lag(ccf, "lag"),
                "contemporaneous_corr": float(ccf.loc[ccf["lag"] == 0, "corr"].iloc[0])},
        "granger": {"sentiment_to_price": g_fwd, "price_to_sentiment": g_rev},
        "rolling": rolling_summary,
    }
    with open(out / f"{prefix}_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=_json_default)
    return summary


def _print_summary(results: list[dict]) -> None:
    header = f"{'coin':<9}{'freq':<5}{'bars':>6}{'cov%':>6} | {'CCF lead(k,r)':<16}{'Granger S->P':<16}{'Granger P->S':<16}"
    logger.info("=" * len(header))
    logger.info(header)
    logger.info("-" * len(header))
    for r in results:
        lead = r["ccf"]["sentiment_leads"]
        fwd, rev = r["granger"]["sentiment_to_price"], r["granger"]["price_to_sentiment"]
        star = lambda sig: "*" if sig else " "  # noqa: E731
        logger.info(
            f"{r['coin']:<9}{r['freq']:<5}{r['data']['n_valid_bars']:>6}{100 * r['data']['coverage']:>6.1f} | "
            f"{str(lead['lag']) + ',' + format(lead['corr'] or 0, '+.3f') + star(lead['significant_bonf']):<16}"
            f"{'L' + str(fwd['chosen_lag']) + ' p=' + format(fwd['p_value_at_chosen'], '.3f') + star(fwd['significant_at_chosen']):<16}"
            f"{'L' + str(rev['chosen_lag']) + ' p=' + format(rev['p_value_at_chosen'], '.3f') + star(rev['significant_at_chosen']):<16}"
        )
    logger.info("* = significant (CCF: Bonferroni; Granger: at BIC-chosen lag)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sentiment -> price lag detection (Week 6)")
    parser.add_argument("--coin", action="append",
                        help="Repeatable; must be in config.yaml `coins:`. Default: all configured coins")
    parser.add_argument("--freq", action="append", help="Repeatable, e.g. --freq 1h")
    parser.add_argument("--sentiment-column", choices=ALLOWED_SENTIMENT_COLUMNS)
    parser.add_argument("--target", choices=ALLOWED_TARGETS)
    parser.add_argument("--start", help="Pin window start (ISO date/time, UTC). Default: first price bar")
    parser.add_argument("--end", help="Pin window end, exclusive (ISO, UTC). Default: last price bar")
    parser.add_argument("--no-rolling", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

    cfg = load_config()
    overrides = {}
    if args.coin:
        unknown = [c for c in args.coin if c not in cfg.coins]
        if unknown:
            parser.error(f"{unknown} not in config.yaml coins {list(cfg.coins)}")
        overrides["coins"] = tuple(args.coin)
    if args.freq:
        overrides["frequencies"] = tuple(args.freq)
    if args.sentiment_column:
        overrides["sentiment_column"] = args.sentiment_column
    if args.target:
        overrides["target"] = args.target
    if args.no_rolling:
        overrides["rolling"] = dataclasses.replace(cfg.rolling, enabled=False)
    if args.start:
        overrides["start"] = to_utc(args.start)
    if args.end:
        overrides["end"] = to_utc(args.end)
    cfg = dataclasses.replace(cfg, **overrides)
    validate(cfg)
    logger.info("Coins %s | freqs %s | window %s -> %s", list(cfg.coins), list(cfg.frequencies),
                cfg.start or "auto", cfg.end or "auto")

    results, failures = [], 0
    conn = connect(load_db_config())
    try:
        for coin in cfg.coins:
            for freq in cfg.frequencies:
                try:
                    results.append(analyse_one(conn, cfg, coin, freq))
                except Exception:
                    failures += 1
                    logger.exception("Analysis failed for %s @ %s", coin, freq)
    finally:
        conn.close()

    if results:
        _print_summary(results)
    logger.info("Reports written to %s", cfg.output_dir)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())