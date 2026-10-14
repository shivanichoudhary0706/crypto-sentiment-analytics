"""
Backtest: replay the signal model hour by hour over recent history and report
how good its BUY/SELL calls were -- with news vs. price-only, against simply
holding the coins.

Run from the project root:
    python -m signals.run_backtest                 # last `test_days` days (config.yaml)
    python -m signals.run_backtest --days 30
    python -m signals.run_backtest --start 2026-08-15 --end 2026-10-14

Outputs (reports/backtest/):
    summary.csv       one row per variant x coin (+ portfolio) -- the results table
    predictions.csv   every simulated signal with its outcome and P&L
    run.json          the exact settings and window used, for reproducibility
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config.logging_config import get_logger
from lag_detection.config import to_utc
from signals import store
from signals.backtest import VARIANTS, evaluate, walk_forward
from signals.config import load_backtest_config, load_config
from signals.dataset import build_model_frame

logger = get_logger(__name__)


def _pct(x) -> str:
    return "   n/a" if pd.isna(x) else f"{100 * x:+6.1f}%"


def _rate(x) -> str:
    return "  n/a" if pd.isna(x) else f"{100 * x:5.1f}%"


def print_report(metrics: pd.DataFrame) -> None:
    header = (f"{'coin':<19}{'variant':<12}{'bars':>5}{'BUY':>5}{'SELL':>5}{'HOLD':>5} | "
              f"{'acc':>6}{'base':>6}{'BUYok':>6}{'SELLok':>7}{'AUC':>6} | "
              f"{'strategy':>9}{'buy&hold':>9}{'trades':>7}{'maxDD':>8}")
    logger.info("=" * len(header))
    logger.info(header)
    logger.info("-" * len(header))
    prev = None
    for r in metrics.itertuples():
        if prev is not None and r.coin != prev:
            logger.info("-" * len(header))
        prev = r.coin
        auc = "  n/a" if pd.isna(r.auc) else f"{r.auc:6.3f}"
        logger.info(f"{r.coin:<19}{r.variant:<12}{r.bars:>5}{r.buy:>5}{r.sell:>5}{r.hold:>5} | "
                    f"{_rate(r.accuracy):>6}{_rate(r.baseline_accuracy):>6}{_rate(r.buy_hit_rate):>6}"
                    f"{_rate(r.sell_hit_rate):>7}{auc:>6} | "
                    f"{_pct(r.strategy_return):>9}{_pct(r.buy_hold_return):>9}{r.trades:>7}"
                    f"{_pct(r.max_drawdown):>8}")
    logger.info("=" * len(header))
    logger.info("acc = direction right (P(up)>0.5 vs actual); base = always guess the majority; "
                "BUYok/SELLok = share of BUY/SELL calls the next hour agreed with; "
                "AUC 0.5 = no skill; strategy/buy&hold = total return after fees; maxDD = worst fall")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Walk-forward backtest of the signal model.")
    p.add_argument("--days", type=int, help="Test on the most recent N days (overrides config)")
    p.add_argument("--start", help="Test window start (UTC date), e.g. 2026-08-15")
    p.add_argument("--end", help="Test window end, exclusive (UTC date)")
    p.add_argument("--retrain-every", type=int, help="Retrain every N bars (overrides config)")
    p.add_argument("--fee-bps", type=float, help="Fee per position change in bps (overrides config)")
    args = p.parse_args(argv)

    cfg, bt = load_config(), load_backtest_config()
    if args.days:
        bt = dataclasses.replace(bt, test_days=args.days)
    if args.retrain_every:
        bt = dataclasses.replace(bt, retrain_every_bars=args.retrain_every)
    if args.fee_bps is not None:
        bt = dataclasses.replace(bt, fee_bps=args.fee_bps)

    from features.config import load_config as load_feature_config
    from streaming.db import get_db_connection
    conn = get_db_connection()
    try:
        features = store.fetch_features(conn, load_feature_config().table,
                                        cfg.targets + cfg.input_coins, cfg.freq, since=None)
    finally:
        conn.close()
    if features.empty:
        raise SystemExit("features_bar is empty -- run `python -m features.run_feature_job` first")

    frame, feature_names = build_model_frame(features, cfg.targets, cfg.input_coins, cfg.horizon_bars)
    last = frame.loc[frame["close"].notna(), "bucket"].max()
    test_end = to_utc(args.end) or (last + cfg.bar)
    test_start = to_utc(args.start) or (pd.Timestamp(test_end) - pd.Timedelta(days=bt.test_days))
    test_start, test_end = pd.Timestamp(test_start), pd.Timestamp(test_end)
    logger.info("Backtest %s -> %s | targets %s | retrain every %d bars | fee %.0f bps | "
                "%d features (%d news)", test_start, test_end, list(cfg.targets),
                bt.retrain_every_bars, bt.fee_bps, len(feature_names),
                sum("_news_" in f for f in feature_names))

    t0 = time.perf_counter()
    preds = walk_forward(frame, feature_names, test_start=test_start, test_end=test_end,
                         bar=cfg.bar, horizon=cfg.horizon, retrain_every_bars=bt.retrain_every_bars,
                         train_window_bars=cfg.train_window_bars, model_params=cfg.model,
                         buy_threshold=cfg.buy_threshold, sell_threshold=cfg.sell_threshold)
    if preds.empty:
        raise SystemExit("No bars to test in that window")
    traded, metrics = evaluate(preds, bt.fee_bps)
    logger.info("Replayed %d signals in %.0fs", len(preds), time.perf_counter() - t0)
    print_report(metrics)

    out = Path(bt.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "summary.csv", index=False)
    traded.to_csv(out / "predictions.csv", index=False)
    with open(out / "run.json", "w", encoding="utf-8") as fh:
        json.dump({"generated_at": datetime.now(timezone.utc).isoformat(),
                   "test_start": test_start.isoformat(), "test_end": test_end.isoformat(),
                   "variants": list(VARIANTS), "backtest": dataclasses.asdict(bt),
                   "signal_model": {k: v for k, v in dataclasses.asdict(cfg).items()},
                   "features": feature_names}, fh, indent=2, default=str)
    logger.info("Saved %s, %s, %s", out / "summary.csv", out / "predictions.csv", out / "run.json")


if __name__ == "__main__":
    main()