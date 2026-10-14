"""
Signal job: features_bar -> model -> BUY / HOLD / SELL -> signals table.

Run from the project root:
    python -m signals.run_signals                # one pass (feature update + signals)
    python -m signals.run_signals --every 300    # keep running, one pass every 5 minutes
    python -m signals.run_signals --skip-features

Each pass first brings features_bar up to date (incremental feature job) unless
--skip-features is given, then retrains the model on the most recent labelled
history and writes one signal per target coin for the newest closed bar.
Re-running for the same bar overwrites that bar's signal (idempotent).
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

import pandas as pd
import psycopg2

from config.logging_config import get_logger
from signals import store
from signals.config import SignalConfig, load_config
from signals.dataset import VOLUME_WINDOW, build_model_frame, training_rows
from signals.model import predict_up, to_action, train

logger = get_logger(__name__)


def generate_signals(features: pd.DataFrame, cfg: SignalConfig) -> tuple[pd.DataFrame, dict]:
    """Pure function: features_bar rows -> one signal per target for its newest bar.

    Returns (signals, info). Targets with too little history are left out and
    listed in info["skipped"].
    """
    frame, feature_names = build_model_frame(features, cfg.targets, cfg.input_coins, cfg.horizon_bars)
    info = {"skipped": {}, "n_train": 0, "base_rate": None, "model_version": None}
    if frame.empty:
        info["skipped"] = {c: "no features" for c in cfg.targets}
        return pd.DataFrame(), info

    latest = (frame[frame["close"].notna()].sort_values("bucket")
              .groupby("coin").tail(1).set_index("coin"))
    if latest.empty:
        return pd.DataFrame(), info

    # One pooled model; the cutoff is set by the OLDEST "latest bar" so that no
    # target's prediction can see a label from its own future.
    cutoff_bucket = latest["bucket"].min()
    rows = training_rows(frame, cutoff_bucket, cfg.horizon, cfg.train_window_bars)
    per_coin = rows.groupby("coin").size()

    eligible = []
    for coin in cfg.targets:
        n = int(per_coin.get(coin, 0))
        if coin not in latest.index:
            info["skipped"][coin] = "no price bars"
        elif n < cfg.min_train_bars:
            info["skipped"][coin] = f"only {n} labelled bars (< min_train_bars {cfg.min_train_bars})"
        else:
            eligible.append(coin)
    if not eligible:
        return pd.DataFrame(), info

    rows = rows[rows["coin"].isin(eligible)]
    model = train(rows, feature_names, cfg.model)
    version = f"hgb-{cfg.freq}-h{cfg.horizon_bars}-{model.train_end:%Y%m%dT%H%M}"
    info.update(n_train=model.n_train, base_rate=model.base_rate, model_version=version)

    now_rows = latest.loc[eligible].reset_index()
    prob = predict_up(model, now_rows)
    out = pd.DataFrame({
        "bucket": now_rows["bucket"],
        "valid_from": now_rows["bucket"] + cfg.bar,
        "symbol": now_rows["coin"].str.replace("/", "", regex=False),
        "coin": now_rows["coin"],
        "freq": cfg.freq,
        "horizon_bars": cfg.horizon_bars,
        "close": now_rows["close"],
        "prob_up": prob,
        "action": to_action(prob, cfg.buy_threshold, cfg.sell_threshold),
        "model_version": version,
        "n_train": model.n_train,
    })
    return out, info


def run_once(conn, cfg: SignalConfig, with_features: bool = True) -> int:
    from features.config import load_config as load_feature_config
    feature_cfg = load_feature_config()
    if with_features:
        # Same steps as `python -m features.run_feature_job`: make sure the table and
        # all configured columns exist, then run one (incremental) feature pass.
        from features import store as feature_store
        from features.run_feature_job import run_once as run_feature_pass
        from features.transforms import feature_columns
        added = feature_store.ensure_schema(conn, feature_cfg.table,
                                            feature_columns(tuple(feature_cfg.news_streams)))
        run_feature_pass(conn, feature_cfg, full_refresh=bool(added))
    store.ensure_schema(conn, cfg.table)

    lookback = cfg.bar * (cfg.train_window_bars + cfg.horizon_bars + VOLUME_WINDOW + 5)
    since = datetime.now(timezone.utc) - lookback
    t0 = time.perf_counter()
    features = store.fetch_features(conn, feature_cfg.table, cfg.targets + cfg.input_coins, cfg.freq, since)
    signals, info = generate_signals(features, cfg)
    n = store.upsert_signals(conn, cfg.table, signals)

    for coin, why in info["skipped"].items():
        logger.warning("No signal for %s: %s", coin, why)
    if not signals.empty:
        # A signal is only useful if it is based on the bar that just closed. If the
        # newest bar with a price is older than that, price ingestion has a gap.
        age = datetime.now(timezone.utc) - (signals["bucket"].max() + cfg.bar)
        if age > 2 * cfg.bar:
            logger.warning("STALE: newest priced bar is %s old -- price data has a gap. Run "
                           "`python -m tools.backfill_market --start <gap start date>` and check "
                           "that market ingestion is running.", str(age).split(".")[0])
    if n:
        logger.info("Signals OK: %d written | model %s | trained on %d bars (up-rate %.1f%%) | %.1fs",
                    n, info["model_version"], info["n_train"], 100 * info["base_rate"],
                    time.perf_counter() - t0)
        for r in signals.itertuples():
            logger.info("  %-9s bar %s  close=%-12.6g P(up)=%.3f  ->  %s",
                        r.coin, r.bucket.strftime("%Y-%m-%d %H:%M"), r.close, r.prob_up, r.action)
    return n


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate BUY/HOLD/SELL signals into the signals table.")
    p.add_argument("--every", type=int, metavar="SECONDS", help="Keep running: one pass every SECONDS")
    p.add_argument("--skip-features", action="store_true",
                   help="Do not run the incremental feature job before generating signals")
    args = p.parse_args(argv)
    if args.every is not None and args.every < 30:
        p.error("--every must be at least 30 seconds")
    return args


def _connect():
    from streaming.db import get_db_connection
    return get_db_connection()


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = load_config()
    logger.info("Signal job: targets=%s inputs=%s freq=%s horizon=%d bar(s) BUY>=%.2f SELL<=%.2f",
                list(cfg.targets), list(cfg.input_coins), cfg.freq, cfg.horizon_bars,
                cfg.buy_threshold, cfg.sell_threshold)
    conn = _connect()
    if not args.every:
        try:
            run_once(conn, cfg, not args.skip_features)
        finally:
            conn.close()
        return

    failures = 0
    try:
        while True:
            wait = args.every
            try:
                if conn is None or conn.closed:
                    conn = _connect()
                run_once(conn, cfg, not args.skip_features)
                failures = 0
            except (psycopg2.OperationalError, psycopg2.InterfaceError) as exc:
                failures += 1
                wait = min(args.every, 15 * 2 ** min(failures, 4))
                logger.error("DB error (%s) -- reconnecting in %ds (failure %d)", exc, wait, failures)
                try:
                    conn.close()
                except Exception:
                    pass
                conn = None
            except Exception:
                failures += 1
                logger.exception("Signal pass failed (%d in a row); retrying in %ds", failures, wait)
                try:
                    conn.rollback()
                except Exception:
                    pass
            time.sleep(wait)
    except KeyboardInterrupt:
        logger.info("Signal job stopped by user.")
    finally:
        if conn is not None and not conn.closed:
            conn.close()


if __name__ == "__main__":
    main()