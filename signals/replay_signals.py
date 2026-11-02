"""
Replay (reprocess) signals for past bars that the live runner never scored.

Why this exists
---------------
The live job (signals.run_signals) only scores the NEWEST closed bar on each pass.
Bars that arrive later through a backfill (tools.backfill_market -> features job)
therefore never get a signal, which leaves holes in the signals table and in the
dashboard.

How it avoids look-ahead bias
-----------------------------
For every missing bar T, this job calls the SAME generate_signals() function the
live job uses, but only passes it features_bar rows with bucket <= T. Rows after T
are never handed to the model code, so a replayed signal cannot be trained on, or
labelled from, anything that happened after T. That is exactly what the live job
would have seen had it been running at T.

Replayed rows are tagged model_version = "replay:<version>" so the dashboard and
the thesis can tell them apart from signals produced live.

Run from the project root:
    python -m signals.replay_signals --start 2026-10-18 --dry-run     # show what would be written
    python -m signals.replay_signals --start 2026-10-18               # fill the gaps
    python -m signals.replay_signals --start 2026-10-18 --compare-live  # check replay vs live, writes nothing
    python -m signals.replay_signals --freq 4h --start 2026-10-18       # 4h signals (default: config freq)

Live signals are never overwritten. Re-running is safe (idempotent): bars that
already have a signal are skipped.
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from config.logging_config import get_logger
from signals import store
from signals.config import FREQS, SignalConfig, load_config
from signals.dataset import VOLUME_WINDOW
from signals.run_signals import generate_signals

logger = get_logger(__name__)

REPLAY_PREFIX = "replay:"
BATCH_SIZE = 200


# --------------------------------------------------------------------------- helpers

def _utc(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def history_window(cfg: SignalConfig) -> pd.Timedelta:
    """How much history one prediction needs. Same formula as run_signals.run_once."""
    return cfg.bar * (cfg.train_window_bars + cfg.horizon_bars + VOLUME_WINDOW + 5)


def target_symbols(cfg: SignalConfig) -> list[str]:
    return [c.replace("/", "") for c in cfg.targets]


# --------------------------------------------------------------------------- core (pure, unit-tested)

def replay(features: pd.DataFrame, cfg: SignalConfig, buckets, skip: set) -> pd.DataFrame:
    """Score each bar in `buckets` using only features with bucket <= that bar.

    features : features_bar rows (any time range, `bucket` tz-aware UTC)
    buckets  : bars to score, in any order (processed oldest first)
    skip     : set of (symbol, bucket) pairs that must NOT be written (existing signals)
    Returns one DataFrame of new signal rows, tagged with the replay prefix.
    """
    window = history_window(cfg)
    out = []
    buckets = sorted(buckets)
    t0 = time.perf_counter()

    for i, t in enumerate(buckets, 1):
        # THE look-ahead guard: nothing after t is ever passed to the model code.
        hist = features[(features["bucket"] <= t) & (features["bucket"] > t - window)]
        sig, info = generate_signals(hist, cfg)

        if sig.empty:
            logger.warning("bar %s: no signal (%s)", t, info.get("skipped"))
            continue

        # A coin with no bar exactly at t would fall back to its older latest bar;
        # that bar is not the one we are filling, so drop it.
        sig = sig[sig["bucket"] == t]
        sig = sig[[(s, t) not in skip for s in sig["symbol"]]]
        if sig.empty:
            continue

        sig = sig.assign(model_version=REPLAY_PREFIX + sig["model_version"].astype(str))
        out.append(sig)

        if i % 50 == 0 or i == len(buckets):
            logger.info("replayed %d/%d bars (%.1fs)", i, len(buckets), time.perf_counter() - t0)

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# --------------------------------------------------------------------------- DB access

def fetch_existing(conn, table: str, freq: str, start, end) -> set:
    """(symbol, bucket) pairs that already have a signal, live or replayed."""
    sql = f"SELECT symbol, bucket FROM {table} WHERE freq = %s AND bucket >= %s AND bucket <= %s"
    with conn.cursor() as cur:
        cur.execute(sql, (freq, start.to_pydatetime(), end.to_pydatetime()))
        return {(sym, _utc(b)) for sym, b in cur.fetchall()}


def fetch_live(conn, table: str, freq: str, start, end) -> pd.DataFrame:
    """Signals written by the live runner (not by a replay)."""
    sql = (f"SELECT symbol, bucket, prob_up, action FROM {table} "
           f"WHERE freq = %s AND bucket >= %s AND bucket <= %s AND model_version NOT LIKE %s")
    with conn.cursor() as cur:
        cur.execute(sql, (freq, start.to_pydatetime(), end.to_pydatetime(), REPLAY_PREFIX + "%"))
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["symbol", "bucket", "prob_up", "action"])
    df["bucket"] = pd.to_datetime(df["bucket"], utc=True)
    return df


def _connect():
    from streaming.db import get_db_connection
    return get_db_connection()


# --------------------------------------------------------------------------- CLI

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fill missing signals for past bars without look-ahead.")
    p.add_argument("--start", required=True, help="First bar to consider, e.g. 2026-10-18 (UTC)")
    p.add_argument("--end", help="Last bar to consider (UTC). Default: newest closed bar")
    p.add_argument("--freq", choices=FREQS,
                   help="Bar size to replay (default: signal_model.freq in config.yaml)")
    p.add_argument("--dry-run", action="store_true", help="Compute and show signals, write nothing")
    p.add_argument("--compare-live", action="store_true",
                   help="Re-score bars that have LIVE signals and report agreement. Writes nothing")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.freq)
    from features.config import load_config as load_feature_config
    fcfg = load_feature_config()

    newest_closed = pd.Timestamp.now(tz="UTC") - cfg.bar
    start = _utc(args.start)
    end = min(_utc(args.end), newest_closed) if args.end else newest_closed
    if start > end:
        raise SystemExit(f"--start {start} is after the newest closed bar {end}")

    logger.info("Signal replay: freq=%s targets=%s window %s -> %s%s",
                cfg.freq, list(cfg.targets), start, end,
                " [DRY RUN]" if args.dry_run else " [COMPARE LIVE]" if args.compare_live else "")

    conn = _connect()
    try:
        store.ensure_schema(conn, cfg.table)
        since = (start - history_window(cfg)).to_pydatetime()
        feats = store.fetch_features(conn, fcfg.table, cfg.targets + cfg.input_coins, cfg.freq, since)
        if feats.empty:
            raise SystemExit("No features_bar rows in that window -- run the feature job first.")
        feats["bucket"] = pd.to_datetime(feats["bucket"], utc=True)

        if args.compare_live:
            _compare_live(conn, cfg, feats, start, end)
            return

        # Bars to fill: priced target bars in [start, end] where any target lacks a signal.
        priced = feats[feats["coin"].isin(list(cfg.targets)) & feats["close"].notna()
                       & (feats["bucket"] >= start) & (feats["bucket"] <= end)]
        existing = fetch_existing(conn, cfg.table, cfg.freq, start, end)
        symbols = target_symbols(cfg)
        todo = sorted(b for b in priced["bucket"].unique()
                      if any((s, _utc(b)) not in existing for s in symbols))
        todo = [_utc(b) for b in todo]
        logger.info("%d bars in window, %d already have signals for all targets, %d to replay",
                    priced["bucket"].nunique(), priced["bucket"].nunique() - len(todo), len(todo))
        if not todo:
            logger.info("Nothing to do.")
            return

        new = replay(feats, cfg, todo, skip=existing)
        if new.empty:
            logger.warning("Replay produced no rows (see warnings above).")
            return

        counts = new.groupby("action").size().to_dict()
        logger.info("Replay produced %d signals: %s", len(new), counts)
        if args.dry_run:
            print(new[["bucket", "symbol", "close", "prob_up", "action", "model_version"]]
                  .to_string(index=False))
            logger.info("Dry run -- nothing written.")
            return

        written = 0
        for i in range(0, len(new), BATCH_SIZE):
            written += store.upsert_signals(conn, cfg.table, new.iloc[i:i + BATCH_SIZE])
        logger.info("Wrote %d replayed signals to %s.", written, cfg.table)
    finally:
        conn.close()


def _compare_live(conn, cfg: SignalConfig, feats: pd.DataFrame, start, end) -> None:
    """Sanity check: does replay reproduce what the live runner decided at the time?"""
    live = fetch_live(conn, cfg.table, cfg.freq, start, end)
    if live.empty:
        logger.info("No live signals in the window to compare against.")
        return
    rep = replay(feats, cfg, [_utc(b) for b in live["bucket"].unique()], skip=set())
    m = live.merge(rep[["symbol", "bucket", "prob_up", "action"]],
                   on=["symbol", "bucket"], suffixes=("_live", "_replay"))
    if m.empty:
        logger.warning("Replay produced no rows for the live bars.")
        return
    agree = (m["action_live"] == m["action_replay"]).mean()
    mad = (m["prob_up_live"] - m["prob_up_replay"]).abs().mean()
    print(m.sort_values(["bucket", "symbol"]).to_string(index=False))
    logger.info("Compared %d live signals: action agreement %.1f%%, mean |dP(up)| %.3f",
                len(m), 100 * agree, mad)


if __name__ == "__main__":
    main()