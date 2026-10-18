"""
Honest tuning: choose settings on one period, prove them on a later, unseen one.

    python -m signals.run_tuning

1. CHOOSE period (older): every candidate in config.yaml `signal_model.tuning.candidates`
   is backtested. The winner is picked by a rule fixed IN ADVANCE: highest after-fee
   portfolio return of the price+news model (ties -> fewer trades).
2. CONFIRM period (newest, never used for choosing): only the winner and the current
   live settings are backtested, next to simply holding the coins.

The CONFIRM numbers are the ones to report. Picking the best-looking settings on
the same data you report would be like seeing the exam answers before the exam.

Each candidate sets: freq ("1h" / "4h") and buy/sell thresholds. Training history
(in days) and the once-a-day retraining are kept equal across candidates, so the
comparison is fair.

Outputs (reports/tuning/): choose.csv, confirm.csv, run.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config.logging_config import get_logger
from signals import store
from signals.backtest import evaluate, walk_forward
from signals.config import FREQ_SECONDS, SignalConfig, load_backtest_config, load_config, parse_signal_config
from signals.dataset import build_model_frame

logger = get_logger(__name__)
PORTFOLIO = "ALL (equal weight)"


@dataclasses.dataclass(frozen=True)
class TuningConfig:
    choose_days: int
    confirm_days: int
    candidates: tuple[dict, ...]
    output_dir: str


def parse_tuning_config(section: dict | None) -> TuningConfig:
    if not section:
        raise KeyError("config.yaml has no `signal_model.tuning:` section")
    missing = [k for k in ("choose_days", "confirm_days", "candidates", "output_dir") if k not in section]
    if missing:
        raise KeyError(f"config.yaml -> signal_model.tuning is missing required key(s): {missing}")
    cands = tuple(dict(c) for c in (section["candidates"] or []))
    if len(cands) < 2:
        raise ValueError("signal_model.tuning.candidates needs at least 2 entries")
    for c in cands:
        if set(c) != {"freq", "buy_threshold", "sell_threshold"}:
            raise ValueError(f"each tuning candidate needs exactly freq, buy_threshold, sell_threshold: {c}")
    cfg = TuningConfig(int(section["choose_days"]), int(section["confirm_days"]), cands,
                       str(section["output_dir"]))
    if cfg.choose_days < 1 or cfg.confirm_days < 1:
        raise ValueError("signal_model.tuning: choose_days and confirm_days must be >= 1")
    return cfg


def candidate_config(base: SignalConfig, cand: dict) -> SignalConfig:
    """Base live config with one candidate's freq/thresholds. History lengths are
    converted so every candidate learns from the same number of DAYS."""
    base_hours = FREQ_SECONDS[base.freq] / 3600
    new_hours = FREQ_SECONDS[cand["freq"]] / 3600
    scale = base_hours / new_hours
    raw = dataclasses.asdict(base)
    raw.update(freq=cand["freq"], buy_threshold=cand["buy_threshold"],
               sell_threshold=cand["sell_threshold"],
               train_window_bars=max(1, round(base.train_window_bars * scale)),
               min_train_bars=max(1, round(base.min_train_bars * scale)))
    return parse_signal_config(raw)          # re-validates the thresholds and sizes


def label(cfg: SignalConfig) -> str:
    return f"{cfg.freq} BUY>={cfg.buy_threshold:.2f} SELL<={cfg.sell_threshold:.2f}"


def select_best(results: pd.DataFrame) -> int:
    """Index of the winning candidate: highest price+news portfolio return after fees,
    ties broken by fewer trades. Fixed before looking at any result."""
    pn = results[results["variant"] == "price+news"]
    best = pn.sort_values(["strategy_return", "trades"], ascending=[False, True]).iloc[0]
    return int(best["candidate"])


def backtest(features_by_freq: dict, cfg: SignalConfig, start, end, fee_bps: float) -> pd.DataFrame:
    frame, names = features_by_freq[cfg.freq]
    retrain_every = max(1, round(24 * 3600 / FREQ_SECONDS[cfg.freq]))      # once per day
    preds = walk_forward(frame, names, test_start=start, test_end=end, bar=cfg.bar,
                         horizon=cfg.horizon, retrain_every_bars=retrain_every,
                         train_window_bars=cfg.train_window_bars, model_params=cfg.model,
                         buy_threshold=cfg.buy_threshold, sell_threshold=cfg.sell_threshold)
    if preds.empty:
        return pd.DataFrame()
    _, metrics = evaluate(preds, fee_bps)
    return metrics[metrics["coin"] == PORTFOLIO]


def _row(name, r) -> str:
    return (f"{name:<42}{r.variant:<12}{r.bars:>6}{r.trades:>7} | {100 * r.accuracy:5.1f}%"
            f"{r.auc:7.3f} | {100 * r.strategy_return:+7.1f}%{100 * r.buy_hold_return:+9.1f}%"
            f"{100 * r.max_drawdown:+8.1f}%")


def _header() -> str:
    return (f"{'settings':<42}{'variant':<12}{'bars':>6}{'trades':>7} | {'acc':>6}{'AUC':>7} | "
            f"{'return':>8}{'buy&hold':>10}{'maxDD':>9}")


def main(argv=None) -> None:
    argparse.ArgumentParser(description="Choose settings on an older period, confirm on a newer one.").parse_args(argv)
    from config.settings import settings
    base, bt = load_config(), load_backtest_config()
    tc = parse_tuning_config((settings.signal_model or {}).get("tuning"))
    cands = [candidate_config(base, c) for c in tc.candidates]

    from features.config import load_config as load_feature_config
    from streaming.db import get_db_connection
    conn = get_db_connection()
    features_by_freq = {}
    try:
        for freq in sorted({c.freq for c in cands} | {base.freq}):
            f = store.fetch_features(conn, load_feature_config().table,
                                     base.targets + base.input_coins, freq, since=None)
            if f.empty:
                raise SystemExit(f"features_bar has no {freq} rows -- add it to features.frequencies")
            features_by_freq[freq] = build_model_frame(f, base.targets, base.input_coins, base.horizon_bars)
    finally:
        conn.close()

    frame_1 = features_by_freq[base.freq][0]
    end = pd.Timestamp(frame_1.loc[frame_1["close"].notna(), "bucket"].max()).floor("D")
    confirm_start = end - pd.Timedelta(days=tc.confirm_days)
    choose_start = confirm_start - pd.Timedelta(days=tc.choose_days)
    logger.info("CHOOSE  %s -> %s | CONFIRM %s -> %s | fee %.0f bps | %d candidates",
                choose_start.date(), confirm_start.date(), confirm_start.date(), end.date(),
                bt.fee_bps, len(cands))

    # ---- 1. choose
    rows = []
    for i, cfg in enumerate(cands):
        logger.info("Choosing: backtesting %s ...", label(cfg))
        m = backtest(features_by_freq, cfg, choose_start, confirm_start, bt.fee_bps)
        rows.append(m.assign(candidate=i, settings=label(cfg)))
    choose = pd.concat(rows, ignore_index=True)
    winner = cands[select_best(choose)]

    logger.info("=" * 100)
    logger.info("CHOOSE period %s -> %s (used ONLY to pick the settings)", choose_start.date(), confirm_start.date())
    logger.info(_header())
    for r in choose.itertuples():
        logger.info(_row(r.settings, r))
    logger.info("Winner (rule: best price+news return after fees): %s", label(winner))

    # ---- 2. confirm
    confirm_rows = []
    for tag, cfg in (("WINNER", winner), ("CURRENT LIVE", base)):
        m = backtest(features_by_freq, cfg, confirm_start, end, bt.fee_bps)
        confirm_rows.append(m.assign(role=tag, settings=label(cfg)))
    confirm = pd.concat(confirm_rows, ignore_index=True)

    logger.info("=" * 100)
    logger.info("CONFIRM period %s -> %s (unseen -- REPORT THESE NUMBERS)", confirm_start.date(), end.date())
    logger.info(_header())
    for r in confirm.itertuples():
        logger.info(_row(f"{r.role}: {r.settings}", r))
    logger.info("=" * 100)
    logger.info("To use the winner live, set in config.yaml -> signal_model:")
    logger.info('  freq: "%s"   buy_threshold: %.2f   sell_threshold: %.2f   train_window_bars: %d   '
                'min_train_bars: %d', winner.freq, winner.buy_threshold, winner.sell_threshold,
                winner.train_window_bars, winner.min_train_bars)

    out = Path(tc.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    choose.to_csv(out / "choose.csv", index=False)
    confirm.to_csv(out / "confirm.csv", index=False)
    with open(out / "run.json", "w", encoding="utf-8") as fh:
        json.dump({"generated_at": datetime.now(timezone.utc).isoformat(),
                   "choose": [choose_start.isoformat(), confirm_start.isoformat()],
                   "confirm": [confirm_start.isoformat(), end.isoformat()],
                   "selection_rule": "max price+news portfolio return after fees; ties -> fewer trades",
                   "fee_bps": bt.fee_bps, "candidates": [label(c) for c in cands],
                   "winner": dataclasses.asdict(winner)}, fh, indent=2, default=str)
    logger.info("Saved %s, %s, %s", out / "choose.csv", out / "confirm.csv", out / "run.json")


if __name__ == "__main__":
    main()