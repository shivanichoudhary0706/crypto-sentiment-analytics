"""
Rolling-window lag scan — tests whether the sentiment->price lag is STABLE
or drifts over time (a core novelty angle: a static "optimal lag" is an
assumption most papers never test).

Windows inherit the full-sample stationarity transform; they are not re-tested.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from lag_detection.stats_tests import cross_correlation, granger_test, peak_lag

logger = logging.getLogger(__name__)


def rolling_lag_scan(df: pd.DataFrame, cause: str, effect: str, window: int, step: int,
                     max_lag: int, alpha: float = 0.05, min_coverage: float = 0.2
                     ) -> tuple[pd.DataFrame, dict]:
    if len(df) < window:
        raise ValueError(f"Series length {len(df)} shorter than window {window}")
    win_lag = max(1, min(max_lag, window // 8))
    rows = []
    for start in range(0, len(df) - window + 1, step):
        w = df.iloc[start:start + window]
        coverage = float((w["n_articles"] > 0).mean()) if "n_articles" in w else np.nan
        row = {"window_start": w.index[0], "window_end": w.index[-1], "coverage": coverage}
        if w[[cause, effect]].isna().any().any():
            row["status"] = "skipped_gap"
            rows.append(row)
            continue
        if np.isclose(w[cause].std(), 0.0) or (not np.isnan(coverage) and coverage < min_coverage):
            row["status"] = "skipped_low_coverage"
            rows.append(row)
            continue
        try:
            ccf = cross_correlation(w[cause], w[effect], win_lag, alpha)
            pk = peak_lag(ccf, "lead")
            g, _ = granger_test(w, cause, effect, win_lag, alpha)
            row.update(status="ok", ccf_peak_lag=pk["lag"], ccf_peak_corr=pk["corr"],
                       ccf_significant_bonf=pk["significant_bonf"],
                       granger_lag=g["chosen_lag"], granger_p=g["p_value_at_chosen"],
                       granger_significant=g["significant_at_chosen"])
        except Exception as exc:  # one bad window must not kill the scan
            logger.warning("Window %s failed: %s", w.index[0], exc)
            row["status"] = f"error: {exc}"
        rows.append(row)

    table = pd.DataFrame(rows)
    return table, summarize_rolling(table)


def summarize_rolling(table: pd.DataFrame) -> dict:
    ok = table[table["status"] == "ok"] if "status" in table else table.iloc[0:0]
    if ok.empty:
        return {"n_windows": int(len(table)), "n_windows_ok": 0,
                "status_counts": {str(k): int(v) for k, v in table["status"].value_counts().items()}
                if "status" in table else {}}
    lags = ok["ccf_peak_lag"].dropna().astype(int)
    status_counts = {str(k): int(v) for k, v in table["status"].value_counts().items()}
    return {
        "n_windows": int(len(table)),
        "n_windows_ok": int(len(ok)),
        "status_counts": status_counts,
        "frac_granger_significant": float(ok["granger_significant"].mean()),
        "frac_ccf_significant_bonf": float(ok["ccf_significant_bonf"].mean()),
        "ccf_peak_lag_mode": int(lags.mode().iloc[0]) if len(lags) else None,
        "ccf_peak_lag_std": float(lags.std()) if len(lags) > 1 else 0.0,
        "ccf_peak_lag_distribution": {int(k): int(v) for k, v in lags.value_counts().sort_index().items()},
    }