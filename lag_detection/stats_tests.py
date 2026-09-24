"""
Statistical core of Week 6.

  1. stationarity_test / make_stationary  -> ADF + KPSS (confirmatory pair)
  2. cross_correlation / peak_lag         -> WHERE the lead-lag link is strongest
  3. granger_test                         -> IS the link predictive (VAR-based F-test)

Multiple-testing control:
  * CCF: Bonferroni-adjusted band over the lags scanned in one direction.
  * Granger: the headline test uses a lag chosen by BIC (chosen WITHOUT looking
    at causality p-values -> one pre-specified test). The per-lag scan is
    exploratory and reported with Benjamini-Hochberg FDR correction.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss

try:
    from statsmodels.tools.sm_exceptions import InterpolationWarning
except ImportError:  # pragma: no cover
    InterpolationWarning = UserWarning

logger = logging.getLogger(__name__)

MIN_OBS = 30


# --------------------------------------------------------------------------- #
# 1. Stationarity
# --------------------------------------------------------------------------- #
def stationarity_test(series: pd.Series, alpha: float = 0.05) -> dict:
    """ADF (H0: unit root) + KPSS (H0: stationary). Agreement = confident verdict.

    Note: KPSS p-values come from a lookup table and are clipped to [0.01, 0.10].
    """
    s = pd.Series(series).dropna().astype(float)
    if len(s) < MIN_OBS:
        raise ValueError(f"Need >= {MIN_OBS} observations, got {len(s)}")
    if np.isclose(s.std(), 0.0):
        return {"verdict": "constant", "n": int(len(s)), "adf_stat": None, "adf_p": None,
                "kpss_stat": None, "kpss_p": None}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        # statsmodels >=0.15 warns about a future return-type change; the tuple
        # layout (stat, p, ...) is unchanged in 0.14 and 0.15, so we silence it.
        warnings.simplefilter("ignore", FutureWarning)
        adf_stat, adf_p = adfuller(s, autolag="AIC")[:2]
        kpss_stat, kpss_p = kpss(s, regression="c", nlags="auto")[:2]

    adf_ok, kpss_ok = adf_p < alpha, kpss_p > alpha
    if adf_ok and kpss_ok:
        verdict = "stationary"
    elif not adf_ok and not kpss_ok:
        verdict = "non-stationary"
    else:
        verdict = "inconclusive"
    return {"verdict": verdict, "n": int(len(s)), "adf_stat": float(adf_stat),
            "adf_p": float(adf_p), "kpss_stat": float(kpss_stat), "kpss_p": float(kpss_p)}


def make_stationary(df: pd.DataFrame, columns: list[str], alpha: float = 0.05
                    ) -> tuple[pd.DataFrame, dict]:
    """First-difference any column whose ADF test fails to reject a unit root."""
    out = df.copy()
    report: dict[str, dict] = {}
    for col in columns:
        before = stationarity_test(out[col], alpha)
        if before["verdict"] == "constant":
            raise ValueError(f"'{col}' is constant — nothing to test (no sentiment variation?)")
        transform, after = "level", None
        if before["adf_p"] >= alpha:
            out[col] = out[col].diff()
            transform = "diff1"
            after = stationarity_test(out[col], alpha)
        report[col] = {"transform": transform, "before": before, "after": after}
        logger.info("Stationarity %-12s -> %-14s transform=%s", col, before["verdict"], transform)
    return out.dropna(subset=columns), report


# --------------------------------------------------------------------------- #
# 2. Cross-correlation
# --------------------------------------------------------------------------- #
def cross_correlation(cause: pd.Series, effect: pd.Series, max_lag: int,
                      alpha: float = 0.05) -> pd.DataFrame:
    """corr(cause_{t-k}, effect_t) for k in [-max_lag, max_lag].

    k > 0 : cause LEADS effect by k bars (sentiment -> price)
    k < 0 : effect leads cause (price -> sentiment, i.e. news reacting to price)

    The +-z/sqrt(n) band is valid when at least one series is ~white noise
    (Bartlett); crypto returns at 1h/4h are close to that. Otherwise prewhiten.
    """
    if not cause.index.equals(effect.index):
        raise ValueError("cause and effect must share the same index")
    z_single = norm.ppf(1 - alpha / 2)
    z_bonf = norm.ppf(1 - alpha / (2 * max_lag))
    rows = []
    for k in range(-max_lag, max_lag + 1):
        pair = pd.concat([cause.shift(k), effect], axis=1).dropna()
        n = len(pair)
        r = float(pair.iloc[:, 0].corr(pair.iloc[:, 1])) if n > 2 else np.nan
        band, band_bonf = z_single / np.sqrt(n), z_bonf / np.sqrt(n)
        rows.append({"lag": k, "corr": r, "n": n, "band": band, "band_bonf": band_bonf,
                     "significant": bool(abs(r) > band),
                     "significant_bonf": bool(abs(r) > band_bonf)})
    return pd.DataFrame(rows)


def peak_lag(ccf: pd.DataFrame, direction: str = "lead") -> dict:
    """Lag with the largest |corr| in one direction ('lead' = k>=1, 'lag' = k<=-1)."""
    sub = ccf[ccf["lag"] >= 1] if direction == "lead" else ccf[ccf["lag"] <= -1]
    sub = sub.dropna(subset=["corr"])
    if sub.empty:
        return {"lag": None, "corr": None, "significant": False, "significant_bonf": False}
    row = sub.loc[sub["corr"].abs().idxmax()]
    return {"lag": int(row["lag"]), "corr": float(row["corr"]),
            "significant": bool(row["significant"]),
            "significant_bonf": bool(row["significant_bonf"])}


# --------------------------------------------------------------------------- #
# 3. Granger causality (VAR-based)
# --------------------------------------------------------------------------- #
def granger_test(df: pd.DataFrame, cause: str, effect: str, max_lag: int,
                 alpha: float = 0.05) -> tuple[dict, pd.DataFrame]:
    """Does the past of `cause` improve prediction of `effect` beyond effect's own past?"""
    data = df[[effect, cause]].dropna().reset_index(drop=True)  # RangeIndex: no freq warnings
    n = len(data)
    if n < MIN_OBS:
        raise ValueError(f"Need >= {MIN_OBS} observations for Granger, got {n}")
    usable = max(1, min(max_lag, n // 10))  # keep enough degrees of freedom
    if usable < max_lag:
        logger.debug("Granger max_lag reduced %d -> %d (n=%d)", max_lag, usable, n)

    model = VAR(data)
    orders = {k: int(v) for k, v in model.select_order(maxlags=usable).selected_orders.items()}

    rows = []
    for p in range(1, usable + 1):
        res = model.fit(p)
        test = res.test_causality(caused=effect, causing=[cause], kind="f")
        rows.append({"lag": p, "f_stat": float(test.test_statistic), "p_value": float(test.pvalue)})
    table = pd.DataFrame(rows)
    reject, p_fdr, *_ = multipletests(table["p_value"], alpha=alpha, method="fdr_bh")
    table["p_fdr"] = p_fdr
    table["significant_fdr"] = reject

    bic_raw = orders.get("bic", 0)
    chosen = max(1, bic_raw)
    chosen_p = float(table.loc[table["lag"] == chosen, "p_value"].iloc[0])
    best = table.loc[table["p_fdr"].idxmin()]
    summary = {
        "cause": cause, "effect": effect, "n_obs": n, "max_lag_tested": usable,
        "ic_orders": orders, "bic_order_raw": bic_raw,  # 0 => no dynamics worth modelling
        "chosen_lag": chosen, "chosen_by": "bic",
        "p_value_at_chosen": chosen_p, "significant_at_chosen": chosen_p < alpha,
        "min_p_fdr": float(best["p_fdr"]), "lag_of_min_p_fdr": int(best["lag"]),
        "any_significant_fdr": bool(table["significant_fdr"].any()),
    }
    return summary, table