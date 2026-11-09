"""
Pure helpers for the signal chart (no Streamlit, no API calls) so they can be unit-tested.

- prepare_history : API rows -> DataFrame with source (live/replayed), change flags, latency
- missing_bars    : how many bars are absent between the first and last bar
- with_gap_breaks : insert an empty row inside each gap so Plotly breaks the line there
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REPLAY_PREFIX = "replay:"   # must match signals.replay_signals.REPLAY_PREFIX


def prepare_history(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["bucket"] = pd.to_datetime(df["bucket"], utc=True)
    df = df.sort_values("bucket").reset_index(drop=True)

    mv = df["model_version"].fillna("").astype(str) if "model_version" in df else pd.Series("", index=df.index)
    df["source"] = np.where(mv.str.startswith(REPLAY_PREFIX), "replayed", "live")
    df["changed"] = df["action"].ne(df["action"].shift())   # first bar counts as a change

    if {"created_at", "valid_from"} <= set(df.columns):
        df["latency_min"] = (pd.to_datetime(df["created_at"], utc=True)
                             - pd.to_datetime(df["valid_from"], utc=True)).dt.total_seconds() / 60
    return df


def missing_bars(df: pd.DataFrame, bar: pd.Timedelta) -> int:
    if len(df) < 2:
        return 0
    steps = (df["bucket"].diff().dropna() / bar).round()
    return int((steps - 1).clip(lower=0).sum())


def with_gap_breaks(df: pd.DataFrame, bar: pd.Timedelta) -> pd.DataFrame:
    """Return a copy with one all-NaN row inside every gap longer than 1.5 bars.

    Plotly draws a line through consecutive points; a NaN close between them makes it
    break instead of drawing a straight line across time that has no data.
    """
    gap = df["bucket"].diff() > 1.5 * bar
    if not gap.any():
        return df
    breaks = pd.DataFrame({"bucket": df.loc[gap, "bucket"] - bar})
    return (pd.concat([df, breaks], ignore_index=True)
            .sort_values("bucket", kind="stable").reset_index(drop=True))