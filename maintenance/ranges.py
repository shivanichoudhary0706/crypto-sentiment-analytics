"""
Pure helpers that turn "what is missing" into "what to fetch". No DB, no network.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Iterable

Range = tuple[datetime, datetime]   # [start, end) in UTC


def market_gaps(history_start: datetime, now: datetime, first: datetime | None,
                last: datetime | None, internal: Iterable[Range],
                live_grace: timedelta) -> list[Range]:
    """All missing [start, end) ranges of 1-minute candles for one symbol.

    first/last: oldest/newest candle at or after history_start (None = no candles).
    internal:   gaps between consecutive candles, as [first missing minute, next candle).
    live_grace: minutes at the very end that the live stream simply hasn't written
                yet; those are not a gap.
    """
    minute = timedelta(minutes=1)
    if first is None:
        return [(history_start, now)]
    gaps = []
    if first - history_start >= minute:
        gaps.append((history_start, first))
    gaps.extend(internal)
    tail_start = last + minute
    if now - tail_start > live_grace:
        gaps.append((tail_start, now))
    return gaps


def missing_minutes(ranges: Iterable[Range]) -> int:
    return int(sum((end - start).total_seconds() // 60 for start, end in ranges))


def coalesce(ranges: Iterable[Range], join_within: timedelta) -> list[Range]:
    """Merge ranges whose distance is under join_within.

    Re-fetching the few existing candles between two nearby gaps is free
    (ON CONFLICT DO NOTHING) and saves one API call per gap, so many small
    gaps become one fetch.
    """
    merged: list[list[datetime]] = []
    for start, end in sorted(ranges):
        if merged and start - merged[-1][1] < join_within:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def day_ranges(days: Iterable[date]) -> list[tuple[date, date]]:
    """Consecutive days -> [(first_day, day_after_last)] runs."""
    runs: list[list[date]] = []
    for d in sorted(set(days)):
        if runs and d == runs[-1][1]:
            runs[-1][1] = d + timedelta(days=1)
        else:
            runs.append([d, d + timedelta(days=1)])
    return [(s, e) for s, e in runs]


def recent_day_ranges(runs: Iterable[tuple[date, date]], since: date) -> list[tuple[date, date]]:
    """Keep only the part of each [first_day, day_after) run on or after `since`."""
    return [(max(s, since), e) for s, e in runs if e > since]


def tags_changed(stored: Iterable[str] | None, fresh: Iterable[str]) -> bool:
    return set(stored or ()) != set(fresh)
