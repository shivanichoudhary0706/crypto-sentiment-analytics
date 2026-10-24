"""
Read-only health checks. Nothing here writes to the database.

Each check appends Findings (what the operator reads) and fills a State
(what repair.py needs to act on). A check that crashes becomes one FAIL
finding; it never stops the other checks.
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from psycopg2 import sql

from maintenance.config import CONTAINERS, OPTIONAL_CONTAINERS, MaintenanceConfig
from maintenance.ranges import Range, day_ranges, market_gaps, missing_minutes, tags_changed

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OK, WARN, FAIL, INFO = "OK", "WARN", "FAIL", "INFO"
AUTO = "auto-repair"   # fixed by this script when run without --check-only


@dataclass
class Finding:
    area: str
    check: str
    status: str
    detail: str = ""
    pending: int = 0          # amount of outstanding work, in `unit`
    unit: str = ""
    action: str = ""          # what fixes it ("auto-fixed", a flag, or a command)


@dataclass(frozen=True)
class ReportSpec:
    name: str
    module: str
    output_dir: Path


@dataclass
class State:
    findings: list[Finding] = field(default_factory=list)
    market_gaps: dict[str, list[Range]] = field(default_factory=dict)
    news_gaps: dict[str, list[tuple[date, date]]] = field(default_factory=dict)
    retag: list[tuple[datetime, str, list[str]]] = field(default_factory=list)  # (time, url, new tags)
    retag_new_scoring: int = 0    # re-tagged articles NOT already in the unscored count
    unscored: int = 0
    features_full_refresh: bool = False
    features_rebuild_from: datetime | None = None
    signals_stale: bool = False
    stale_reports: list[ReportSpec] = field(default_factory=list)

    def add(self, *args, **kwargs) -> Finding:
        f = Finding(*args, **kwargs)
        self.findings.append(f)
        return f

    @property
    def worst(self) -> str:
        statuses = {f.status for f in self.findings}
        return FAIL if FAIL in statuses else WARN if WARN in statuses else OK


# ------------------------------------------------------------------ helpers
def fmt_age(delta: timedelta | None) -> str:
    if delta is None:
        return "never"
    mins = int(delta.total_seconds() // 60)
    if mins < 60:
        return f"{mins}m"
    hours, mins = divmod(mins, 60)
    if hours < 48:
        return f"{hours}h {mins}m"
    return f"{hours // 24}d {hours % 24}h"


def ago(delta: timedelta | None) -> str:
    return "never" if delta is None else f"{fmt_age(delta)} ago"


def _rows(conn, query, params=None) -> list[tuple]:
    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()
    conn.commit()   # never leave a read transaction open
    return rows


def _table_exists(conn, table: str) -> bool:
    return bool(_rows(conn, "SELECT to_regclass(%s) IS NOT NULL", (table,))[0][0])


def _guard(st: State, conn, area: str, name: str, fn: Callable[[], None]) -> None:
    try:
        fn()
    except Exception as exc:   # one broken check must not hide the others
        if conn is not None:
            try:
                conn.rollback()
            except Exception:
                pass
        st.add(area, name, FAIL, f"check crashed: {type(exc).__name__}: {str(exc).strip()[:160]}")


# ------------------------------------------------------------------ infrastructure
def check_docker(st: State) -> None:
    area = "Infrastructure"
    try:
        out = subprocess.run(["docker", "ps", "--format", "{{.Names}}|{{.Status}}"],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as exc:
        st.add(area, "Docker", FAIL, f"not reachable ({type(exc).__name__}) - start Docker Desktop",
               action="start Docker Desktop, then: docker compose up -d")
        return
    if out.returncode != 0:
        st.add(area, "Docker", FAIL, "daemon not reachable - is Docker Desktop running?",
               action="start Docker Desktop, then: docker compose up -d")
        return
    running = dict(line.split("|", 1) for line in out.stdout.splitlines() if "|" in line)
    for name in CONTAINERS + OPTIONAL_CONTAINERS:
        status = running.get(name)
        if status is None:
            st.add(area, f"Container {name}", INFO if name in OPTIONAL_CONTAINERS else FAIL,
                   "not running", action="docker compose up -d")
        elif "unhealthy" in status.lower():
            st.add(area, f"Container {name}", WARN, status, action=f"docker logs {name}")
        else:
            st.add(area, f"Container {name}", OK, status)


def check_kafka(st: State, cfg: MaintenanceConfig) -> None:
    from config.settings import settings
    from kafka import KafkaAdminClient, KafkaConsumer

    area = "Infrastructure"
    bootstrap = settings.kafka_bootstrap_servers
    logging.getLogger("kafka").setLevel(logging.CRITICAL)   # its retry tracebacks are noise here
    try:
        admin = KafkaAdminClient(bootstrap_servers=bootstrap, client_id="health-check",
                                 request_timeout_ms=5000)
    except Exception as exc:
        st.add(area, "Kafka broker", FAIL, f"not reachable at {bootstrap} ({type(exc).__name__})",
               action="docker compose up -d")
        return
    try:
        st.add(area, "Kafka broker", OK, bootstrap)
        topics = set(admin.list_topics())
        for topic in settings.kafka_topics.values():
            st.add(area, f"Topic {topic}", OK if topic in topics else FAIL,
                   "" if topic in topics else "missing (created on first produce)")
        group = settings.kafka_consumer_group
        offsets = admin.list_consumer_group_offsets(group)
        if not offsets:
            st.add(area, f"Consumer lag [{group}]", WARN, "no committed offsets - consumers never ran?",
                   action="python -m streaming.run_storage_consumers")
            return
        consumer = KafkaConsumer(bootstrap_servers=bootstrap, client_id="health-check")
        try:
            ends = consumer.end_offsets(list(offsets))
        finally:
            consumer.close()
        lag = sum(max(0, ends.get(tp, 0) - om.offset) for tp, om in offsets.items() if om.offset >= 0)
        status = OK if lag <= cfg.kafka_max_lag else WARN
        st.add(area, f"Consumer lag [{group}]", status, f"{lag} messages behind",
               pending=lag if status != OK else 0, unit="messages",
               action="" if status == OK else "python -m streaming.run_storage_consumers")
    finally:
        admin.close()


# ------------------------------------------------------------------ market data
_MARKET_RANGE_SQL = """
SELECT symbol, min(time), max(time) FROM market_data
WHERE symbol = ANY(%(symbols)s) AND time >= %(since)s GROUP BY symbol
"""
_MARKET_GAPS_SQL = """
SELECT symbol, time + interval '1 minute', next_time FROM (
    SELECT symbol, time, lead(time) OVER (PARTITION BY symbol ORDER BY time) AS next_time
    FROM market_data WHERE symbol = ANY(%(symbols)s) AND time >= %(since)s
) t
WHERE next_time - time > interval '1 minute'
ORDER BY symbol, time
"""


def check_market(st: State, conn, cfg: MaintenanceConfig, now: datetime) -> None:
    area = "Market data"
    params = {"symbols": list(cfg.market_symbols), "since": cfg.history_start}
    span = {s: (first, last) for s, first, last in _rows(conn, _MARKET_RANGE_SQL, params)}
    internal: dict[str, list[Range]] = {}
    for symbol, start, end in _rows(conn, _MARKET_GAPS_SQL, params):
        internal.setdefault(symbol, []).append((start, end))

    grace = timedelta(minutes=cfg.stale_minutes["market"])
    newest = None
    for symbol in cfg.market_symbols:
        first, last = span.get(symbol, (None, None))
        gaps = market_gaps(cfg.history_start, now, first, last, internal.get(symbol, []), grace)
        st.market_gaps[symbol] = gaps
        mins = missing_minutes(gaps)
        if last is not None:
            newest = last if newest is None else max(newest, last)
        if not gaps:
            st.add(area, f"Candles {symbol}", OK, f"complete since {cfg.history_start:%Y-%m-%d}")
        else:
            largest = max(gaps, key=lambda g: g[1] - g[0])
            st.add(area, f"Candles {symbol}", WARN,
                   f"{len(gaps)} gap(s), largest {fmt_age(largest[1] - largest[0])} "
                   f"from {largest[0]:%Y-%m-%d %H:%M}",
                   pending=mins, unit="candles", action=AUTO)

    age = None if newest is None else now - newest
    live_ok = age is not None and age <= grace
    st.add(area, "Live market stream", OK if live_ok else FAIL,
           f"newest candle {fmt_age(age)} old",
           action="" if live_ok else r"scripts\start_pipeline.bat (market-ingestion + storage-consumers)")


# ------------------------------------------------------------------ news + tagging + scoring
_NEWS_GAP_SQL = """
SELECT d::date, count(r.url)
FROM generate_series(%(since)s::date, %(until)s::date - 1, interval '1 day') AS d
LEFT JOIN raw_sentiment r
       ON r.time >= d AND r.time < d + interval '1 day' AND %(coin)s = ANY(r.coin_tags)
GROUP BY 1 HAVING count(r.url) < %(min)s ORDER BY 1
"""


def check_news(st: State, conn, cfg: MaintenanceConfig, now: datetime) -> None:
    area = "News"
    for coin in cfg.news_backfill_coins:
        rows = _rows(conn, _NEWS_GAP_SQL, {"since": cfg.history_start, "until": now.date(),
                                           "coin": coin, "min": cfg.min_articles_per_day})
        runs = day_ranges(d for d, _ in rows)
        st.news_gaps[coin] = runs
        if not runs:
            st.add(area, f"Daily coverage {coin}", OK, f"every day since {cfg.history_start:%Y-%m-%d}")
        else:
            longest = max(runs, key=lambda r: r[1] - r[0])
            st.add(area, f"Daily coverage {coin}", WARN,
                   f"{len(rows)} day(s) without news, longest run {longest[0]} -> "
                   f"{longest[1] - timedelta(days=1)}",
                   pending=len(rows), unit="days",
                   action=(f"{AUTO} (last {cfg.auto_news_backfill_days} days); older: --with-news"
                           if cfg.auto_news_backfill_days else "--with-news (uses CryptoCompare quota)"))

    last = _rows(conn, "SELECT max(ingested_at) FROM raw_sentiment WHERE source_type = 'news_rss'")[0][0]
    age = None if last is None else now - last
    live_ok = age is not None and age <= timedelta(minutes=cfg.stale_minutes["news"])
    st.add(area, "Live RSS poller", OK if live_ok else WARN, f"last new article {ago(age)}",
           action="" if live_ok else r"scripts\start_pipeline.bat (sentiment-ingestion)")


def check_tagging(st: State, conn, cfg: MaintenanceConfig) -> None:
    from ingestion.sentiment.coin_tagger import tag_coins

    rows = _rows(conn, """
        SELECT rs.time, rs.url, rs.source_type, rs.title, rs.summary, rs.coin_tags,
               EXISTS (SELECT 1 FROM sentiment_scores ss WHERE ss.url = rs.url AND ss.time = rs.time)
        FROM raw_sentiment rs WHERE rs.source_type = ANY(%s)""", (list(cfg.retag_source_types),))
    untagged = 0
    for time_, url, source_type, title, summary, stored, scored in rows:
        # Same text the pollers tag: GDELT has titles only, RSS uses title + summary.
        text = title if source_type == "gdelt" else f"{title} {summary or ''}"
        fresh = tag_coins(text or "")
        if tags_changed(stored, fresh):
            untagged += not stored
            # Already counted as unscored: tagged but never scored. New scoring work:
            # untagged articles (skipped by the scorer today) and scored ones whose
            # scores re-tagging deletes.
            st.retag_new_scoring += (not stored) or scored
            st.retag.append((time_, url, fresh))
    n = len(st.retag)
    st.add("Tagging", "Coin tags", OK if n == 0 else WARN,
           f"{len(rows)} keyword-tagged articles checked" if n == 0 else
           f"{untagged} untagged + {n - untagged} tagged differently by the current tagger",
           pending=n, unit="articles", action=AUTO if n else "")


_UNSCORED_SQL = """
SELECT count(*) FROM raw_sentiment rs
WHERE rs.coin_tags != '{}'
  AND NOT EXISTS (SELECT 1 FROM sentiment_scores ss WHERE ss.url = rs.url AND ss.time = rs.time)
"""


def check_scoring(st: State, conn, cfg: MaintenanceConfig, now: datetime) -> None:
    area = "Scoring"
    st.unscored = _rows(conn, _UNSCORED_SQL)[0][0]
    extra = st.retag_new_scoring
    pending = st.unscored + extra
    detail = f"{st.unscored} unscored" + (f" (+{extra} after re-tagging)" if extra else "")
    st.add(area, "FinBERT + VADER backlog", OK if pending == 0 else WARN, detail,
           pending=pending, unit="articles", action=AUTO if pending else "")

    last = _rows(conn, "SELECT max(scored_at) FROM sentiment_scores")[0][0]
    age = None if last is None else now - last
    idle_ok = st.unscored == 0 or (age is not None and age <= timedelta(minutes=cfg.stale_minutes["scoring"]))
    st.add(area, "Live scorer", OK if idle_ok else WARN,
           f"last score {ago(age)}" + ("" if st.unscored else " (nothing waiting)"),
           action="" if idle_ok else r"scripts\start_pipeline.bat (sentiment-scoring)")


# ------------------------------------------------------------------ features
def check_features(st: State, conn, cfg: MaintenanceConfig, now: datetime) -> None:
    from features.config import FREQ_TO_SECONDS, load_config

    area = "Features"
    fcfg = load_config()
    if not _table_exists(conn, fcfg.table):
        st.features_full_refresh = True
        st.add(area, f"Table {fcfg.table}", WARN, "missing - never built", pending=1,
               unit="full build", action=AUTO)
        return

    table = sql.Identifier(fcfg.table)
    base_freq = min(fcfg.frequencies, key=FREQ_TO_SECONDS.get)
    step = timedelta(seconds=FREQ_TO_SECONDS[base_freq])
    # Bars this recent are re-computed by every incremental pass anyway.
    until = now - max(fcfg.lookback(f) for f in fcfg.frequencies)
    params = {"symbols": list(fcfg.symbol_to_coin), "coins": list(fcfg.score_coins),
              "since": cfg.history_start, "until": until, "step": step, "freq": base_freq}

    # A bar is stale if it is missing, or if any input row arrived after the bar
    # was computed (market backfill, late scores, re-tagging).
    market_q = sql.SQL("""
WITH m AS (
    SELECT symbol, date_bin(%(step)s, time, TIMESTAMPTZ '1970-01-01') AS b, max(ingested_at) AS changed
    FROM market_data
    WHERE symbol = ANY(%(symbols)s) AND time >= %(since)s AND time < %(until)s
    GROUP BY 1, 2)
SELECT min(m.b), count(*) FROM m
LEFT JOIN {t} f ON f.freq = %(freq)s AND f.symbol = m.symbol AND f.bucket = m.b
WHERE f.bucket IS NULL OR m.changed > f.computed_at
""").format(t=table)
    news_q = sql.SQL("""
WITH s AS (
    SELECT date_bin(%(step)s, time, TIMESTAMPTZ '1970-01-01') AS b, max(scored_at) AS changed
    FROM sentiment_scores
    WHERE coin = ANY(%(coins)s) AND time >= %(since)s AND time < %(until)s
    GROUP BY 1),
f AS (
    SELECT bucket AS b, min(computed_at) AS computed FROM {t}
    WHERE freq = %(freq)s AND bucket >= %(since)s GROUP BY 1)
SELECT min(s.b), count(*) FROM s JOIN f USING (b) WHERE s.changed > f.computed
""").format(t=table)
    m_from, m_n = _rows(conn, market_q, params)[0]
    s_from, s_n = _rows(conn, news_q, params)[0]
    starts = [t for t in (m_from, s_from) if t is not None]
    st.features_rebuild_from = min(starts) if starts else None
    if st.features_rebuild_from is None:
        st.add(area, f"Historical bars ({base_freq})", OK, f"in sync with inputs since {cfg.history_start:%Y-%m-%d}")
    else:
        st.add(area, f"Historical bars ({base_freq})", WARN,
               f"{m_n} out of date vs prices, {s_n} vs news; rebuild from "
               f"{st.features_rebuild_from:%Y-%m-%d %H:%M}",
               pending=m_n + s_n, unit="bars", action=AUTO)

    marks = dict(_rows(conn, sql.SQL("SELECT freq, max(bucket) FROM {} GROUP BY freq").format(table)))
    for freq in fcfg.frequencies:
        bar = timedelta(seconds=FREQ_TO_SECONDS[freq])
        age = None if marks.get(freq) is None else now - marks[freq]
        fresh = age is not None and age <= 2 * bar
        st.add(area, f"Newest {freq} bar", OK if fresh else WARN, f"{fmt_age(age)} old",
               action="" if fresh else AUTO + r"; live loop: scripts\start_pipeline.bat (signals)")


# ------------------------------------------------------------------ signals + model
def check_signals(st: State, conn, cfg: MaintenanceConfig, now: datetime) -> None:
    from features.config import load_config as load_feature_config
    from signals.config import load_config

    area = "Signals + model"
    scfg = load_config()
    bar = scfg.bar.to_pytimedelta()
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    latest_closed = epoch + ((now - epoch) // bar) * bar - bar

    if not _table_exists(conn, scfg.table):
        st.signals_stale = True
        st.add(area, f"Table {scfg.table}", WARN, "missing - signals never generated",
               pending=len(scfg.targets), unit="coins", action=AUTO)
    else:
        table = sql.Identifier(scfg.table)
        latest = {coin: (b, c) for coin, b, c in _rows(conn, sql.SQL("""
            SELECT coin, max(bucket), max(created_at) FROM {} WHERE freq = %s AND coin = ANY(%s)
            GROUP BY coin""").format(table), (scfg.freq, list(scfg.targets)))}
        week_ago = now - timedelta(days=7)
        history = dict(_rows(conn, sql.SQL("""
            SELECT coin, count(*) FROM {} WHERE freq = %s AND coin = ANY(%s) AND bucket >= %s
            GROUP BY coin""").format(table), (scfg.freq, list(scfg.targets), week_ago)))
        expected_week = int(timedelta(days=7) / bar)
        for coin in scfg.targets:
            b, _ = latest.get(coin, (None, None))
            behind = b is None or b < latest_closed - bar
            st.signals_stale |= behind
            st.add(area, f"Latest signal {coin}", WARN if behind else OK,
                   f"bar {b:%Y-%m-%d %H:%M} ({fmt_age(now - b)} ago)" if b else "none yet",
                   pending=int(behind), unit="coins", action=AUTO if behind else "")
            st.add(area, f"7-day signal history {coin}", INFO,
                   f"{history.get(coin, 0)}/{expected_week} bars have a signal "
                   "(past bars are not back-filled; the backtest covers history)")
        last_created = max((c for _, c in latest.values() if c), default=None)
        age = None if last_created is None else now - last_created
        live_ok = age is not None and age <= timedelta(minutes=cfg.stale_minutes["signals"])
        st.add(area, "Live signal runner", OK if live_ok else WARN, f"last write {ago(age)}",
               action="" if live_ok else r"scripts\start_pipeline.bat (signals)")
        model = _rows(conn, sql.SQL("SELECT model_version, n_train FROM {} ORDER BY created_at DESC LIMIT 1")
                      .format(table))
        if model:
            st.add(area, "Model", INFO, f"{model[0][0]}, trained on {model[0][1]} bars "
                                        "(retrained on every signal pass)")

    fcfg = load_feature_config()
    if _table_exists(conn, fcfg.table):
        since = now - bar * scfg.train_window_bars
        counts = dict(_rows(conn, sql.SQL("""
            SELECT coin, count(*) FROM {} WHERE freq = %s AND coin = ANY(%s) AND bucket >= %s
              AND close IS NOT NULL GROUP BY coin""").format(sql.Identifier(fcfg.table)),
            (scfg.freq, list(scfg.targets), since)))
        for coin in scfg.targets:
            n = counts.get(coin, 0)
            ok = n >= scfg.min_train_bars
            st.add(area, f"Training history {coin}", OK if ok else WARN,
                   f"{n} bars in the last {scfg.train_window_bars} (needs >= {scfg.min_train_bars})",
                   pending=0 if ok else scfg.min_train_bars - n, unit="bars",
                   action="" if ok else "fills as market backfill + feature rebuild run")


# ------------------------------------------------------------------ offline reports
def report_specs() -> list[ReportSpec]:
    from config.settings import settings
    sm = settings.signal_model or {}
    specs = [
        ("Lag analysis", "lag_detection.run_analysis", (settings.lag_detection or {}).get("output_dir")),
        ("Event study", "lag_detection.run_event_study", (settings.event_study or {}).get("output_dir")),
        ("Backtest", "signals.run_backtest", (sm.get("backtest") or {}).get("output_dir")),
        ("Tuning", "signals.run_tuning", (sm.get("tuning") or {}).get("output_dir")),
    ]
    return [ReportSpec(name, module, PROJECT_ROOT / out) for name, module, out in specs if out]


def check_reports(st: State, conn, now: datetime) -> None:
    from features.config import load_config

    area = "Reports"
    fcfg = load_config()
    changed = None
    if _table_exists(conn, fcfg.table):
        # Live passes only touch the last day; a newer computed_at on OLDER bars
        # means history was repaired, so every analysis over it is out of date.
        changed = _rows(conn, sql.SQL("SELECT max(computed_at) FROM {} WHERE bucket < %s")
                        .format(sql.Identifier(fcfg.table)), (now - timedelta(days=2),))[0][0]
    history_pending = st.features_full_refresh or st.features_rebuild_from is not None
    for spec in report_specs():
        files = [p for p in spec.output_dir.rglob("*") if p.is_file()] if spec.output_dir.exists() else []
        if not files:
            st.stale_reports.append(spec)
            st.add(area, spec.name, WARN, "never run", pending=1, unit="reports", action="--with-reports")
            continue
        written = datetime.fromtimestamp(max(p.stat().st_mtime for p in files), tz=timezone.utc)
        stale = history_pending or (changed is not None and written < changed)
        if stale:
            st.stale_reports.append(spec)
        st.add(area, spec.name, WARN if stale else OK,
               f"written {fmt_age(now - written)} ago" + (", data changed since" if stale else ""),
               pending=int(stale), unit="reports", action="--with-reports" if stale else "")


# ------------------------------------------------------------------ entry point
def inspect_all(conn, cfg: MaintenanceConfig, now: datetime | None = None,
                infrastructure: bool = True) -> State:
    now = now or datetime.now(timezone.utc)
    st = State()
    if infrastructure:
        _guard(st, None, "Infrastructure", "Docker", lambda: check_docker(st))
        _guard(st, None, "Infrastructure", "Kafka", lambda: check_kafka(st, cfg))
    if conn is None:
        st.add("Database", "TimescaleDB", FAIL, "no connection - data checks skipped",
               action="docker compose up -d")
        return st
    _guard(st, conn, "Market data", "Market checks", lambda: check_market(st, conn, cfg, now))
    _guard(st, conn, "News", "News checks", lambda: check_news(st, conn, cfg, now))
    _guard(st, conn, "Tagging", "Tagging check", lambda: check_tagging(st, conn, cfg))
    _guard(st, conn, "Scoring", "Scoring checks", lambda: check_scoring(st, conn, cfg, now))
    _guard(st, conn, "Features", "Feature checks", lambda: check_features(st, conn, cfg, now))
    _guard(st, conn, "Signals + model", "Signal checks", lambda: check_signals(st, conn, cfg, now))
    _guard(st, conn, "Reports", "Report checks", lambda: check_reports(st, conn, now))
    return st
