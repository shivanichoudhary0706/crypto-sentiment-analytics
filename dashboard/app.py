"""
Crypto Signal Desk - Streamlit dashboard. Reads everything through the FastAPI layer.
Run from project root (API must be running):   streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow 'dashboard.' imports

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard.api_client import ApiError, get

COLOR = {"BUY": "#22c55e", "SELL": "#ef4444", "HOLD": "#f59e0b"}
POS, NEG = "#22c55e", "#ef4444"

st.set_page_config(page_title="Crypto Signal Desk", page_icon="📈", layout="wide")


def short(sym: str) -> str:
    return sym.replace("/USDT", "").removesuffix("USDT")


def fmt_ts(ts) -> str:
    return pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M UTC")


def sig_windows(curve: pd.DataFrame) -> list[tuple[int, int]]:
    """Contiguous horizon ranges where the FDR-corrected test is significant."""
    runs, start, prev = [], None, None
    for h, ok in zip(curve["horizon_h"], curve["significant_fdr"]):
        if ok and start is None:
            start = h
        if not ok and start is not None:
            runs.append((start, prev)); start = None
        prev = h
    if start is not None:
        runs.append((start, prev))
    return runs


def fmt_windows(runs) -> str:
    return ", ".join(f"{a}–{b} h" if a != b else f"{a} h" for a, b in runs)


# ---------------- header + shared data ----------------
st.title("Crypto Signal Desk")
st.caption("News-aware BUY / HOLD / SELL for small-cap targets · inputs: BTC & ETH prices "
           "+ BTC / ETH / MARKET news sentiment (FinBERT)")

if st.sidebar.button("Refresh now"):
    st.cache_data.clear()

try:
    health = get("/health")
    meta = get("/meta")
except ApiError as exc:
    st.error(str(exc))
    st.info("Start the API first:  uvicorn api.main:app --host 127.0.0.1 --port 8000")
    st.stop()

if health["ok"]:
    st.sidebar.success("All pipeline stages fresh")
else:
    stale = ", ".join(s["stage"] for s in health["stages"] if not s["ok"])
    st.sidebar.warning(f"Stale: {stale}")
st.sidebar.caption(f"Model: {meta['freq']} bars · BUY ≥ {meta['buy_threshold']} · "
                   f"SELL ≤ {meta['sell_threshold']} · fee {meta['fee_bps']} bps")
st.sidebar.caption("Data cached 60 s")

tab_sig, tab_news, tab_eval, tab_health = st.tabs(
    ["Live signals", "News → exit window", "Model evaluation", "System health"])

# ---------------- live signals ----------------
with tab_sig:
    latest = get("/signals/latest", freq=meta["freq"]) or []
    if not latest:
        st.info("No signals yet - is the signal runner window open?")
    for col, s in zip(st.columns(max(len(latest), 1)), latest):
        with col, st.container(border=True):
            st.markdown(f"### {short(s['symbol'])}")
            st.markdown(f"<div style='font-size:34px;font-weight:700;color:{COLOR[s['action']]}'>"
                        f"{s['action']}</div>", unsafe_allow_html=True)
            st.metric("P(up next bar)", f"{s['prob_up']:.3f}")
            close = f"{s['close']:.4g}" if s.get("close") is not None else "n/a"
            st.caption(f"Bar starting {fmt_ts(s['bucket'])} · close {close}\n\n{s['model_version']}")
    st.caption("A signal is for the bar that has just closed, so ~1–2 h behind wall-clock time is normal.")

    st.subheader("Price & signal history")
    c1, c2 = st.columns(2)
    coin = c1.selectbox("Coin", meta["targets"], format_func=short)
    hours = c2.select_slider("Window", options=[24, 48, 96, 168, 336], value=96,
                             format_func=lambda h: f"{h} h")
    hist = get("/signals/history", coin=coin, hours=hours, freq=meta["freq"])
    df = pd.DataFrame(hist["rows"]) if hist else pd.DataFrame()
    if df.empty:
        st.info("No signal history in this window yet.")
    else:
        df["bucket"] = pd.to_datetime(df["bucket"])
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.65, 0.35], vertical_spacing=0.06)
        fig.add_trace(go.Scatter(x=df.bucket, y=df.close, name="Close",
                                 line=dict(color="#60a5fa")), 1, 1)
        for a, symbol in (("BUY", "triangle-up"), ("SELL", "triangle-down")):
            d = df[df.action == a]
            fig.add_trace(go.Scatter(x=d.bucket, y=d.close, mode="markers", name=a,
                                     marker=dict(symbol=symbol, size=10, color=COLOR[a])), 1, 1)
        fig.add_trace(go.Bar(x=df.bucket, y=df.prob_up, showlegend=False, name="P(up)",
                             marker_color=[COLOR[a] for a in df.action]), 2, 1)
        for thr in (meta["buy_threshold"], meta["sell_threshold"]):
            fig.add_hline(y=thr, line_dash="dot", line_color="#8a93a6", row=2, col=1)
        lo, hi = min(0.3, df.prob_up.min() - 0.02), max(0.7, df.prob_up.max() + 0.02)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="P(up)", range=[lo, hi], row=2, col=1)
        fig.update_layout(height=540, margin=dict(l=10, r=10, t=10, b=10),
                          legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig)
        changes = int((df.action != df.action.shift()).sum()) - 1
        st.caption(f"{len(df)} bars · {changes} action changes in this window")

# ---------------- news -> exit window ----------------
with tab_news:
    pairs = get("/event-study") or []
    if not pairs:
        st.info("No event-study results found in reports/event_study.")
    else:
        c1, c2 = st.columns(2)
        causes = sorted({p["cause"] for p in pairs})
        cause = c1.selectbox("News about", causes, format_func=short)
        targets = sorted({p["target"] for p in pairs if p["cause"] == cause})
        preferred = [t for t in targets if t in {m.replace("/", "") for m in meta["targets"]}]
        target = c2.selectbox("Price of", targets, format_func=short,
                              index=targets.index(preferred[0]) if preferred else 0)

        es = get(f"/event-study/{cause}/{target}")
        sm = es["summary"]
        pos, neg = pd.DataFrame(es["positive_curve"]), pd.DataFrame(es["negative_curve"])
        pos_w = sig_windows(pos) if not pos.empty else []
        neg_w = sig_windows(neg) if not neg.empty else []
        p, n = sm["positive"], sm["negative"]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Positive-news events", p["n_events"])
        k2.metric("Positive: median best exit", f"{p['median_optimal_exit_hour']:.0f} h",
                  f"{p['mean_car_at_median_exit'] * 100:+.2f}% avg")
        k3.metric("Negative-news events", n["n_events"])
        k4.metric("Negative: significant window", fmt_windows(neg_w) if neg_w else "none")

        if pos_w:
            st.success(f"After strongly positive {short(cause)} news, {short(target)} moved "
                       f"significantly in **{fmt_windows(pos_w)}**. Median optimal exit ≈ "
                       f"**{p['median_optimal_exit_hour']:.0f} h** "
                       f"(IQR {p['p25_optimal_exit_hour']:.0f}–{p['p75_optimal_exit_hour']:.0f} h).")
        else:
            st.info(f"Positive {short(cause)} news: no statistically significant reaction in "
                    f"{short(target)} within {sm['horizon_hours']} h.")
        if neg_w:
            st.warning(f"After strongly negative {short(cause)} news, {short(target)} moved "
                       f"significantly in **{fmt_windows(neg_w)}**.")
        else:
            st.info(f"Negative {short(cause)} news: no statistically significant reaction in "
                    f"{short(target)} - no exit rule is recommended.")

        fig = go.Figure()
        for df_c, name, color in ((pos, "after positive news", POS), (neg, "after negative news", NEG)):
            if df_c.empty:
                continue
            fig.add_trace(go.Scatter(x=df_c.horizon_h, y=df_c.mean_car * 100, name=name,
                                     line=dict(color=color), customdata=df_c.p_fdr,
                                     hovertemplate="%{x} h: %{y:.2f}%<br>p (FDR) = %{customdata:.3f}"))
            s = df_c[df_c.significant_fdr]
            fig.add_trace(go.Scatter(x=s.horizon_h, y=s.mean_car * 100, mode="markers",
                                     marker=dict(color=color, size=8), showlegend=False,
                                     hoverinfo="skip"))
        fig.add_hline(y=0, line_dash="dot", line_color="#8a93a6")
        fig.add_vline(x=p["median_optimal_exit_hour"], line_dash="dash", line_color=POS,
                      annotation_text="median exit", annotation_position="top")
        fig.update_layout(height=420, xaxis_title="Hours after news",
                          yaxis_title="Mean cumulative return %",
                          margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig)
        st.caption(f"Dots = significant after Benjamini–Hochberg FDR (α = {sm['alpha']}). Events: "
                   f"|FinBERT| ≥ {sm['threshold']}, clustered within {sm['cluster_window_hours']:.0f} h. "
                   f"Window {fmt_ts(sm['window_start'])} → {fmt_ts(sm['window_end'])}.")

    st.subheader("Recent scored news")
    news = get("/news/recent", limit=25) or []
    if news:
        nd = pd.DataFrame(news)
        nd["time"] = pd.to_datetime(nd["time"]).dt.strftime("%m-%d %H:%M")
        st.dataframe(nd[["time", "coin", "finbert_compound", "title", "source_name"]],
                     hide_index=True,
                     column_config={"finbert_compound": st.column_config.NumberColumn(
                         "FinBERT", format="%.2f")})

# ---------------- evaluation ----------------
with tab_eval:
    bt = get("/backtest/summary")
    if not bt:
        st.info("No backtest summary - run: python -m signals.run_backtest")
    else:
        run = bt.get("run") or {}
        fee = run.get("fee_bps") if isinstance(run, dict) else None
        if fee == 0:
            st.warning("summary.csv is the zero-fee diagnostic run. Restore the 10 bps version "
                       "for reporting.")
        df = pd.DataFrame(bt["rows"])
        allrow = df[df.coin.str.startswith("ALL")].set_index("variant")
        if {"price+news", "price_only"} <= set(allrow.index):
            pn, po = allrow.loc["price+news"], allrow.loc["price_only"]
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("price + news", f"{pn.strategy_return * 100:+.1f}%",
                      f"{(pn.strategy_return - po.strategy_return) * 100:+.1f} pp vs price only")
            k2.metric("price only", f"{po.strategy_return * 100:+.1f}%")
            k3.metric("buy & hold", f"{pn.buy_hold_return * 100:+.1f}%")
            k4.metric("AUC (news / price)", f"{pn.auc:.3f}", f"{pn.auc - po.auc:+.3f}")

        fig = go.Figure()
        per_coin = df[~df.coin.str.startswith("ALL")]
        for v, color in (("price+news", "#a78bfa"), ("price_only", "#64748b")):
            d = per_coin[per_coin.variant == v]
            fig.add_bar(x=d.coin.map(short), y=d.strategy_return * 100, name=v, marker_color=color)
        d = per_coin[per_coin.variant == "price+news"]
        fig.add_bar(x=d.coin.map(short), y=d.buy_hold_return * 100, name="buy & hold",
                    marker_color="#60a5fa")
        fig.update_layout(barmode="group", height=380, yaxis_title="Return % (net of fees)",
                          margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig)

        show = df[["coin", "variant", "accuracy", "baseline_accuracy", "auc", "strategy_return",
                   "buy_hold_return", "max_drawdown", "trades", "time_in_market"]]
        pct = st.column_config.NumberColumn(format="%.3f")
        st.dataframe(show, hide_index=True,
                     column_config={c: pct for c in ["accuracy", "baseline_accuracy", "auc",
                                                     "strategy_return", "buy_hold_return",
                                                     "max_drawdown", "time_in_market"]})

    tune = get("/backtest/tuning")
    if tune:
        st.subheader("Out-of-sample check of tuned settings")
        cf = pd.DataFrame(tune["confirm"])
        st.dataframe(cf[["role", "settings", "variant", "auc", "strategy_return",
                         "buy_hold_return", "trades"]], hide_index=True)
        st.caption("Settings chosen on the first window, re-tested on the next unseen window.")

# ---------------- system health ----------------
with tab_health:
    hd = pd.DataFrame(health["stages"])
    hd["latest"] = hd["latest"].map(lambda t: fmt_ts(t) if t else "-")
    hd["status"] = hd["ok"].map({True: "✅ OK", False: "⚠️ Stale"})
    st.dataframe(hd[["stage", "latest", "lag_minutes", "max_lag_minutes", "status"]],
                 hide_index=True)
    st.caption(f"Checked {fmt_ts(health['checked_at'])}")