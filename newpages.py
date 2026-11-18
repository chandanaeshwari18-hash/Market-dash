"""Extra read-only pages for the dashboard: Heatmap, Futures activity, Scanners, Rotation (RRG / RMI / ratio) and Watchlist.
Data comes from Kite quotes and daily candles. No order placement. Called from app.py only when one of these pages is chosen,
so the Greeks page code is not affected by anything in this file."""
import datetime as dt
import time as _time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

IST = ZoneInfo("Asia/Kolkata")
VIEWS = ["Signals (Greeks)", "Heatmap", "Futures", "Scanners", "Rotation", "Watchlist", "Charts", "IPO"]
INDEX_NAMES = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50", "SENSEX", "BANKEX"}
BENCH = "NIFTY 50"
SECTORS = {"Bank": "NIFTY BANK", "Pvt Bank": "NIFTY PVT BANK", "PSU Bank": "NIFTY PSU BANK", "Fin Services": "NIFTY FIN SERVICE",
           "IT": "NIFTY IT", "Auto": "NIFTY AUTO", "FMCG": "NIFTY FMCG", "Pharma": "NIFTY PHARMA", "Metal": "NIFTY METAL",
           "Energy": "NIFTY ENERGY", "Realty": "NIFTY REALTY", "Media": "NIFTY MEDIA", "Infra": "NIFTY INFRA",
           "Oil & Gas": "NIFTY OIL AND GAS", "Healthcare": "NIFTY HEALTHCARE"}
LOAD_GAP = 0.34          # Kite allows about 3 history calls per second


def _now():
    return dt.datetime.now(IST)


def _market_open():
    t = _now()
    return t.weekday() < 5 and dt.time(9, 15) <= t.time() <= dt.time(15, 30)


# =====================================================================
# Pure functions (no Streamlit): tested separately
# =====================================================================
def classify_buildup(chg, oi_chg):
    """Price change % and OI change % -> the usual four futures states."""
    if chg != chg or oi_chg != oi_chg:
        return "-"
    if chg >= 0 and oi_chg >= 0:
        return "Long build-up"
    if chg < 0 and oi_chg >= 0:
        return "Short build-up"
    if chg >= 0 and oi_chg < 0:
        return "Short covering"
    return "Long unwinding"


def rsi_wilder(close, n=14):
    c = pd.Series(close, dtype=float)
    d = c.diff()
    up, dn = d.clip(lower=0), (-d).clip(lower=0)
    au, ad = up.ewm(alpha=1 / n, adjust=False).mean(), dn.ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + au / ad.replace(0, np.nan))


def rmi(close, n=14, m=5):
    """Relative Momentum Index: like RSI but momentum is measured over m bars instead of 1."""
    c = pd.Series(close, dtype=float)
    d = c - c.shift(m)
    up, dn = d.clip(lower=0), (-d).clip(lower=0)
    au, ad = up.ewm(alpha=1 / n, adjust=False).mean(), dn.ewm(alpha=1 / n, adjust=False).mean()
    return 100 * au / (au + ad).replace(0, np.nan)


def rrg_points(sector, bench, n=60, smooth=5, mom=10):
    """Simplified Relative Rotation Graph (NOT the licensed JdK formula).
    RS = sector / benchmark. RS-Ratio = RS relative to its own n-day average (x100, smoothed). Momentum = RS-Ratio now vs `mom` days ago (x100)."""
    s, b = pd.Series(sector, dtype=float), pd.Series(bench, dtype=float)
    j = pd.concat([s, b], axis=1, join="inner").dropna()
    if len(j) < n + mom + 5:
        return pd.DataFrame(columns=["ratio", "momentum"])
    rs = j.iloc[:, 0] / j.iloc[:, 1]
    ratio = (100 * rs / rs.rolling(n).mean()).ewm(span=smooth, adjust=False).mean()
    momentum = 100 * ratio / ratio.shift(mom)
    return pd.DataFrame({"ratio": ratio, "momentum": momentum}).dropna()


def quadrant(ratio, momentum):
    if ratio >= 100 and momentum >= 100:
        return "Leading"
    if ratio >= 100:
        return "Weakening"
    if momentum >= 100:
        return "Improving"
    return "Lagging"


def daily_stats(candles, today):
    """candles: list of Kite daily dicts (date, open, high, low, close, volume). Uses only days before `today`."""
    df = pd.DataFrame(candles)
    if df.empty:
        return None
    ts = pd.to_datetime(df["date"])
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_localize(None)
    df["d"] = ts.dt.date
    df = df[df["d"] < today]
    if len(df) < 30:
        return None
    c = df["close"].astype(float)
    last250 = df.tail(250)
    return dict(hi52=float(last250["high"].max()), lo52=float(last250["low"].min()), prev_close=float(c.iloc[-1]),
                avgvol=float(df["volume"].tail(20).mean()), rsi=float(rsi_wilder(c).iloc[-1]),
                sma20=float(c.tail(20).mean()), sma50=float(c.tail(50).mean()),
                sma200=float(c.tail(200).mean()) if len(c) >= 200 else float("nan"))


def apply_scan(df, name):
    """df columns: Stock, Price, Day %, Volume x, 52w high %, 52w low %, RSI, vs SMA. Returns the matching rows, sorted."""
    d = df.copy()
    if name.startswith("Volume spike"):
        return d[d["Volume x"] >= 2.0].sort_values("Volume x", ascending=False)
    if name.startswith("Near 52-week high"):
        return d[d["To 52w high %"] <= 2.0].sort_values("To 52w high %")
    if name.startswith("Near 52-week low"):
        return d[d["From 52w low %"] <= 2.0].sort_values("From 52w low %")
    if name.startswith("Top gainers"):
        return d.sort_values("Day %", ascending=False).head(30)
    if name.startswith("Top losers"):
        return d.sort_values("Day %").head(30)
    if name.startswith("RSI above 70"):
        return d[d["RSI"] > 70].sort_values("RSI", ascending=False)
    if name.startswith("RSI below 30"):
        return d[d["RSI"] < 30].sort_values("RSI")
    if name.startswith("Above 20"):
        return d[d["Trend"] == "Above 20/50/200"].sort_values("Day %", ascending=False)
    if name.startswith("Below 20"):
        return d[d["Trend"] == "Below 20/50/200"].sort_values("Day %")
    return d


SCANS = ["Volume spike (2x the 20-day average or more)", "Near 52-week high (within 2%)", "Near 52-week low (within 2%)",
         "Top gainers", "Top losers", "RSI above 70 (daily)", "RSI below 30 (daily)",
         "Above 20/50/200-day averages", "Below 20/50/200-day averages"]


# =====================================================================
# Kite data helpers (cached)
# =====================================================================
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _nse_inst(_kite, day):
    return pd.DataFrame(_kite.instruments("NSE"))


def _eq_tokens(_kite, day):
    df = _nse_inst(_kite, day)
    eq = df[(df["segment"] == "NSE") & (df["instrument_type"] == "EQ")]
    return dict(zip(eq["tradingsymbol"], eq["instrument_token"]))


def _idx_tokens(_kite, day):
    df = _nse_inst(_kite, day)
    ix = df[df["segment"] == "INDICES"]
    return dict(zip(ix["tradingsymbol"], ix["instrument_token"]))


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _fno_futs(_kite, day):
    df = pd.DataFrame(_kite.instruments("NFO"))
    f = df[(df["instrument_type"] == "FUT") & (~df["name"].isin(INDEX_NAMES))]
    f = f[f["expiry"] >= _now().date()].sort_values("expiry").groupby("name").first().reset_index()
    return f[["name", "tradingsymbol", "instrument_token", "lot_size", "expiry"]]


@st.cache_data(ttl=20, show_spinner=False)
def _quotes(_kite, keys):
    out, ks = {}, list(keys)
    for i in range(0, len(ks), 400):
        out.update(_kite.quote(ks[i:i + 400]))
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def _hist(_kite, token, day, days=420):
    to = _now().date()
    d = _kite.historical_data(int(token), to - dt.timedelta(days=days), to, "day")
    df = pd.DataFrame(d)
    if df.empty:
        return df
    ix = pd.to_datetime(df["date"])
    if ix.dt.tz is not None:
        ix = ix.dt.tz_localize(None)
    df.index = ix.dt.normalize()
    return df[["open", "high", "low", "close", "volume"]]


@st.cache_resource
def _store():
    return {}


def _chg(q):
    last = q.get("last_price") or 0
    prev = (q.get("ohlc") or {}).get("close") or 0
    return last, prev, ((last / prev - 1) * 100 if prev else float("nan"))


def _live(body, every=30):
    """Run `body` as a self-refreshing fragment while the market is open (only this part re-runs), otherwise once."""
    if _market_open() and hasattr(st, "fragment"):
        st.fragment(run_every=every)(body)()
    else:
        body()


def _members(ctx, kind):
    try:
        return list(ctx["members"](kind)[0])
    except Exception:
        return []


# =====================================================================
# Pages
# =====================================================================
def page_heatmap(kite, ctx):
    day = _now().strftime("%Y-%m-%d")
    futs = _fno_futs(kite, day)
    fno = list(futs["name"])
    n50, bn = _members(ctx, "nifty50"), _members(ctx, "banknifty")
    c1, c2 = st.columns(2)
    uni = c1.radio("Stocks", ["Nifty 50", "Bank Nifty", "F&O stocks"], horizontal=True, key="hm_uni")
    size_by = c2.radio("Tile size", ["Traded value", "Equal"], horizontal=True, key="hm_size")
    all_syms = sorted(set(n50) | set(bn) | set(fno))

    def body():
        q = _quotes(kite, tuple("NSE:" + s for s in all_syms))
        rows = {}
        for s in all_syms:
            d = q.get("NSE:" + s)
            if not d:
                continue
            last, prev, ch = _chg(d)
            if last:
                rows[s] = dict(sym=s, price=last, chg=ch, val=last * (d.get("volume") or 0))
        df = pd.DataFrame(rows.values())
        if df.empty:
            st.warning("No quotes yet.")
            return
        # advance / decline
        st.markdown("**Advance / decline**")
        a1, a2, a3 = st.columns(3)
        for col, label, lst in ((a1, "Nifty 50", n50), (a2, "Bank Nifty", bn), (a3, "F&O stocks", fno)):
            d = df[df["sym"].isin(lst)]
            adv, dec = int((d["chg"] > 0).sum()), int((d["chg"] < 0).sum())
            col.metric(label, f"{adv} up / {dec} down", f"{len(d) - adv - dec} unchanged", delta_color="off")
        pick = {"Nifty 50": n50, "Bank Nifty": bn, "F&O stocks": fno}[uni]
        d = df[df["sym"].isin(pick)].copy()
        if d.empty:
            st.info("No stocks for this list.")
            return
        vals = d["val"].where(d["val"] > 0, 1.0) if size_by == "Traded value" else pd.Series(1.0, index=d.index)
        fig = go.Figure(go.Treemap(
            labels=d["sym"], parents=[""] * len(d), values=vals,
            text=[f"{c:+.2f}%" for c in d["chg"]], textinfo="label+text",
            marker=dict(colors=d["chg"].clip(-3, 3), colorscale=[[0, "#b91c1c"], [0.5, "#334155"], [1, "#15803d"]],
                        cmin=-3, cmax=3, line=dict(width=1, color="#0f172a")),
            hovertemplate="%{label}: %{text}<extra></extra>"))
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=520)
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Green = up, red = down (colour range -3% to +3%). Updated {_now().strftime('%H:%M:%S')} IST.")
        top = d.sort_values("chg")
        c_a, c_b = st.columns(2)
        c_a.markdown("**Top gainers**")
        c_a.dataframe(top.tail(8).iloc[::-1][["sym", "price", "chg"]].rename(columns={"sym": "Stock", "price": "Price", "chg": "Day %"}).round(2),
                      hide_index=True, width="stretch")
        c_b.markdown("**Top losers**")
        c_b.dataframe(top.head(8)[["sym", "price", "chg"]].rename(columns={"sym": "Stock", "price": "Price", "chg": "Day %"}).round(2),
                      hide_index=True, width="stretch")
    _live(body)


def _load_prev_oi(kite, futs, store, bar):
    today = _now().date()
    n = len(futs)
    for i, r in enumerate(futs.itertuples()):
        if r.name not in store:
            try:
                c = kite.historical_data(int(r.instrument_token), today - dt.timedelta(days=10), today, "day", oi=True)
                prev = [x for x in c if pd.Timestamp(x["date"]).date() < today]
                if prev:
                    store[r.name] = float(prev[-1].get("oi") or 0)
            except Exception:
                pass
            _time.sleep(LOAD_GAP)
        bar.progress((i + 1) / n)


def page_futures(kite, ctx):
    day = _now().strftime("%Y-%m-%d")
    futs = _fno_futs(kite, day)
    st_all = _store()
    oi_prev = st_all.setdefault(("prev_oi", day), {})
    if len(oi_prev) < 0.8 * len(futs):
        st.info(f"Yesterday's open interest is needed to tell long build-up from short covering. Loaded {len(oi_prev)} of {len(futs)}.")
        if st.button("Load yesterday's OI (about 1-2 minutes, once a day)", key="load_oi"):
            bar = st.progress(0.0)
            _load_prev_oi(kite, futs, oi_prev, bar)
            st.rerun()
    cat = st.radio("Show", ["Long build-up", "Short build-up", "Short covering", "Long unwinding"], horizontal=True, key="fu_cat")

    def body():
        q = _quotes(kite, tuple("NFO:" + s for s in futs["tradingsymbol"]))
        rows = []
        for r in futs.itertuples():
            d = q.get("NFO:" + r.tradingsymbol)
            if not d:
                continue
            last, prev, ch = _chg(d)
            oi = float(d.get("oi") or 0)
            po = oi_prev.get(r.name)
            oich = (oi / po - 1) * 100 if po else float("nan")
            rows.append(dict(Stock=r.name, Price=last, **{"Price %": ch, "OI": oi, "OI %": oich, "Volume": d.get("volume") or 0,
                                                          "State": classify_buildup(ch, oich)}))
        df = pd.DataFrame(rows)
        if df.empty:
            st.warning("No futures quotes yet.")
            return
        counts = df["State"].value_counts()
        m = st.columns(4)
        for col, k in zip(m, ["Long build-up", "Short build-up", "Short covering", "Long unwinding"]):
            col.metric(k, int(counts.get(k, 0)))
        sel = df[df["State"] == cat].sort_values("OI %", ascending=False).head(15)
        if sel.empty:
            st.info("Nothing in this group yet" + ("" if len(oi_prev) else " (load yesterday's OI first)") + ".")
        else:
            fig = go.Figure()
            fig.add_bar(y=sel["Stock"], x=sel["Price %"], orientation="h", name="Price %", marker_color="#3b82f6")
            fig.add_bar(y=sel["Stock"], x=sel["OI %"], orientation="h", name="OI %", marker_color="#8b5cf6")
            fig.update_layout(barmode="group", height=60 + 34 * len(sel), margin=dict(l=0, r=0, t=10, b=0),
                              yaxis=dict(autorange="reversed"), legend=dict(orientation="h"))
            st.plotly_chart(fig, width="stretch")
            st.dataframe(sel.round(2), hide_index=True, width="stretch")
        st.caption("Price % is against yesterday's close; OI % is against yesterday's closing OI of the same near-month contract. "
                   f"Updated {_now().strftime('%H:%M:%S')} IST.")
    _live(body)


def _load_daily_stats(kite, tokens, syms, store, bar):
    today = _now().date()
    n = len(syms)
    for i, s in enumerate(syms):
        if s not in store and s in tokens:
            try:
                c = kite.historical_data(int(tokens[s]), today - dt.timedelta(days=420), today, "day")
                stt = daily_stats(c, today)
                if stt:
                    store[s] = stt
            except Exception:
                pass
            _time.sleep(LOAD_GAP)
        bar.progress((i + 1) / n)


def page_scanners(kite, ctx):
    day = _now().strftime("%Y-%m-%d")
    futs = _fno_futs(kite, day)
    syms = sorted(futs["name"])
    tokens = _eq_tokens(kite, day)
    stats = _store().setdefault(("daily", day), {})
    if len(stats) < 0.8 * len(syms):
        st.info(f"Daily history (52-week range, average volume, RSI, averages) is needed. Loaded {len(stats)} of {len(syms)}.")
        if st.button("Load daily data (about 1-2 minutes, once a day)", key="load_daily"):
            bar = st.progress(0.0)
            _load_daily_stats(kite, tokens, syms, stats, bar)
            st.rerun()
    scan = st.radio("Scanner", SCANS, key="sc_pick")

    def body():
        q = _quotes(kite, tuple("NSE:" + s for s in syms))
        rows = []
        for s in syms:
            d, h = q.get("NSE:" + s), stats.get(s)
            if not d or not h:
                continue
            last, prev, ch = _chg(d)
            if not last:
                continue
            vol = d.get("volume") or 0
            above = all(last > h[k] for k in ("sma20", "sma50", "sma200") if h[k] == h[k])
            below = all(last < h[k] for k in ("sma20", "sma50", "sma200") if h[k] == h[k])
            rows.append(dict(Stock=s, Price=last, **{"Day %": ch, "Volume x": vol / h["avgvol"] if h["avgvol"] else float("nan"),
                                                    "To 52w high %": (h["hi52"] / last - 1) * 100, "From 52w low %": (last / h["lo52"] - 1) * 100,
                                                    "RSI": h["rsi"], "Trend": "Above 20/50/200" if above else "Below 20/50/200" if below else "Mixed"}))
        df = pd.DataFrame(rows)
        if df.empty:
            st.warning("No data yet: load the daily data first.")
            return
        out = apply_scan(df, scan)
        st.write(f"**{len(out)} stocks** match.")
        st.dataframe(out.round(2).head(60), hide_index=True, width="stretch")
        st.caption("Volume x = today's volume so far / 20-day average daily volume (early in the day it reads low). "
                   f"Stocks scanned: the F&O list. Updated {_now().strftime('%H:%M:%S')} IST.")
    _live(body, every=60)


def _series_options(kite, day):
    ix, eq = _idx_tokens(kite, day), _eq_tokens(kite, day)
    return ix, eq


def _close_of(kite, key, day, ix, eq):
    tok = ix.get(key) if key in ix else eq.get(key)
    if tok is None:
        return None
    h = _hist(kite, int(tok), day)
    return None if h is None or h.empty else h["close"]


def page_rotation(kite, ctx):
    day = _now().strftime("%Y-%m-%d")
    ix, eq = _series_options(kite, day)
    mode = st.radio("View", ["Sector rotation (RRG)", "RMI", "Ratio chart"], horizontal=True, key="ro_mode")
    if mode.startswith("Sector"):
        bench = _close_of(kite, BENCH, day, ix, eq)
        if bench is None:
            st.error(f"Could not load {BENCH} history.")
            return
        c1, c2 = st.columns(2)
        n = c1.slider("Average window (days)", 20, 120, 60, 5, key="rrg_n")
        tail = c2.slider("Tail length (days)", 3, 15, 6, 1, key="rrg_tail")
        c3, c4 = st.columns(2)
        zoom = c3.radio("Zoom", ["Centre", "Fit all"], horizontal=True, key="rrg_zoom")
        labels_on = c4.checkbox("Show labels on chart", value=True, key="rrg_lab")
        fig = go.Figure()
        rows, lasts = [], []
        for label, sym in SECTORS.items():
            s = _close_of(kite, sym, day, ix, eq) if sym in ix else None
            if s is None:
                continue
            p_ = rrg_points(s, bench, n=n)
            if p_.empty:
                continue
            t_ = p_.tail(tail)
            last = t_.iloc[-1]
            q_ = quadrant(last["ratio"], last["momentum"])
            rows.append(dict(Sector=label, **{"RS-Ratio": last["ratio"], "Momentum": last["momentum"], "Quadrant": q_}))
            lasts.append((label, float(last["ratio"]), float(last["momentum"]), q_))
            fig.add_trace(go.Scatter(x=t_["ratio"], y=t_["momentum"], mode="lines+markers", name=label, line=dict(width=1.2),
                                     marker=dict(size=[3] * (len(t_) - 1) + [9]), hoverinfo="name", showlegend=False, opacity=0.8))
        if not rows:
            st.warning("No sector index data found in Kite instruments.")
            return
        if labels_on:                       # labels on the last point only, nudged to different sides so they overlap less
            pos = ["top center", "bottom center", "middle right", "middle left"]
            order = sorted(range(len(lasts)), key=lambda i: (round(lasts[i][1]), lasts[i][2]))
            lab_pos = {i: pos[k % 4] for k, i in enumerate(order)}
            fig.add_trace(go.Scatter(x=[l[1] for l in lasts], y=[l[2] for l in lasts], mode="text", text=[l[0] for l in lasts],
                                     textposition=[lab_pos[i] for i in range(len(lasts))], textfont=dict(size=9),
                                     hoverinfo="skip", showlegend=False))
        xs = [l[1] for l in lasts]
        ys = [l[2] for l in lasts]
        if zoom == "Centre":
            half_x = max(4.0, min(10.0, 1.15 * max(abs(v - 100) for v in xs)))
            half_y = max(4.0, min(10.0, 1.15 * max(abs(v - 100) for v in sorted(ys, key=lambda v: abs(v - 100))[:max(1, int(len(ys) * 0.85))])))
            xr, yr = [100 - half_x, 100 + half_x], [100 - half_y, 100 + half_y]
        else:
            xr = [min(xs + [99]) - 1.5, max(xs + [101]) + 1.5]
            yr = [min(ys + [99]) - 1.5, max(ys + [101]) + 1.5]
        for (x0, x1, y0, y1, col) in ((100, 300, 100, 300, "rgba(34,197,94,0.10)"), (100, 300, -100, 100, "rgba(245,158,11,0.10)"),
                                      (-100, 100, -100, 100, "rgba(239,68,68,0.10)"), (-100, 100, 100, 300, "rgba(59,130,246,0.10)")):
            fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, fillcolor=col, line=dict(width=0), layer="below")
        for txt, ax, ay, xa, ya in (("Improving", 0.01, 0.99, "left", "top"), ("Leading", 0.99, 0.99, "right", "top"),
                                    ("Lagging", 0.01, 0.01, "left", "bottom"), ("Weakening", 0.99, 0.01, "right", "bottom")):
            fig.add_annotation(xref="paper", yref="paper", x=ax, y=ay, text=txt, showarrow=False, xanchor=xa, yanchor=ya,
                               font=dict(size=11, color="#94a3b8"))
        fig.add_hline(y=100, line_dash="dot", line_color="gray")
        fig.add_vline(x=100, line_dash="dot", line_color="gray")
        fig.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
                          xaxis=dict(title="RS-Ratio (strength vs Nifty 50)", range=xr), yaxis=dict(title="Momentum", range=yr))
        st.plotly_chart(fig, width="stretch")
        if zoom == "Centre":
            out_ = [l[0] for l in lasts if not (xr[0] <= l[1] <= xr[1] and yr[0] <= l[2] <= yr[1])]
            if out_:
                st.caption("Outside this zoom (see the table or switch to 'Fit all'): " + ", ".join(out_))
        st.dataframe(pd.DataFrame(rows).sort_values("RS-Ratio", ascending=False).round(2), hide_index=True, width="stretch")
        st.caption("Top right = Leading, bottom right = Weakening, bottom left = Lagging, top left = Improving. "
                   "This is a simplified RRG (relative strength against its own average), not the licensed JdK formula.")
        return
    options = sorted(set(ix) | set(eq))
    if mode == "RMI":
        c1, c2 = st.columns([2, 1])
        key = c1.selectbox("Index or stock", options, index=options.index("NIFTY BANK") if "NIFTY BANK" in options else 0, key="rmi_sym")
        mlen = c2.number_input("Momentum bars", 1, 20, 5, key="rmi_m")
        s = _close_of(kite, key, day, ix, eq)
        if s is None:
            st.error("No history.")
            return
        r = rmi(s, 14, int(mlen))
        s_, r_ = s.tail(160), r.tail(160)
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.04)
        fig.add_trace(go.Scatter(x=s_.index, y=s_, name=key, line=dict(color="#3b82f6")), row=1, col=1)
        fig.add_trace(go.Scatter(x=r_.index, y=r_, name="RMI", line=dict(color="#a855f7")), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#ef4444", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#22c55e", row=2, col=1)
        fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig, width="stretch")
        v = float(r.iloc[-1])
        st.metric("RMI now", f"{v:.1f}", "overbought (above 70)" if v > 70 else "oversold (below 30)" if v < 30 else "neutral", delta_color="off")
        return
    c1, c2 = st.columns(2)
    a = c1.selectbox("Numerator", options, index=options.index("NIFTY BANK") if "NIFTY BANK" in options else 0, key="rt_a")
    b = c2.selectbox("Denominator", options, index=options.index(BENCH) if BENCH in options else 0, key="rt_b")
    sa, sb = _close_of(kite, a, day, ix, eq), _close_of(kite, b, day, ix, eq)
    if sa is None or sb is None:
        st.error("No history for one of the symbols.")
        return
    j = pd.concat([sa, sb], axis=1, join="inner").dropna()
    ratio = (j.iloc[:, 0] / j.iloc[:, 1]).tail(250)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ratio.index, y=ratio, name=f"{a} / {b}", line=dict(color="#3b82f6")))
    fig.add_trace(go.Scatter(x=ratio.index, y=ratio.rolling(20).mean(), name="20-day average", line=dict(color="#f59e0b", dash="dot")))
    fig.update_layout(height=460, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
    st.caption("Rising line = the numerator is outperforming the denominator.")


def _wl_read(ctx):
    SL = ctx["SL"]
    try:
        if SL._can_sheet():
            ws = SL._gc().open_by_key(st.secrets["SHEET_ID"]).worksheet("Watchlist")
            v = ws.get("A2")
            if v and v[0]:
                return [x.strip().upper() for x in v[0][0].split(",") if x.strip()]
    except Exception:
        pass
    return None


def _wl_write(ctx, syms):
    SL = ctx["SL"]
    try:
        if SL._can_sheet():
            sh = SL._gc().open_by_key(st.secrets["SHEET_ID"])
            try:
                ws = sh.worksheet("Watchlist")
            except Exception:
                ws = sh.add_worksheet("Watchlist", rows=5, cols=2)
            ws.update(values=[["symbols"], [",".join(syms)]], range_name="A1:A2")
            return True
    except Exception:
        pass
    return False


def page_watchlist(kite, ctx):
    day = _now().strftime("%Y-%m-%d")
    eq = _eq_tokens(kite, day)
    if "wl" not in st.session_state:
        st.session_state["wl"] = _wl_read(ctx) or ["RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS"]
    txt = st.text_input("Stocks (NSE symbols, comma separated)", ", ".join(st.session_state["wl"]), key="wl_txt")
    if st.button("Save watchlist", key="wl_save"):
        syms = [x.strip().upper() for x in txt.split(",") if x.strip()]
        bad = [s for s in syms if s not in eq]
        syms = [s for s in syms if s in eq]
        st.session_state["wl"] = syms
        ok = _wl_write(ctx, syms)
        st.success("Saved" + (" to the Google Sheet." if ok else " for this session only (Sheet not available)."))
        if bad:
            st.warning("Not found on NSE: " + ", ".join(bad))
    syms = st.session_state["wl"]
    stats = _store().setdefault(("daily", day), {})

    def body():
        if not syms:
            st.info("Add some symbols above.")
            return
        q = _quotes(kite, tuple("NSE:" + s for s in syms))
        rows = []
        for s in syms:
            d = q.get("NSE:" + s)
            if not d:
                continue
            last, prev, ch = _chg(d)
            h = stats.get(s)
            rows.append({"Stock": s, "Price": last, "Day %": ch, "Volume": d.get("volume") or 0,
                         "To 52w high %": (h["hi52"] / last - 1) * 100 if h and last else "",
                         "RSI": h["rsi"] if h else ""})
        st.dataframe(pd.DataFrame(rows).round(2), hide_index=True, width="stretch")
        st.caption(f"52-week and RSI columns fill in after 'Load daily data' on the Scanners page. Updated {_now().strftime('%H:%M:%S')} IST.")
    _live(body)


# =====================================================================
# Charts (TradingView widget) and IPO list
# =====================================================================
TV_PRESETS = ["NSE:NIFTY", "NSE:BANKNIFTY", "NSE:CNXFINANCE", "NSE:CNXIT", "NSE:RELIANCE", "NSE:HDFCBANK", "NSE:ICICIBANK",
              "NSE:SBIN", "NSE:INFY", "NSE:TCS", "NSE:AXISBANK", "NSE:KOTAKBANK", "BSE:SENSEX"]


def _tv_html(symbol, interval, height):
    import json
    cfg = dict(autosize=True, symbol=symbol, interval=interval, timezone="Asia/Kolkata", theme="dark", style="1", locale="en",
               enable_publishing=False, allow_symbol_change=True, hide_side_toolbar=False, container_id="tv_chart")
    return ('<div style="height:%dpx;width:100%%"><div id="tv_chart" style="height:100%%;width:100%%"></div></div>'
            '<script src="https://s3.tradingview.com/tv.js"></script><script>new TradingView.widget(%s);</script>' % (height, json.dumps(cfg)))


KITE_IVL = {"1m": ("minute", 3), "5m": ("5minute", 7), "15m": ("15minute", 15), "1h": ("60minute", 40), "Day": ("day", 400)}


@st.cache_data(ttl=20, show_spinner=False)
def _candles(_kite, token, interval, days):
    to = _now().replace(tzinfo=None)
    d = _kite.historical_data(int(token), to - dt.timedelta(days=days), to, interval)
    df = pd.DataFrame(d)
    if df.empty:
        return df
    ix = pd.to_datetime(df["date"])
    if ix.dt.tz is not None:
        ix = ix.dt.tz_convert(IST).dt.tz_localize(None)
    df.index = ix
    cols = ["open", "high", "low", "close"] + (["volume"] if "volume" in df.columns else [])
    return df[cols]


def page_charts(kite, ctx):
    import streamlit.components.v1 as components
    day = _now().strftime("%Y-%m-%d")
    mode = st.radio("Chart", ["Kite chart (indices and stocks)", "TradingView (stocks only)"], horizontal=True, key="ch_mode")
    if mode.startswith("TradingView"):
        c1, c2 = st.columns([2, 1])
        pick = c1.selectbox("Symbol", [s for s in TV_PRESETS if not s.endswith(("NIFTY", "BANKNIFTY", "CNXFINANCE", "CNXIT", "SENSEX"))],
                            key="tv_pick2")
        own = c1.text_input("Or type a TradingView stock symbol (e.g. NSE:TATASTEEL)", "", key="tv_own")
        interval = c2.selectbox("Interval", ["1", "5", "15", "60", "D"], index=1, key="tv_int")
        st.caption("TradingView's free widget does not show NSE index data (it says 'Invalid symbol'), so use the Kite chart for Nifty / Bank Nifty. "
                   "Stocks may work; the data is TradingView's and can be delayed. Heavy for a phone, so it loads only when you tick the box.")
        if st.checkbox("Load the chart", value=False, key="tv_on"):
            components.html(_tv_html((own.strip() or pick), interval, 600), height=620)
        return
    ix, eq = _series_options(kite, day)
    idx_first = [k for k in ("NIFTY BANK", "NIFTY 50", "NIFTY FIN SERVICE", "NIFTY IT") if k in ix]
    options = idx_first + sorted(set(ix) - set(idx_first)) + sorted(eq)
    c1, c2 = st.columns([2, 1])
    sym = c1.selectbox("Index or stock", options, key="kc_sym")
    ivl = c2.radio("Interval", list(KITE_IVL), index=1, key="kc_ivl", horizontal=True)
    o1, o2 = st.columns(2)
    show_ema = o1.checkbox("EMA 9 and EMA 21", value=True, key="kc_ema")
    show_vwap = o2.checkbox("VWAP (stocks, intraday)", value=False, key="kc_vwap")
    kite_ivl, days = KITE_IVL[ivl]
    tok = ix.get(sym) if sym in ix else eq.get(sym)

    def body():
        df = _candles(kite, int(tok), kite_ivl, days)
        if df is None or df.empty:
            st.warning("No candles returned.")
            return
        fig = go.Figure(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name=sym,
                                       increasing_line_color="#22c55e", decreasing_line_color="#ef4444"))
        if show_ema:
            fig.add_trace(go.Scatter(x=df.index, y=df["close"].ewm(span=9, adjust=False).mean(), name="EMA9", line=dict(width=1.2, color="#f59e0b")))
            fig.add_trace(go.Scatter(x=df.index, y=df["close"].ewm(span=21, adjust=False).mean(), name="EMA21", line=dict(width=1.2, color="#3b82f6")))
        if show_vwap and kite_ivl != "day" and "volume" in df and float(df["volume"].sum()) > 0:
            tp = (df["high"] + df["low"] + df["close"]) / 3
            g = pd.Series(df.index.date, index=df.index)
            vw = (tp * df["volume"]).groupby(g).cumsum() / df["volume"].groupby(g).cumsum().replace(0, np.nan)
            fig.add_trace(go.Scatter(x=df.index, y=vw, name="VWAP", line=dict(width=1.2, color="#a855f7", dash="dot")))
        breaks = [dict(bounds=["sat", "mon"])]
        if kite_ivl != "day":
            breaks.append(dict(bounds=[15.5, 9.25], pattern="hour"))
        fig.update_xaxes(rangebreaks=breaks, rangeslider_visible=False)
        fig.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h"), dragmode="pan")
        st.plotly_chart(fig, width="stretch")
        last = df.iloc[-1]
        chg = (float(last["close"]) / float(df["close"].iloc[-2]) - 1) * 100 if len(df) > 1 else 0.0
        st.caption(f"{sym}  last {float(last['close']):,.2f} ({chg:+.2f}% vs the previous candle). Kite data, updated {_now().strftime('%H:%M:%S')} IST. "
                   "Indices have no volume, so VWAP is for stocks.")
    _live(body, every=30 if kite_ivl != "day" else 120)


def _nse_get(path):
    import requests
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
          "Accept": "application/json,text/plain,*/*", "Referer": "https://www.nseindia.com/"}
    ss = requests.Session()
    ss.get("https://www.nseindia.com/", headers=ua, timeout=8)
    r = ss.get("https://www.nseindia.com" + path, headers=ua, timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    return r.json()


@st.cache_data(ttl=1800, show_spinner=False)
def _ipo_data():
    out, err = {}, {}
    for key, path in (("Open now", "/api/ipo-current-issue"), ("Upcoming", "/api/all-upcoming-issues?category=ipo")):
        try:
            js = _nse_get(path)
            if isinstance(js, dict):
                for k in ("data", "issues", "rows"):
                    if isinstance(js.get(k), list):
                        js = js[k]
                        break
            out[key] = js if isinstance(js, list) else []
        except Exception as e:
            err[key] = f"{type(e).__name__}: {str(e)[:80]}"
    return out, err


IPO_COLS = ["symbol", "companyName", "series", "issueStartDate", "issueEndDate", "issuePrice", "priceBand", "issueSize",
            "noOfSharesOffered", "status", "noOfTime", "sr_no"]


def page_ipo(kite, ctx):
    data, err = _ipo_data()
    shown = False
    for key in ("Open now", "Upcoming"):
        rows = data.get(key)
        if rows:
            df = pd.DataFrame(rows)
            cols = [c for c in IPO_COLS if c in df.columns and c != "sr_no"] or list(df.columns)[:10]
            st.markdown(f"**{key}** ({len(df)})")
            st.dataframe(df[cols], hide_index=True, width="stretch")
            shown = True
        elif key in data:
            st.markdown(f"**{key}**: no IPOs listed right now.")
            shown = True
    for key, msg in err.items():
        st.caption(f"{key}: could not read NSE ({msg}).")
    if not shown or err:
        st.info("If the list is empty or NSE did not answer, open the lists directly:")
    st.link_button("NSE: current issues", "https://www.nseindia.com/market-data/all-upcoming-issues-ipo")
    st.link_button("Chittorgarh IPO list", "https://www.chittorgarh.com/ipo/ipo_dashboard.asp")
    st.caption("IPO data comes from NSE's public pages and is cached for 30 minutes. Subscription status, GMP and allotment are not included. "
               "Check the offer document (RHP) before any decision: this is information, not advice.")


PAGES = {"Heatmap": page_heatmap, "Futures": page_futures, "Scanners": page_scanners,
         "Rotation": page_rotation, "Watchlist": page_watchlist, "Charts": page_charts, "IPO": page_ipo}


def render(view, kite, ctx):
    fn = PAGES.get(view)
    if fn is None:
        return
    try:
        fn(kite, ctx)
    except Exception as e:
        st.error(f"{view} page error: {type(e).__name__}: {str(e)[:300]}")
        st.caption("The Greeks page is not affected. Pick 'Signals (Greeks)' above to go back.")
