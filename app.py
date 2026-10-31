"""Read-only market dashboard (Zerodha Kite Connect + global cues). NO order placement code."""
import hmac
import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from kiteconnect import KiteConnect
from streamlit_autorefresh import st_autorefresh

import engine as E

IST = ZoneInfo("Asia/Kolkata")
st.set_page_config(page_title="Market Dashboard", layout="wide", initial_sidebar_state="collapsed")

UND = {
    "BANKNIFTY": dict(spot="NSE:NIFTY BANK", yf="^NSEBANK", step=100),
    "NIFTY": dict(spot="NSE:NIFTY 50", yf="^NSEI", step=50),
}
GLOBAL = {
    "S&P 500": "^GSPC", "Nasdaq": "^IXIC", "Nikkei": "^N225", "Hang Seng": "^HSI",
    "DAX": "^GDAXI", "FTSE": "^FTSE", "Crude (WTI)": "CL=F", "Brent": "BZ=F",
    "USD/INR": "INR=X", "US 10Y": "^TNX", "India VIX": "^INDIAVIX",
}


@st.cache_resource
def store():
    """Server-side memory shared by this single-user app: daily token + OI baseline."""
    return {}


S = store()


def now():
    return dt.datetime.now(IST)


# ---------------- access control ----------------
def gate():
    pw = st.secrets.get("APP_PASSWORD", "")
    if not pw:
        st.error("Set APP_PASSWORD in the app secrets.")
        st.stop()
    if st.session_state.get("ok"):
        return
    x = st.text_input("Password", type="password")
    if x and hmac.compare_digest(x, pw):
        st.session_state.ok = True
        st.rerun()
    st.stop()


def get_kite():
    kite = KiteConnect(api_key=st.secrets["KITE_API_KEY"])
    today = now().strftime("%Y-%m-%d")
    tok = S.get("token")
    if tok and tok[0] == today:
        kite.set_access_token(tok[1])
        return kite
    rt = st.query_params.get("request_token")
    if rt:
        try:
            data = kite.generate_session(rt, api_secret=st.secrets["KITE_API_SECRET"])
            S["token"] = (today, data["access_token"])
            st.query_params.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Kite login failed: {e}")
    st.info("Kite login is needed once per day.")
    st.link_button("Login to Kite", kite.login_url())
    st.stop()


# ---------------- data ----------------
@st.cache_data(ttl=6 * 3600)
def load_instruments(_kite, day):
    return pd.DataFrame(_kite.instruments("NFO"))


@st.cache_data(ttl=120)
def global_data():
    px = yf.download(list(GLOBAL.values()), period="7d", interval="1d", progress=False, auto_adjust=True)["Close"]
    rows, ch = [], {}
    for name, sym in GLOBAL.items():
        try:
            s = px[sym].dropna()
            last, prev = float(s.iloc[-1]), float(s.iloc[-2])
            if name == "US 10Y":
                chg = (last - prev) * 100  # basis points
                unit = "bp"
            else:
                chg = (last / prev - 1) * 100
                unit = "%"
            ch[name] = chg
            rows.append(dict(Market=name, Last=round(last, 2), Change=f"{chg:+.2f}{unit}"))
        except Exception:
            rows.append(dict(Market=name, Last=np.nan, Change="n/a"))
    return pd.DataFrame(rows), ch


def get_hist(kite, token, yf_sym):
    try:
        to = now().replace(tzinfo=None)
        d = kite.historical_data(token, to - dt.timedelta(days=7), to, "5minute")
        df = pd.DataFrame(d)
        idx = pd.to_datetime(df["date"])
        if idx.dt.tz is not None:
            idx = idx.dt.tz_convert(IST).dt.tz_localize(None)
        df.index = idx
        return df[["open", "high", "low", "close"]], "Kite"
    except Exception:
        try:
            df = yf.download(yf_sym, period="5d", interval="5m", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            df.index = df.index.tz_convert(IST).tz_localize(None) if df.index.tz is not None else df.index
            return df[["open", "high", "low", "close"]].dropna(), "Yahoo (may be delayed)"
        except Exception:
            return None, "none"


def build_chain(kite, inst, name, expiry, spot, step, width):
    df = inst[(inst.name == name) & (inst.expiry == expiry) & (inst.instrument_type.isin(["CE", "PE"]))]
    atm = round(spot / step) * step
    df = df[(df.strike >= atm - width * step) & (df.strike <= atm + width * step)]
    q = kite.quote(["NFO:" + s for s in df.tradingsymbol])
    rows = []
    for _, r in df.iterrows():
        d = q.get("NFO:" + r.tradingsymbol)
        if d:
            rows.append(dict(strike=float(r.strike), type=r.instrument_type, ltp=d["last_price"],
                             oi=d.get("oi", 0), vol=d.get("volume", 0), sym=r.tradingsymbol))
    c = pd.DataFrame(rows)
    ce = c[c.type == "CE"].set_index("strike").drop(columns="type").add_prefix("ce_")
    pe = c[c.type == "PE"].set_index("strike").drop(columns="type").add_prefix("pe_")
    return ce.join(pe, how="outer").sort_index(), atm


# ---------------- app ----------------
gate()
kite = get_kite()

with st.sidebar:
    name = st.selectbox("Index", list(UND))
    strike_pref = st.radio("Option to suggest", ["ATM", "1 strike OTM"])
    width = st.slider("Strikes each side of ATM", 6, 20, 12)
    gift = st.number_input("Gift Nifty change % (type from your app, 0 = ignore)", value=0.0, step=0.1)
    refresh = st.selectbox("Auto refresh (sec)", [30, 60, 120, 0], index=1)
    block_expiry = st.checkbox("No signals on expiry day", value=False)

cfg = UND[name]
t = now()
market_open = t.weekday() < 5 and dt.time(9, 15) <= t.time() <= dt.time(15, 30)
in_window = t.weekday() < 5 and dt.time(9, 30) <= t.time() <= dt.time(15, 0)
if refresh and market_open:
    st_autorefresh(interval=refresh * 1000, key="auto")

inst = load_instruments(kite, t.strftime("%Y-%m-%d"))
expiries = sorted(e for e in inst[inst.name == name].expiry.unique() if e >= t.date())
with st.sidebar:
    expiry = st.selectbox("Expiry", expiries[:6], index=0)

try:
    ltp = kite.ltp([cfg["spot"]])[cfg["spot"]]
    spot, token = float(ltp["last_price"]), ltp["instrument_token"]
    fut_row = inst[(inst.name == name) & (inst.instrument_type == "FUT") & (inst.expiry >= t.date())].sort_values("expiry").iloc[0]
    fq = kite.quote(["NFO:" + fut_row.tradingsymbol])["NFO:" + fut_row.tradingsymbol]
    F = float(fq["last_price"])
    chain, atm = build_chain(kite, inst, name, expiry, spot, cfg["step"], width)
except Exception as e:
    st.error(f"Kite data error: {e}")
    st.stop()

exp_dt = dt.datetime.combine(expiry, dt.time(15, 30), tzinfo=IST)
T = max((exp_dt - t).total_seconds() / (365 * 86400), 1e-6)
chain = E.add_greeks(chain, F, T)

# OI baseline = first snapshot of the day (OI change is measured from here)
bkey = (name, expiry, t.strftime("%Y-%m-%d"))
base = S.setdefault("base", {})
if bkey not in base:
    base[bkey] = dict(chain=chain[["ce_oi", "pe_oi"]].copy(), fut_px=F, fut_oi=fq.get("oi", 0), at=t.strftime("%H:%M"))
b = base[bkey]
chain["ce_chg"] = chain["ce_oi"] - b["chain"]["ce_oi"].reindex(chain.index).fillna(chain["ce_oi"])
chain["pe_chg"] = chain["pe_oi"] - b["chain"]["pe_oi"].reindex(chain.index).fillna(chain["pe_oi"])

summ = E.oi_summary(chain)
hist, hsrc = get_hist(kite, token, cfg["yf"])
tr = E.trend_state(hist)
gdf, gch = global_data()
gscore = E.global_score(gch, gift)
gdir = E.sign_dir(gscore, 0.3)
fut_px_chg = F - b["fut_px"]
fut_oi_chg = fq.get("oi", 0) - b["fut_oi"]
odir, oscore, onotes = E.oi_direction(summ["pcr"], fut_px_chg, fut_oi_chg, spot, summ["support"], summ["resistance"])

is_expiry_day = expiry == t.date()
sig = E.make_signal(spot, tr, odir, gdir, summ["support"], summ["resistance"],
                    in_window and not (block_expiry and is_expiry_day), market_open)

# ---------- header ----------
st.caption(f"{t.strftime('%a %d %b %Y %H:%M:%S')} IST | {'MARKET OPEN' if market_open else 'MARKET CLOSED (last data)'}"
           f" | OI baseline taken {b['at']} | trend data: {hsrc}")
c1, c2, c3, c4 = st.columns(4)
c1.metric(name, f"{spot:,.1f}")
c2.metric("PCR", f"{summ['pcr']:.2f}")
c3.metric("OI support / resistance", f"{summ['support']:.0f} / {summ['resistance']:.0f}")
c4.metric("Max pain", f"{summ['max_pain']:.0f}")

# ---------- signal card ----------
if sig["action"] == "NO TRADE":
    st.warning(f"**NO TRADE** - {sig['reason']}")
else:
    L = sig["levels"]
    side, mult = sig["side"], (0 if strike_pref == "ATM" else 1)
    k = atm + (mult * cfg["step"] if side == "CE" else -mult * cfg["step"])
    k = min(chain.index, key=lambda x: abs(x - k))
    ltp_o, dlt = chain.loc[k, f"{side.lower()}_ltp"], chain.loc[k, f"{side.lower()}_delta"]
    ol = E.option_levels(sig, ltp_o, dlt)
    (st.success if side == "CE" else st.error)(f"**{sig['action']}** - {name} {k:.0f} {side} ({expiry})")
    a, b_, c_ = st.columns(3)
    a.metric("Index entry / SL", f"{L['entry']:.0f} / {L['sl']:.0f}")
    b_.metric("Index T1 / T2", f"{L['t1']:.0f} / {L['t2']:.0f}")
    c_.metric("Option premium now", f"{ltp_o:.1f}", f"delta {dlt:.2f}")
    st.write(f"Approx. option levels (delta based): SL **{ol['sl']:.1f}**, T1 **{ol['t1']:.1f}**, T2 **{ol['t2']:.1f}**. "
             "Signal is void if index closes a 5-min candle beyond the SL level. You place the order yourself.")

with st.expander("Why? (checklist)", expanded=sig["action"] == "NO TRADE"):
    chk = pd.DataFrame(sig["checks"])
    chk["ok"] = chk["ok"].map({True: "YES", False: "NO"})
    st.dataframe(chk, hide_index=True, use_container_width=True)
    st.write(f"OI score {oscore:+.1f}: " + "; ".join(onotes))
    st.write(f"Global score {gscore:+.2f} (Gift Nifty {'included' if gift else 'not included'})")

# ---------- tabs ----------
t1, t2, t3, t4 = st.tabs(["Chart", "Option chain", "OI chart", "Global"])
with t1:
    d = tr["df"]
    if d is not None and len(d) > 5:
        d = d.iloc[-150:]
        fig = go.Figure(go.Candlestick(x=d.index, open=d.open, high=d.high, low=d.low, close=d.close, name=name))
        if "ema20" in d:
            fig.add_scatter(x=d.index, y=d.ema20, name="EMA20")
            fig.add_scatter(x=d.index, y=d.ema50, name="EMA50")
        for lvl, nm in ((summ["support"], "OI support"), (summ["resistance"], "OI resistance")):
            if lvl == lvl:
                fig.add_hline(y=lvl, line_dash="dot", annotation_text=nm)
        fig.update_layout(height=420, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=10, b=0))
        fig.update_xaxes(type="category", showticklabels=False)
        st.plotly_chart(fig, use_container_width=True)
    st.write(f"5-min trend: **{tr['label']}**")
with t2:
    show = chain.reset_index().rename(columns={"strike": "STRIKE"})
    cols = ["ce_oi", "ce_chg", "ce_iv", "ce_delta", "ce_vega", "ce_ltp", "STRIKE",
            "pe_ltp", "pe_vega", "pe_delta", "pe_iv", "pe_chg", "pe_oi"]
    st.dataframe(show[cols].round(2), hide_index=True, use_container_width=True, height=520)
    st.caption("OI chg = change since the first snapshot today. IV/delta/vega computed here from futures price (approximate).")
with t3:
    fig = go.Figure()
    fig.add_bar(x=chain.index, y=chain.ce_oi, name="Call OI", marker_color="#d9534f")
    fig.add_bar(x=chain.index, y=chain.pe_oi, name="Put OI", marker_color="#2e8b57")
    fig.add_vline(x=spot, line_dash="dash")
    fig.update_layout(barmode="group", height=380, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
with t4:
    st.dataframe(gdf, hide_index=True, use_container_width=True)
    st.caption("Daily change from Yahoo Finance (can lag). Yield change in basis points. Gift Nifty: enter in sidebar.")

st.caption("Educational analysis only. Not investment advice. Read-only: this app cannot place orders.")
