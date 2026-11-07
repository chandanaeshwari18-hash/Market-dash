"""Read-only market dashboard (Zerodha Kite Connect + global cues). NO order placement code."""
import hmac
import time as _time
import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
from kiteconnect import KiteConnect

import importlib
import engine as E
import stocks as K
import sheets_log as SL
import market_view as MV
for _m in (E, K, SL, MV):          # pick up freshly uploaded files without a manual reboot
    importlib.reload(_m)

IST = ZoneInfo("Asia/Kolkata")
st.set_page_config(page_title="Market Dashboard", layout="wide", initial_sidebar_state="collapsed")

# ---------- colourful look (CSS only; works on both the light and the dark theme) ----------
try:                                           # Streamlit 1.46+ tells us which theme the viewer is using
    _THEME = str(st.context.theme.type)
except Exception:
    _THEME = ""
_DARK = _THEME == "dark"
_BG = {"dark": ("#0a0f1f", "#101a33", "#0c1c2e", "#14122e", "#0f2038"),
       "light": ("#f4f7ff", "#eef6ff", "#f1fbf6", "#ece9ff", "#e3f2ff")}.get(_THEME)
_BG_CSS = ("" if not _BG else
           f".stApp {{ background: linear-gradient(160deg, {_BG[0]} 0%, {_BG[1]} 50%, {_BG[2]} 100%); }}"
           f"[data-testid=\"stSidebar\"] {{ background: linear-gradient(180deg, {_BG[3]} 0%, {_BG[4]} 100%); }}")
st.markdown("""
<style>
""" + _BG_CSS + """
[data-testid="stSidebar"] { border-right: 2px solid #8b5cf655; }
h1, h2, h3 { background: linear-gradient(90deg, #0ea5e9, #8b5cf6 55%, #ec4899); -webkit-background-clip: text;
             -webkit-text-fill-color: transparent; font-weight: 800; }
[data-testid="stMetric"] { background: rgba(127,127,127,0.12); border: 1px solid rgba(127,127,127,0.25);
             border-left: 5px solid #0ea5e9; border-radius: 14px; padding: 10px 14px; }
[data-testid="column"]:nth-child(4n+2) [data-testid="stMetric"], [data-testid="stColumn"]:nth-child(4n+2) [data-testid="stMetric"] { border-left-color: #8b5cf6; }
[data-testid="column"]:nth-child(4n+3) [data-testid="stMetric"], [data-testid="stColumn"]:nth-child(4n+3) [data-testid="stMetric"] { border-left-color: #f59e0b; }
[data-testid="column"]:nth-child(4n+4) [data-testid="stMetric"], [data-testid="stColumn"]:nth-child(4n+4) [data-testid="stMetric"] { border-left-color: #ec4899; }
[data-baseweb="tab-list"] { gap: 6px; }
[data-baseweb="tab"] { background: rgba(127,127,127,0.14); border-radius: 10px 10px 0 0; padding: 6px 14px; }
[data-baseweb="tab"][aria-selected="true"] { background: linear-gradient(90deg, #6d5dfc, #0ea5e9); color: #ffffff; font-weight: 700; }
[data-baseweb="tab-highlight"] { background: #ec4899 !important; }
[data-testid="stRadio"] [role="radiogroup"] { gap: 8px; }
[data-testid="stRadio"] [role="radiogroup"] > label { background: rgba(127,127,127,0.15); border: 1px solid rgba(127,127,127,0.30);
             border-radius: 999px; padding: 6px 16px; }
[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) { background: linear-gradient(90deg, #6d5dfc, #0ea5e9);
             color: #ffffff; font-weight: 700; border-color: transparent; }
[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) * { color: #ffffff; }
[data-testid="stAlert"] { border-radius: 14px; border-left: 6px solid #f59e0b; }
[data-testid="stExpander"] { border: 1px solid #8b5cf666; border-radius: 12px; background: rgba(139,92,246,0.08); }
[data-testid="stDataFrame"] { border: 1px solid #0ea5e966; border-radius: 12px; overflow: hidden; }
.stButton > button { background: linear-gradient(90deg, #6d5dfc, #0ea5e9); color: white; border: 0; border-radius: 10px; font-weight: 700; }
.mdchip { display: inline-block; padding: 4px 12px; border-radius: 999px; margin: 2px 6px 2px 0; font-weight: 700; color: #fff; font-size: 0.9rem; }
.mdg { background: #16a34a; } .mdr { background: #dc2626; } .mda { background: #d97706; } .mdn { background: #64748b; }
.mdmood { padding: 14px 18px; border-radius: 16px; color: #fff; font-weight: 800; font-size: 1.15rem; margin: 6px 0 10px 0; }
.mdbar { display: flex; height: 22px; border-radius: 11px; overflow: hidden; margin: 6px 0; }
.mdbanner { padding: 10px 16px; border-radius: 14px; margin: 4px 0 10px 0; font-weight: 800; font-size: 1.15rem; color: #fff;
            background: linear-gradient(90deg, #6d5dfc 0%, #0ea5e9 45%, #10b981 75%, #f59e0b 100%); }
</style>
""", unsafe_allow_html=True)

# table colours: translucent fills keep the normal text colour, so they read well on a light AND a dark background
_GREEN = "background-color:rgba(34,197,94,0.30);font-weight:700"
_RED = "background-color:rgba(244,63,94,0.30);font-weight:700"
_AMBER = "background-color:rgba(245,158,11,0.32);font-weight:600"
_BLUE = "background-color:rgba(14,165,233,0.25)"
_POS, _NEG = "#16a34a" if not _DARK else "#4ade80", "#e11d48" if not _DARK else "#fb7185"


def _cell(v):
    s_ = str(v)
    if s_ in ("ALERT", "Bullish", "BUY CE", "T2", "Positive", "Broad strength") or s_.startswith("T1"):
        return _GREEN
    if s_ in ("SL", "Bearish", "BUY PE", "Negative", "Broad weakness") or s_.startswith("Conflict"):
        return _RED
    if s_ in ("YES", "illiquid", "Mixed", "OPEN", "WAITING"):
        return _AMBER
    if s_.startswith("Day end") or s_ == "Neutral":
        return _BLUE
    return ""


def _num(v):
    try:
        x_ = float(v)
    except Exception:
        return ""
    return f"color:{_POS};font-weight:600" if x_ > 0 else f"color:{_NEG};font-weight:600" if x_ < 0 else ""


def _num_inv(v):
    try:
        x_ = float(v)
    except Exception:
        return ""
    return f"color:{_NEG};font-weight:600" if x_ > 0 else f"color:{_POS};font-weight:600" if x_ < 0 else ""


def paint(df, num_cols=(), yes_no=False, inv_cols=(), int_cols=()):
    """Colour a table: green = bullish / profit, red = bearish / loss, amber = wait. Falls back to the plain table."""
    try:
        sty = df.style
        mp = getattr(sty, "map", None) or sty.applymap
        sty = mp(_cell) if not yes_no else mp(lambda v: _GREEN if str(v) == "YES" else _RED if str(v) == "NO" else "")
        nc = [c_ for c_ in num_cols if c_ in df.columns]
        if nc:
            sty = (getattr(sty, "map", None) or sty.applymap)(_num, subset=nc)
        ic = [c_ for c_ in inv_cols if c_ in df.columns]
        if ic:
            sty = (getattr(sty, "map", None) or sty.applymap)(_num_inv, subset=ic)
        sty = sty.format(precision=2, na_rep="-")
        ints = [c_ for c_ in int_cols if c_ in df.columns]
        if ints:
            sty = sty.format("{:,.0f}", subset=ints, na_rep="-")
        return sty
    except Exception:
        return df

UND = {
    "BANKNIFTY": dict(spot="NSE:NIFTY BANK", yf="^NSEBANK", step=100),
    "NIFTY": dict(spot="NSE:NIFTY 50", yf="^NSEI", step=50),
    "FINNIFTY": dict(spot="NSE:NIFTY FIN SERVICE", yf="NIFTY_FIN_SERVICE.NS", step=50),
    "MIDCPNIFTY": dict(spot="NSE:NIFTY MID SELECT", yf="^NSEMDCP50", step=25),
    "SENSEX": dict(spot="BSE:SENSEX", yf="^BSESN", step=100),
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
    # "Remember me": a token in the page address lets a reopened tab skip the password. Bookmark the address after login.
    token = hmac.new(pw.encode(), b"market-dash-remember", "sha256").hexdigest()[:32]
    if hmac.compare_digest(str(st.query_params.get("t", "")), token):
        st.session_state.ok = True
        return
    x = st.text_input("Password", type="password")
    if x and hmac.compare_digest(x, pw):
        st.session_state.ok = True
        st.query_params["t"] = token
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
            for _k in ('request_token', 'action', 'status', 'type'):
                st.query_params.pop(_k, None)
            st.rerun()
        except Exception as e:
            st.error(f"Kite login failed: {e}")
    st.info("Kite login is needed once per day.")
    st.link_button("Login to Kite", kite.login_url())
    st.stop()


# ---------------- data ----------------
@st.cache_data(ttl=6 * 3600)
def load_instruments(_kite, day):
    df = pd.DataFrame(_kite.instruments("NFO"))
    try:                                               # SENSEX options/futures trade on BSE (BFO)
        b = pd.DataFrame(_kite.instruments("BFO"))
        df = pd.concat([df, b[b.name.isin(["SENSEX"])]], ignore_index=True)
    except Exception:
        pass
    if "exchange" not in df.columns:
        df["exchange"] = "NFO"
    return df


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


@st.cache_data(ttl=5)
def kite_candles(_kite, token):
    """5-minute candles with volume (use the futures token: indices have no volume)."""
    to = now().replace(tzinfo=None)
    df = pd.DataFrame(_kite.historical_data(token, to - dt.timedelta(days=7), to, "5minute"))
    idx = pd.to_datetime(df["date"])
    if idx.dt.tz is not None:
        idx = idx.dt.tz_convert(IST).dt.tz_localize(None)
    df.index = idx
    return df[["open", "high", "low", "close", "volume"]]


def _r1(x):
    try:
        return round(float(x), 1) if x == x else ""
    except Exception:
        return ""


@st.cache_data(ttl=120)
def stock_candles(_kite, token):
    """5-minute futures candles with volume and open interest, last few sessions."""
    to = now().replace(tzinfo=None)
    df = pd.DataFrame(_kite.historical_data(token, to - dt.timedelta(days=5), to, "5minute", oi=True))
    ix = pd.to_datetime(df["date"])
    if ix.dt.tz is not None:
        ix = ix.dt.tz_convert(IST).dt.tz_localize(None)
    df.index = ix
    return df[[c for c in ("open", "high", "low", "close", "volume", "oi") if c in df.columns]]


@st.cache_data(ttl=300)
def mood_data():
    """Last price and daily change for global indices, commodities, currencies, bond yields and VIX (Yahoo Finance)."""
    px = yf.download([m_[2] for m_ in MV.MOOD], period="10d", interval="1d", progress=False, auto_adjust=True)["Close"]
    out = {}
    for _g, nm, sym, _w, _th, unit in MV.MOOD:
        try:
            s_ = px[sym].dropna()
            last, prev = float(s_.iloc[-1]), float(s_.iloc[-2])
            out[nm] = (last, (last - prev) * 100 if unit == "bp" else (last / prev - 1) * 100)
        except Exception:
            out[nm] = (np.nan, np.nan)
    return out


@st.cache_data(ttl=60)
def fo_breadth(_kite, syms):
    """Advance / decline across all F&O stocks (one or two Kite quote calls)."""
    q = {}
    for i_ in range(0, len(syms), 400):
        q.update(_kite.quote(list(syms[i_:i_ + 400])))
    return MV.breadth(q)


_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
       "Accept": "text/csv,application/json,*/*", "Referer": "https://www.nseindia.com/"}


@st.cache_data(ttl=1800)
def fetch_participant_oi():
    """Latest two trading days of NSE participant-wise open interest (FII / DII / Client / Pro). Returns (list, error)."""
    import requests
    got, tries, err, exc = [], 0, "", 0
    for i_ in range(0, 12):
        d_ = now().date() - dt.timedelta(days=i_)
        if d_.weekday() >= 5:
            continue
        for host in ("nsearchives.nseindia.com", "archives.nseindia.com"):
            tries += 1
            try:
                r = requests.get(f"https://{host}/content/nsccl/fao_participant_oi_{d_:%d%m%Y}.csv", headers=_UA, timeout=6)
                if r.status_code == 200:
                    pr = MV.parse_participant_oi(r.text)
                    if pr:
                        got.append((d_.strftime("%d %b %Y"), pr))
                        break
                else:
                    err = f"HTTP {r.status_code}"
            except Exception as e_:
                err = type(e_).__name__
                exc += 1
        if len(got) >= 2 or (not got and (tries >= 6 or exc >= 2)):      # stop early when NSE is not reachable
            break
    return got, ("" if got else (err or "no file found"))


@st.cache_data(ttl=900)
def fetch_cash_flows():
    import requests
    try:
        ss = requests.Session()
        ss.get("https://www.nseindia.com/", headers=_UA, timeout=8)
        r = ss.get("https://www.nseindia.com/api/fiidiiTradeReact", headers=_UA, timeout=8)
        if r.status_code == 200:
            cf = MV.cash_flows(r.json())
            return cf, ("" if cf else "empty reply")
        return {}, f"HTTP {r.status_code}"
    except Exception as e_:
        return {}, type(e_).__name__


@st.cache_data(ttl=120)
def fetch_gift():
    """GIFT Nifty from NSE's marketStatus feed (same NSE session approach as FII/DII). Returns (dict or None, error text)."""
    import requests
    try:
        ss = requests.Session()
        ss.get("https://www.nseindia.com/", headers=_UA, timeout=8)
        r = ss.get("https://www.nseindia.com/api/marketStatus", headers=_UA, timeout=8)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}"
        g = MV.parse_gift(r.json())
        return g, ("" if g else "no GIFT Nifty block in reply")
    except Exception as e_:
        return None, type(e_).__name__


HEAVY = {
    "BANKNIFTY": ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK"],
    "NIFTY": ["RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", "ITC", "LT", "BHARTIARTL"],
    "FINNIFTY": ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK", "BAJFINANCE"],
    "SENSEX": ["RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", "ITC", "LT", "BHARTIARTL"],
}


@st.cache_data(ttl=30)
def heavy_confirm(_kite, idx_name):
    """Day change of the index heavyweights: (up, down, flat, [(name, %)]) - do the big stocks agree with the index?"""
    syms = HEAVY.get(idx_name)
    if not syms:
        return None
    try:
        q = _kite.quote([f"NSE:{s_}" for s_ in syms])
        rows = []
        for s_ in syms:
            d_ = q.get(f"NSE:{s_}")
            if d_ and d_.get("ohlc", {}).get("close"):
                rows.append((s_, (d_["last_price"] / d_["ohlc"]["close"] - 1) * 100))
        if not rows:
            return None
        return (sum(1 for _, p_ in rows if p_ > 0.05), sum(1 for _, p_ in rows if p_ < -0.05),
                sum(1 for _, p_ in rows if abs(p_) <= 0.05), rows)
    except Exception:
        return None


@st.cache_data(ttl=3600)
def fetch_events():
    """Upcoming results / board meetings from NSE (same NSE session approach). Returns (rows, error text)."""
    import requests
    errs = []
    for url in ("https://www.nseindia.com/api/event-calendar",
                "https://www.nseindia.com/api/corporate-board-meetings?index=equities"):
        try:
            ss = requests.Session()
            ss.get("https://www.nseindia.com/", headers=_UA, timeout=8)
            r = ss.get(url, headers=_UA, timeout=10)
            if r.status_code != 200:
                errs.append(f"HTTP {r.status_code}")
                continue
            rows = MV.parse_events(r.json(), now().date(), 10, set(K.STOCKS.values()))
            if rows:
                return rows, ""
            errs.append("no results events in reply")
        except Exception as e_:
            errs.append(type(e_).__name__)
    return [], "; ".join(errs)


def get_fut_hist(kite, token):
    try:
        return kite_candles(kite, token)
    except Exception:
        return None


def build_chain(kite, inst, name, expiry, spot, step, width):
    df = inst[(inst.name == name) & (inst.expiry == expiry) & (inst.instrument_type.isin(["CE", "PE"]))]
    atm = round(spot / step) * step
    df = df[(df.strike >= atm - width * step) & (df.strike <= atm + width * step)]
    q = kite.quote([f"{r.exchange}:{r.tradingsymbol}" for r in df.itertuples()])
    rows = []
    for _, r in df.iterrows():
        d = q.get(f"{r.exchange}:{r.tradingsymbol}")
        if d:
            rows.append(dict(strike=float(r.strike), type=r.instrument_type, ltp=d["last_price"],
                             oi=d.get("oi", 0), vol=d.get("volume", 0), sym=r.tradingsymbol,
                             bid=((d.get("depth") or {}).get("buy") or [{}])[0].get("price", 0),
                             ask=((d.get("depth") or {}).get("sell") or [{}])[0].get("price", 0)))
    c = pd.DataFrame(rows)
    ce = c[c.type == "CE"].set_index("strike").drop(columns="type").add_prefix("ce_")
    pe = c[c.type == "PE"].set_index("strike").drop(columns="type").add_prefix("pe_")
    return ce.join(pe, how="outer").sort_index(), atm


def SCORE_NEED(mode):
    return E.SCORE_MODES[mode]["aplus"]


# ---------------- app ----------------
gate()
kite = get_kite()

with st.sidebar:
    strike_pref = st.radio("Option to suggest", ["ATM", "1 strike OTM"])
    width = st.slider("Strikes each side of ATM", 6, 20, 12)
    gift_auto = st.checkbox("Gift Nifty: fetch automatically", value=True)
    gift_manual = st.number_input("Gift Nifty change % (manual, used if auto is off or fails; 0 = ignore)", value=0.0, step=0.1)
    gift, gift_info = gift_manual, "manual entry"
    if gift_auto:
        _gd, _ge = fetch_gift()
        if _gd:
            gift = round(_gd["pct"], 2)
            gift_info = f"auto {gift:+.2f}%" + (f" at {_gd['last']:,.0f}" if _gd["last"] == _gd["last"] else "") + (f" ({_gd['time']})" if _gd["time"] else "")
        else:
            gift_info = f"auto failed ({_ge}), using manual"
    st.caption(f"Gift Nifty: {gift_info}")
    refresh = st.selectbox("Auto refresh (sec)", [5, 10, 15, 30, 60, 120, 0], index=4)
    block_expiry = st.checkbox("No signals on expiry day", value=False)
    rule = st.radio("Build-up rule", ["My rule", "Standard"],
                    help="My rule: CE long buildup + PE short covering = BUY CE; PE short buildup + CE long unwinding = BUY PE. "
                         "Standard: bullish = CE (long buildup/short covering) with PE (short buildup/long unwinding); bearish is the mirror.")
    lookback = st.selectbox("Build-up lookback (min)", [5, 15, 30, 60], index=1)
    confirm_n = st.number_input("Confirm after N refreshes", 1, 5, 2)
    stock_scan = st.checkbox("Stock scanner (alerts only on clear momentum + direction)", value=True)
    keep_awake = st.checkbox("Keep screen on (while this page is open)", value=True)
    sound_on = st.checkbox("Sound + vibration when an alert fires", value=True,
                           help="Needs this page open and you must have tapped something on it once (phone browser rule). Turn the phone volume up.")
    if sound_on and st.button("Test alert sound"):
        S["test_sound"] = True
    g_mode = "10-point (final scanner)"            # only mode in use: alerts at 10/10, 9/10, 8/10
    st.caption("Greeks scoring: 10-point. Entry only at 10/10, 9/10 or 8/10 (7 and below = no alert).")
    g_strike = st.radio("Greeks scanner strike", ["ATM", "1 strike ITM (safer)"])
    st.markdown("**Paper trade size**")
    cap_amt = st.number_input("Capital (Rs)", min_value=10000, value=100000, step=10000)
    risk_pct = st.number_input("Risk per trade (% of capital)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
    fixed_lots = st.number_input("Lots (0 = auto from risk)", min_value=0, max_value=100, value=0, step=1)
    slip_pct = st.number_input("Slippage per side (% of premium)", min_value=0.0, max_value=5.0, value=1.0, step=0.25,
                               help="Real fills are usually a little worse than the signal price. 1% is a fair guess for liquid index options.")

st.markdown('<div class="mdbanner">Market Dashboard &nbsp;|&nbsp; read-only signals, paper-trade tracker</div>', unsafe_allow_html=True)
name = st.radio("Index", list(UND), horizontal=True, key="idx", label_visibility="collapsed")

cfg = UND[name]
t = now()
market_open = t.weekday() < 5 and dt.time(9, 15) <= t.time() <= dt.time(15, 30)
in_window = t.weekday() < 5 and dt.time(9, 20) <= t.time() <= dt.time(15, 15)
if refresh and market_open:
    pass                                         # refresh now happens at the end of the script (no extra component)

inst = load_instruments(kite, t.strftime("%Y-%m-%d"))


def size_cols(sym, expiry_, entry_, sl_):
    """Lots / quantity / amounts for one alert, as Sheet columns (lot size comes from the Kite instrument list)."""
    try:
        ls = inst[(inst.name == sym) & (inst.expiry.astype(str) == str(expiry_)) & inst.instrument_type.isin(["CE", "PE"])]
        lot = int(ls.iloc[0].lot_size) if not ls.empty else 0
    except Exception:
        lot = 0
    return E.position_size(entry_, sl_, lot, cap_amt, risk_pct, fixed_lots)
expiries = sorted(e for e in inst[inst.name == name].expiry.unique() if e >= t.date())
with st.sidebar:
    expiry = st.selectbox("Expiry", expiries[:6], index=0)

try:
    ltp = kite.ltp([cfg["spot"]])[cfg["spot"]]
    spot, token = float(ltp["last_price"]), ltp["instrument_token"]
    fut_row = inst[(inst.name == name) & (inst.instrument_type == "FUT") & (inst.expiry >= t.date())].sort_values("expiry").iloc[0]
    _fk = f"{fut_row.exchange}:{fut_row.tradingsymbol}"
    fq = kite.quote([_fk])[_fk]
    try:                                               # strike gap straight from the instrument list
        _ks = inst[(inst.name == name) & (inst.expiry == expiry) & (inst.instrument_type == "CE")].strike
        _st = K.pick_step(_ks, spot)
        if _st:
            cfg = dict(cfg, step=float(_st))
    except Exception:
        pass
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

summ = E.oi_summary(chain, spot)
hist, hsrc = get_hist(kite, token, cfg["yf"])
tr = E.trend_state(hist)

# ---------- volume, VWAP, buyer / seller zones (futures data, shifted to index scale) ----------
fut_hist = get_fut_hist(kite, int(fut_row.instrument_token))
basis = F - spot
zones, vp = [], None
if fut_hist is not None and len(fut_hist) > 30:
    recent = fut_hist.iloc[-300:]
    vp = E.volume_profile(recent)
    for z in E.volume_spike_zones(recent):
        z["low"] -= basis
        z["high"] -= basis
        zones.append(z)
zones += E.oi_wall_zones(chain, spot, cfg["step"])
gdf, gch = global_data()
gscore = E.global_score(gch, gift)
gdir = E.sign_dir(gscore, 0.3)
fut_px_chg = F - b["fut_px"]
fut_oi_chg = fq.get("oi", 0) - b["fut_oi"]
odir, oscore, onotes = E.oi_direction(summ["pcr"], fut_px_chg, fut_oi_chg, spot, summ["support"], summ["resistance"])

is_expiry_day = expiry == t.date()
sig = E.make_signal(spot, tr, odir, gdir, summ["support"], summ["resistance"],
                    in_window and not (block_expiry and is_expiry_day), market_open)

# ---------- per-strike build-up signal ----------
snaps = S.setdefault("snaps", {}).setdefault(bkey, [])
if not snaps or (t - snaps[-1]["t"]).total_seconds() >= 20:
    snaps.append(dict(t=t, chain=chain[["ce_ltp", "pe_ltp", "ce_oi", "pe_oi"]].copy()))
    del snaps[:-500]
ref, ref_full = E.pick_reference(snaps, t, lookback)
bu_df = E.buildup_analysis(chain, ref, atm, cfg["step"], 3)
bu_sides = E.side_states(bu_df)
bsig = E.buildup_signal(bu_sides, rule, spot, atm, cfg["step"], tr, summ,
                        in_window and not (block_expiry and is_expiry_day), chain)
c_side, c_n = S.get("bu_conf", (None, 0))
if bsig["side"]:
    c_n = c_n + 1 if c_side == bsig["side"] else 1
    c_side = bsig["side"]
else:
    c_side, c_n = None, 0
S["bu_conf"] = (c_side, c_n)
bu_confirmed = bool(bsig["side"]) and c_n >= confirm_n
fired = S.setdefault("fired", [])
new_alert = False
_n_fired0 = None            # set after the Sheet reload below; anything added after this is a NEW alert this run
today = t.strftime("%Y-%m-%d")
if SL.enabled() and S.get("sheet_day") != today:  # after a restart, reload today's alerts so they are not repeated
    S["sheet_day"] = today
    for r in SL.read_alerts():
        if str(r.get("date")) == today:
            try:
                fk0 = (today, r.get("index"), str(r.get("action", "")).replace("BUY ", ""), float(r.get("strike")))
                if str(r.get("source")) in ("Greeks", "Stock"):
                    fk0 = fk0 + (str(r.get("source")),)
            except Exception:
                continue
            if not any(f["key"] == fk0 for f in fired):
                fired.append(dict(key=fk0, time=r.get("time"), index=r.get("index"), action=r.get("action"),
                                  strike=r.get("strike"), entry=r.get("entry"), sl=r.get("sl"),
                                  t1=r.get("t1"), t2=r.get("t2")))
_n_fired0 = len(fired)
if bu_confirmed:
    fk = (t.strftime("%Y-%m-%d"), name, bsig["side"], bsig["strike"])
    if not any(f["key"] == fk for f in fired):
        fired.append(dict(key=fk, time=t.strftime("%H:%M:%S"), index=name, action=bsig["action"],
                          strike=bsig["strike"], entry=bsig["opt"]["entry"], sl=bsig["opt"]["sl"],
                          t1=bsig["opt"]["t1"], t2=bsig["opt"]["t2"]))
        new_alert = True
        if SL.enabled():
            ok, err = SL.append_alert(dict(date=today, time=t.strftime("%H:%M:%S"), index=name, expiry=str(expiry),
                                           action=bsig["action"], strike=bsig["strike"], source="Build-up",
                                           entry=round(bsig["opt"]["entry"], 1), sl=round(bsig["opt"]["sl"], 1),
                                           t1=round(bsig["opt"]["t1"], 1), t2=round(bsig["opt"]["t2"], 1),
                                           **{k_: v_ for k_, v_ in size_cols(name, expiry, bsig["opt"]["entry"], bsig["opt"]["sl"]).items()
                                              if k_ != "note"}))
            if not ok:
                st.warning(f"Could not write to Google Sheet: {err}")
            else:
                SL.read_alerts.clear()

# ---------- Greeks scanner (price action = trigger, OI = confirmation, Greeks = strike selection) ----------
try:
    _a = chain.loc[atm]
    atm_iv = float(np.nanmean([_a["ce_iv"], _a["pe_iv"]]))
except Exception:
    atm_iv = float("nan")
ivh = S.setdefault("ivh", {}).setdefault(bkey, [])
if atm_iv == atm_iv and (not ivh or (t - ivh[-1][0]).total_seconds() >= 20):
    ivh.append((t, atm_iv))
    del ivh[:-500]
iv_ref = next((v for (tt, v) in reversed(ivh) if (t - tt).total_seconds() >= lookback * 60), None)
vwap_idx, vol_ratio = float("nan"), None
if fut_hist is not None and len(fut_hist) > 25:
    try:
        vwap_idx = float((E.vwap_daily(fut_hist) - basis).iloc[-1])
        vv = fut_hist["volume"]
        j = -2 if market_open else -1          # skip the candle still forming
        base_v = vv.iloc[:j].tail(20).mean()
        vol_ratio = float(vv.iloc[j] / base_v) if base_v and base_v > 0 else None
    except Exception:
        pass
ind_idx = E.indicators(fut_hist) if fut_hist is not None else {}
vix_v, vix_c = float("nan"), float("nan")
try:                                                   # India VIX from Kite (live), used for a premium-cost warning
    _vq = kite.quote(["NSE:INDIA VIX"])["NSE:INDIA VIX"]
    vix_v = float(_vq["last_price"])
    _vp = float(_vq.get("ohlc", {}).get("close") or 0)
    vix_c = (vix_v / _vp - 1) * 100 if _vp > 0 else float("nan")
except Exception:
    pass
vix_w = E.vix_warning(vix_v, vix_c)
oi_ok = {sd: E.oi_confirmation(sd, chain, ref, summ) for sd in ("CE", "PE")}
try:
    gres, gbest = E.greeks_scan(chain, atm, cfg["step"], spot, tr, vwap_idx, atm_iv, iv_ref, vol_ratio, oi_ok,
                                is_expiry_day, itm=g_strike.startswith("1"),
                                live=in_window and not (block_expiry and is_expiry_day),
                                closed=market_open, mode=g_mode)
    for _sd in ("CE", "PE"):                       # stop loss and targets from ATR (volatility), not a fixed %
        E.apply_atr_levels(gres[_sd]["levels"], ind_idx.get("atr", float("nan")))
except Exception as e:
    gres, gbest = None, None
    st.warning(f"Greeks scanner could not run: {e}")

# ---------- near-miss tracker: why no alert today? (kept in server memory, resets daily / on reboot) ----------
NM = S.setdefault("near", {})
if NM.get("day") != today:
    NM.clear(); NM["day"] = today
if gres and market_open:
    _rec = NM.setdefault(name, dict(scans=0, ce_max=0, pe_max=0, ce_t="", pe_t="", ge8=0, why={}))
    _rec["scans"] += 1
    for _sd in ("CE", "PE"):
        _r = gres[_sd]
        if _r["score"] > _rec[_sd.lower() + "_max"]:
            _rec[_sd.lower() + "_max"], _rec[_sd.lower() + "_t"] = _r["score"], t.strftime("%H:%M")
        if _r["score"] >= 8 and not _r["entry_ok"]:
            _rec["ge8"] += 1
            _why = ("outside entry window" if not in_window else _r.get("conflict") and "opposite side too high" or
                    "trend/VWAP/EMA/breakout not all agreeing" if not _r.get("direction_ok") else
                    "option spread above 3%" if not _r.get("spread_ok") else "other")
            _rec["why"][_why] = _rec["why"].get(_why, 0) + 1

# ---------- log Greeks entries to the Sheet (so the outcome tracker can follow them) ----------
dup_blocked = False
if gbest:
    gk = (today, name, gbest["side"], float(gbest["strike"]), "Greeks")
    if not any(f["key"] == gk for f in fired) and E.recent_same_side(fired, name, gbest["side"], t, 60):
        dup_blocked = True                              # same side already alerted within 60 min: same move, not logged again
    if not dup_blocked and not any(f["key"] == gk for f in fired):
        GL = gbest["levels"]
        SZ_G = size_cols(name, expiry, GL["entry"], GL["sl_premium"])
        fired.append(dict(key=gk, time=t.strftime("%H:%M:%S"), index=name, action=f"BUY {gbest['side']}",
                          strike=gbest["strike"], entry=GL["entry"], sl=GL["sl_premium"], t1=GL["t1"], t2=GL["t2"]))
        if SL.enabled():
            ok, err = SL.append_alert(dict(date=today, time=t.strftime("%H:%M:%S"), index=name, expiry=str(expiry),
                                           action=f"BUY {gbest['side']}", strike=gbest["strike"], source="Greeks",
                                           entry=round(GL["entry"], 1), sl=round(GL["sl_premium"], 1),
                                           t1=round(GL["t1"], 1), t2=round(GL["t2"], 1),
                                           adx=_r1(ind_idx.get("adx")), rsi=_r1(ind_idx.get("rsi")), atr=_r1(ind_idx.get("atr")),
                                           note="; ".join(x for x in (E.ind_warnings(gbest["side"], ind_idx), vix_w) if x),
                                           **{k_: v_ for k_, v_ in SZ_G.items() if k_ != "note"}))
            if ok:
                SL.read_alerts.clear()
            else:
                st.warning(f"Could not write Greeks entry to Google Sheet: {err}")

if keep_awake:   # best effort: asks the browser not to dim the screen; ignored if the browser refuses
    import streamlit.components.v1 as _c
    _c.html("<script>let l=null;async function w(){try{l=await navigator.wakeLock.request('screen');}catch(e){}}"
            "w();document.addEventListener('visibilitychange',()=>{if(document.visibilityState==='visible')w();});</script>",
            height=0)

# ---------- header ----------
st.caption(f"{t.strftime('%a %d %b %Y %H:%M:%S')} IST | {'MARKET OPEN' if market_open else 'MARKET CLOSED (last data)'}"
           f" | OI baseline taken {b['at']} | trend data: {hsrc}"
           f"{f' | auto-refresh {refresh}s' if refresh and market_open else ''}")
tsig, tmv, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10 = st.tabs(["Signals", "Market view", "Chart", "Option chain", "OI chart", "Global",
                                                         "Build-up", "Zones", "Tracker", "Stocks", "Pre-market", "Earnings"])
import streamlit.components.v1 as _swipe                 # swipe left / right on the page to move to the next / previous tab
_swipe.html("""<script>
(function () {
  const d = window.parent.document;
  if (d.__mdSwipe) return;
  d.__mdSwipe = true;
  let x0 = null, y0 = 0, t0 = 0, skip = false;
  const bad = el => el.closest && el.closest('[data-testid="stDataFrame"], .js-plotly-plot, canvas, input, [role="slider"], [data-baseweb="select"], [data-testid="stSidebar"]');
  d.addEventListener('touchstart', e => {
    if (e.touches.length !== 1) { x0 = null; return; }
    const p = e.touches[0]; x0 = p.clientX; y0 = p.clientY; t0 = Date.now(); skip = !!bad(e.target);
  }, {passive: true});
  d.addEventListener('touchend', e => {
    if (x0 === null || skip) return;
    const p = e.changedTouches[0], dx = p.clientX - x0, dy = p.clientY - y0; x0 = null;
    if (Math.abs(dx) < 90 || Math.abs(dy) > 60 || Date.now() - t0 > 700) return;
    const list = d.querySelector('[data-baseweb="tab-list"]'); if (!list) return;
    const tabs = [...list.querySelectorAll('[role="tab"]')];
    const i = tabs.findIndex(b => b.getAttribute('aria-selected') === 'true');
    const j = dx < 0 ? i + 1 : i - 1;
    if (j >= 0 && j < tabs.length) {
      tabs[j].click(); tabs[j].scrollIntoView({inline: 'center', block: 'nearest'});
      const m = d.querySelector('[data-testid="stMain"]') || d.querySelector('section.main'); if (m) m.scrollTo({top: 0});
    }
  }, {passive: true});
})();
</script>""", height=0)
with tsig:
    _ms = S.get("mv_sum")
    if _ms:
        st.markdown("".join(f'<span class="mdchip {k_}">{txt_}</span>' for txt_, k_ in _ms), unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(name, f"{spot:,.1f}")
    c2.metric("PCR", f"{summ['pcr']:.2f}")
    c3.metric("OI support / resistance", f"{summ['support']:.0f} / {summ['resistance']:.0f}")
    c4.metric("Max pain", f"{summ['max_pain']:.0f}")
    sell_up = sorted([z for z in zones if z["kind"] == "SELL" and z["high"] >= spot], key=lambda z: z["low"])
    buy_dn = sorted([z for z in zones if z["kind"] == "BUY" and z["low"] <= spot], key=lambda z: -z["high"])
    nz = []
    if sell_up:
        nz.append(f"Nearest seller zone above: {sell_up[0]['low']:.0f}-{sell_up[0]['high']:.0f} ({sell_up[0]['source']})")
    if buy_dn:
        nz.append(f"Nearest buyer zone below: {buy_dn[0]['low']:.0f}-{buy_dn[0]['high']:.0f} ({buy_dn[0]['source']})")
    if nz:
        st.caption(" | ".join(nz))

    # ---------- build-up signal card ----------
    if new_alert:
        st.toast(f"{bsig['action']} {name} {bsig['strike']:.0f}", icon="🔔")
        import streamlit.components.v1 as components
        components.html("<script>try{const c=new (window.AudioContext||window.webkitAudioContext)();"
                        "const o=c.createOscillator();o.connect(c.destination);o.frequency.value=880;"
                        "o.start();setTimeout(()=>o.stop(),700);}catch(e){}</script>", height=0)
    # futures market signal (price + OI), heavyweight confirmation and data freshness - from the scanner blueprint PDF
    try:
        _dp = fut_px_chg / b["fut_px"] * 100 if b["fut_px"] else 0.0
        _doi = fut_oi_chg / b["fut_oi"] * 100 if b["fut_oi"] else 0.0
        _fs = E.classify(_dp, _doi, 0.03, 0.1)
        st.caption(f"Futures market signal (live estimate, not FII): **{E.STATE_NAME[_fs]}** (price {_dp:+.2f}%, OI {_doi:+.2f}% since {b['at']}). "
                   f"Participant (FII) positions are end-of-day only, see Market view.")
    except Exception:
        pass
    _hc = heavy_confirm(kite, name)
    if _hc:
        _up, _dn, _fl, _rows = _hc
        _agree = _up if fut_px_chg >= 0 else _dn
        _tot = len(_rows)
        st.caption(f"Heavyweights ({name}): {_up} up / {_dn} down of {_tot} - "
                   + ", ".join(f"{s_} {p_:+.1f}%" for s_, p_ in _rows)
                   + (" | confirm the futures move" if _agree >= _tot * 0.6 else " | do NOT clearly confirm the futures move (caution)"))
    try:
        _ts = fq.get("timestamp") or fq.get("last_trade_time")
        if _ts is not None and market_open:
            _age = (t.replace(tzinfo=None) - _ts.replace(tzinfo=None)).total_seconds()
            if _age > 180:
                st.warning(f"DATA DELAYED: futures quote is {_age / 60:.0f} min old. Do not trust signals until it refreshes.")
    except Exception:
        pass
    ce_s, pe_s = bsig["ce"], bsig["pe"]
    st.caption(f"Build-up ({rule}, lookback {lookback} min{'' if ref_full or ref is None else ', short history so far'}): "
               f"CE = {E.STATE_NAME[ce_s['state']]} | PE = {E.STATE_NAME[pe_s['state']]}")
    if bsig["side"] and bu_confirmed:
        (st.success if bsig["side"] == "CE" else st.error)(
            f"**BUILD-UP SIGNAL: {bsig['action']} - {name} {bsig['strike']:.0f} {bsig['side']} ({expiry})**")
        o, ix = bsig["opt"], bsig["idx"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Entry (premium)", f"{o['entry']:.1f}")
        m2.metric("Stop loss", f"{o['sl']:.1f}")
        m3.metric("Target 1", f"{o['t1']:.1f}")
        m4.metric("Target 2", f"{o['t2']:.1f}")
        st.write(f"Index: entry {ix['entry']:.0f}, SL {ix['sl']:.0f}, T1 {ix['t1']:.0f}, T2 {ix['t2']:.0f}. "
                 f"Pattern strongest at strike {bsig['pattern_strike']:.0f}.")
        for line in bsig["exits"]:
            st.write("- " + line)
        zw = E.zone_warning(bsig["side"], bsig["idx"], zones)
        if zw:
            st.warning(zw)
    elif bsig["side"]:
        st.info(f"Pattern forming for {bsig['action']} - waiting for confirmation ({c_n}/{confirm_n}).")
    else:
        st.warning(f"No build-up signal - {bsig['reason']}")
    if fired:
        with st.expander(f"Today's build-up alerts ({len([f for f in fired if f['key'][0] == t.strftime('%Y-%m-%d')])})"):
            st.dataframe(pd.DataFrame([{k: v for k, v in f.items() if k != "key"} for f in fired[::-1]]),
                         hide_index=True, use_container_width=True)

    if SL.enabled():
        with st.expander("All past alerts (Google Sheet)"):
            past = SL.read_alerts()
            if past:
                st.dataframe(pd.DataFrame(past).iloc[::-1], hide_index=True, use_container_width=True)
            else:
                st.write("No alerts logged yet (or the sheet could not be read).")
    else:
        st.caption("Google Sheet logging is off (add SHEET_ID and gcp_service_account to Secrets to turn it on).")

    # ---------- Greeks scanner card ----------
    if gres:
        st.subheader("Greeks scanner")
        gc1, gc2 = st.columns(2)
        for col, sd in ((gc1, "CE"), (gc2, "PE")):
            r = gres[sd]
            col.metric(f"{sd} {r['strike']:.0f}", f"{r['score']}/{r['max']}", r["grade"], delta_color="off")
        if ind_idx.get("adx", float("nan")) == ind_idx.get("adx", float("nan")):
            st.caption(f"{name} 5-min: RSI {ind_idx['rsi']:.0f} | ADX {ind_idx['adx']:.0f} (+DI {ind_idx['pdi']:.0f} / -DI {ind_idx['mdi']:.0f})"
                       f" | ATR {ind_idx['atr']:.1f} pts. Stop loss and targets below are ATR-based "
                       f"(stop = 3 x ATR x delta, kept between 15% and 35% of premium; T1 = 1R, T2 = 1.8R).")
        if vix_v == vix_v:
            st.caption(f"India VIX {vix_v:.2f}" + (f" ({vix_c:+.1f}% today)" if vix_c == vix_c else "")
                       + (" | " + vix_w if vix_w else " | normal range"))
        if dup_blocked:
            st.info("Repeat signal: same side already alerted in the last 60 minutes, so it is not logged again (same move, not a new setup).")
        if gbest:
            L, side = gbest["levels"], gbest["side"]
            (st.success if side == "CE" else st.error)(
                f"**GREEKS ENTRY - BUY {side}** {name} {gbest['strike']:.0f} {side} ({expiry}) - score {gbest['score']}/{gbest['max']}")
            a, b_, c_ = st.columns(3)
            a.metric("Entry / premium SL", f"{L['entry']:.1f} / {L['sl_premium']:.1f}")
            b_.metric("T1 / T2", f"{L['t1']:.1f} / {L['t2']:.1f}")
            c_.metric("Delta", f"{L['delta']:.2f}")
            for line in E.greeks_exit_rules(gbest):
                st.write("- " + line)
            _sz = size_cols(name, expiry, L["entry"], L["sl_premium"])
            if _sz["lots"] != "":
                st.info(f"**Paper size:** {_sz['lots']} lot(s) x {_sz['lot_size']} = qty {_sz['qty']} | amount Rs {_sz['cost']:,} "
                        f"| loss if stop hits about Rs {_sz['risk_amt']:,} (capital Rs {cap_amt:,}, risk {risk_pct}%)"
                        + (f" | {_sz['note']}" if _sz["note"] else ""))
            _w = "; ".join(x for x in (E.ind_warnings(side, ind_idx), vix_w) if x)
            if _w:
                st.warning("Heads-up (does not block the entry): " + _w)
            st.caption("Estimate only. You place the order yourself; keep the stop loss.")
        else:
            why = []
            for sd in ("CE", "PE"):
                r = gres[sd]
                if r.get("conflict"):
                    why.append(f"{sd}: {r['conflict']}")
                elif r["score"] >= SCORE_NEED(g_mode) and not r["direction_ok"]:
                    why.append(f"{sd}: score is enough but trend/VWAP/EMA/breakout do not all agree")
                elif r["score"] >= SCORE_NEED(g_mode) and not r["spread_ok"]:
                    why.append(f"{sd}: bid-ask spread is above 3%")
            st.warning("**No Greeks entry** - " + ("; ".join(why) if why else
                       f"needs a score of {SCORE_NEED(g_mode)}+ with trend, VWAP, EMA and a candle-close breakout in the same direction"
                       + ("" if in_window else " (and outside the 9:20-15:15 entry window)")))
        with st.expander("Greeks checklists (CE and PE)"):
            for sd in ("CE", "PE"):
                r = gres[sd]
                st.write(f"**{sd} {r['strike']:.0f}** - {r['score']}/{r['max']} ({g_mode}), option price {r['ltp']:.1f}"
                         + (f", bid-ask spread {r['spread']:.1f}%" if r["spread"] is not None else ""))
                gdf = pd.DataFrame(r["items"])
                gdf["ok"] = gdf["ok"].map({True: "YES", False: "NO", None: "n/a"})
                st.dataframe(gdf, hide_index=True, use_container_width=True)

    _nm = NM.get(name)
    if _nm and _nm["scans"] > 0:
        with st.expander(f"Why no alert? Today's scans for {name} ({_nm['scans']} checks)"):
            st.write(f"Best CE score today: **{_nm['ce_max']}/10** (at {_nm['ce_t'] or '-'}) | best PE score: **{_nm['pe_max']}/10** (at {_nm['pe_t'] or '-'})")
            if _nm["ge8"]:
                st.write(f"Score reached 8+ but entry was blocked **{_nm['ge8']}** time(s): "
                         + ", ".join(f"{k} x{v}" for k, v in sorted(_nm["why"].items(), key=lambda x: -x[1])))
            else:
                st.write("Score never reached 8, so there was simply no setup (not a blocked alert).")
            st.caption("Kept in memory while the app runs: a reboot or a closed page resets it. Only counts checks done while this page was open.")

    # ---------- signal card ----------
    if sig["action"] == "NO TRADE":
        WHY = {"Market open": "Market is closed",
               "Entry window (9:20-15:15 IST)": "outside the 9:20-15:15 IST entry window",
               "Trend is one-sided": "5-min trend is sideways",
               "OI agrees with trend": "OI does not agree with the trend",
               "Global cues not against": "global cues are against the trend"}
        fails = [WHY.get(c["check"], c["check"]) for c in sig["checks"] if not c["ok"]]
        st.warning("**NO TRADE** - " + ("; ".join(fails) if fails else sig["reason"]))
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
        _al = dict(entry=float(ltp_o), delta=abs(float(dlt)) if dlt == dlt else 0.5)
        if E.apply_atr_levels(_al, ind_idx.get("atr", float("nan"))):
            st.write(f"Option levels (ATR based): SL **{_al['sl_premium']:.1f}** ({-_al['risk_pct']:.0f}%), T1 **{_al['t1']:.1f}**, "
                     f"T2 **{_al['t2']:.1f}**. Signal is void if index closes a 5-min candle beyond the SL level. You place the order yourself.")
        else:
            st.write(f"Approx. option levels (delta based): SL **{ol['sl']:.1f}**, T1 **{ol['t1']:.1f}**, T2 **{ol['t2']:.1f}**. "
                     "Signal is void if index closes a 5-min candle beyond the SL level. You place the order yourself.")
        _w = E.ind_warnings(side, ind_idx)
        if _w:
            st.error("Caution: " + _w + ". This card is a technical signal, not a Greeks entry.")
        try:
            _gs = gres[side]
            if not _gs.get("entry_ok"):
                st.warning(f"Greeks scanner says WAIT: {side} {_gs['score']}/{_gs['max']} ({_gs.get('grade', '')}). "
                           "Paper trade only when the Greeks scanner gives an entry.")
            else:
                st.success(f"Greeks scanner agrees: {side} {_gs['score']}/{_gs['max']}.")
        except Exception:
            pass

    with st.expander("Why? (checklist)", expanded=sig["action"] == "NO TRADE"):
        chk = pd.DataFrame(sig["checks"])
        chk["ok"] = chk["ok"].map({True: "YES", False: "NO"})
        st.dataframe(chk, hide_index=True, use_container_width=True)
        st.write(f"OI score {oscore:+.1f}: " + "; ".join(onotes))
        st.write(f"Global score {gscore:+.2f} (Gift Nifty {'included' if gift else 'not included'})")

# ---------- outcome tracker: replay each alert's option candles and write WIN / LOSS back to the Sheet ----------
trk_live = S.setdefault("trk_live", {})
if SL.enabled() and (S.get("trk_at") is None or (t - S["trk_at"]).total_seconds() >= 60):
    S["trk_at"] = t
    try:
        def _fetch(r, at):
            side = "CE" if "CE" in str(r.get("action")) else "PE"
            m = inst[(inst.name == r.get("index")) & (inst.expiry.astype(str) == str(r.get("expiry")))
                     & (inst.strike == float(r.get("strike"))) & (inst.instrument_type == side)]
            if m.empty:
                return None
            frm = dt.datetime.combine(at.date(), dt.time(9, 15))
            d = kite.historical_data(int(m.iloc[0].instrument_token), frm, t.replace(tzinfo=None), "5minute")
            df = pd.DataFrame(d)
            if df.empty:
                return None
            ix = pd.to_datetime(df["date"])
            if ix.dt.tz is not None:
                ix = ix.dt.tz_convert(IST).dt.tz_localize(None)
            df.index = ix
            return df[["high", "low", "close"]]
        ups, live_now = E.track_rows(SL.read_alerts(), _fetch, t.replace(tzinfo=None), slip_pct=slip_pct)
        trk_live.clear()
        trk_live.update(live_now)
        if ups:
            for row_no, ev in ups:
                SL.update_outcome(row_no, ev)
            SL.read_alerts.clear()
    except Exception as e:
        st.caption(f"Outcome tracker: {type(e).__name__}: {str(e)[:120]}")

# ---------- stock scanner: momentum + clear direction, then option pick, liquidity check, alert ----------
STK = S.setdefault("stk", dict(rows=[], at=None, alerts=[], new=False))
STK["new"] = False
if stock_scan and (STK["at"] is None or (t - STK["at"]).total_seconds() >= 120) and (market_open or not STK["rows"]):
    STK["at"] = t
    rows_ = []
    futs_ = inst[(inst.instrument_type == "FUT") & (inst.expiry >= t.date()) & (inst.exchange == "NFO")]
    try:                                                   # spot previous closes (true day-change reference)
        _sq = kite.quote([f"NSE:{s_}" for s_ in K.STOCKS.values()])
        spot_prev_ = {k_[4:]: v_.get("ohlc", {}).get("close") for k_, v_ in _sq.items()}
        spot_last_ = {k_[4:]: v_.get("last_price") for k_, v_ in _sq.items()}
    except Exception:
        spot_prev_, spot_last_ = {}, {}
    _ev_rows, _ = fetch_events()
    _res_soon = {r_["symbol"]: r_["date"] for r_ in _ev_rows if r_["date"] <= t.date() + dt.timedelta(days=1)}
    _fno = set(futs_.name)
    for disp, sym in [(d_, s_) for d_, s_ in K.STOCKS.items() if s_ in _fno]:
        row_ = dict(stock=disp, symbol=sym, direction="", clear="no", last=np.nan, day_pct=np.nan, state="-",
                    vol_x=np.nan, score=0, note="", checks=[], rsi=np.nan, adx=np.nan, atr_pct=np.nan)
        try:
            fr_ = futs_[futs_.name == sym].sort_values("expiry")
            if fr_.empty:
                row_["note"] = "no futures contract (not in F&O?)"
                rows_.append(row_)
                continue
            fr_ = fr_.iloc[0]
            _time.sleep(0.34)                                  # Kite allows about 3 history calls per second
            sdf = stock_candles(kite, int(fr_.instrument_token))
            _rtag = (f"RESULTS {'today' if _res_soon[sym] == t.date() else 'tomorrow'}: gap/volatility risk. " if sym in _res_soon else "")
            a_ = K.analyze_stock(sdf, prev_close=spot_prev_.get(sym), spot_last=spot_last_.get(sym))
            row_.update(direction={"BULL": "Bullish", "BEAR": "Bearish"}.get(a_["direction"], "none"), last=a_["last"],
                        day_pct=a_["day_pct"], state=K.OI_NAMES.get(a_["state"], "-"), vol_x=a_["vol_ratio"],
                        score=a_["score"], note=_rtag + a_["why"], checks=a_["checks"],
                        rsi=a_["ind"].get("rsi", np.nan), adx=a_["ind"].get("adx", np.nan), atr_pct=a_["ind"].get("atr_pct", np.nan))
            if a_["clear"] and in_window and t.time() >= dt.time(9, 30):
                row_["clear"] = "YES"
                exps_ = sorted(e for e in inst[(inst.name == sym) & inst.instrument_type.isin(["CE", "PE"])].expiry.unique()
                               if e >= t.date())
                ex_ = exps_[0]
                ks_ = inst[(inst.name == sym) & (inst.expiry == ex_) & (inst.instrument_type == "CE")].strike
                step_ = K.pick_step(ks_, a_["last"])
                ch_, _ = build_chain(kite, inst, sym, ex_, a_["last"], step_, 3)
                T_ = max((dt.datetime.combine(ex_, dt.time(15, 30), tzinfo=IST) - t).total_seconds() / (365 * 86400), 1e-6)
                ch_ = E.add_greeks(ch_, a_["last"], T_)
                atm_ = min(ch_.index, key=lambda k_: abs(k_ - a_["last"]))
                aiv_ = float(np.nanmean([ch_.at[atm_, "ce_iv"], ch_.at[atm_, "pe_iv"]]))
                bull_ = a_["direction"] == "BULL"
                gr_, gb_ = E.greeks_scan(ch_, atm_, step_, a_["last"], a_["trend"], a_["vwap"], aiv_, None, a_["vol_ratio"],
                                         {"CE": bull_, "PE": not bull_}, ex_ == t.date(), itm=False, live=True,
                                         closed=True, mode=g_mode)
                side_ = "CE" if bull_ else "PE"
                pick_ = gr_[side_]
                for _s2 in ("CE", "PE"):
                    E.apply_atr_levels(gr_[_s2]["levels"], a_["ind"].get("atr", float("nan")))
                ind_ = a_["ind"]
                warn_ = E.ind_warnings(side_, ind_)
                lq_ok, lq_txt = K.option_liquidity(ch_, pick_["strike"], side_)
                row_["note"] = _rtag + f"{side_} {pick_['strike']:.0f}: score {pick_['score']}/{pick_['max']}; {lq_txt}"
                if gb_ and gb_["side"] == side_ and lq_ok:
                    row_["clear"] = "ALERT"
                    gk_ = (today, sym, side_, float(pick_["strike"]), "Stock")
                    if not any(f["key"] == gk_ for f in fired) and not E.recent_same_side(fired, sym, side_, t, 60):
                        L_ = pick_["levels"]
                        sz_ = size_cols(sym, ex_, L_["entry"], L_["sl_premium"])
                        fired.append(dict(key=gk_, time=t.strftime("%H:%M:%S"), index=sym, action=f"BUY {side_}",
                                          strike=pick_["strike"], entry=L_["entry"], sl=L_["sl_premium"], t1=L_["t1"], t2=L_["t2"]))
                        STK["alerts"].append(dict(time=t.strftime("%H:%M:%S"), stock=disp, action=f"BUY {side_}",
                                                  strike=pick_["strike"], expiry=str(ex_), entry=round(L_["entry"], 1),
                                                  sl=round(L_["sl_premium"], 1), t1=round(L_["t1"], 1), t2=round(L_["t2"], 1),
                                                  sl_pct=round((L_["sl_premium"] / L_["entry"] - 1) * 100, 0),
                                                  rsi=_r1(ind_.get("rsi")), adx=_r1(ind_.get("adx")), warn=warn_,
                                                  lots=sz_["lots"], qty=sz_["qty"], cost=sz_["cost"], risk_Rs=sz_["risk_amt"]))
                        STK["new"] = True
                        if SL.enabled():
                            ok_, err_ = SL.append_alert(dict(date=today, time=t.strftime("%H:%M:%S"), index=sym, expiry=str(ex_),
                                                             action=f"BUY {side_}", strike=pick_["strike"], source="Stock",
                                                             entry=round(L_["entry"], 1), sl=round(L_["sl_premium"], 1),
                                                             t1=round(L_["t1"], 1), t2=round(L_["t2"], 1),
                                                             adx=_r1(ind_.get("adx")), rsi=_r1(ind_.get("rsi")),
                                                             atr=_r1(ind_.get("atr")), note=warn_,
                                                             **{k_: v_ for k_, v_ in sz_.items() if k_ != "note"}))
                            if ok_:
                                SL.read_alerts.clear()
                elif not lq_ok:
                    row_["clear"] = "illiquid"
        except Exception as e_:
            row_["note"] = f"{type(e_).__name__}: {str(e_)[:80]}"
        rows_.append(row_)
    STK["rows"] = rows_
if STK["new"]:
    st.toast("Stock alert - see the Stocks tab", icon="🔔")

# ---------- tabs ----------
with t1:
    d = tr["df"]
    if d is not None and len(d) > 5:
        d = d.iloc[-150:]
        xs = list(d.index.strftime("%d %b %H:%M"))
        has_vol = fut_hist is not None and len(fut_hist) > 20
        fig = make_subplots(rows=2 if has_vol else 1, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                            row_heights=[0.74, 0.26] if has_vol else [1.0])
        fig.add_trace(go.Candlestick(x=xs, open=d.open, high=d.high, low=d.low, close=d.close, name=name), row=1, col=1)
        if "ema20" in d:
            fig.add_scatter(x=xs, y=d.ema20, name="EMA20", row=1, col=1)
            fig.add_scatter(x=xs, y=d.ema50, name="EMA50", row=1, col=1)
        if has_vol:
            vol = fut_hist["volume"].reindex(d.index).fillna(0)
            vavg = fut_hist["volume"].rolling(20).mean().reindex(d.index)
            vw = (E.vwap_daily(fut_hist) - basis).reindex(d.index)
            fig.add_scatter(x=xs, y=vw, name="VWAP", line=dict(color="#ffffff", dash="dot", width=1.5), row=1, col=1)
            fig.add_bar(x=xs, y=vol, name="Volume (futures)",
                        marker_color=np.where(d.close >= d.open, "#2e8b57", "#d9534f"), row=2, col=1)
            fig.add_scatter(x=xs, y=vavg, name="Vol avg 20", line=dict(color="#ffc107", width=1.5), row=2, col=1)
        lo, hi = float(d.low.min()), float(d.high.max())
        last = float(d.close.iloc[-1])
        shown = sorted([z for z in zones if abs((z["low"] + z["high"]) / 2 - last) / last < 0.012],
                       key=lambda z: abs((z["low"] + z["high"]) / 2 - last))[:6]
        for z in shown:
            sell = z["kind"] == "SELL"
            fig.add_hrect(y0=z["low"], y1=z["high"], row=1, col=1,
                          fillcolor="rgba(255,193,7,0.30)" if sell else "rgba(255,255,255,0.25)",
                          line=dict(color="#ffc107" if sell else "#ffffff", width=1),
                          annotation_text=("SELL " if sell else "BUY ") + f"{(z['low'] + z['high']) / 2:.0f}",
                          annotation_position="top right", annotation_font_size=10,
                          annotation_font_color="#ffc107" if sell else "#ffffff")
            lo, hi = min(lo, z["low"]), max(hi, z["high"])
        for lvl, txt, col in ((summ["resistance"], "R", "#d9534f"), (summ["support"], "S", "#2e8b57")):
            if lvl == lvl and abs(lvl - last) / last < 0.012:
                fig.add_hline(y=lvl, line_dash="dot", line_color=col, row=1, col=1,
                              annotation_text=f"{txt} {lvl:.0f}", annotation_position="bottom left",
                              annotation_font_size=10)
                lo, hi = min(lo, lvl), max(hi, lvl)
        pad = (hi - lo) * 0.04
        fig.update_yaxes(range=[lo - pad, hi + pad], row=1, col=1)
        fig.update_layout(template="plotly_dark", height=560 if has_vol else 460, bargap=0.1,
                          margin=dict(l=0, r=0, t=10, b=0),
                          legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0))
        fig.update_xaxes(type="category", tickmode="array", tickvals=xs[::30], ticktext=xs[::30],
                         tickangle=-45, rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Gold boxes = seller (supply) zones, white boxes = buyer (demand) zones. Volume and VWAP are from the "
                   "futures contract, shifted to the index scale by the current basis. Zones are inferred from volume "
                   "spikes and OI walls; Kite does not reveal who is trading." if has_vol else
                   "Futures volume data not available, so volume, VWAP and volume-spike zones are hidden (OI zones still show).")
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
    st.caption("Daily change from Yahoo Finance (can lag). Yield change in basis points. Gift Nifty: automatic (sidebar).")

with t5:
    if len(bu_df):
        v = bu_df.copy()
        v["state"] = v["state"].map(E.STATE_NAME)
        v = v.rename(columns={"dp": "price chg %", "doi": "OI chg %", "d_oi": "OI chg"}).round(2)
        st.dataframe(v.sort_values(["side", "strike"]), hide_index=True, use_container_width=True, height=480)
        st.caption("Price and OI change compared with the snapshot taken about the lookback minutes ago (ATM +/- 3 strikes). "
                   "A change under 2% counts as no clear change. History resets if the server restarts.")
    else:
        st.info("Collecting snapshots - build-up needs at least a minute of history.")

with t6:
    if zones:
        rows = []
        for z in sorted(zones, key=lambda z: -(z["low"] + z["high"]) / 2):
            rows.append(dict(
                side="SELL (gold)" if z["kind"] == "SELL" else "BUY (white)",
                zone=f"{z['low']:.0f} - {z['high']:.0f}", source=z["source"],
                strength=(f"{z['strength']:.1f}x average volume" if z["source"] == "Volume spike" else f"OI {z['strength']:,.0f}"),
                when=(z["time"].strftime("%d %b %H:%M") if z["time"] is not None else "-")))
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("No zones found yet.")
    if vp:
        centers = vp["centers"] - basis
        fig = go.Figure(go.Bar(x=vp["prof"], y=centers, orientation="h", name="Volume",
                               marker_color=np.where(vp["hvn"], "#ffc107", "#6c757d")))
        for val, txt in ((vp["poc"], "POC"), (vp["vah"], "VAH"), (vp["val"], "VAL")):
            fig.add_hline(y=val - basis, line_dash="dot", line_color="#ffffff",
                          annotation_text=f"{txt} {val - basis:.0f}", annotation_position="top right",
                          annotation_font_size=10)
        fig.add_hline(y=spot, line_color="#00bcd4", annotation_text=f"Spot {spot:.0f}",
                      annotation_position="bottom right", annotation_font_size=10)
        fig.update_layout(template="plotly_dark", height=520, bargap=0.05, margin=dict(l=0, r=0, t=10, b=0),
                          xaxis_title="Volume", yaxis_title="Price (index scale)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Volume profile from roughly the last 4 sessions of futures 5-min candles (volume spread across each "
                   "candle's range, so it is an estimate). Gold bars = high-volume nodes. POC = price with the most volume; "
                   "VAH/VAL = range holding about 70% of volume.")
    else:
        st.info("Futures volume data not available.")

with t7:
    if not SL.enabled():
        st.info("Turn on Google Sheet logging first - the tracker reads and writes the Alerts sheet.")
    else:
        recs = SL.read_alerts()
        if not recs:
            st.write("No alerts yet. Every build-up alert and Greeks entry is followed here automatically.")
        else:
            tdf = pd.DataFrame(recs)
            for c_ in ("source", "outcome", "pnl_pct", "exit", "lots", "qty", "cost", "risk_amt", "pnl_amt", "charges", "slippage", "net_amt"):
                if c_ not in tdf.columns:
                    tdf[c_] = ""
            tdf["source"] = tdf["source"].replace("", "Build-up")
            for c_ in ("outcome", "pnl_pct", "exit", "pnl_amt", "charges", "slippage", "net_amt"):      # these hold text and numbers; keep them as generic objects
                tdf[c_] = tdf[c_].astype(object)
            for i_, r_ in tdf.iterrows():                    # show the running status of unfinished alerts
                if str(r_["outcome"]) in E.TRACK_RETRY:
                    lv = trk_live.get((str(r_["date"]), str(r_["time"]), str(r_["index"]), str(r_["action"]), str(r_["strike"])))
                    if lv:
                        tdf.at[i_, "outcome"], tdf.at[i_, "pnl_pct"], tdf.at[i_, "exit"] = lv["outcome"], lv["pnl_pct"], lv["exit"]
                        tdf.at[i_, "pnl_amt"] = E.pnl_amount(lv["pnl_pct"], r_["cost"])
                        for k_, v_ in E.trade_costs(r_["cost"], lv["pnl_pct"], 2, slip_pct).items():
                            if k_ != "gross_amt":
                                tdf.at[i_, {"charges": "charges", "slippage": "slippage", "net_amt": "net_amt"}[k_]] = v_
            pick = st.radio("Show", ["All", "Build-up", "Greeks"], horizontal=True)
            if pick != "All":
                tdf = tdf[tdf["source"] == pick]
            sm = E.tracker_summary(tdf.to_dict("records"))
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Finished", sm["n"])
            m2.metric("Win rate", f"{sm['win_rate']:.0f}%")
            m3.metric("Avg result", f"{sm['avg']:+.1f}%")
            m4.metric("Total (premium %)", f"{sm['total']:+.0f}%")
            if sm["total_amt"] or sm["total_net"]:
                n1, n2 = st.columns(2)
                n1.metric("Gross result (Rs)", f"Rs {sm['total_amt']:+,.0f}")
                n2.metric("NET after charges + slippage", f"Rs {sm['total_net']:+,.0f}",
                          f"{sm['total_net'] / cap_amt * 100:+.1f}% of Rs {cap_amt:,} capital")
            if sm["n"]:
                st.caption(f"Best {sm['best']:+.1f}% | Worst {sm['worst']:+.1f}% (percent of option premium, per trade)")
            show = ["date", "time", "index", "action", "strike", "source", "lots", "qty", "cost", "entry", "sl", "risk_amt",
                    "t1", "t2", "outcome", "pnl_pct", "pnl_amt", "charges", "slippage", "net_amt"]
            st.dataframe(paint(tdf[show].iloc[::-1], ("pnl_pct", "pnl_amt", "net_amt")), hide_index=True, use_container_width=True)
            st.caption("How it is judged: option 5-min candles after the alert. SL touched = loss. T1 touched = half booked and the "
                       "stop moves to entry; the rest exits at T2, at entry, or at the day-end price. If SL and T1 are both inside one "
                       "candle it counts as SL. Percent is on the option premium, before charges and slippage. "
                       "'T1 hit, runner open' and 'OPEN' are still running and are not counted yet. "
                       "cost = amount invested (premium x quantity); risk_amt = loss in Rs if the stop hits; "
                       "pnl_amt = result in Rs (percent x cost, before costs); charges = Zerodha brokerage, STT, exchange fee, GST, stamp "
                       "duty; slippage = your sidebar % on both fills; net_amt = pnl_amt - charges - slippage. "
                       "Alerts logged before this update have no lot data.")

with t8:
    if not stock_scan:
        st.info("Turn on 'Stock scanner' in the sidebar.")
    else:
        st.caption("An alert needs ALL of: one-sided 5-min trend, price on the right side of VWAP, candle-close breakout, "
                   "volume surge, and futures build-up in the same direction - plus a liquid ATM option (spread 3% or less, "
                   "traded value 25 lakh or more) and the Greeks score. Alerts only after 9:30 and before 15:15 IST.")
        if STK["alerts"]:
            st.subheader("Stock alerts today")
            st.dataframe(paint(pd.DataFrame(STK["alerts"]).iloc[::-1]), hide_index=True, use_container_width=True)
        rr = pd.DataFrame(STK["rows"])
        if rr.empty:
            st.info("Scanning... (first scan takes about 10 seconds; it refreshes every 2 minutes)")
        else:
            rr["_o"] = rr["clear"].map({"ALERT": 0, "YES": 1, "illiquid": 2}).fillna(3)
            rr["_m"] = rr["day_pct"].abs().fillna(0)
            rr = rr.sort_values(["_o", "_m"], ascending=[True, False])
            v = rr[["stock", "clear", "direction", "day_pct", "last", "state", "vol_x", "score", "rsi", "adx", "atr_pct", "note"]].copy()
            v.columns = ["Stock", "Signal", "Direction", "Day %", "Price", "Futures OI", "Volume x", "Checks (of 7)",
                         "RSI", "ADX", "ATR %", "Note"]
            for c_ in ("Day %", "Price", "Volume x", "RSI", "ADX", "ATR %"):
                v[c_] = v[c_].round(2)
            st.dataframe(paint(v, ("Day %",)), hide_index=True, use_container_width=True)
            pk = st.selectbox("See the checklist for", list(rr["stock"]))
            ck = rr[rr["stock"] == pk].iloc[0]["checks"]
            if ck:
                cdf = pd.DataFrame(ck)[["check", "ok", "detail"]]
                cdf["ok"] = cdf["ok"].map({True: "YES", False: "NO"})
                st.dataframe(paint(cdf, yes_no=True), hide_index=True, use_container_width=True)
        if STK["at"] is not None:
            st.caption(f"Last scan {STK['at'].strftime('%H:%M:%S')} IST. Illiquid = direction was clear but the option is too thin to trade.")

with tmv:
    _kind = lambda x_: "mdg" if x_ > 0 else "mdr" if x_ < 0 else "mdn"
    chips = []
    # ---- 1. global / commodity / currency / bond mood ----
    st.subheader("Global mood for today")
    try:
        md_ = mood_data()
        mv = MV.mood_from_data(md_, gift)
        col_ = "#16a34a" if mv["label"] == "Positive" else "#dc2626" if mv["label"] == "Negative" else "#d97706"
        meaning = {"Positive": "Global cues lean POSITIVE: a steady-to-higher start is more likely.",
                   "Negative": "Global cues lean NEGATIVE: a weak start is more likely, be careful with CE buying.",
                   "Neutral / mixed": "Global cues are MIXED: no clear help from outside, rely on the live signals."}[mv["label"]]
        st.markdown(f'<div class="mdmood" style="background:{col_}">{mv["label"].upper()} ({mv["score"]:+.2f}) - {meaning}</div>',
                    unsafe_allow_html=True)
        st.markdown("".join(f'<span class="mdchip {_kind(v_ if abs(v_) >= 0.15 else 0)}">{g_} {v_:+.2f}</span>'
                            for g_, v_ in mv["groups"].items()), unsafe_allow_html=True)
        rows_m = pd.DataFrame(mv["rows"])
        up_ = rows_m[rows_m["Impact"] == "Positive"]
        dn_ = rows_m[rows_m["Impact"] == "Negative"]
        fmt_ = lambda r_: f"{r_['Market']} {r_['Change']:+.2f}{r_['Unit']}"
        if len(up_):
            st.markdown("**Supporting:** " + ", ".join(fmt_(r_) for _, r_ in up_.head(6).iterrows()))
        if len(dn_):
            st.markdown("**Against:** " + ", ".join(fmt_(r_) for _, r_ in dn_.head(6).iterrows()))
        _lim = {"Equities": 3.5, "Commodities": 6.0, "Currency": 1.5, "Bonds": 25.0, "Volatility": 40.0}
        odd_ = [f"{r_['Market']} {r_['Change']:+.2f}{r_['Unit']}" for _, r_ in rows_m.iterrows()
                if r_["Change"] == r_["Change"] and abs(r_["Change"]) > _lim.get(r_["Group"], 99)]
        if odd_:
            st.warning("Unusually large moves, please cross-check in another app before trusting them (a futures contract roll or "
                       "stale Yahoo data can cause this): " + ", ".join(odd_))
        shown = rows_m[["Group", "Market", "Last", "Change", "Unit", "Impact"]].copy()
        shown.columns = ["Group", "Market", "Last", "Change (% or bp)", "Unit", "Impact on India"]
        shown["Impact on India"] = shown["Impact on India"].replace({"n/a": "-"})
        st.dataframe(paint(shown, ("Change (% or bp)",)), hide_index=True, use_container_width=True, height=420)
        st.caption("Daily change from Yahoo Finance (can lag; Asian markets may still be live). Crude, dollar, yields and VIX going UP "
                   "count as negative for India; the number colour shows direction, the Impact column shows what it means for India. "
                   "Gift Nifty is fetched automatically (sidebar shows the source).")
        chips.append((f"Global {mv['label']} {mv['score']:+.2f}", "mdg" if mv["label"] == "Positive" else "mdr" if mv["label"] == "Negative" else "mda"))
    except Exception as e_:
        st.warning(f"Global data could not be loaded: {type(e_).__name__}: {str(e_)[:100]}")

    # ---- 2. advance / decline (breadth) ----
    st.subheader("Advance / decline (F&O stocks)")
    try:
        fo_names = sorted(set(inst[(inst.instrument_type == "FUT") & (inst.exchange == "NFO")].name) - {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"})
        bd = fo_breadth(kite, tuple("NSE:" + n_ for n_ in fo_names))
        if bd.get("total"):
            kcol = "mdg" if bd["label"] == "Broad strength" else "mdr" if bd["label"] == "Broad weakness" else "mdn" if bd["label"] == "Market not open yet" else "mda"
            st.markdown(f'<span class="mdchip mdg">Advances {bd["adv"]} ({bd["adv_pct"]:.0f}%)</span>'
                        f'<span class="mdchip mdr">Declines {bd["dec"]} ({bd["dec_pct"]:.0f}%)</span>'
                        f'<span class="mdchip mdn">Unchanged {bd["unch"]}</span>'
                        f'<span class="mdchip {kcol}">{bd["label"]}</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="mdbar"><div style="width:{bd["adv_pct"]}%;background:#16a34a"></div>'
                        f'<div style="width:{100 - bd["adv_pct"] - bd["dec_pct"]}%;background:#94a3b8"></div>'
                        f'<div style="width:{bd["dec_pct"]}%;background:#dc2626"></div></div>', unsafe_allow_html=True)
            ratio_ = "-" if bd["ratio"] != bd["ratio"] else "all up" if bd["ratio"] == float("inf") else f"{bd['ratio']:.2f}"
            if bd["label"] == "Market not open yet":
                st.info("Market has not opened yet, so every stock still shows yesterday's close (0% change). Live advance / decline starts after 9:15 IST.")
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Stocks counted", bd["total"])
            b2.metric("Advance : decline", ratio_)
            b3.metric("Average change", f"{bd['avg']:+.2f}%")
            b4.metric("Up 2%+ / down 2%+", f"{bd['strong_up']} / {bd['strong_dn']}")
            g1, g2 = st.columns(2)
            gdf2 = pd.DataFrame([(a_.replace("NSE:", ""), round(c_, 2)) for a_, c_ in bd["gainers"]], columns=["Top gainers", "Change %"])
            ldf2 = pd.DataFrame([(a_.replace("NSE:", ""), round(c_, 2)) for a_, c_ in bd["losers"]], columns=["Top losers", "Change %"])
            if bd["label"] != "Market not open yet":
                g1.dataframe(paint(gdf2, ("Change %",)), hide_index=True, use_container_width=True)
                g2.dataframe(paint(ldf2, ("Change %",)), hide_index=True, use_container_width=True)
            st.caption("Counted over all F&O stocks (the liquid ones), live from Kite. Above 60% advancing = broad strength, below 40% = broad weakness. "
                       "A rise in the index with few advances means only a few heavy stocks are pulling it.")
            chips.append((f"Breadth {bd['adv']}↑ {bd['dec']}↓", kcol))
        else:
            st.info("No quotes received for breadth right now.")
    except Exception as e_:
        st.warning(f"Breadth could not be loaded: {type(e_).__name__}: {str(e_)[:100]}")

    # ---- 3. FII / DII ----
    st.subheader("FII / DII positions")
    got_, err_ = fetch_participant_oi()
    if not got_:
        st.warning(f"FII / DII futures data could not be loaded ({err_}). NSE sometimes refuses cloud servers. Try again later; "
                   "the same table is on NSE's website under Reports > F&O > Participant wise open interest.")
        if st.button("Retry FII / DII"):
            fetch_participant_oi.clear()
            fetch_cash_flows.clear()
            st.rerun()
    else:
        d0, today_oi = got_[0]
        prev_oi = got_[1][1] if len(got_) > 1 else None
        prow = MV.participant_table(today_oi, prev_oi)
        fii_ = next((r_ for r_ in prow if r_["Participant"] == "FII"), None)
        if fii_:
            improving = fii_["dShort"] == fii_["dShort"] and (fii_["dShort"] < 0 or fii_["dLong"] > 0)
            weakening = fii_["dShort"] == fii_["dShort"] and (fii_["dShort"] > 0 or fii_["dLong"] < 0)
            # net short but covering = amber; net long but cutting = amber; otherwise red (short) / green (long)
            c_fii = (("#d97706" if improving else "#dc2626") if fii_["Net"] < 0 else ("#d97706" if weakening and not improving else "#16a34a"))
            st.markdown(f'<div class="mdmood" style="background:{c_fii}">FII index futures: {fii_["Reading"]}</div>', unsafe_allow_html=True)
            _nz = lambda v: 0 if v != v else v
            _dn = fii_["dNet"]
            if _dn == _dn:
                _tag = ("Short exposure reduced" if _dn > 0 else "Short exposure increased") if fii_["Net"] < 0 else ("Long exposure increased" if _dn > 0 else "Long exposure reduced")
                _good = _dn > 0
                _tagc = "#16a34a" if _good else "#dc2626"
                _sub = f'<span style="background:{_tagc}22;color:{_tagc};padding:2px 10px;border-radius:10px;font-weight:600;font-size:0.85rem">{_tag}</span> <span style="color:{_tagc};font-weight:600">{abs(_dn):,.0f} {"fewer" if (_dn > 0 and fii_["Net"] < 0) else "more" if fii_["Net"] < 0 else "more" if _dn > 0 else "fewer"} net {"shorts" if fii_["Net"] < 0 else "longs"}</span>'
            else:
                _sub = ""
            def _chg(v):
                if v != v: return ""
                return f'<div style="color:{"#16a34a" if v > 0 else "#dc2626"};font-size:0.85rem">{v:+,.0f}</div>'
            _lc, _sc = fii_["dLong"], fii_["dShort"]
            _lcol = "#16a34a" if _nz(_lc) >= 0 else "#dc2626"
            _scol = "#16a34a" if _nz(_sc) <= 0 else "#dc2626"
            _chgl = "" if _lc != _lc else f'<div style="color:{_lcol};font-size:0.85rem">{_lc:+,.0f}</div>'
            _chgs = "" if _sc != _sc else f'<div style="color:{_scol};font-size:0.85rem">{_sc:+,.0f}</div>'
            st.markdown(
                f'<div style="border:1px solid #8884;border-radius:14px;padding:14px 16px;margin:8px 0">'
                f'<div style="opacity:.7;font-size:0.85rem">FII net index-futures position</div>'
                f'<div style="font-size:1.8rem;font-weight:700;color:{"#dc2626" if fii_["Net"] < 0 else "#16a34a"}">{fii_["Net"]:+,} contracts</div>'
                f'<div style="margin:4px 0 10px 0">{_sub}</div>'
                f'<div style="display:flex;justify-content:space-between;gap:12px">'
                f'<div><div style="opacity:.7;font-size:0.85rem">FII long contracts</div><div style="font-size:1.3rem;font-weight:700">{fii_["Long"]:,}</div>{_chgl}</div>'
                f'<div style="text-align:right"><div style="opacity:.7;font-size:0.85rem">FII short contracts</div><div style="font-size:1.3rem;font-weight:700">{fii_["Short"]:,}</div>{_chgs}</div></div>'
                f'<div class="mdbar" style="margin-top:12px"><div style="width:{fii_["LongPct"]:.1f}%;background:#16a34a"></div><div style="width:{fii_["ShortPct"]:.1f}%;background:#dc2626"></div></div>'
                f'<div style="display:flex;justify-content:space-between;font-size:0.85rem"><span style="color:#16a34a">{fii_["LongPct"]:.1f}% long</span><span style="color:#dc2626">{fii_["ShortPct"]:.1f}% short</span></div>'
                f'<div style="opacity:.6;font-size:0.78rem;margin-top:8px">NSE participant-wise index futures as on {d0}. Covers Nifty, Bank Nifty and other index futures together.</div>'
                f'</div>', unsafe_allow_html=True)
            ob = fii_["OptBull"]
            st.markdown(f"**FII index options net bullish exposure:** {ob:+,.0f} contracts"
                        + (f" (change {fii_['dOptBull']:+,.0f})" if fii_["dOptBull"] == fii_["dOptBull"] else "")
                        + " (positive = more calls bought / puts sold)")
            chips.append((f"FII fut {'NET SHORT' if fii_['Net'] < 0 else 'NET LONG'} {abs(fii_['Net']) / 1000:.0f}k", "mda" if c_fii == "#d97706" else "mdr" if fii_["Net"] < 0 else "mdg"))
        pdf = pd.DataFrame(prow)[["Participant", "Long", "Short", "Net", "ShortPct", "LongPct", "dLong", "dShort", "dNet", "Reading"]]
        pdf.columns = ["Who", "Long", "Short", "Net", "Short %", "Long %", "Change in long", "Change in short", "Change in net", "Reading"]
        st.dataframe(paint(pdf, ("Net", "Change in long", "Change in net"), inv_cols=("Change in short",),
                           int_cols=("Long", "Short", "Net", "Change in long", "Change in short", "Change in net")),
                     hide_index=True, use_container_width=True)
        st.caption(f"Index futures open interest as on {d0}" + (f" (change is versus {got_[1][0]})" if prev_oi else "")
                   + ". NSE publishes this after the market closes, so in the morning you see yesterday's positions. "
                   "More short contracts = red (bearish pressure), fewer shorts / more longs = green. Net = long minus short.")
    cf_, cerr_ = fetch_cash_flows()
    if cf_:
        k1, k2 = st.columns(2)
        for col_, who_ in ((k1, "FII"), (k2, "DII")):
            if who_ in cf_:
                v_ = cf_[who_]
                col_.markdown(f'<div class="mdmood" style="background:{"#16a34a" if v_["net"] > 0 else "#dc2626"}">'
                              f'{who_} cash: {"BUYING" if v_["net"] > 0 else "SELLING"} Rs {abs(v_["net"]):,.0f} crore net'
                              f'<br><span style="font-weight:500;font-size:0.9rem">buy {v_["buy"]:,.0f} | sell {v_["sell"]:,.0f} | {v_["date"]}</span></div>',
                              unsafe_allow_html=True)
        if "FII" in cf_:
            chips.append((f"FII cash {cf_['FII']['net']:+,.0f} cr", "mdg" if cf_["FII"]["net"] > 0 else "mdr"))
    else:
        st.caption(f"FII / DII cash-market figures could not be loaded ({cerr_}).")
    if chips:
        S["mv_sum"] = chips

with t9:
    st.caption("Gap versus yesterday's close and where yesterday closed inside its range. Best viewed 9:00-9:15 IST "
               "(pre-open prices) - it still works later in the day (then it shows the current gap). Bias is a starting "
               "point for the day, not an entry signal: entries still need the Stocks / Greeks alerts.")
    if st.button("Load / refresh pre-market data") or t.time() < dt.time(9, 30):
        try:
            qk = ["NSE:" + v_ for v_ in K.STOCKS.values()] + ["NSE:NIFTY 50", "NSE:NIFTY BANK"]
            qq = kite.quote(qk)
            pm, pm_err = [], ""
            for disp, sym in list(K.STOCKS.items()) + [("NIFTY 50", "NIFTY 50"), ("BANKNIFTY", "NIFTY BANK")]:
                q1 = qq.get("NSE:" + sym) or {}
                last_ = q1.get("last_price") or 0
                pc_ = (q1.get("ohlc") or {}).get("close") or 0
                ph_ = pl_ = float("nan")
                try:                                            # yesterday's high/low from DAILY candles of the stock itself
                    _time.sleep(0.34)
                    dd9 = pd.DataFrame(kite.historical_data(int(q1["instrument_token"]), t.date() - dt.timedelta(days=10),
                                                            t.date(), "day"))
                    dd9["d"] = pd.to_datetime(dd9["date"]).dt.date
                    dd9 = dd9[dd9["d"] < t.date()]
                    ph_, pl_ = float(dd9["high"].iloc[-1]), float(dd9["low"].iloc[-1])
                except Exception as e9:
                    pm_err = pm_err or f"{sym}: {type(e9).__name__}: {str(e9)[:100]}"
                r9 = K.premarket_row(last_, pc_, ph_, pl_)
                pm.append(dict(Stock=disp, Price=round(last_, 2), PrevClose=round(pc_, 2), GapPct=round(r9["gap"], 2),
                               PrevClosePos=round(r9["pos"], 0) if r9["pos"] == r9["pos"] else None,
                               Bias=r9["bias"], Note=r9["note"]))
            S["pm"] = dict(rows=pm, at=t, err=pm_err)
        except Exception as e:
            st.warning(f"Pre-market data could not be loaded: {type(e).__name__}: {str(e)[:120]}")
    pm_ = S.get("pm")
    if pm_ and pm_["rows"]:
        pdf_ = pd.DataFrame(pm_["rows"])
        idx_rows = pdf_[pdf_["Stock"].isin(["NIFTY 50", "BANKNIFTY"])]
        stk_rows = pdf_[~pdf_["Stock"].isin(["NIFTY 50", "BANKNIFTY"])].copy()
        c_a, c_b = st.columns(2)
        for col_, (_, r_) in zip((c_a, c_b), idx_rows.iterrows()):
            col_.metric(r_["Stock"], f"{r_['Price']:.0f}", f"{r_['GapPct']:+.2f}% gap")
        if gift:
            st.write(f"Gift Nifty: {gift:+.2f}% ({gift_info})")
        stk_rows["_o"] = stk_rows["Bias"].map({"Bullish": 0, "Bearish": 0, "Mixed": 1, "Flat": 2}).fillna(3)
        stk_rows["_g"] = stk_rows["GapPct"].abs().fillna(0)
        stk_rows = stk_rows.sort_values(["_o", "_g"], ascending=[True, False]).drop(columns=["_o", "_g"])
        stk_rows.columns = ["Stock", "Price", "Prev close", "Gap %", "Prev close in range % (100 = at high)", "Bias", "Note"]
        st.dataframe(paint(stk_rows, ("Gap %",)), hide_index=True, use_container_width=True)
        if pm_.get("err"):
            st.caption(f"Yesterday's range could not be read for some stocks (first error - {pm_['err']}).")
        st.caption(f"Loaded {pm_['at'].strftime('%H:%M:%S')} IST. Bullish = gap up of 0.5%+ and yesterday closed in the upper 60% of its range; "
                   "Bearish = the mirror. Mixed = gap goes against yesterday's close, wait for confirmation.")
    else:
        st.info("Press the button to load gap data.")

with t10:
    st.subheader("Earnings calendar (results, next 10 days)")
    ev_rows, ev_err = fetch_events()
    if not ev_rows:
        st.warning(f"Earnings calendar could not be loaded ({ev_err or 'no results scheduled'}). NSE sometimes refuses cloud servers. "
                   "The same list is on NSE's website under Corporate Filings > Event Calendar.")
        if st.button("Retry earnings calendar"):
            fetch_events.clear()
            st.rerun()
    else:
        only_mine = st.checkbox("Only stocks in my scanner list", value=True)
        show = [r_ for r_ in ev_rows if r_["watch"]] if only_mine else ev_rows
        today_ = now().date()
        soon = [r_ for r_ in ev_rows if r_["watch"] and r_["date"] <= today_ + dt.timedelta(days=1)]
        if soon:
            st.warning("Results today / tomorrow in your list: " + ", ".join(
                f"{r_['symbol']} ({'today' if r_['date'] == today_ else 'tomorrow'})" for r_ in soon)
                + ". Expect gaps and wild option premiums around results; the stock scanner marks these.")
        if show:
            edf = pd.DataFrame([dict(Date=r_["date"].strftime("%a %d %b"), Stock=r_["symbol"], Company=r_["company"],
                                     Purpose=r_["purpose"], Mine="YES" if r_["watch"] else "") for r_ in show])
            st.dataframe(edf, hide_index=True, use_container_width=True)
        else:
            st.info("No results from your scanner list in the next 10 days.")
        st.caption("Source: NSE event calendar (refreshed hourly). Results days bring gaps and IV crush, so avoid fresh option buying on that stock unless you planned for it.")

st.caption("Educational analysis only. Not investment advice. Read-only: this app cannot place orders.")

# ---------- sound + vibration + toast for a NEW alert (this run only) ----------
_new_n = (len(fired) - _n_fired0) if _n_fired0 is not None else 0
if sound_on and (_new_n > 0 or S.pop("test_sound", False)):
    try:
        import io, wave
        sr_ = 22050
        tone_ = np.concatenate([np.concatenate([0.6 * np.sin(2 * np.pi * f_ * np.arange(int(sr_ * 0.22)) / sr_),
                                                np.zeros(int(sr_ * 0.08))]) for f_ in (880, 1175, 880, 1175, 1480)])
        buf_ = io.BytesIO()
        with wave.open(buf_, "wb") as w_:
            w_.setnchannels(1); w_.setsampwidth(2); w_.setframerate(sr_)
            w_.writeframes((tone_ * 32767).astype(np.int16).tobytes())
        try:
            st.audio(buf_.getvalue(), format="audio/wav", autoplay=True)
        except TypeError:
            st.audio(buf_.getvalue(), format="audio/wav")
        st.toast(f"{_new_n} new alert(s) - check Signals / Stocks" if _new_n else "Test sound", icon="🔔")
        import streamlit.components.v1 as _cv
        _cv.html("<script>try{navigator.vibrate&&navigator.vibrate([300,150,300,150,500]);}catch(e){}</script>", height=0)
    except Exception:
        pass

if refresh and market_open:                        # built-in auto refresh: wait, then rerun the script
    _time.sleep(refresh)
    st.rerun()
