"""Read-only market dashboard (Zerodha Kite Connect + global cues). NO order placement code."""
import hmac
import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
from kiteconnect import KiteConnect
from streamlit_autorefresh import st_autorefresh

import engine as E
import sheets_log as SL

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


def get_fut_hist(kite, token):
    try:
        return kite_candles(kite, token)
    except Exception:
        return None


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
    name = st.selectbox("Index", list(UND))
    strike_pref = st.radio("Option to suggest", ["ATM", "1 strike OTM"])
    width = st.slider("Strikes each side of ATM", 6, 20, 12)
    gift = st.number_input("Gift Nifty change % (type from your app, 0 = ignore)", value=0.0, step=0.1)
    refresh = st.selectbox("Auto refresh (sec)", [5, 10, 15, 30, 60, 120, 0], index=4)
    block_expiry = st.checkbox("No signals on expiry day", value=False)
    rule = st.radio("Build-up rule", ["My rule", "Standard"],
                    help="My rule: CE long buildup + PE short covering = BUY CE; PE short buildup + CE long unwinding = BUY PE. "
                         "Standard: bullish = CE (long buildup/short covering) with PE (short buildup/long unwinding); bearish is the mirror.")
    lookback = st.selectbox("Build-up lookback (min)", [5, 15, 30, 60], index=1)
    confirm_n = st.number_input("Confirm after N refreshes", 1, 5, 2)
    g_mode = st.selectbox("Greeks scanner scoring", list(E.SCORE_MODES), index=0)
    g_strike = st.radio("Greeks scanner strike", ["ATM", "1 strike ITM (safer)"])

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
today = t.strftime("%Y-%m-%d")
if SL.enabled() and S.get("sheet_day") != today:  # after a restart, reload today's alerts so they are not repeated
    S["sheet_day"] = today
    for r in SL.read_alerts():
        if str(r.get("date")) == today:
            try:
                fk0 = (today, r.get("index"), str(r.get("action", "")).replace("BUY ", ""), float(r.get("strike")))
            except Exception:
                continue
            if not any(f["key"] == fk0 for f in fired):
                fired.append(dict(key=fk0, time=r.get("time"), index=r.get("index"), action=r.get("action"),
                                  strike=r.get("strike"), entry=r.get("entry"), sl=r.get("sl"),
                                  t1=r.get("t1"), t2=r.get("t2")))
if bu_confirmed:
    fk = (t.strftime("%Y-%m-%d"), name, bsig["side"], bsig["strike"])
    if not any(f["key"] == fk for f in fired):
        fired.append(dict(key=fk, time=t.strftime("%H:%M:%S"), index=name, action=bsig["action"],
                          strike=bsig["strike"], entry=bsig["opt"]["entry"], sl=bsig["opt"]["sl"],
                          t1=bsig["opt"]["t1"], t2=bsig["opt"]["t2"]))
        new_alert = True
        if SL.enabled():
            ok, err = SL.append_alert(dict(date=today, time=t.strftime("%H:%M:%S"), index=name, expiry=str(expiry),
                                           action=bsig["action"], strike=bsig["strike"],
                                           entry=round(bsig["opt"]["entry"], 1), sl=round(bsig["opt"]["sl"], 1),
                                           t1=round(bsig["opt"]["t1"], 1), t2=round(bsig["opt"]["t2"], 1)))
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
oi_ok = {sd: E.oi_confirmation(sd, chain, ref, summ) for sd in ("CE", "PE")}
try:
    gres, gbest = E.greeks_scan(chain, atm, cfg["step"], spot, tr, vwap_idx, atm_iv, iv_ref, vol_ratio, oi_ok,
                                is_expiry_day, itm=g_strike.startswith("1"),
                                live=in_window and not (block_expiry and is_expiry_day),
                                closed=market_open, mode=g_mode)
except Exception as e:
    gres, gbest = None, None
    st.warning(f"Greeks scanner could not run: {e}")

# ---------- header ----------
st.caption(f"{t.strftime('%a %d %b %Y %H:%M:%S')} IST | {'MARKET OPEN' if market_open else 'MARKET CLOSED (last data)'}"
           f" | OI baseline taken {b['at']} | trend data: {hsrc}")
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
        st.caption("Estimate only. You place the order yourself; keep the stop loss.")
    else:
        why = []
        for sd in ("CE", "PE"):
            r = gres[sd]
            if r["score"] >= SCORE_NEED(g_mode) and not r["direction_ok"]:
                why.append(f"{sd}: score is enough but trend/VWAP/EMA/breakout do not all agree")
            elif r["score"] >= SCORE_NEED(g_mode) and not r["spread_ok"]:
                why.append(f"{sd}: bid-ask spread is above 3%")
        st.warning("**No Greeks entry** - " + ("; ".join(why) if why else
                   f"needs a score of {SCORE_NEED(g_mode)}+ with trend, VWAP, EMA and a candle-close breakout in the same direction"
                   + ("" if in_window else " (and outside the 9:30-15:00 entry window)")))
    with st.expander("Greeks checklists (CE and PE)"):
        for sd in ("CE", "PE"):
            r = gres[sd]
            st.write(f"**{sd} {r['strike']:.0f}** - {r['score']}/{r['max']} ({g_mode}), option price {r['ltp']:.1f}"
                     + (f", bid-ask spread {r['spread']:.1f}%" if r["spread"] is not None else ""))
            gdf = pd.DataFrame(r["items"])
            gdf["ok"] = gdf["ok"].map({True: "YES", False: "NO", None: "n/a"})
            st.dataframe(gdf, hide_index=True, use_container_width=True)

# ---------- signal card ----------
if sig["action"] == "NO TRADE":
    WHY = {"Market open": "Market is closed",
           "Entry window (9:30-15:00 IST)": "outside the 9:30-15:00 IST entry window",
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
    st.write(f"Approx. option levels (delta based): SL **{ol['sl']:.1f}**, T1 **{ol['t1']:.1f}**, T2 **{ol['t2']:.1f}**. "
             "Signal is void if index closes a 5-min candle beyond the SL level. You place the order yourself.")

with st.expander("Why? (checklist)", expanded=sig["action"] == "NO TRADE"):
    chk = pd.DataFrame(sig["checks"])
    chk["ok"] = chk["ok"].map({True: "YES", False: "NO"})
    st.dataframe(chk, hide_index=True, use_container_width=True)
    st.write(f"OI score {oscore:+.1f}: " + "; ".join(onotes))
    st.write(f"Global score {gscore:+.2f} (Gift Nifty {'included' if gift else 'not included'})")

# ---------- tabs ----------
t1, t2, t3, t4, t5, t6 = st.tabs(["Chart", "Option chain", "OI chart", "Global", "Build-up", "Zones"])
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
    st.caption("Daily change from Yahoo Finance (can lag). Yield change in basis points. Gift Nifty: enter in sidebar.")

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

st.caption("Educational analysis only. Not investment advice. Read-only: this app cannot place orders.")
