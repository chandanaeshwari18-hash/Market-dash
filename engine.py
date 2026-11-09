"""Pure calculation logic: Black-Scholes greeks, OI analytics, trend, signal engine.
No Streamlit / network code here so it can be tested offline."""
import math
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq


# ---------- Black-Scholes on futures price (r ~ 0) ----------
def bs_price(F, K, T, sig, cp):
    if T <= 0 or sig <= 0:
        return max(0.0, (F - K) if cp == "CE" else (K - F))
    s = sig * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sig * sig * T) / s
    d2 = d1 - s
    if cp == "CE":
        return F * norm.cdf(d1) - K * norm.cdf(d2)
    return K * norm.cdf(-d2) - F * norm.cdf(-d1)


def implied_vol(price, F, K, T, cp):
    intrinsic = max(0.0, (F - K) if cp == "CE" else (K - F))
    if price is None or price <= 0.05 or T <= 0 or price <= intrinsic:
        return float("nan")
    try:
        return brentq(lambda s: bs_price(F, K, T, s, cp) - price, 0.01, 3.0, xtol=1e-5)
    except Exception:
        return float("nan")


def greeks(F, K, T, sig, cp):
    """delta, vega (per 1 vol point)."""
    if not (sig == sig) or sig <= 0 or T <= 0:
        return float("nan"), float("nan")
    s = sig * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sig * sig * T) / s
    delta = norm.cdf(d1) if cp == "CE" else norm.cdf(d1) - 1
    vega = F * norm.pdf(d1) * math.sqrt(T) / 100.0
    return delta, vega


def greeks_full(F, K, T, sig, cp):
    """delta, gamma (per index point), theta (per day, rupees), vega (per 1 vol point). r = 0."""
    nan = float("nan")
    if not (sig == sig) or sig <= 0 or T <= 0:
        return nan, nan, nan, nan
    s = sig * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sig * sig * T) / s
    pdf = norm.pdf(d1)
    delta = norm.cdf(d1) if cp == "CE" else norm.cdf(d1) - 1
    return delta, pdf / (F * s), -F * pdf * sig / (2 * math.sqrt(T)) / 365.0, F * pdf * math.sqrt(T) / 100.0


def add_greeks(chain, F, T):
    """chain indexed by strike with ce_ltp / pe_ltp -> adds iv, delta, gamma, theta, vega columns."""
    for side, cp in (("ce", "CE"), ("pe", "PE")):
        ivs, dels, gams, ths, vegs = [], [], [], [], []
        for k, row in chain.iterrows():
            iv = implied_vol(row.get(f"{side}_ltp"), F, k, T, cp)
            d, g, th, v = greeks_full(F, k, T, iv, cp)
            ivs.append(iv * 100 if iv == iv else float("nan"))
            dels.append(d); gams.append(g); ths.append(th); vegs.append(v)
        chain[f"{side}_iv"] = ivs
        chain[f"{side}_delta"] = dels
        chain[f"{side}_gamma"] = gams
        chain[f"{side}_theta"] = ths
        chain[f"{side}_vega"] = vegs
    return chain


# ---------- OI analytics ----------
def oi_summary(chain, spot=None):
    """Support = biggest Put OI at/below spot, resistance = biggest Call OI at/above spot."""
    ce_oi = chain["ce_oi"].fillna(0)
    pe_oi = chain["pe_oi"].fillna(0)
    pcr = float(pe_oi.sum() / ce_oi.sum()) if ce_oi.sum() > 0 else float("nan")
    ce_use, pe_use = ce_oi, pe_oi
    if spot is not None:
        up, dn = ce_oi[ce_oi.index >= spot], pe_oi[pe_oi.index <= spot]
        if len(up) and up.max() > 0:
            ce_use = up
        if len(dn) and dn.max() > 0:
            pe_use = dn
    resistance = float(ce_use.idxmax()) if ce_use.max() > 0 else float("nan")
    support = float(pe_use.idxmax()) if pe_use.max() > 0 else float("nan")
    strikes = chain.index.values.astype(float)
    pain = []
    for k in strikes:
        ce_pay = (np.maximum(0, k - strikes) * ce_oi.values).sum()
        pe_pay = (np.maximum(0, strikes - k) * pe_oi.values).sum()
        pain.append(ce_pay + pe_pay)
    max_pain = float(strikes[int(np.argmin(pain))]) if len(pain) else float("nan")
    return dict(pcr=pcr, resistance=resistance, support=support, max_pain=max_pain)


# ---------- Trend ----------
def trend_state(df):
    """df: 5-min OHLC. Returns dict with label UP/DOWN/SIDEWAYS and helpers."""
    out = dict(label="NA", ema20=float("nan"), ema50=float("nan"), swing_high=float("nan"),
               swing_low=float("nan"), df=df)
    if df is None or len(df) < 60:
        return out
    d = df.copy()
    d["ema20"] = d["close"].ewm(span=20, adjust=False).mean()
    d["ema50"] = d["close"].ewm(span=50, adjust=False).mean()
    c, e20, e50 = d["close"].iloc[-1], d["ema20"].iloc[-1], d["ema50"].iloc[-1]
    e20_prev = d["ema20"].iloc[-7]
    gap = abs(e20 - e50) / c
    if c > e20 > e50 and e20 > e20_prev and gap > 0.0008:
        label = "UP"
    elif c < e20 < e50 and e20 < e20_prev and gap > 0.0008:
        label = "DOWN"
    else:
        label = "SIDEWAYS"
    out.update(label=label, ema20=float(e20), ema50=float(e50),
               swing_high=float(d["high"].iloc[-12:].max()),
               swing_low=float(d["low"].iloc[-12:].min()), df=d)
    return out


# ---------- Global cues ----------
def global_score(changes, gift_pct=0.0):
    """changes: dict name -> pct change (yield in bp). Returns score in [-1, 1]."""
    pts, wts = 0.0, 0.0

    def add(sign_val, thresh, w, invert=False):
        nonlocal pts, wts
        if sign_val is None or sign_val != sign_val:
            return
        wts += w
        if abs(sign_val) >= thresh:
            s = 1 if sign_val > 0 else -1
            pts += w * (-s if invert else s)

    for k in ("S&P 500", "Nasdaq", "Nikkei", "Hang Seng", "DAX", "FTSE"):
        add(changes.get(k), 0.3, 1)
    add(changes.get("Crude (WTI)"), 1.0, 1, invert=True)
    add(changes.get("USD/INR"), 0.2, 1, invert=True)
    add(changes.get("US 10Y"), 4.0, 1, invert=True)  # basis points
    if gift_pct:
        add(gift_pct, 0.3, 2)
    return pts / wts if wts else 0.0


def sign_dir(x, thresh):
    return 1 if x >= thresh else (-1 if x <= -thresh else 0)


# ---------- OI direction ----------
def oi_direction(pcr, fut_px_chg, fut_oi_chg, spot, support, resistance):
    score, notes = 0.0, []
    if pcr == pcr:
        if pcr >= 1.15:
            score += 1; notes.append(f"PCR {pcr:.2f} high (put writing, bullish)")
        elif pcr <= 0.85:
            score -= 1; notes.append(f"PCR {pcr:.2f} low (call writing, bearish)")
        else:
            notes.append(f"PCR {pcr:.2f} neutral")
    if fut_px_chg > 0 and fut_oi_chg > 0:
        score += 1; notes.append("Futures: long build-up")
    elif fut_px_chg < 0 and fut_oi_chg > 0:
        score -= 1; notes.append("Futures: short build-up")
    elif fut_px_chg > 0 and fut_oi_chg < 0:
        score += 0.5; notes.append("Futures: short covering")
    elif fut_px_chg < 0 and fut_oi_chg < 0:
        score -= 0.5; notes.append("Futures: long unwinding")
    else:
        notes.append("Futures: no clear build-up")
    if resistance == resistance and 0 <= (resistance - spot) / spot < 0.0025:
        score -= 0.5; notes.append("Spot right under max call OI wall")
    if support == support and 0 <= (spot - support) / spot < 0.0025:
        score += 0.5; notes.append("Spot right above max put OI wall")
    return sign_dir(score, 1.0), score, notes


# ---------- Signal ----------
def make_signal(spot, trend, oi_dir, glob_dir, support, resistance, in_window,
                market_open, min_rr=1.5):
    checks = []

    def chk(name, ok, detail=""):
        checks.append(dict(check=name, ok=bool(ok), detail=detail))

    chk("Market open", market_open, "" if market_open else "Closed - no live signals")
    chk("Entry window (9:20-15:15 IST)", in_window)
    t = trend["label"]
    tdir = {"UP": 1, "DOWN": -1}.get(t, 0)
    chk("Trend is one-sided", tdir != 0, f"5-min trend: {t}")
    chk("OI agrees with trend", oi_dir == tdir and tdir != 0)
    chk("Global cues not against", glob_dir != -tdir if tdir else False)

    base = dict(action="NO TRADE", side=None, checks=checks, levels=None, reason="")
    if not all(c["ok"] for c in checks):
        base["reason"] = "; ".join(c["check"] for c in checks if not c["ok"]) + " - not satisfied"
        return base

    if tdir == 1:
        cands = [x for x in (trend["swing_low"], support) if x == x and x < spot * (1 - 0.0025)]
        sl = max(cands) if cands else spot * (1 - 0.004)
        r = spot - sl
        t1 = spot + 1.5 * r
        if resistance == resistance and resistance - spot < min_rr * r:
            base["reason"] = f"Call OI wall {resistance:.0f} too close for {min_rr}R reward"
            return base
        t2 = max(spot + 2.5 * r, resistance) if resistance == resistance else spot + 2.5 * r
        action, side = "BUY CE", "CE"
    else:
        cands = [x for x in (trend["swing_high"], resistance) if x == x and x > spot * (1 + 0.0025)]
        sl = min(cands) if cands else spot * (1 + 0.004)
        r = sl - spot
        t1 = spot - 1.5 * r
        if support == support and spot - support < min_rr * r:
            base["reason"] = f"Put OI wall {support:.0f} too close for {min_rr}R reward"
            return base
        t2 = min(spot - 2.5 * r, support) if support == support else spot - 2.5 * r
        action, side = "BUY PE", "PE"
    base.update(action=action, side=side, reason="All conditions confirmed",
                levels=dict(entry=spot, sl=sl, t1=t1, t2=t2, risk=r))
    return base


def option_levels(sig, opt_ltp, opt_delta):
    """First-order premium levels from spot levels using delta (approximate)."""
    L = sig["levels"]
    d = abs(opt_delta) if opt_delta == opt_delta else 0.5
    sgn = 1 if sig["side"] == "CE" else -1
    f = lambda lvl: opt_ltp + d * sgn * (lvl - L["entry"])
    return dict(entry=opt_ltp, sl=max(0.05, f(L["sl"])), t1=f(L["t1"]), t2=f(L["t2"]))


# =====================================================================
# Per-strike OI build-up detection (CE/PE long build-up, short covering ...)
# =====================================================================
import datetime as _dt

STATE_NAME = {"LB": "Long buildup", "SB": "Short buildup", "SC": "Short covering",
              "LU": "Long unwinding", "-": "No clear change"}


def classify(dp, doi, thr_p=2.0, thr_oi=2.0):
    """dp, doi are % changes of option price and OI."""
    if abs(dp) < thr_p or abs(doi) < thr_oi:
        return "-"
    if dp > 0 and doi > 0:
        return "LB"
    if dp < 0 and doi > 0:
        return "SB"
    if dp > 0 and doi < 0:
        return "SC"
    return "LU"


def pick_reference(snaps, now, lookback_min):
    """Return (reference snapshot, has_full_lookback). None if not enough history yet."""
    if len(snaps) < 2:
        return None, False
    target = now - _dt.timedelta(minutes=lookback_min)
    older = [s for s in snaps if s["t"] <= target]
    ref = older[-1] if older else snaps[0]
    if (now - ref["t"]).total_seconds() < 60:
        return None, False
    return ref, bool(older)


BU_COLS = ["strike", "side", "ltp", "dp", "doi", "d_oi", "state"]


def buildup_analysis(chain, ref, atm, step, window=3, thr_p=2.0, thr_oi=2.0):
    rows = []
    if ref is None:
        return pd.DataFrame(columns=BU_COLS)
    rc = ref["chain"]
    for k in chain.index:
        if abs(k - atm) > window * step + 1e-9 or k not in rc.index:
            continue
        for side in ("ce", "pe"):
            p0, o0 = rc.at[k, f"{side}_ltp"], rc.at[k, f"{side}_oi"]
            p1, o1 = chain.at[k, f"{side}_ltp"], chain.at[k, f"{side}_oi"]
            if not (p0 == p0 and o0 == o0 and p1 == p1 and o1 == o1) or p0 <= 0 or o0 <= 0:
                continue
            dp, doi = (p1 / p0 - 1) * 100, (o1 / o0 - 1) * 100
            rows.append(dict(strike=float(k), side=side.upper(), ltp=float(p1), dp=dp, doi=doi,
                             d_oi=float(o1 - o0), state=classify(dp, doi, thr_p, thr_oi)))
    return pd.DataFrame(rows, columns=BU_COLS)


def side_states(df):
    """Dominant build-up state per side (weighted by absolute OI change) and its strongest strike."""
    out = {}
    for side in ("CE", "PE"):
        d = df[(df["side"] == side) & (df["state"] != "-")] if len(df) else df
        if len(d) == 0:
            out[side] = dict(state="-", strike=float("nan"), share=0.0, n=0)
            continue
        w = d.assign(w=d["d_oi"].abs()).groupby("state")["w"].sum()
        top = w.idxmax()
        share = float(w.max() / w.sum()) if w.sum() > 0 else 0.0
        dd = d[d["state"] == top]
        best = float(dd.loc[dd["d_oi"].abs().idxmax(), "strike"])
        out[side] = dict(state=top if share >= 0.5 else "-", strike=best, share=share, n=int(len(dd)))
    return out


def index_levels(side, spot, swing_low, swing_high, support, resistance, min_rr=1.5):
    """Index-level entry / SL / targets. Returns (levels or None, reason)."""
    if side == "CE":
        c = [x for x in (swing_low, support) if x == x and x < spot * (1 - 0.0025)]
        sl = max(c) if c else spot * (1 - 0.004)
        r = spot - sl
        if resistance == resistance and 0 < resistance - spot < min_rr * r:
            return None, f"Call OI wall {resistance:.0f} too close for {min_rr}R reward"
        return dict(entry=spot, sl=sl, t1=spot + 1.5 * r, t2=spot + 2.5 * r, risk=r), ""
    c = [x for x in (swing_high, resistance) if x == x and x > spot * (1 + 0.0025)]
    sl = min(c) if c else spot * (1 + 0.004)
    r = sl - spot
    if support == support and 0 < spot - support < min_rr * r:
        return None, f"Put OI wall {support:.0f} too close for {min_rr}R reward"
    return dict(entry=spot, sl=sl, t1=spot - 1.5 * r, t2=spot - 2.5 * r, risk=r), ""


def buildup_signal(sides, rule, spot, atm, step, trend, summ, live, chain):
    ce, pe = sides["CE"], sides["PE"]
    res = dict(side=None, action="NO SIGNAL", reason="", strike=None, idx=None, opt=None,
               ce=ce, pe=pe, exits=[])
    if not live:
        res["reason"] = "Market closed or outside 9:20-15:15 IST window"
        return res
    side = None
    if rule == "My rule":
        if ce["state"] == "LB" and pe["state"] == "SC":
            side = "CE"
        elif pe["state"] == "SB" and ce["state"] == "LU":
            side = "PE"
    else:
        if ce["state"] in ("LB", "SC") and pe["state"] in ("SB", "LU"):
            side = "CE"
        elif pe["state"] in ("LB", "SC") and ce["state"] in ("SB", "LU"):
            side = "PE"
    if side is None:
        res["reason"] = (f"Pattern not formed. CE: {STATE_NAME[ce['state']]}, PE: {STATE_NAME[pe['state']]}")
        return res
    pattern_strike = ce["strike"] if side == "CE" else pe["strike"]
    strike = min(max(pattern_strike, atm - step), atm + step)
    if strike not in chain.index:
        strike = atm
    lv, why = index_levels(side, spot, trend.get("swing_low", float("nan")), trend.get("swing_high", float("nan")),
                           summ["support"], summ["resistance"])
    if lv is None:
        res["reason"] = "Pattern formed but " + why
        return res
    ltp = float(chain.at[strike, f"{side.lower()}_ltp"])
    dlt = float(chain.at[strike, f"{side.lower()}_delta"])
    opt = option_levels(dict(side=side, levels=lv), ltp, dlt)
    res.update(side=side, action=f"BUY {side}", strike=float(strike), idx=lv, opt=opt,
               pattern_strike=float(pattern_strike), reason="Pattern confirmed",
               exits=[
                   f"SL: exit if a 5-min candle closes beyond index {lv['sl']:.0f} (option about {opt['sl']:.1f}).",
                   f"T1 {lv['t1']:.0f} (option about {opt['t1']:.1f}): book half, move SL to entry.",
                   f"T2 {lv['t2']:.0f} (option about {opt['t2']:.1f}): exit the rest.",
                   "Exit early if the opposite build-up pattern appears.",
                   "Time exit by 15:15 IST; avoid holding through event risk.",
               ])
    return res


# =====================================================================
# Volume: VWAP, volume profile, buyer / seller zones
# =====================================================================
def vwap_daily(df):
    """VWAP that restarts every day. df needs high, low, close, volume and a datetime index."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    day = pd.Series(df.index.date, index=df.index)
    pv = (tp * df["volume"]).groupby(day).cumsum()
    v = df["volume"].groupby(day).cumsum()
    return pv / v.replace(0, np.nan)


def volume_profile(df, nbins=60):
    """Spread each candle's volume over its high-low range. Returns POC, value area and bins."""
    lo, hi = float(df["low"].min()), float(df["high"].max())
    if not (hi > lo):
        return None
    edges = np.linspace(lo, hi, nbins + 1)
    prof = np.zeros(nbins)
    for h, l, v in zip(df["high"].values, df["low"].values, df["volume"].values):
        if not v or v <= 0:
            continue
        if h <= l:
            i = int(min(nbins - 1, max(0, np.searchsorted(edges, h) - 1)))
            prof[i] += v
            continue
        a = np.clip(np.minimum(edges[1:], h) - np.maximum(edges[:-1], l), 0, None)
        prof += v * a / (h - l)
    if prof.sum() <= 0:
        return None
    centers = (edges[:-1] + edges[1:]) / 2
    poc_i = int(np.argmax(prof))
    lo_i = hi_i = poc_i
    acc, target = prof[poc_i], 0.7 * prof.sum()
    while acc < target and (lo_i > 0 or hi_i < nbins - 1):
        up = prof[hi_i + 1] if hi_i < nbins - 1 else -1.0
        dn = prof[lo_i - 1] if lo_i > 0 else -1.0
        if up >= dn:
            hi_i += 1
            acc += prof[hi_i]
        else:
            lo_i -= 1
            acc += prof[lo_i]
    hvn = prof >= np.percentile(prof, 80)
    return dict(edges=edges, centers=centers, prof=prof, poc=float(centers[poc_i]),
                vah=float(edges[hi_i + 1]), val=float(edges[lo_i]), hvn=hvn)


def volume_spike_zones(df, mult=2.0, win=20, keep=3, max_width_pct=0.8):
    """Candles with unusually high volume that closed strongly one way mark where big players acted.
    Strong up candle -> BUY (demand) zone, strong down candle -> SELL (supply) zone.
    A zone is dropped once price has closed through it."""
    d = df.copy()
    d["vavg"] = d["volume"].rolling(win, min_periods=10).mean().shift(1)
    rng = (d["high"] - d["low"]).replace(0, np.nan)
    d["pos"] = (d["close"] - d["low"]) / rng
    buys, sells = [], []
    n = len(d)
    for i in range(n):
        r = d.iloc[i]
        if not (r["vavg"] == r["vavg"]) or r["vavg"] <= 0 or not (r["pos"] == r["pos"]):
            continue
        if r["volume"] < mult * r["vavg"]:
            continue
        later = d.iloc[i + 1:]
        minw = r["close"] * 0.0003
        if r["close"] > r["open"] and r["pos"] >= 0.65:
            lo_, hi_ = r["low"], max(r["open"], r["low"] + minw)
            if len(later) and (later["close"] < lo_).any():
                continue
            if (hi_ - lo_) / r["close"] * 100 > max_width_pct:
                hi_ = lo_ + r["close"] * max_width_pct / 100
            buys.append(dict(kind="BUY", low=float(lo_), high=float(hi_), time=d.index[i],
                             strength=float(r["volume"] / r["vavg"]), source="Volume spike"))
        elif r["close"] < r["open"] and r["pos"] <= 0.35:
            hi_, lo_ = r["high"], min(r["open"], r["high"] - minw)
            if len(later) and (later["close"] > hi_).any():
                continue
            if (hi_ - lo_) / r["close"] * 100 > max_width_pct:
                lo_ = hi_ - r["close"] * max_width_pct / 100
            sells.append(dict(kind="SELL", low=float(lo_), high=float(hi_), time=d.index[i],
                              strength=float(r["volume"] / r["vavg"]), source="Volume spike"))
    top = lambda z: sorted(z, key=lambda x: -x["strength"])[:keep]
    return top(buys) + top(sells)


def oi_wall_zones(chain, spot, step, top=2):
    """Heaviest Call OI above spot = writers selling (SELL zone); heaviest Put OI below spot = BUY zone."""
    out = []
    ce = chain["ce_oi"].fillna(0)
    pe = chain["pe_oi"].fillna(0)
    for k, oi in ce[ce.index >= spot].nlargest(top).items():
        if oi > 0:
            out.append(dict(kind="SELL", low=float(k - 0.15 * step), high=float(k + 0.15 * step),
                            time=None, strength=float(oi), source="Call OI wall"))
    for k, oi in pe[pe.index <= spot].nlargest(top).items():
        if oi > 0:
            out.append(dict(kind="BUY", low=float(k - 0.15 * step), high=float(k + 0.15 * step),
                            time=None, strength=float(oi), source="Put OI wall"))
    return out


def zone_warning(side, lv, zones):
    """Warn when an opposing zone sits between entry and Target 1."""
    if not lv or not zones:
        return ""
    if side == "CE":
        blk = [z for z in zones if z["kind"] == "SELL" and lv["entry"] <= z["low"] <= lv["t1"]]
        if blk:
            z = min(blk, key=lambda z: z["low"])
            return (f"Seller zone {z['low']:.0f}-{z['high']:.0f} ({z['source']}) lies before Target 1 - "
                    "consider booking early there.")
    else:
        blk = [z for z in zones if z["kind"] == "BUY" and lv["t1"] <= z["high"] <= lv["entry"]]
        if blk:
            z = max(blk, key=lambda z: z["high"])
            return (f"Buyer zone {z['low']:.0f}-{z['high']:.0f} ({z['source']}) lies before Target 1 - "
                    "consider booking early there.")
    return ""


# =====================================================================
# Greeks 10-point scanner (price action = trigger, option chain = confirmation, Greeks = option selection)
# =====================================================================
def structure(df, n=12):
    """Higher-high + higher-low (or lower-high + lower-low) of the last n candles vs the n before."""
    if df is None or len(df) < 2 * n + 2:
        return None
    a, b = df.iloc[-n:], df.iloc[-2 * n:-n]
    return dict(hh_hl=bool(a["high"].max() > b["high"].max() and a["low"].min() > b["low"].min()),
                ll_lh=bool(a["high"].max() < b["high"].max() and a["low"].min() < b["low"].min()))


def breakout_info(df, n=12, closed=False):
    """Close vs the high/low of the previous n candles. closed=True judges the last COMPLETED candle
    (ignores the candle still forming), which is the 'candle close' trigger."""
    if df is None:
        return None
    d = df.iloc[:-1] if closed else df
    if len(d) < n + 2:
        return None
    prev = d.iloc[-n - 1:-1]
    last = float(d["close"].iloc[-1])
    ph, pl = float(prev["high"].max()), float(prev["low"].min())
    return dict(prev_high=ph, prev_low=pl, last=last, up=last > ph, down=last < pl)


def oi_confirmation(side, chain, ref, summ, thr=2.0):
    """CE: call OI at resistance unwinding or put OI at support building. PE: the mirror. None = not enough history."""
    if ref is None:
        return None
    rc = ref["chain"]

    def chg(col, k):
        if k != k or k not in chain.index or k not in rc.index:
            return None
        o0, o1 = rc.at[k, col], chain.at[k, col]
        return (o1 / o0 - 1) * 100 if o0 and o0 > 0 else None
    r, s = chg("ce_oi", summ["resistance"]), chg("pe_oi", summ["support"])
    if r is None and s is None:
        return None
    if side == "CE":
        return bool((r is not None and r <= -thr) or (s is not None and s >= thr))
    return bool((s is not None and s <= -thr) or (r is not None and r >= thr))


def greeks_checklist(side, chain, strike, spot, tr, vwap_idx, atm_iv, iv_ref, vol_ratio, oi_ok, expiry_day, closed=False):
    s = side.lower()
    row = chain.loc[strike]
    ltp, delta = float(row[f"{s}_ltp"]), float(row[f"{s}_delta"])
    gamma, theta, iv = float(row[f"{s}_gamma"]), float(row[f"{s}_theta"]), float(row[f"{s}_iv"])
    bull = side == "CE"
    items = []

    def add(name, ok, detail):
        items.append(dict(factor=name, ok=ok, detail=detail))
    st_, bo = structure(tr["df"]), breakout_info(tr["df"], closed=closed)
    struct_ok = None if st_ is None else (st_["hh_hl"] if bull else st_["ll_lh"])
    trend_ok = bool(struct_ok)
    add("1 Trend (5-min structure)", trend_ok,
        f"{'HH+HL' if bull else 'LL+LH'} structure: {'n/a' if struct_ok is None else struct_ok}")
    if vwap_idx == vwap_idx:
        vwap_ok = bool(spot > vwap_idx if bull else spot < vwap_idx)
        add("2 VWAP", vwap_ok, f"price {spot:.0f} vs VWAP {vwap_idx:.0f}")
    else:
        vwap_ok = False
        add("2 VWAP", None, "futures data not available")
    e20, e50 = tr.get("ema20", float("nan")), tr.get("ema50", float("nan"))
    ema_ok = bool(e20 == e20 and e50 == e50 and (e20 > e50 if bull else e20 < e50))
    add("3 EMA 20 vs 50", ema_ok, f"EMA20 {e20:.0f} vs EMA50 {e50:.0f}" if e20 == e20 else "not enough candles")
    broke = None if bo is None else (bo["up"] if bull else bo["down"])
    breakout_ok = bool(broke)
    add("4 Breakout (candle close)" if closed else "4 Breakout (price now)", None if broke is None else breakout_ok,
        "no candle data" if bo is None else
        f"close {bo['last']:.0f} vs previous {'high' if bull else 'low'} {bo['prev_high'] if bull else bo['prev_low']:.0f}")
    lo, hi = (0.45, 0.60) if expiry_day else (0.45, 0.65)
    add("5 Delta", bool(lo <= abs(delta) <= hi), f"{abs(delta):.2f} (want {lo:.2f}-{hi:.2f}; best 0.50-0.60)")
    gmax = float(chain[f"{s}_gamma"].max())
    add("6 Gamma", bool(gamma == gamma and gmax == gmax and gamma >= 0.8 * gmax),
        f"{gamma:.5f} vs strongest strike {gmax:.5f} (needs 80% or more; no fixed minimum)")
    lim = 1.5 if expiry_day else 1.0
    tpct = abs(theta) / ltp * 100 if ltp > 0 else float("nan")
    add("7 Theta/Premium", bool(tpct == tpct and tpct < lim), f"{tpct:.2f}% per day (limit {lim}%); theta {theta:.1f}/day")
    iv_ok = bool(iv == iv and atm_iv == atm_iv and iv <= 1.2 * atm_iv and (iv_ref is None or atm_iv <= 1.10 * iv_ref))
    add("8 IV", iv_ok, f"option IV {iv:.1f} vs ATM IV {atm_iv:.1f}"
        + (f"; ATM IV change over lookback {(atm_iv / iv_ref - 1) * 100:+.1f}%" if iv_ref else "; spike check needs more history"))
    add("9 OI confirmation", oi_ok, "call OI at resistance unwinding / put OI at support building" if bull
        else "put OI at support unwinding / call OI at resistance building")
    add("10 Volume", None if vol_ratio is None else bool(vol_ratio > 1.0),
        "futures volume not available" if vol_ratio is None else f"candle volume {vol_ratio:.1f}x recent average")
    score = sum(1 for i in items if i["ok"] is True)
    grade = "A+ trade - entry allowed" if score >= 8 else "Watch - wait for breakout confirmation" if score >= 6 else "Avoid"
    spread = None
    bid, ask = row.get(f"{s}_bid", 0), row.get(f"{s}_ask", 0)
    if bid and ask and bid > 0 and ask > 0:
        spread = (ask - bid) / ((ask + bid) / 2) * 100
    spread_ok = spread is None or spread <= 3.0
    direction_ok = trend_ok and vwap_ok and ema_ok and breakout_ok
    entry_ok = bool(score >= 8 and direction_ok and spread_ok)
    level = None if bo is None else (bo["prev_high"] if bull else bo["prev_low"])
    levels = dict(entry=ltp, sl_premium=ltp * 0.75, sl_premium_tight=ltp * 0.80, t1=ltp * 1.25, t2=ltp * 1.45,
                  index_sl=level, delta=abs(delta) if delta == delta else 0.5)
    return dict(side=side, strike=float(strike), items=items, score=score, grade=grade, entry_ok=entry_ok,
                direction_ok=direction_ok, trend_ok=trend_ok, breakout_ok=breakout_ok, spread=spread,
                spread_ok=spread_ok, levels=levels, ltp=ltp)


SCORE_MODES = {
    "10-point (final scanner)": dict(max=10, aplus=8, good=6),
    "8-point (5-Greek A+)": dict(max=8, aplus=7, good=6),
    "7-point (simple)": dict(max=7, aplus=6, good=5),
}


def apply_mode(r, mode):
    """Re-score a 10-point checklist with the 8-point or 7-point version of the strategy.
    Direction filters (trend, VWAP, EMA, breakout) and the spread check always stay mandatory."""
    if mode not in SCORE_MODES or mode.startswith("10"):
        r["max"] = 10
        r["mode"] = mode
        return r
    it = {i["factor"][:2].strip(): i["ok"] for i in r["items"]}
    g = lambda k: it.get(k) is True
    brk_vol = g("4") and it.get("10") is not False          # 'Volume/Price breakout'
    oi_vol = g("9") or g("10")                              # 'Volume/OI'
    if mode.startswith("8"):
        parts = [g("1"), g("2"), g("5"), g("6"), g("7"), g("8"), g("9"), brk_vol]
    else:
        parts = [g("1"), g("2"), g("5"), g("6"), g("8"), g("7"), oi_vol]
    m = SCORE_MODES[mode]
    r["score"], r["max"], r["mode"] = sum(parts), m["max"], mode
    r["grade"] = ("A+ trade - entry allowed" if r["score"] >= m["aplus"] else
                  "Good - possible with smaller size" if r["score"] >= m["good"] else "Avoid / wait")
    r["entry_ok"] = bool(r["score"] >= m["aplus"] and r["direction_ok"] and r["spread_ok"])
    return r


def conflict_limit(score):
    """Sai's rule (10-point scanner, revised 6 Nov): the OPPOSITE side must stay quiet.
    Score 10 or 9 allows the opposite side up to 4; score 8 allows up to 3."""
    return 4 if score >= 9 else 3 if score == 8 else None


def apply_conflict_rule(res):
    """Block an entry when the other side scores too high (mixed signals). 10-point mode only. In place."""
    for sd, od in (("CE", "PE"), ("PE", "CE")):
        r, o = res[sd], res[od]
        r["conflict"] = ""
        if r.get("max") != 10 or not r.get("entry_ok"):
            continue
        lim = conflict_limit(r["score"])
        if lim is not None and o["score"] > lim:
            r["entry_ok"] = False
            r["conflict"] = f"{od} scores {o['score']}/10 but must be {lim} or less when {sd} is {r['score']}/10"
            r["grade"] = f"Conflict - {od} {o['score']}/10 (needs {lim} or less)"


def greeks_scan(chain, atm, step, spot, tr, vwap_idx, atm_iv, iv_ref, vol_ratio, oi_ok, expiry_day,
                itm=False, live=True, closed=False, mode="10-point (final scanner)", conflict_rule=True):
    """Score the CE and PE candidate strikes (ATM, or 1 strike ITM). Returns (results by side, best entry or None)."""
    res = {}
    for sd in ("CE", "PE"):
        k = atm
        if itm:
            k = atm - step if sd == "CE" else atm + step
        if k not in chain.index:
            k = atm
        res[sd] = apply_mode(greeks_checklist(sd, chain, k, spot, tr, vwap_idx, atm_iv, iv_ref, vol_ratio,
                                              oi_ok[sd], expiry_day, closed), mode)
    if conflict_rule:
        apply_conflict_rule(res)
    cands = [r for r in res.values() if r["entry_ok"]]
    return res, (max(cands, key=lambda r: r["score"]) if live and cands else None)


def greeks_exit_rules(r):
    bull, L = r["side"] == "CE", r["levels"]
    lvl = f" ({L['index_sl']:.0f})" if L.get("index_sl") else ""
    idx_move = lambda pct: L["entry"] * pct / L["delta"]
    return [
        f"Exit if a 5-min candle closes {'below' if bull else 'above'} VWAP, or the breakout level{lvl} fails.",
        f"{'ATR-based' if L.get('atr_based') else 'Emergency'} premium stop: {L['sl_premium']:.1f} ({(L['sl_premium'] / L['entry'] - 1) * 100:.0f}%); tighter option {L['sl_premium_tight']:.1f}. Whichever comes first.",
        f"T1 {L['t1']:.1f} ({(L['t1'] / L['entry'] - 1) * 100:+.0f}%, about {idx_move(L['t1'] / L['entry'] - 1):.0f} index points): book half, move stop to entry {L['entry']:.1f}.",
        f"Runner: T2 {L['t2']:.1f} ({(L['t2'] / L['entry'] - 1) * 100:+.0f}%) or trail with VWAP and the previous candle {'low' if bull else 'high'}.",
    ]


# =====================================================================
# Indicators (Wilder RSI / ATR / ADX) and ATR-based option levels
# =====================================================================
def _wilder(x, n):
    return x.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(close, n=14):
    d = close.diff()
    up, dn = d.clip(lower=0), (-d).clip(lower=0)
    rs = _wilder(up, n) / _wilder(dn, n).replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(~((_wilder(dn, n) == 0) & (_wilder(up, n) > 0)), 100.0)


def true_range(df):
    pc = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)


def atr(df, n=14):
    return _wilder(true_range(df), n)


def adx(df, n=14):
    """Returns (adx, +DI, -DI) series."""
    up, dn = df["high"].diff(), -df["low"].diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    a = _wilder(true_range(df), n)
    pdi, mdi = 100 * _wilder(pdm, n) / a, 100 * _wilder(mdm, n) / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return _wilder(dx.fillna(0), n), pdi, mdi


def indicators(df, n=14):
    """Latest RSI / ADX / ATR from 5-min candles (NaN when there is not enough data)."""
    nan = float("nan")
    out = dict(rsi=nan, adx=nan, atr=nan, atr_pct=nan, pdi=nan, mdi=nan, atr_raw=nan)
    try:
        if df is None or len(df) < 3 * n:
            return out
        a_, p_, m_ = adx(df, n)
        tr_ = true_range(df)
        cap_ = 2.5 * tr_.rolling(50, min_periods=10).median()          # one wild opening candle must not blow up the stop
        at = float(_wilder(tr_.clip(upper=cap_.fillna(tr_.max())), n).iloc[-1])
        out["atr_raw"] = float(atr(df, n).iloc[-1])
        out.update(rsi=float(rsi(df["close"], n).iloc[-1]), adx=float(a_.iloc[-1]), atr=at,
                   atr_pct=at / float(df["close"].iloc[-1]) * 100, pdi=float(p_.iloc[-1]), mdi=float(m_.iloc[-1]))
    except Exception:
        pass
    return out


def ind_warnings(side, ind):
    """Plain-language warnings (display and log only - they do NOT block an alert)."""
    w = []
    a, r = ind.get("adx", float("nan")), ind.get("rsi", float("nan"))
    if a == a and a < 20:
        w.append(f"ADX {a:.0f}: weak/sideways trend")
    if r == r:
        if side == "CE" and r > 75:
            w.append(f"RSI {r:.0f}: overbought, chasing risk")
        elif side == "PE" and r < 25:
            w.append(f"RSI {r:.0f}: oversold, chasing risk")
        elif side == "CE" and r < 50:
            w.append(f"RSI {r:.0f}: bullish momentum weak")
        elif side == "PE" and r > 50:
            w.append(f"RSI {r:.0f}: bearish momentum weak")
    p, m = ind.get("pdi", float("nan")), ind.get("mdi", float("nan"))
    if p == p and m == m:
        if side == "CE" and m > p:
            w.append(f"-DI {m:.0f} above +DI {p:.0f}: sellers stronger than buyers")
        elif side == "PE" and p > m:
            w.append(f"+DI {p:.0f} above -DI {m:.0f}: buyers stronger than sellers")
    return "; ".join(w)


RSI_BLOCK = 80   # CE alert blocked when RSI > 80; PE blocked when RSI < 20 (extreme chasing)


def rsi_blocked(side, ind, limit=RSI_BLOCK):
    """True when the entry would chase an extreme move: CE with RSI above limit, PE with RSI below 100-limit."""
    r = ind.get("rsi", float("nan")) if ind else float("nan")
    if r != r:
        return False
    return (side == "CE" and r > limit) or (side == "PE" and r < 100 - limit)


def vix_warning(vix, chg_pct):
    """India VIX: high or jumping VIX means expensive option premium (display and log only)."""
    w = []
    if vix == vix and vix is not None:
        if vix >= 20:
            w.append(f"India VIX {vix:.1f}: high, premiums expensive")
        if chg_pct == chg_pct and chg_pct is not None and chg_pct >= 5:
            w.append(f"India VIX up {chg_pct:.1f}% today: fear rising, premium inflated")
    return "; ".join(w)


def recent_same_side(fired, index, side, now_t, minutes=60, source=None):
    """True when an alert for the same index and side already fired in the last `minutes` (same move, not a new setup).
    ORB paper alerts are a separate strategy: they only count when source="ORB", and other strategies ignore them."""
    for f in fired:
        try:
            if f.get("index") != index or side not in str(f.get("action", "")):
                continue
            if (f["key"][-1] == "ORB") != (source == "ORB"):
                continue
            hh, mm, ss = [int(x) for x in str(f.get("time")).split(":")[:3]]
            ft = now_t.replace(hour=hh, minute=mm, second=ss, microsecond=0)
            if f["key"][0] == now_t.strftime("%Y-%m-%d") and 0 <= (now_t - ft).total_seconds() < minutes * 60:
                return True
        except Exception:
            continue
    return False


def apply_atr_levels(levels, atr_und, sl_mult=3.0, min_pct=0.15, max_pct=0.35, t1_r=1.0, t2_r=1.8):
    """Replace the fixed -25% / +25% / +45% premium levels with ATR-based ones, in place.
    Stop distance in the underlying = sl_mult x ATR(5-min); premium risk = delta x that distance,
    kept between min_pct and max_pct of the premium. T1 = 1R, T2 = 1.8R. Returns True when applied."""
    try:
        e, d = float(levels["entry"]), float(levels.get("delta", 0.5))
        if not (atr_und == atr_und and atr_und > 0 and e > 0):
            return False
        risk = min(max(d * sl_mult * atr_und, min_pct * e), max_pct * e)
        levels.update(sl_premium=e - risk, sl_premium_tight=e - 0.8 * risk, t1=e + t1_r * risk, t2=e + t2_r * risk,
                      risk_pct=risk / e * 100, atr_based=True, atr=float(atr_und))
        return True
    except Exception:
        return False


# =====================================================================
# Paper-trade position sizing (lots, rupees) and rupee P&L
# =====================================================================
def position_size(entry, sl, lot_size, capital, risk_pct=2.0, fixed_lots=0):
    """How many lots, and what it costs / risks. Auto mode: lots = capital x risk% / (loss per lot at the stop),
    never more than the capital can buy. At least 1 lot is always shown (note says when 1 lot is above the capital)."""
    out = dict(lot_size="", lots="", qty="", cost="", capital=capital, risk_amt="", note="")
    try:
        entry, sl, lot_size = float(entry), float(sl), int(lot_size)
        if not (entry > 0 and lot_size > 0):
            return out
        per_lot_cost = entry * lot_size
        per_lot_risk = max(entry - sl, 0.0) * lot_size
        if fixed_lots and fixed_lots > 0:
            lots = int(fixed_lots)
        else:
            lots = int((capital * risk_pct / 100.0) // per_lot_risk) if per_lot_risk > 0 else 1
            lots = min(lots, int(capital // per_lot_cost))
        if lots < 1:
            lots = 1
            out["note"] = f"1 lot costs Rs {per_lot_cost:,.0f}, above capital or risk limit"
        qty = lots * lot_size
        out.update(lot_size=lot_size, lots=lots, qty=qty, cost=round(entry * qty), risk_amt=round((entry - sl) * qty))
    except Exception:
        pass
    return out


def pnl_amount(pnl_pct, cost):
    """Rupee result of a trade = percent result x amount invested (premium x quantity). '' when unknown."""
    try:
        return round(float(pnl_pct) / 100.0 * float(cost))
    except Exception:
        return ""


# Zerodha NSE equity-option charges (from zerodha.com/charge-list). Rates can change, edit here if they do.
CH_BROKERAGE, CH_STT_SELL, CH_EXCH, CH_SEBI, CH_STAMP_BUY, CH_IPFT, CH_GST = 20.0, 0.0015, 0.0003553, 1e-6, 0.00003, 1e-9, 0.18


def trade_costs(cost, pnl_pct, orders=2, slip_pct=1.0):
    """Charges and slippage for one round trip on the premium. cost = amount invested (premium x qty);
    sell turnover follows from the percent result. orders = 2 (buy + one sell) or 3 when half is booked at T1.
    Returns gross, charges, slippage and net (all Rs). '' when the inputs are unknown."""
    try:
        buy = float(cost)
        sell = buy * (1 + float(pnl_pct) / 100.0)
        gross = sell - buy
    except Exception:
        return dict(gross_amt="", charges="", slippage="", net_amt="")
    brok = CH_BROKERAGE * orders
    exch = CH_EXCH * (buy + sell)
    sebi = CH_SEBI * (buy + sell)
    ipft = CH_IPFT * (buy + sell)
    gst = CH_GST * (brok + exch + sebi + ipft)
    charges = brok + CH_STT_SELL * sell + exch + sebi + ipft + CH_STAMP_BUY * buy + gst
    slip = slip_pct / 100.0 * (buy + sell)
    return dict(gross_amt=round(gross), charges=round(charges), slippage=round(slip), net_amt=round(gross - charges - slip))


# =====================================================================
# Outcome tracker: what happened to an alert after it fired (option premium, 5-min candles)
# =====================================================================
def evaluate_outcome(candles, entry, sl, t1, t2, alert_time, now_t, day_over=False):
    """Replay option candles after the alert.
    Plan modelled (same as the exit rules): stop loss = SL; at T1 book half and move the stop to entry;
    the other half exits at T2, at entry (stop), or at the last close if the day ends.
    If SL and T1 are both touched inside one 5-min candle we assume SL first (conservative).
    Returns dict(outcome, pnl_pct, exit, resolved). pnl_pct is on the premium, before charges and slippage."""
    def res(outcome, pnl, ex, resolved):
        return dict(outcome=outcome, pnl_pct=round(pnl, 1), exit=round(float(ex), 1), resolved=resolved)
    entry, sl, t1, t2 = float(entry), float(sl), float(t1), float(t2)
    if candles is None or len(candles) == 0 or entry <= 0:
        return dict(outcome="NO DATA", pnl_pct=0.0, exit=0.0, resolved=False)
    start = (alert_time + _dt.timedelta(minutes=5)).replace(second=0, microsecond=0)
    start = start.replace(minute=start.minute - start.minute % 5)
    c = candles[candles.index >= start]
    if len(c) == 0:
        return dict(outcome="WAITING", pnl_pct=0.0, exit=entry, resolved=False)
    pct = lambda p: (p / entry - 1) * 100
    t1_hit = False
    for _, r in c.iterrows():
        hi, lo = float(r["high"]), float(r["low"])
        if not t1_hit:
            if lo <= sl:
                return res("SL", pct(sl), sl, True)
            if hi >= t1:
                t1_hit = True
                continue                      # a T2 / stop check on the same candle would be a guess
        else:
            if lo <= entry:
                return res("T1 then stop at entry", 0.5 * pct(t1), entry, True)
            if hi >= t2:
                return res("T2", 0.5 * pct(t1) + 0.5 * pct(t2), t2, True)
    last = float(c["close"].iloc[-1])
    if t1_hit:
        out = res("T1 hit, runner open", 0.5 * pct(t1) + 0.5 * pct(last), last, day_over)
        if day_over:
            out["outcome"] = "T1 then day end"
        return out
    out = res("OPEN", pct(last), last, day_over)
    if day_over:
        out["outcome"] = "Day end, no level hit"
    return out


def tracker_summary(rows):
    """rows: dicts with outcome / pnl_pct. Only resolved outcomes count."""
    done = [r for r in rows if str(r.get("outcome", "")) not in ("", "OPEN", "WAITING")
            and not str(r.get("outcome", "")).startswith("NO DATA")
            and "runner open" not in str(r.get("outcome", ""))]
    pn, amt, net = [], 0.0, 0.0
    for r in done:
        try:
            pn.append(float(r.get("pnl_pct")))
        except Exception:
            continue
        try:
            amt += float(r.get("pnl_amt"))
        except Exception:
            pass
        try:
            net += float(r.get("net_amt"))
        except Exception:
            pass
    if not pn:
        return dict(n=0, wins=0, win_rate=0.0, avg=0.0, total=0.0, best=0.0, worst=0.0, total_amt=0.0, total_net=0.0)
    w = sum(1 for p in pn if p > 0)
    return dict(n=len(pn), wins=w, win_rate=100.0 * w / len(pn), avg=sum(pn) / len(pn), total=sum(pn),
                best=max(pn), worst=min(pn), total_amt=amt, total_net=net)


TRACK_RETRY = ("", "OPEN", "WAITING", "NO DATA", "T1 hit, runner open")


def track_rows(rows, fetch, now_t, max_rows=6, slip_pct=1.0):
    """Work out outcomes for alerts that are not finished yet.
    fetch(row, alert_time) -> 5-min candles of that option (high/low/close) or None.
    Returns (updates, live): updates = [(sheet_row, outcome dict)] to write; live = {key: outcome dict} still running."""
    updates, live, n = [], {}, 0
    for r in rows:
        if str(r.get("outcome") or "") not in TRACK_RETRY:
            continue
        try:
            entry, sl, t1, t2 = (float(r[k]) for k in ("entry", "sl", "t1", "t2"))
            at = pd.Timestamp(f"{r['date']} {r['time']}").to_pydatetime()
        except Exception:
            continue
        if n >= max_rows:
            break
        n += 1
        old = at.date() < now_t.date()
        day_over = old or now_t.time() >= _dt.time(15, 30)
        try:
            c = fetch(r, at)
        except Exception:
            c = None
        if c is None or len(c) == 0:
            if old and r.get("_row"):
                updates.append((r["_row"], dict(outcome="NO DATA (expired)", pnl_pct="", exit="", pnl_amt="", charges="", slippage="", net_amt="")))
            continue
        ev = evaluate_outcome(c, entry, sl, t1, t2, at, now_t, day_over)
        ev["pnl_amt"] = pnl_amount(ev.get("pnl_pct"), r.get("cost"))
        oc = str(ev.get("outcome", ""))
        ev.update({k: v for k, v in trade_costs(r.get("cost"), ev.get("pnl_pct"), 3 if (oc.startswith("T1") or oc == "T2") else 2,
                                                slip_pct).items() if k != "gross_amt"})
        if ev["resolved"] and r.get("_row"):
            updates.append((r["_row"], ev))
        else:
            live[(str(r["date"]), str(r["time"]), str(r.get("index")), str(r.get("action")), str(r.get("strike")))] = ev
    return updates, live


# =====================================================================
# Entry context (logged and shown only - none of this blocks an alert yet)
# =====================================================================
HEAVY_W = {   # approximate index weights in % (rounded; refresh now and then). Only the relative size matters.
    "BANKNIFTY": {"HDFCBANK": 29, "ICICIBANK": 24, "SBIN": 9, "AXISBANK": 8, "KOTAKBANK": 8},
    "FINNIFTY": {"HDFCBANK": 30, "ICICIBANK": 24, "SBIN": 9, "AXISBANK": 8, "KOTAKBANK": 8, "BAJFINANCE": 6},
    "NIFTY": {"HDFCBANK": 13, "ICICIBANK": 9, "RELIANCE": 8, "INFY": 5, "TCS": 3.5, "ITC": 3.5, "LT": 3.5, "BHARTIARTL": 4},
    "SENSEX": {"HDFCBANK": 13, "ICICIBANK": 9, "RELIANCE": 8, "INFY": 5, "TCS": 3.5, "ITC": 3.5, "LT": 3.5, "BHARTIARTL": 4},
}


def heavy_bias(rows, weights, thr=0.15):
    """rows = [(symbol, day % change)]. Weighted average move of the heavyweights and a label.
    Returns dict(w_pct, label, up, down) or None."""
    num = den = 0.0
    up = dn = 0
    for s, p in rows or []:
        w = (weights or {}).get(s, 1.0)
        num += w * p
        den += w
        up += p > 0.05
        dn += p < -0.05
    if den <= 0:
        return None
    w_pct = num / den
    label = "Bullish" if w_pct >= thr else "Bearish" if w_pct <= -thr else "Mixed"
    return dict(w_pct=w_pct, label=label, up=int(up), down=int(dn))


def heavy_agrees(side, bias):
    """True / False / None (mixed or unknown) - do the heavyweights back the trade side?"""
    if not bias or bias["label"] == "Mixed":
        return None
    return (bias["label"] == "Bullish") == (side == "CE")


def entry_context(df, side, atr, n=12, look=24):
    """How stretched is the move when the alert fires?
    ext_vwap / ext_open: distance of the last completed close from VWAP / day open in ATR units
    (positive = already moved in the trade direction). brk_age: completed candles since the latest breakout
    close beyond the previous n candles (0 = the last candle). Missing values come back as None."""
    out = dict(ext_vwap=None, ext_open=None, brk_age=None)
    try:
        d = df.iloc[:-1]
        if len(d) < n + 2 or not atr or atr != atr or atr <= 0:
            return out
        sgn = 1.0 if side == "CE" else -1.0
        last = float(d["close"].iloc[-1])
        vw = float(vwap_daily(d).iloc[-1])
        day = d[d.index.date == d.index[-1].date()]
        if vw == vw:
            out["ext_vwap"] = round(sgn * (last - vw) / atr, 2)
        if len(day):
            out["ext_open"] = round(sgn * (last - float(day["open"].iloc[0])) / atr, 2)
        for age in range(0, min(look, len(d) - n - 1)):
            i = len(d) - 1 - age
            prev = d.iloc[i - n:i]
            c = float(d["close"].iloc[i])
            if (side == "CE" and c > float(prev["high"].max())) or (side == "PE" and c < float(prev["low"].min())):
                out["brk_age"] = age
                break
    except Exception:
        pass
    return out


def premium_runup(df, lookback=3):
    """Option premium run-up before the alert: last close vs the lowest low of the previous `lookback` candles (before the last), in %."""
    try:
        if df is None or len(df) < lookback + 1:
            return None
        low = float(df["low"].iloc[-lookback - 1:-1].min())
        return round((float(df["close"].iloc[-1]) / low - 1) * 100, 1) if low > 0 else None
    except Exception:
        return None


# =====================================================================
# Second strategy (paper): Opening Range Breakout + first retest of the range edge / VWAP
# =====================================================================
ORB_RANGE_MIN, ORB_RANGE_MAX = 0.12, 0.40    # opening range (9:15-9:30) as a share of the average daily range of the last days
ORB_END = _dt.time(11, 30)                   # last entry candle must close by this time
ORB_ADX_MIN = 20
ORB_MAX_PER_DAY = 2


def orb_signal(df, atr5):
    """5-min futures candles -> dict(ok, reason, ...). Looks only at COMPLETED candles.
    Rules (CE; PE is the mirror): opening range 9:15-9:30 sized 0.12-0.40 of the average daily range; a 5-min close above the range
    high (breakout); later a candle dips back to the range high (retest) but closes above it; the NEXT candle closes up
    (close > open and > the retest close) while above VWAP, EMA20 rising and ADX >= 20. Signal only if that entry candle
    is the last (or the one before the last) completed candle and closed by 11:30. Stop = retest extreme."""
    res = dict(ok=False, reason="")
    try:
        d = df.iloc[:-1]
        if len(d) < 40 or not (atr5 == atr5 and atr5 > 0):
            res["reason"] = "not enough candles yet"
            return res
        today = d.index[-1].date()
        day = d[d.index.date == today]
        orc = day[day.index.time < _dt.time(9, 30)]
        post = day[day.index.time >= _dt.time(9, 30)]
        if len(orc) < 3:
            res["reason"] = "opening range (9:15-9:30) not complete"
            return res
        if post.empty:
            res["reason"] = "waiting for the 9:30 candle"
            return res
        orh, orl = float(orc["high"].max()), float(orc["low"].min())
        prev = d[d.index.date < today]
        rng = (prev["high"].groupby(prev.index.date).max() - prev["low"].groupby(prev.index.date).min()).tail(5)
        if len(rng) < 2:
            res["reason"] = "not enough previous days for the range test"
            return res
        ratio = (orh - orl) / float(rng.mean())
        res.update(orh=orh, orl=orl, range_ratio=round(ratio, 2))
        if not (ORB_RANGE_MIN <= ratio <= ORB_RANGE_MAX):
            res["reason"] = f"opening range is {ratio:.2f} of the daily range (needs {ORB_RANGE_MIN}-{ORB_RANGE_MAX}): skip today"
            return res
        vw = vwap_daily(d)
        ema20 = d["close"].ewm(span=20, adjust=False).mean()
        adx_s = adx(d)[0]
        tol = 0.15 * atr5
        pos = {ts: i for i, ts in enumerate(d.index)}
        n = len(post)
        for side in ("CE", "PE"):
            sg = 1 if side == "CE" else -1
            edge = orh if side == "CE" else orl
            b = next((i for i in range(n) if sg * (float(post["close"].iloc[i]) - edge) > 0), None)
            if b is None:
                continue
            for k in range(b + 1, n - 1):
                r, e = post.iloc[k], post.iloc[k + 1]
                touched = (float(r["low"]) <= edge + tol) if side == "CE" else (float(r["high"]) >= edge - tol)
                held = sg * (float(r["close"]) - edge) > 0
                if not (touched and held):
                    continue
                up_ok = sg * (float(e["close"]) - float(e["open"])) > 0 and sg * (float(e["close"]) - float(r["close"])) > 0 \
                    and sg * (float(e["close"]) - edge) > 0
                if not up_ok:
                    continue
                ei = pos[post.index[k + 1]]
                if sg * (float(e["close"]) - float(vw.iloc[ei])) <= 0:
                    continue
                if sg * (float(ema20.iloc[ei]) - float(ema20.iloc[ei - 1])) <= 0:
                    continue
                if float(adx_s.iloc[ei]) < ORB_ADX_MIN:
                    continue
                if k + 1 < n - 2:
                    continue                                    # entry candle is older than the previous candle: stale
                close_t = (post.index[k + 1] + _dt.timedelta(minutes=5)).time()
                if close_t > ORB_END:
                    res["reason"] = "setup formed after 11:30: skipped"
                    continue
                stop = float(r["low"]) if side == "CE" else float(r["high"])
                entry_idx = float(e["close"])
                risk = max(abs(entry_idx - stop), 0.5 * atr5)
                if risk > 4 * atr5:
                    res["reason"] = "retest stop is too far (over 4 ATR): skipped"
                    continue
                res.update(ok=True, reason="", side=side, entry_idx=entry_idx, sl_idx=stop, risk_idx=risk,
                           brk_time=post.index[b].strftime("%H:%M"), retest_time=post.index[k].strftime("%H:%M"),
                           adx=round(float(adx_s.iloc[ei]), 1))
                return res
        if not res["reason"]:
            res["reason"] = "no breakout + retest + confirmation yet"
    except Exception as ex:
        res["reason"] = f"ORB check error: {type(ex).__name__}"
    return res


def orb_option_levels(entry, delta, risk_idx):
    """Premium stop / targets from the index risk: risk_prem = delta x index risk, kept between 10% and 35% of the premium.
    T1 = 1R, T2 = 2R."""
    d = float(delta) if delta == delta and delta else 0.5
    risk = min(max(abs(d) * risk_idx, 0.10 * entry), 0.35 * entry)
    return dict(entry=entry, sl=entry - risk, t1=entry + risk, t2=entry + 2 * risk, risk_pct=risk / entry * 100)
