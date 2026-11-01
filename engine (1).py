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
    chk("Entry window (9:30-15:00 IST)", in_window)
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
        res["reason"] = "Market closed or outside 9:30-15:00 IST window"
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


def greeks_scan(chain, atm, step, spot, tr, vwap_idx, atm_iv, iv_ref, vol_ratio, oi_ok, expiry_day,
                itm=False, live=True, closed=False, mode="10-point (final scanner)"):
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
    cands = [r for r in res.values() if r["entry_ok"]]
    return res, (max(cands, key=lambda r: r["score"]) if live and cands else None)


def greeks_exit_rules(r):
    bull, L = r["side"] == "CE", r["levels"]
    lvl = f" ({L['index_sl']:.0f})" if L.get("index_sl") else ""
    idx_move = lambda pct: L["entry"] * pct / L["delta"]
    return [
        f"Exit if a 5-min candle closes {'below' if bull else 'above'} VWAP, or the breakout level{lvl} fails.",
        f"Emergency premium stop: {L['sl_premium']:.1f} (-25%); tighter option {L['sl_premium_tight']:.1f} (-20%). Whichever comes first.",
        f"T1 {L['t1']:.1f} (+25%, about {idx_move(0.25):.0f} index points): book half, move stop to entry {L['entry']:.1f}.",
        f"Runner: T2 {L['t2']:.1f} (+45%) or trail with VWAP and the previous candle {'low' if bull else 'high'}.",
    ]
