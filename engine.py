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


def add_greeks(chain, F, T):
    """chain indexed by strike with ce_ltp / pe_ltp -> adds iv, delta, vega columns."""
    for side, cp in (("ce", "CE"), ("pe", "PE")):
        ivs, dels, vegs = [], [], []
        for k, row in chain.iterrows():
            iv = implied_vol(row.get(f"{side}_ltp"), F, k, T, cp)
            d, v = greeks(F, k, T, iv, cp)
            ivs.append(iv * 100 if iv == iv else float("nan"))
            dels.append(d)
            vegs.append(v)
        chain[f"{side}_iv"] = ivs
        chain[f"{side}_delta"] = dels
        chain[f"{side}_vega"] = vegs
    return chain


# ---------- OI analytics ----------
def oi_summary(chain):
    ce_oi = chain["ce_oi"].fillna(0)
    pe_oi = chain["pe_oi"].fillna(0)
    pcr = float(pe_oi.sum() / ce_oi.sum()) if ce_oi.sum() > 0 else float("nan")
    resistance = float(ce_oi.idxmax()) if ce_oi.max() > 0 else float("nan")
    support = float(pe_oi.idxmax()) if pe_oi.max() > 0 else float("nan")
    # max pain inside the loaded window
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
