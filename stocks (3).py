"""Stock momentum / direction scanner (futures 5-min candles with volume and OI). Pure functions, no Streamlit."""
import numpy as np
import pandas as pd
import engine as E

# display name -> NSE symbol (futures + options trade under this name)
STOCKS = {
    "BSE": "BSE", "CDSL": "CDSL", "MCX": "MCX", "Reliance": "RELIANCE", "HDFC Bank": "HDFCBANK",
    "Axis Bank": "AXISBANK", "ICICI Bank": "ICICIBANK", "Kotak Bank": "KOTAKBANK", "SBI": "SBIN",
    "Bajaj Finance": "BAJFINANCE",
    "PB Fintech (Policybazaar)": "POLICYBZR", "TCS": "TCS",
    "Infosys": "INFY", "L&T": "LT", "ITC": "ITC", "Bharti Airtel": "BHARTIARTL",
    "Adani Enterprises": "ADANIENT", "Adani Ports": "ADANIPORTS",
    "Maruti Suzuki": "MARUTI", "M&M": "M&M", "Sun Pharma": "SUNPHARMA", "Tata Steel": "TATASTEEL",
    "Wipro": "WIPRO", "HCL Tech": "HCLTECH", "NTPC": "NTPC", "ONGC": "ONGC", "JSW Steel": "JSWSTEEL",
    "Hindalco": "HINDALCO", "Titan": "TITAN", "Bajaj Finserv": "BAJAJFINSV", "HAL": "HAL", "BEL": "BEL",
    "Eternal (Zomato)": "ETERNAL", "Trent": "TRENT", "Jio Financial": "JIOFIN", "IndusInd Bank": "INDUSINDBK",
    "HUL": "HINDUNILVR", "Tech Mahindra": "TECHM",
}

OI_NAMES = {"LB": "Long build-up", "SB": "Short build-up", "SC": "Short covering", "LU": "Long unwinding", "-": "No clear change"}


def oi_state(price_chg_pct, oi_chg_pct, thr_p=0.15, thr_o=1.0):
    if abs(price_chg_pct) < thr_p or abs(oi_chg_pct) < thr_o:
        return "-"
    if price_chg_pct > 0:
        return "LB" if oi_chg_pct > 0 else "SC"
    return "SB" if oi_chg_pct > 0 else "LU"


def analyze_stock(df, lookback=6, min_today=9, prev_close=None, spot_last=None):
    """df: 5-min futures candles (open high low close volume oi) covering a few sessions.
    Direction is 'clear' only when five core checks agree: one-sided trend, VWAP side, candle-close breakout,
    volume surge and futures build-up (long build-up for bulls, short build-up for bears)."""
    out = dict(direction=None, clear=False, checks=[], score=0, state="-", vol_ratio=float("nan"),
               day_pct=float("nan"), last=float("nan"), vwap=float("nan"), why="not enough candles", ind={})
    if df is None or len(df) < 70:
        return out
    day = df.index[-1].date()
    today = df[df.index.date == day]
    if len(today) < min_today:
        out["why"] = "market just opened - waiting for more candles"
        return out
    prev = df[df.index.date < day]
    last = float(df["close"].iloc[-1])
    if prev_close and prev_close > 0:                      # spot previous close is the true reference
        day_pct = ((spot_last or last) / float(prev_close) - 1) * 100
    else:
        day_pct = (last / float(prev["close"].iloc[-1]) - 1) * 100 if len(prev) else float("nan")
    tr = E.trend_state(df)
    vw = float(E.vwap_daily(df).iloc[-1])
    bo = E.breakout_info(df, closed=True)
    done = df.iloc[:-1]                                  # last COMPLETED candle for volume / OI
    v_now = float(done["volume"].iloc[-1])
    v_avg = float(done["volume"].iloc[-21:-1].mean())
    vol_ratio = v_now / v_avg if v_avg > 0 else float("nan")
    st_ = "-"
    if "oi" in done.columns and len(done) > lookback and done["oi"].iloc[-1 - lookback] > 0:
        p0, p1 = float(done["close"].iloc[-1 - lookback]), float(done["close"].iloc[-1])
        o0, o1 = float(done["oi"].iloc[-1 - lookback]), float(done["oi"].iloc[-1])
        st_ = oi_state((p1 / p0 - 1) * 100, (o1 / o0 - 1) * 100)
    d = "BULL" if tr["label"] == "UP" else "BEAR" if tr["label"] == "DOWN" else None
    bull = d == "BULL"
    chk = []

    def add(name, ok, detail, core=True):
        chk.append(dict(check=name, ok=bool(ok), detail=detail, core=core))
    add("5-min trend one-sided", d is not None, f"trend {tr['label']}")
    add("Price on the right side of VWAP", d is not None and ((last > vw) if bull else (last < vw)), f"price {last:.1f} vs VWAP {vw:.1f}")
    brk = bo is not None and ((bo["up"]) if bull else (bo["down"])) and d is not None
    add("Candle-close breakout", brk, "no data" if bo is None else
        f"close {bo['last']:.1f} vs previous {'high' if bull else 'low'} {(bo['prev_high'] if bull else bo['prev_low']):.1f}")
    add("Volume surge (>1.5x recent average)", vol_ratio == vol_ratio and vol_ratio > 1.5, f"{vol_ratio:.1f}x")
    want = "LB" if bull else "SB"
    add("Futures build-up agrees", d is not None and st_ == want,
        f"{OI_NAMES.get(st_)} over last {lookback * 5} min (need {OI_NAMES[want]})" if d else OI_NAMES.get(st_))
    mv = day_pct == day_pct and d is not None and ((day_pct >= 0.7) if bull else (day_pct <= -0.7))
    add("Day move of 0.7% or more in this direction", mv, f"{day_pct:+.2f}% vs yesterday close", core=False)
    ext = abs(last / vw - 1) * 100 if vw == vw else float("nan")
    add("Not stretched (within 2% of VWAP)", ext == ext and ext <= 2.0, f"{ext:.2f}% from VWAP", core=False)
    core_ok = all(c["ok"] for c in chk if c["core"])
    extras = sum(1 for c in chk if not c["core"] and c["ok"])
    out.update(direction=d, checks=chk, state=st_, vol_ratio=vol_ratio, day_pct=day_pct, last=last, vwap=vw,
               score=sum(1 for c in chk if c["ok"]), clear=bool(core_ok and extras >= 1), trend=tr, ind=E.indicators(df),
               why="" if core_ok and extras >= 1 else "; ".join(c["check"] for c in chk if not c["ok"])[:160])
    return out


def pick_step(strikes, spot):
    """Strike gap near the money from the instrument master (differs per stock)."""
    s = np.array(sorted(set(float(x) for x in strikes)))
    if len(s) < 3:
        return None
    near = s[np.abs(s - spot) <= spot * 0.06]
    near = near if len(near) >= 3 else s
    gaps = np.diff(near)
    gaps = gaps[gaps > 0]
    return float(np.min(gaps)) if len(gaps) else None


MIN_OPT_VALUE = 2_500_000      # rupees traded today in the chosen option (price x quantity); thinner than this = skip
MAX_SPREAD = 3.0               # percent bid-ask spread


def option_liquidity(chain, strike, side):
    """Is the chosen option liquid enough to trade? Returns (ok, text)."""
    s = side.lower()
    row = chain.loc[strike]
    val = float(row.get(f"{s}_ltp", 0) or 0) * float(row.get(f"{s}_vol", 0) or 0)
    bid, ask = float(row.get(f"{s}_bid", 0) or 0), float(row.get(f"{s}_ask", 0) or 0)
    spread = (ask - bid) / ((ask + bid) / 2) * 100 if bid > 0 and ask > 0 else float("nan")
    ok = val >= MIN_OPT_VALUE and spread == spread and spread <= MAX_SPREAD
    return ok, f"traded today {val / 1e5:.0f} lakh, bid-ask spread {spread:.1f}%" if spread == spread else \
        f"traded today {val / 1e5:.0f} lakh, no bid/ask"


def premarket_row(last, prev_close, prev_high, prev_low, gap_thr=0.5):
    """Gap vs yesterday's close, where yesterday closed inside its own range, and a simple bias.
    Bullish: gap up AND yesterday closed in the upper 60% of its range. Bearish: the mirror. Otherwise Mixed/Flat."""
    nan = float("nan")
    gap = (last / prev_close - 1) * 100 if prev_close and prev_close > 0 else nan
    pos = (prev_close - prev_low) / (prev_high - prev_low) * 100 if prev_high and prev_high > prev_low else nan
    if gap != gap:
        return dict(gap=nan, pos=pos, bias="-", note="no price")
    if gap >= gap_thr and pos == pos and pos >= 60:
        bias, note = "Bullish", "gap up + strong previous close"
    elif gap <= -gap_thr and pos == pos and pos <= 40:
        bias, note = "Bearish", "gap down + weak previous close"
    elif abs(gap) >= gap_thr:
        bias, note = "Mixed", "gap against yesterday's close position - needs confirmation"
    else:
        bias, note = "Flat", "no meaningful gap"
    return dict(gap=gap, pos=pos, bias=bias, note=note)
