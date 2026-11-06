"""Market view: global / commodity / currency / bond mood, F&O breadth (advance-decline) and FII/DII positions.
Pure functions (no Streamlit, no network) so they can be tested offline."""
import re
import numpy as np

# group, name, Yahoo symbol, weight (+ = rising is good for Indian equities, - = rising is bad), threshold, unit
MOOD = [
    ("Equities", "S&P 500", "^GSPC", +1.5, 0.3, "%"), ("Equities", "Nasdaq", "^IXIC", +1.0, 0.4, "%"),
    ("Equities", "Dow Jones", "^DJI", +0.5, 0.3, "%"), ("Equities", "Nikkei", "^N225", +1.0, 0.3, "%"),
    ("Equities", "Hang Seng", "^HSI", +1.0, 0.4, "%"), ("Equities", "Shanghai", "000001.SS", +0.5, 0.3, "%"),
    ("Equities", "KOSPI", "^KS11", +0.5, 0.4, "%"), ("Equities", "FTSE", "^FTSE", +0.5, 0.3, "%"),
    ("Equities", "DAX", "^GDAXI", +0.5, 0.3, "%"), ("Equities", "CAC 40", "^FCHI", +0.5, 0.3, "%"),
    ("Commodities", "Crude (WTI)", "CL=F", -1.0, 1.0, "%"), ("Commodities", "Brent", "BZ=F", -1.0, 1.0, "%"),
    ("Commodities", "Gold", "GC=F", -0.5, 0.7, "%"), ("Commodities", "Silver", "SI=F", 0.0, 1.0, "%"),
    ("Commodities", "Copper", "HG=F", +0.5, 1.0, "%"), ("Commodities", "Natural gas", "NG=F", 0.0, 2.0, "%"),
    ("Currency", "USD/INR", "INR=X", -1.5, 0.2, "%"), ("Currency", "Dollar index", "DX-Y.NYB", -1.0, 0.3, "%"),
    ("Currency", "EUR/USD", "EURUSD=X", 0.0, 0.3, "%"), ("Currency", "USD/JPY", "JPY=X", 0.0, 0.3, "%"),
    ("Bonds", "US 10Y yield", "^TNX", -1.0, 4.0, "bp"), ("Bonds", "US 5Y yield", "^FVX", -0.5, 4.0, "bp"),
    ("Bonds", "US 3M yield", "^IRX", 0.0, 4.0, "bp"),
    ("Volatility", "US VIX", "^VIX", -1.0, 3.0, "%"), ("Volatility", "India VIX", "^INDIAVIX", -1.5, 3.0, "%"),
]
GROUPS = ["Equities", "Commodities", "Currency", "Bonds", "Volatility"]


def mood_from_data(data, gift_pct=0.0):
    """data: {name: (last, change)} with change in % (yields in basis points).
    Returns dict(score in [-1, 1], label, groups {group: score}, rows [...]). Each row has an Impact for India:
    Positive / Negative / Neutral (small moves count as Neutral)."""
    rows, gp, gw = [], {g: 0.0 for g in GROUPS}, {g: 0.0 for g in GROUPS}
    pts = wts = 0.0
    for grp, name, _sym, w, thr, unit in MOOD:
        last, chg = data.get(name, (np.nan, np.nan))
        ok = chg == chg and chg is not None
        impact = "n/a"
        if ok:
            if abs(chg) < thr:
                impact = "Neutral"
            else:
                s = (1 if chg > 0 else -1) * (1 if w > 0 else -1 if w < 0 else 0)
                impact = "Positive" if s > 0 else "Negative" if s < 0 else "Neutral"
                if w != 0:
                    pts += abs(w) * s
                    gp[grp] += abs(w) * s
            if w != 0:
                wts += abs(w)
                gw[grp] += abs(w)
        rows.append(dict(Group=grp, Market=name, Last=last, Change=chg, Unit=unit, Impact=impact))
    if gift_pct:
        wts += 2.0
        if abs(gift_pct) >= 0.3:
            pts += 2.0 * (1 if gift_pct > 0 else -1)
    score = pts / wts if wts else 0.0
    groups = {g: (gp[g] / gw[g] if gw[g] else 0.0) for g in GROUPS}
    return dict(score=score, label=mood_label(score), groups=groups, rows=rows)


def mood_label(score):
    return "Positive" if score >= 0.30 else "Negative" if score <= -0.30 else "Neutral / mixed"


def breadth(quotes, strong=2.0):
    """quotes: {symbol: {'last_price': x, 'ohlc': {'close': prev_close}}} -> advance / decline counts."""
    chg = {}
    for sym, q in quotes.items():
        try:
            last, prev = float(q["last_price"]), float(q["ohlc"]["close"])
            if last > 0 and prev > 0:
                chg[sym] = (last / prev - 1) * 100
        except Exception:
            continue
    if not chg:
        return dict(total=0)
    vals = np.array(list(chg.values()))
    adv, dec = int((vals > 0.0001).sum()), int((vals < -0.0001).sum())
    total = len(vals)
    pct = 100.0 * adv / total
    srt = sorted(chg.items(), key=lambda kv: kv[1])
    return dict(total=total, adv=adv, dec=dec, unch=total - adv - dec, adv_pct=pct, dec_pct=100.0 * dec / total,
                ratio=(adv / dec) if dec else (float("inf") if adv else float("nan")), avg=float(vals.mean()),
                strong_up=int((vals >= strong).sum()), strong_dn=int((vals <= -strong).sum()),
                label=("Market not open yet" if (total - adv - dec) / total >= 0.95 else
                       "Broad strength" if pct >= 60 else "Broad weakness" if pct <= 40 else "Mixed / stock-specific"),
                gainers=srt[::-1][:5], losers=srt[:5])


# ---------------- NSE participant-wise open interest (FII / DII / Client / Pro) ----------------
def _norm(c):
    return re.sub(r"\s+", " ", str(c).replace('"', "").strip().lower())


def _int(x):
    try:
        return int(float(str(x).replace('"', "").replace(",", "").strip()))
    except Exception:
        return 0


def parse_participant_oi(text):
    """Parse NSE's fao_participant_oi CSV. Returns {'FII': {'future index long': n, ...}, 'DII': ..., 'CLIENT': ..., 'PRO': ...}
    (empty dict when the file is not in the expected shape)."""
    lines = [ln for ln in str(text).splitlines() if ln.strip()]
    h = next((i for i, ln in enumerate(lines) if _norm(ln.split(",")[0]) == "client type"), None)
    if h is None:
        return {}
    head = [_norm(c) for c in lines[h].split(",")]
    out = {}
    for ln in lines[h + 1:]:
        cells = ln.split(",")
        who = _norm(cells[0]).upper()
        if who in ("CLIENT", "DII", "FII", "PRO", "TOTAL"):
            out[who] = {head[i]: _int(cells[i]) for i in range(1, min(len(head), len(cells)))}
    return out if "FII" in out else {}


def _fut(d):
    lg, sh = d.get("future index long", 0), d.get("future index short", 0)
    return lg, sh


def _opt_bull(d):
    """Net bullish exposure through index options = (call long - call short) - (put long - put short)."""
    return ((d.get("option index call long", 0) - d.get("option index call short", 0))
            - (d.get("option index put long", 0) - d.get("option index put short", 0)))


def participant_table(today, prev=None):
    """One row per participant: index-futures long / short, net, short %, daily change in each, and a plain reading."""
    rows = []
    for who in ("FII", "DII", "CLIENT", "PRO"):
        d = today.get(who)
        if not d:
            continue
        lg, sh = _fut(d)
        tot = lg + sh
        row = dict(Participant=who, Long=lg, Short=sh, Net=lg - sh, LongPct=100.0 * lg / tot if tot else np.nan,
                   ShortPct=100.0 * sh / tot if tot else np.nan, OptBull=_opt_bull(d))
        if prev and prev.get(who):
            pl, ps = _fut(prev[who])
            row.update(dLong=lg - pl, dShort=sh - ps, dNet=(lg - sh) - (pl - ps), dOptBull=_opt_bull(d) - _opt_bull(prev[who]))
        else:
            row.update(dLong=np.nan, dShort=np.nan, dNet=np.nan, dOptBull=np.nan)
        row["Reading"] = reading(row)
        rows.append(row)
    return rows


def reading(r):
    side = "NET SHORT" if r["Net"] < 0 else "NET LONG"
    txt = f"{side} {abs(r['Net']):,} (short {r['ShortPct']:.0f}%)"
    dl, ds = r.get("dLong"), r.get("dShort")
    if dl == dl and ds == ds:
        if ds > 0 and dl <= 0:
            txt += ": adding shorts (bearish)"
        elif ds < 0 and dl >= 0:
            txt += ": covering shorts, adding longs (bullish)"
        elif ds < 0:
            txt += ": covering shorts (supportive)"
        elif dl > 0 and ds <= 0:
            txt += ": adding longs (bullish)"
        elif dl < 0 and ds >= 0:
            txt += ": cutting longs (bearish)"
        elif dl > 0 and ds > 0:
            txt += ": adding both sides"
    return txt


def cash_flows(payload):
    """NSE FII/DII cash-market list -> {'FII': {'buy', 'sell', 'net', 'date'}, 'DII': ...} in Rs crore."""
    out = {}
    try:
        for it in payload:
            cat = str(it.get("category", "")).upper()
            key = "FII" if "FII" in cat or "FPI" in cat else "DII" if "DII" in cat else None
            if key:
                out[key] = dict(buy=float(str(it.get("buyValue")).replace(",", "")), sell=float(str(it.get("sellValue")).replace(",", "")),
                                net=float(str(it.get("netValue")).replace(",", "")), date=str(it.get("date", "")))
    except Exception:
        return {}
    return out


def parse_gift(payload):
    """GIFT Nifty from NSE's marketStatus JSON -> dict(last, pct, time) or None.
    The payload carries a 'giftnifty' block (keys such as LASTPRICE / PERCHANGE / TIMESTMP, case may vary)."""
    try:
        blk = None
        if isinstance(payload, dict):
            for k, v in payload.items():
                if "gift" in str(k).lower():
                    blk = v
                    break
        if isinstance(blk, list):
            blk = blk[0] if blk else None
        if not isinstance(blk, dict):
            return None
        low = {str(k).lower(): v for k, v in blk.items()}
        num = lambda x: float(str(x).replace(",", "").replace("%", ""))
        pct = next((low[k] for k in ("perchange", "pchange", "percentchange", "changepercent") if k in low), None)
        last = next((low[k] for k in ("lastprice", "last", "ltp", "lastpricerupee") if k in low), None)
        if pct is None and last is not None and "daychange" in low:
            ch, lp = num(low["daychange"]), num(last)
            pct = ch / (lp - ch) * 100 if lp - ch else None
        if pct is None:
            return None
        return dict(last=num(last) if last is not None else float("nan"), pct=num(pct),
                    time=str(low.get("timestmp") or low.get("timestamp") or ""))
    except Exception:
        return None
