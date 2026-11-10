"""Nifty 50 / Bank Nifty stock trackers: moving averages, 52-week position, RSI/ADX/ROC, volume, breakouts.
Pure pandas (testable) plus helpers that write the result into Google Sheet tabs. Read-only data, no orders."""
import datetime as _dt
import numpy as np
import pandas as pd
import engine as E

NAN = float("nan")
MA_SPEC = [("5 DMA", 5, "sma"), ("10 DMA", 10, "sma"), ("20 DMA", 20, "sma"), ("50 DMA", 50, "sma"),
           ("100 DMA", 100, "sma"), ("200 DMA", 200, "sma")]      # DMA = simple moving average of closes; 200 DMA = 200 SMA (one line)
NEAR_HIGH_PCT, FAR_HIGH_PCT = 5.0, 20.0      # within 5% of the 52-week high = near; 20%+ below it = far
VOL_BREAKOUT_X = 1.5


def yahoo_symbol(sym):
    return sym + ".NS"


def _f(x, nd=2):
    try:
        x = float(x)
        return round(x, nd) if x == x and np.isfinite(x) else ""
    except Exception:
        return ""


def compute_row(sym, df):
    """df: daily candles (open/high/low/close/volume, date index, oldest first). Returns a dict, or None if too short."""
    if df is None:
        return None
    df = df.dropna(subset=["close"])
    if len(df) < 30:
        return None
    c = df["close"].astype(float)
    cmp_, prev = float(c.iloc[-1]), float(c.iloc[-2])
    row = {"Stock": sym, "CMP": cmp_, "Day %": (cmp_ / prev - 1) * 100}
    n_above = n_avail = 0
    for label, n, kind in MA_SPEC:
        if len(c) >= n:
            v = float(c.rolling(n).mean().iloc[-1]) if kind == "sma" else float(c.ewm(span=n, adjust=False).mean().iloc[-1])
            st_ = "Above" if cmp_ > v else "Below"
            n_avail += 1
            n_above += st_ == "Above"
        else:
            v, st_ = NAN, "n/a"
        row[label] = v
        row[label + " status"] = st_
    row["MAs above (of 6)"] = n_above
    row["All 6 above"] = "Yes" if (n_avail == 6 and n_above == 6) else "No"
    row["All 6 below"] = "Yes" if (n_avail == 6 and n_above == 0) else "No"
    hi, lo = float(df["high"].tail(252).max()), float(df["low"].tail(252).min())
    row["52W High"], row["52W Low"] = hi, lo
    row["% from 52W High"] = (cmp_ / hi - 1) * 100
    row["% above 52W Low"] = (cmp_ / lo - 1) * 100
    pf = row["% from 52W High"]
    row["52W zone"] = "Near high" if pf >= -NEAR_HIGH_PCT else "Far from high" if pf <= -FAR_HIGH_PCT else "Middle"
    try:
        row["RSI 14"] = float(E.rsi(c, 14).iloc[-1])
        row["ADX 14"] = float(E.adx(df.astype(float), 14)[0].iloc[-1])
    except Exception:
        row["RSI 14"], row["ADX 14"] = NAN, NAN
    row["ROC 12"] = (cmp_ / float(c.iloc[-13]) - 1) * 100 if len(c) > 13 else NAN
    row["ROC 21"] = (cmp_ / float(c.iloc[-22]) - 1) * 100 if len(c) > 22 else NAN
    vol = float(df["volume"].iloc[-1]) if "volume" in df else NAN
    avg20 = float(df["volume"].iloc[-21:-1].mean()) if "volume" in df and len(df) > 21 else NAN
    row["Volume"], row["Avg vol 20D"] = vol, avg20
    row["Vol x avg"] = vol / avg20 if avg20 and avg20 == avg20 and avg20 > 0 else NAN
    h20, l20 = float(df["high"].iloc[-21:-1].max()), float(df["low"].iloc[-21:-1].min())
    h50 = float(df["high"].iloc[-51:-1].max()) if len(df) > 51 else NAN
    row["20D High"], row["20D Low"], row["50D High"] = h20, l20, h50
    vx = row["Vol x avg"]
    if cmp_ > h20:
        brk = "Breakout 20D" + (" + volume" if vx == vx and vx >= VOL_BREAKOUT_X else "")
    elif cmp_ >= h20 * 0.99:
        brk = "Near 20D high"
    elif cmp_ < l20:
        brk = "Breakdown 20D"
    else:
        brk = "-"
    row["Breakout status"] = brk
    return row


COLS = ["Stock", "CMP", "Day %", "5 DMA", "5 DMA status", "10 DMA", "10 DMA status", "20 DMA", "20 DMA status", "50 DMA", "50 DMA status",
        "100 DMA", "100 DMA status", "200 DMA", "200 DMA status", "MAs above (of 6)", "All 6 above",
        "52W High", "52W Low", "% from 52W High", "% above 52W Low", "52W zone", "RSI 14", "ADX 14", "ROC 12", "ROC 21",
        "Volume", "Avg vol 20D", "Vol x avg", "20D High", "20D Low", "50D High", "Breakout status"]


def summary(rows):
    """[(label, count, percent)] breadth numbers for the stocks in `rows`."""
    n = len(rows)
    out = []

    def add(label, k):
        out.append((label, int(k), round(100 * k / n, 1) if n else 0.0))
    for label, _, _ in MA_SPEC:
        add(f"Above {label}", sum(r[label + " status"] == "Above" for r in rows))
    add("Above ALL 6 DMAs", sum(r["All 6 above"] == "Yes" for r in rows))
    add("Below ALL 6 DMAs", sum(r["All 6 below"] == "Yes" for r in rows))
    add(f"Near 52W high (within {NEAR_HIGH_PCT:.0f}%)", sum(r["52W zone"] == "Near high" for r in rows))
    add(f"Far from 52W high ({FAR_HIGH_PCT:.0f}%+ below)", sum(r["52W zone"] == "Far from high" for r in rows))
    add("20-day breakout", sum(str(r["Breakout status"]).startswith("Breakout") for r in rows))
    add("20-day breakout with volume", sum("volume" in str(r["Breakout status"]) for r in rows))
    add("Near 20-day high", sum(r["Breakout status"] == "Near 20D high" for r in rows))
    add("20-day breakdown", sum(r["Breakout status"] == "Breakdown 20D" for r in rows))
    add("RSI above 70", sum(r["RSI 14"] == r["RSI 14"] and r["RSI 14"] > 70 for r in rows))
    add("RSI below 30", sum(r["RSI 14"] == r["RSI 14"] and r["RSI 14"] < 30 for r in rows))
    add("ADX above 25 (trending)", sum(r["ADX 14"] == r["ADX 14"] and r["ADX 14"] > 25 for r in rows))
    add("Up today", sum(r["Day %"] > 0 for r in rows))
    add("Down today", sum(r["Day %"] < 0 for r in rows))
    return out


def to_values(rows, title, updated, source_note):
    """Two 2-D lists ready for a Sheet: (summary_values, stocks_values)."""
    n = len(rows)
    avg_from_high = float(np.nanmean([r["% from 52W High"] for r in rows])) if rows else NAN
    summ = [[title], [f"Updated {updated} | {n} stocks | {source_note}"], [],
            ["Measure", "Stocks", "% of stocks"]]
    summ += [[lbl, k, p] for lbl, k, p in summary(rows)]
    summ += [["Average distance from 52W high (%)", _f(avg_from_high), ""]]
    rows_sorted = sorted(rows, key=lambda r: (-r["MAs above (of 6)"], r["% from 52W High"] * -1))
    st_vals = [COLS] + [[(r[c] if isinstance(r[c], str) else _f(r[c], 0 if c in ("Volume", "Avg vol 20D", "MAs above (of 6)") else 2))
                         for c in COLS] for r in rows_sorted]
    return summ, st_vals


def build_rows(symbols, hist):
    """hist: {symbol: daily df}. Returns (rows, missing_symbols)."""
    rows, missing = [], []
    for s in symbols:
        r = compute_row(s, hist.get(s))
        (rows if r else missing).append(r or s)
    return rows, missing


def fetch_history(symbols, days=420):
    """Daily candles from Yahoo Finance (NSE .NS symbols) -> {symbol: df}."""
    import yfinance as yf
    tickers = [yahoo_symbol(s) for s in symbols]
    raw = yf.download(tickers, period="2y", interval="1d", group_by="ticker", progress=False, auto_adjust=False, threads=True)
    out = {}
    for s, tk in zip(symbols, tickers):
        try:
            d = raw[tk] if len(tickers) > 1 else raw
            d = d.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
            d.index = pd.to_datetime(d.index).tz_localize(None)
            if len(d):
                out[s] = d.tail(days)
        except Exception:
            continue
    return out


def write_book(book, prefix, summ, st_vals):
    """Write the two tabs into a gspread Spreadsheet. prefix '' = own sheet (tabs Summary/Stocks), otherwise tab-name prefix."""
    names = (prefix + "Summary", prefix + "Stocks")
    for nm, vals in zip(names, (summ, st_vals)):
        try:
            ws = book.worksheet(nm)
        except Exception:
            ws = book.add_worksheet(nm, rows=max(len(vals) + 5, 60), cols=max(max(len(r) for r in vals), 6))
        ws.clear()
        if ws.row_count < len(vals) + 2:
            ws.add_rows(len(vals) + 2 - ws.row_count)
        if ws.col_count < max(len(r) for r in vals):
            ws.add_cols(max(len(r) for r in vals) - ws.col_count)
        ws.update(values=vals, range_name="A1", value_input_option="USER_ENTERED")
        try:
            sid = ws.id
            hdr_row = 3 if nm.endswith("Summary") else 0
            reqs = [dict(repeatCell=dict(range=dict(sheetId=sid, startRowIndex=hdr_row, endRowIndex=hdr_row + 1),
                                         cell=dict(userEnteredFormat=dict(textFormat=dict(bold=True),
                                                                          backgroundColor=dict(red=.9, green=.93, blue=1))),
                                         fields="userEnteredFormat(textFormat,backgroundColor)")),
                    dict(updateSheetProperties=dict(properties=dict(sheetId=sid, gridProperties=dict(frozenRowCount=hdr_row + 1)),
                                                    fields="gridProperties.frozenRowCount"))]
            if nm.endswith("Stocks"):
                rng = [dict(sheetId=sid, startRowIndex=1, endRowIndex=len(vals), startColumnIndex=0, endColumnIndex=len(COLS))]
                for txt, col in (("Above", (.78, .93, .8)), ("Below", (.97, .8, .8)), ("Breakout", (.78, .93, .8)),
                                 ("Breakdown", (.97, .8, .8)), ("Near high", (.78, .93, .8)), ("Far from high", (.97, .8, .8))):
                    reqs.append(dict(addConditionalFormatRule=dict(index=0, rule=dict(
                        ranges=rng, booleanRule=dict(condition=dict(type="TEXT_CONTAINS", values=[dict(userEnteredValue=txt)]),
                                                     format=dict(backgroundColor=dict(red=col[0], green=col[1], blue=col[2])))))))
            book.batch_update(dict(requests=reqs))
        except Exception:
            pass                                          # formatting is cosmetic; the data is already written
    return names


def parse_index_csv(text):
    """NSE index constituent CSV (Company Name, Industry, Symbol, Series, ISIN Code) -> list of symbols."""
    import csv
    import io
    rd = csv.DictReader(io.StringIO(text))
    return [r["Symbol"].strip() for r in rd if r.get("Symbol")]
