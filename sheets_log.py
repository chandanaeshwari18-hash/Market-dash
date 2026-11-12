"""Optional Google Sheet logging of build-up alerts (service account). Fails soft."""
import streamlit as st

HEADERS = ["date", "time", "index", "expiry", "action", "strike", "entry", "sl", "t1", "t2",
           "source", "outcome", "pnl_pct", "exit", "adx", "rsi", "atr", "note",
           "lot_size", "lots", "qty", "cost", "capital", "risk_amt", "pnl_amt", "charges", "slippage", "net_amt",
           "ext_vwap", "ext_open", "brk_age", "prem_runup", "heavy"]


def _info():
    """Service-account info: either the whole JSON pasted as GCP_JSON, or a [gcp_service_account] table."""
    import json
    if "GCP_JSON" in st.secrets:
        return json.loads(st.secrets["GCP_JSON"])
    return dict(st.secrets["gcp_service_account"])


def enabled():
    """True when Sheet logging is configured. When it is not, shows exactly why (names only, never values)."""
    try:
        keys = list(st.secrets.keys())
    except Exception as e:
        st.caption(f"Sheet check: could not read Secrets ({type(e).__name__})")
        return False
    why = None
    if "SHEET_ID" not in keys:
        why = "SHEET_ID not found at top level"
    elif "GCP_JSON" in keys:
        try:
            import json
            info = json.loads(st.secrets["GCP_JSON"])
            if "private_key" not in info or "client_email" not in info:
                why = "GCP_JSON parsed but private_key/client_email missing"
        except Exception as e:
            why = f"GCP_JSON is not valid JSON ({type(e).__name__}: {str(e)[:80]})"
    elif "gcp_service_account" not in keys:
        why = "neither GCP_JSON nor gcp_service_account found at top level"
    if why:
        st.caption(f"Sheet check: {why}. Secret names seen: {keys}")
        return False
    return True


@st.cache_resource
def _ws():
    import gspread
    from google.oauth2.service_account import Credentials
    creds = Credentials.from_service_account_info(
        _info(),
        scopes=["https://www.googleapis.com/auth/spreadsheets"])
    sh = gspread.authorize(creds).open_by_key(st.secrets["SHEET_ID"])
    try:
        ws = sh.worksheet("Alerts")
    except Exception:
        ws = sh.add_worksheet("Alerts", rows=2000, cols=len(HEADERS))
    if ws.col_count < len(HEADERS):
        ws.add_cols(len(HEADERS) - ws.col_count)
    if ws.row_values(1) != HEADERS:                      # also adds the new tracker columns to an older sheet
        ws.update(values=[HEADERS], range_name="A1")
    return ws


def append_alert(row):
    try:
        _ws().append_row([row.get(h, "") for h in HEADERS], value_input_option="USER_ENTERED")
        return True, ""
    except Exception as e:
        return False, str(e)[:200]


def update_outcome(row, ev):
    """Write the outcome of one alert (sheet row number): outcome / pnl_pct / exit and the rupee result pnl_amt."""
    try:
        from gspread.utils import rowcol_to_a1
        r = int(row)
        pc = rowcol_to_a1(r, HEADERS.index("pnl_amt") + 1) + ":" + rowcol_to_a1(r, HEADERS.index("net_amt") + 1)
        _ws().batch_update([
            dict(range=f"L{r}:N{r}", values=[[ev.get("outcome", ""), ev.get("pnl_pct", ""), ev.get("exit", "")]]),
            dict(range=pc, values=[[ev.get("pnl_amt", ""), ev.get("charges", ""), ev.get("slippage", ""), ev.get("net_amt", "")]]),
        ])
        return True
    except Exception as e:
        return False


@st.cache_data(ttl=60)
def read_alerts():
    try:
        recs = _ws().get_all_records()
        for i, r in enumerate(recs):
            r["_row"] = i + 2                                # sheet row number, used to write outcomes back
        return recs
    except Exception as e:
        st.caption(f"Sheet error: {type(e).__name__}: {str(e)[:160] or 'no details'}")
        return []


# ---------- "My events": a small list of dates Sai wants warnings for (SEBI / budget / tax / any news) ----------
EV_HEADERS = ["date", "time", "title", "impact"]


@st.cache_resource
def _events_ws():
    import gspread
    from google.oauth2.service_account import Credentials
    creds = Credentials.from_service_account_info(_info(), scopes=["https://www.googleapis.com/auth/spreadsheets"])
    sh = gspread.authorize(creds).open_by_key(st.secrets["SHEET_ID"])
    try:
        ws = sh.worksheet("MyEvents")
    except Exception:
        ws = sh.add_worksheet("MyEvents", rows=500, cols=len(EV_HEADERS))
    if ws.row_values(1) != EV_HEADERS:
        ws.update(values=[EV_HEADERS], range_name="A1")
    return ws


@st.cache_data(ttl=60)
def read_my_events():
    """Rows of the MyEvents tab as dicts, each with its sheet row number in 'row'."""
    try:
        vals = _events_ws().get_all_values()
    except Exception:
        return []
    out = []
    for i, r in enumerate(vals[1:], start=2):
        if len(r) >= 3 and r[0].strip():
            out.append(dict(row=i, date=r[0].strip(), time=(r[1] if len(r) > 1 else "").strip(), title=r[2].strip(),
                            impact=(r[3] if len(r) > 3 else "High").strip() or "High"))
    return out


def add_my_event(date_s, time_s, title, impact):
    try:
        _events_ws().append_row([date_s, time_s, title, impact], value_input_option="USER_ENTERED")
        read_my_events.clear()
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:100]}"


def delete_my_event(row_no):
    try:
        _events_ws().delete_rows(int(row_no))
        read_my_events.clear()
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:100]}"


@st.cache_resource
def _gc():
    import gspread
    from google.oauth2.service_account import Credentials
    return gspread.authorize(Credentials.from_service_account_info(
        _info(), scopes=["https://www.googleapis.com/auth/spreadsheets"]))


def tracker_book(secret_name):
    """(spreadsheet, own). A dedicated Google Sheet when `secret_name` (its ID) is in Secrets, otherwise the main SHEET_ID
    (then the caller uses prefixed tab names). The Sheet must be shared with the service account as Editor."""
    own = secret_name in st.secrets
    return _gc().open_by_key(st.secrets[secret_name] if own else st.secrets["SHEET_ID"]), own


ORBDAY_HEADERS = ["date", "index", "or_low", "or_high", "or_width", "range_ratio", "status", "logged_at"]


def log_orb_day(row):
    """One row per day and index in the 'ORB_Days' tab: opening range and its size vs the average daily range.
    Skips when today's row for that index already exists. Returns True if a row was added."""
    try:
        sh = _gc().open_by_key(st.secrets["SHEET_ID"])
        try:
            ws = sh.worksheet("ORB_Days")
        except Exception:
            ws = sh.add_worksheet("ORB_Days", rows=500, cols=len(ORBDAY_HEADERS))
            ws.update(values=[ORBDAY_HEADERS], range_name="A1")
        for r in ws.get_all_values()[1:]:
            if len(r) > 1 and r[0] == row["date"] and r[1] == row["index"]:
                return False
        ws.append_row([row.get(h, "") for h in ORBDAY_HEADERS], value_input_option="USER_ENTERED")
        return True
    except Exception:
        return False


PREOPEN_HEADERS = ["date", "time", "index", "fut_price", "prev_close", "gap_pct", "volume", "vol_added", "buy_qty", "sell_qty",
                   "imbalance_pct", "oi", "heavy_wtd_pct", "heavy_view", "nse_est_pct", "nse_est_view", "stocks_pre_vol", "advances", "declines"]


def log_preopen(rows):
    """Append pre-open minute rows (list of dicts keyed by PREOPEN_HEADERS) to the 'PreOpen' tab. True on success."""
    try:
        sh = _gc().open_by_key(st.secrets["SHEET_ID"])
        try:
            ws = sh.worksheet("PreOpen")
        except Exception:
            ws = sh.add_worksheet("PreOpen", rows=2000, cols=len(PREOPEN_HEADERS))
            ws.update(values=[PREOPEN_HEADERS], range_name="A1")
        if ws.col_count < len(PREOPEN_HEADERS):
            ws.resize(cols=len(PREOPEN_HEADERS))
        if ws.row_values(1) != PREOPEN_HEADERS:                  # older tab without the new columns: refresh the header row
            ws.update(values=[PREOPEN_HEADERS], range_name="A1")
        ws.append_rows([[r.get(h, "") for h in PREOPEN_HEADERS] for r in rows], value_input_option="USER_ENTERED")
        return True
    except Exception:
        return False


FIIDII_HEADERS = ["oi_date", "cash_date", "fii_fut_long", "fii_fut_short", "fii_fut_net", "fii_fut_dnet", "fii_long_pct",
                  "dii_fut_net", "fii_cash_buy", "fii_cash_sell", "fii_cash_net", "dii_cash_buy", "dii_cash_sell", "dii_cash_net", "logged_at"]


def log_fiidii(row):
    """One row per NSE report date in the 'FII_DII' tab (FII/DII index-futures positions and cash flows). Skips a date already
    stored. Returns True if a row was added."""
    try:
        sh = _gc().open_by_key(st.secrets["SHEET_ID"])
        try:
            ws = sh.worksheet("FII_DII")
        except Exception:
            ws = sh.add_worksheet("FII_DII", rows=500, cols=len(FIIDII_HEADERS))
            ws.update(values=[FIIDII_HEADERS], range_name="A1")
        for r in ws.get_all_values()[1:]:
            if r and r[0] == row["oi_date"]:
                return "exists"
        ws.append_row([row.get(h, "") for h in FIIDII_HEADERS], value_input_option="USER_ENTERED")
        return "added"
    except Exception as e:
        return f"failed: {type(e).__name__} {str(e)[:90]}"


def _can_sheet():
    try:
        return "SHEET_ID" in st.secrets and ("GCP_JSON" in st.secrets or "gcp_service_account" in st.secrets)
    except Exception:
        return False


def save_token(day, token):
    """Keep today's Kite login in a private 'Session' tab so an app restart does not ask for login again. Fails soft."""
    if not _can_sheet():
        return False
    try:
        sh = _gc().open_by_key(st.secrets["SHEET_ID"])
        try:
            ws = sh.worksheet("Session")
        except Exception:
            ws = sh.add_worksheet("Session", rows=5, cols=3)
        ws.update(values=[[day, token]], range_name="A1:B1")
        return True
    except Exception:
        return False


def load_token(today):
    """(day, token) when the saved login is from today, otherwise None."""
    if not _can_sheet():
        return None
    try:
        ws = _gc().open_by_key(st.secrets["SHEET_ID"]).worksheet("Session")
        v = ws.get("A1:B1")
        if v and len(v[0]) >= 2 and v[0][0] == today and v[0][1]:
            return (today, v[0][1])
    except Exception:
        pass
    return None


def clear_token():
    if not _can_sheet():
        return
    try:
        _gc().open_by_key(st.secrets["SHEET_ID"]).worksheet("Session").update(values=[["", ""]], range_name="A1:B1")
    except Exception:
        pass
