"""Optional Google Sheet logging of build-up alerts (service account). Fails soft."""
import streamlit as st

HEADERS = ["date", "time", "index", "expiry", "action", "strike", "entry", "sl", "t1", "t2"]


def _info():
    """Service-account info: either the whole JSON pasted as GCP_JSON, or a [gcp_service_account] table."""
    import json
    if "GCP_JSON" in st.secrets:
        return json.loads(st.secrets["GCP_JSON"])
    return dict(st.secrets["gcp_service_account"])


def enabled():
    try:
        return ("GCP_JSON" in st.secrets or "gcp_service_account" in st.secrets) and "SHEET_ID" in st.secrets
    except Exception:
        return False


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
    if ws.acell("A1").value != "date":
        ws.update("A1", [HEADERS])
    return ws


def append_alert(row):
    try:
        _ws().append_row([row.get(h, "") for h in HEADERS], value_input_option="USER_ENTERED")
        return True, ""
    except Exception as e:
        return False, str(e)[:200]


@st.cache_data(ttl=60)
def read_alerts():
    try:
        return _ws().get_all_records()
    except Exception:
        return []
