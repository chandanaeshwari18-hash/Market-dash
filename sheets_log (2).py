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
