"""AI judge (shadow mode): reads a Greeks alert and says TAKE / SKIP with a short reason.
Uses Google Gemini free tier (Flash model). Never blocks an alert; fails soft."""
import json
import urllib.request

MODEL = "gemini-2.5-flash"
URL = "https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent" % MODEL

PROMPT = (
    "You are a strict, sceptical Indian index-options trade reviewer. A rule-based scanner (10-point Greeks checklist) "
    "has just raised a BUY alert on an option. Judge whether this is a sound intraday entry using ONLY the data given. "
    "Do not invent data. If the data is thin or conflicting, lean SKIP. Reply with JSON only: "
    '{"verdict":"TAKE" or "SKIP","confidence":0-100,"reason":"max 25 words, plain English","risk":"biggest risk, max 12 words"}'
)


def judge(api_key, facts, timeout=20):
    """facts: dict of plain values. Returns (verdict, confidence, reason, risk, error)."""
    if not api_key:
        return "", "", "", "", "no key"
    body = {
        "contents": [{"parts": [{"text": PROMPT + "\n\nDATA:\n" + json.dumps(facts, default=str)}]}],
        "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
    }
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "x-goog-api-key": api_key})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read().decode())
        txt = out["candidates"][0]["content"]["parts"][0]["text"]
        j = json.loads(txt)
        v = str(j.get("verdict", "")).upper()
        if v not in ("TAKE", "SKIP"):
            return "", "", "", "", "bad verdict"
        return v, j.get("confidence", ""), str(j.get("reason", ""))[:200], str(j.get("risk", ""))[:100], ""
    except Exception as e:
        return "", "", "", "", str(e)[:120]
