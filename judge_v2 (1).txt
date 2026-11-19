"""AI judge (shadow mode): reads a Greeks alert and says TAKE / SKIP with a short reason.
Uses the Google Gemini free tier. The model name is looked up from Google's own model list
(so a retired model name cannot break it) with a fixed fallback list. Never blocks an alert; fails soft."""
import json
import re
import urllib.error
import urllib.request

BASE = "https://generativelanguage.googleapis.com/v1beta/"
FALLBACK = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-2.5-flash"]
_cache = {"models": None}

PROMPT = (
    "You are a strict, sceptical Indian index-options trade reviewer. A rule-based scanner (10-point Greeks checklist) "
    "has just raised a BUY alert on an option. Judge whether this is a sound intraday entry using ONLY the data given. "
    "Do not invent data. If the data is thin or conflicting, lean SKIP. Reply with JSON only: "
    '{"verdict":"TAKE" or "SKIP","confidence":0-100,"reason":"max 25 words, plain English","risk":"biggest risk, max 12 words"}'
)


def _get(url, key, timeout):
    req = urllib.request.Request(url, headers={"x-goog-api-key": key})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _models(key, timeout=10):
    """Flash models this key can call, newest stable first (cached for the life of the app)."""
    if _cache["models"] is not None:
        return _cache["models"]
    found = []
    try:
        for m in _get(BASE + "models?pageSize=200", key, timeout).get("models", []):
            n = m.get("name", "").split("/")[-1]
            if "generateContent" not in m.get("supportedGenerationMethods", []):
                continue
            if "flash" not in n or any(x in n for x in ("lite", "tts", "live", "image", "audio", "native", "omni", "computer", "thinking", "exp")):
                continue
            v = [int(x) for x in re.findall(r"\d+", n.split("flash")[0])] or [0]
            found.append((("preview" not in n), v, n))
    except Exception:
        pass
    found.sort(reverse=True)
    out = [n for _, _, n in found] + [n for n in FALLBACK if n not in [f[2] for f in found]]
    _cache["models"] = out[:6]
    return _cache["models"]


def judge(api_key, facts, timeout=25):
    """facts: dict of plain values. Returns (verdict, confidence, reason, risk, error)."""
    if not api_key:
        return "", "", "", "", "no key"
    body = json.dumps({
        "contents": [{"parts": [{"text": PROMPT + "\n\nDATA:\n" + json.dumps(facts, default=str)}]}],
        "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
    }).encode()
    last = ""
    for model in _models(api_key):
        req = urllib.request.Request(BASE + "models/%s:generateContent" % model, data=body, method="POST",
                                     headers={"Content-Type": "application/json", "x-goog-api-key": api_key})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.loads(r.read().decode())
            j = json.loads(out["candidates"][0]["content"]["parts"][0]["text"])
            v = str(j.get("verdict", "")).upper()
            if v not in ("TAKE", "SKIP"):
                last = "bad verdict from " + model
                continue
            return v, j.get("confidence", ""), "[" + model + "] " + str(j.get("reason", ""))[:200], str(j.get("risk", ""))[:100], ""
        except urllib.error.HTTPError as e:
            last = "%s: HTTP %s" % (model, e.code)
            if e.code in (404, 400, 429, 503):
                continue                                    # model gone / busy / quota: try the next one
            break                                           # 401/403 etc: key problem, no point retrying
        except Exception as e:
            last = "%s: %s" % (model, str(e)[:80])
            continue
    _cache["models"] = None                                 # re-read the model list next time
    return "", "", "", "", last or "no model answered"
