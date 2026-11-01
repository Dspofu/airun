"""Compares airun's HTTP API with llama-server's, request by request.

Runs a fixed set of requests against two servers that serve the same model
and reports, per request, whether the status codes match and which JSON keys
one side returns and the other does not (recursively; for a stream, over the
union of its events). Values are not compared -- the engines differ -- only
the shape a client relies on.

    python tests/api_conformance.py --ref http://127.0.0.1:8090 --test http://127.0.0.1:8091

or, when the two cannot run at once (one GPU), record each and compare later:

    python tests/api_conformance.py --record ref.json --url http://127.0.0.1:8090
    python tests/api_conformance.py --record airun.json --url http://127.0.0.1:8091
    python tests/api_conformance.py --compare ref.json airun.json

Only the standard library is used.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

MSG = [{"role": "user", "content": "Say hi in one word."}]

CASES = [
    ("GET", "/health", None),
    ("GET", "/v1/health", None),
    ("GET", "/props", None),
    ("GET", "/models", None),
    ("GET", "/v1/models", None),
    ("POST", "/completion", {"prompt": "The capital of France is", "n_predict": 8, "temperature": 0}),
    ("POST", "/completion", {"prompt": "The capital of France is", "n_predict": 8, "temperature": 0, "stream": True}),
    ("POST", "/completions", {"prompt": "The capital of France is", "n_predict": 8, "temperature": 0}),
    ("POST", "/v1/completions", {"prompt": "The capital of France is", "max_tokens": 8, "temperature": 0}),
    ("POST", "/v1/completions", {"prompt": "The capital of France is", "max_tokens": 8, "temperature": 0,
                                 "stream": True}),
    ("POST", "/v1/chat/completions", {"messages": MSG, "max_tokens": 64, "temperature": 0}),
    ("POST", "/v1/chat/completions", {"messages": MSG, "max_tokens": 64, "temperature": 0, "stream": True,
                                      "stream_options": {"include_usage": True}}),
    ("POST", "/chat/completions", {"messages": MSG, "max_tokens": 64, "temperature": 0}),
    ("POST", "/tokenize", {"content": "Hello world"}),
    ("POST", "/tokenize", {"content": "Hello world", "with_pieces": True}),
    ("POST", "/detokenize", {"tokens": [9707, 1879]}),
    ("POST", "/apply-template", {"messages": MSG}),
    ("GET", "/slots", None),
    ("POST", "/v1/chat/completions", {"max_tokens": 4}),
    ("POST", "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "What is the weather in Paris? Use the tool."}],
        "max_tokens": 200, "temperature": 0,
        "tools": [{"type": "function", "function": {
            "name": "get_weather", "description": "Current weather of a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]}),
]


def call(base, method, path, body):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            status, raw, ctype = r.status, r.read(), r.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        status, raw, ctype = e.code, e.read(), e.headers.get("Content-Type", "")
    except Exception as e:  # connection refused, timeout
        return None, f"<{type(e).__name__}: {e}>"
    text = raw.decode("utf-8", "replace")
    if "text/event-stream" in ctype or text.startswith("data:"):
        events = []
        for line in text.splitlines():
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload and payload != "[DONE]":
                    try:
                        events.append(json.loads(payload))
                    except json.JSONDecodeError:
                        events.append({"<unparsed>": payload[:40]})
        return status, {"<stream>": events, "<done>": "[DONE]" in text}
    try:
        return status, json.loads(text)
    except json.JSONDecodeError:
        return status, f"<text {len(text)} bytes>"


def keys(v, prefix=""):
    """Dotted key paths of a JSON value; list items are merged under []."""
    out = set()
    if isinstance(v, dict):
        for k, x in v.items():
            p = f"{prefix}.{k}" if prefix else k
            out.add(p)
            out |= keys(x, p)
    elif isinstance(v, list):
        for x in v:
            out |= keys(x, prefix + "[]")
    return out


def label_of(method, path, body):
    return f"{method} {path}" + (" stream" if body and body.get("stream") else "") +         (" tools" if body and body.get("tools") else "") +         (" with_pieces" if body and body.get("with_pieces") else "") +         (" (bad request)" if body is not None and path.endswith("chat/completions") and "messages" not in body else "")


def record(url):
    return [{"label": label_of(m, p, b), "status": s, "body": r}
            for (m, p, b) in CASES for (s, r) in [call(url, m, p, b)]]


def compare(ref, test, show=False):
    failures = 0
    for r, t in zip(ref, test):
        rk, tk = keys(r["body"]), keys(t["body"])
        missing, extra = sorted(rk - tk), sorted(tk - rk)
        ok = r["status"] == t["status"] and not missing
        failures += not ok
        print(f"{'OK  ' if ok else 'DIFF'} {r['label']}: status {r['status']} vs {t['status']}")
        if missing:
            print(f"       missing in airun: {', '.join(missing[:40])}{' ...' if len(missing) > 40 else ''}")
        if extra and not ok:
            print(f"       only in airun:    {', '.join(extra[:20])}")
        if show:
            print("       ref :", json.dumps(r["body"])[:600])
            print("       test:", json.dumps(t["body"])[:600])
    print(f"\n{len(ref) - failures}/{len(ref)} requests match llama-server's shape")
    return 1 if failures else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", help="llama-server base URL")
    ap.add_argument("--test", help="airun base URL")
    ap.add_argument("--url", help="with --record: the server to record")
    ap.add_argument("--record", help="write the responses of --url to this file")
    ap.add_argument("--compare", nargs=2, metavar=("REF", "TEST"), help="compare two recordings")
    ap.add_argument("--show", action="store_true", help="print both bodies")
    a = ap.parse_args()
    if a.record:
        json.dump(record(a.url), open(a.record, "w", encoding="utf-8"), indent=1)
        return 0
    if a.compare:
        ref, test = (json.load(open(f, encoding="utf-8")) for f in a.compare)
        return compare(ref, test, a.show)
    return compare(record(a.ref), record(a.test), a.show)


if __name__ == "__main__":
    sys.exit(main())
