"""Compares airun's HTTP API with llama-server's, request by request.

Runs a fixed set of requests against two servers that serve the same model
and reports, per request, whether the status codes match and which JSON keys
one side returns and the other does not (recursively; for a stream, over the
union of its events, with each event's SSE name as `<event>`). Values are not
compared -- the engines differ -- only the shape a client relies on.

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
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}
WEATHER_Q = "What is the weather in Paris? Use the tool."
WEATHER_PARAMS = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
OAI_TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather of a city",
                                               "parameters": WEATHER_PARAMS}}]
PROMPT = "The capital of France is"
SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
          "required": ["name", "age"]}

# (label, method, path, body)
CASES = [
    ("health", "GET", "/health", None),
    ("v1 health", "GET", "/v1/health", None),
    ("props", "GET", "/props", None),
    ("models", "GET", "/models", None),
    ("v1 models", "GET", "/v1/models", None),
    ("completion", "POST", "/completion", {"prompt": PROMPT, "n_predict": 8, "temperature": 0}),
    ("completion stream", "POST", "/completion", {"prompt": PROMPT, "n_predict": 8, "temperature": 0,
                                                  "stream": True}),
    ("completions", "POST", "/completions", {"prompt": PROMPT, "n_predict": 8, "temperature": 0}),
    ("completion n_probs", "POST", "/completion", {"prompt": PROMPT, "n_predict": 3, "temperature": 0,
                                                   "n_probs": 2}),
    ("completion n_probs stream", "POST", "/completion", {"prompt": PROMPT, "n_predict": 3, "temperature": 0,
                                                          "n_probs": 2, "stream": True}),
    ("completion tokens prompt", "POST", "/completion", {"prompt": [760, 6511, 314, 9338, 369], "n_predict": 4,
                                                         "temperature": 0, "return_tokens": True}),
    ("completion ignore_eos", "POST", "/completion", {"prompt": "Say OK.", "n_predict": 6, "temperature": 0,
                                                      "ignore_eos": True}),
    ("completion samplers", "POST", "/completion", {"prompt": PROMPT, "n_predict": 6, "temperature": 0.7,
                                                    "seed": 7, "xtc_probability": 0.5, "xtc_threshold": 0.1,
                                                    "dry_multiplier": 0.8, "dynatemp_range": 0.3,
                                                    "mirostat": 0}),
    ("completion grammar", "POST", "/completion", {"prompt": "Is the sky blue? Answer:", "n_predict": 8,
                                                   "temperature": 0, "grammar": 'root ::= " yes" | " no"'}),
    ("completion json_schema", "POST", "/completion", {"prompt": "A person as JSON:", "n_predict": 48,
                                                       "temperature": 0, "json_schema": SCHEMA}),
    ("v1 completions", "POST", "/v1/completions", {"prompt": PROMPT, "max_tokens": 8, "temperature": 0}),
    ("v1 completions stream", "POST", "/v1/completions", {"prompt": PROMPT, "max_tokens": 8, "temperature": 0,
                                                          "stream": True}),
    ("v1 completions logprobs", "POST", "/v1/completions", {"prompt": PROMPT, "max_tokens": 3, "temperature": 0,
                                                            "logprobs": 2}),
    ("chat", "POST", "/v1/chat/completions", {"messages": MSG, "max_tokens": 64, "temperature": 0}),
    ("chat stream", "POST", "/v1/chat/completions", {"messages": MSG, "max_tokens": 64, "temperature": 0,
                                                     "stream": True, "stream_options": {"include_usage": True}}),
    ("chat alias", "POST", "/chat/completions", {"messages": MSG, "max_tokens": 64, "temperature": 0}),
    ("chat bad request", "POST", "/v1/chat/completions", {"max_tokens": 4}),
    ("chat tools", "POST", "/v1/chat/completions", {
        "messages": [{"role": "user", "content": WEATHER_Q}], "max_tokens": 200, "temperature": 0,
        "tools": OAI_TOOLS}),
    ("chat tools stream", "POST", "/v1/chat/completions", {
        "messages": [{"role": "user", "content": WEATHER_Q}], "max_tokens": 200, "temperature": 0,
        "tools": OAI_TOOLS, "stream": True}),
    ("chat tool_choice none", "POST", "/v1/chat/completions", {
        "messages": [{"role": "user", "content": WEATHER_Q}], "max_tokens": 32, "temperature": 0,
        "tools": OAI_TOOLS, "tool_choice": "none", **NOTHINK}),
    ("chat logprobs", "POST", "/v1/chat/completions", {"messages": MSG, "max_tokens": 4, "temperature": 0,
                                                       "logprobs": True, "top_logprobs": 2, **NOTHINK}),
    ("chat logprobs stream", "POST", "/v1/chat/completions", {"messages": MSG, "max_tokens": 4, "temperature": 0,
                                                              "logprobs": True, "top_logprobs": 2,
                                                              "stream": True, **NOTHINK}),
    ("chat json_object", "POST", "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "Give a JSON object with a key 'a' set to 1."}],
        "max_tokens": 32, "temperature": 0, "response_format": {"type": "json_object"}, **NOTHINK}),
    ("chat json_schema", "POST", "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "Invent a person."}], "max_tokens": 48, "temperature": 0,
        "response_format": {"type": "json_schema", "json_schema": {"name": "person", "schema": SCHEMA}},
        **NOTHINK}),
    ("chat input_tokens", "POST", "/v1/chat/completions/input_tokens", {"messages": MSG}),
    ("responses", "POST", "/v1/responses", {"input": "Say hi in one word.", "max_output_tokens": 64,
                                            "temperature": 0}),
    ("responses stream", "POST", "/v1/responses", {"input": "Say hi in one word.", "max_output_tokens": 64,
                                                   "temperature": 0, "stream": True}),
    ("responses tools", "POST", "/v1/responses", {
        "input": WEATHER_Q, "max_output_tokens": 200, "temperature": 0,
        "tools": [{"type": "function", "name": "get_weather", "description": "Current weather of a city",
                   "parameters": WEATHER_PARAMS}]}),
    ("responses input items", "POST", "/v1/responses", {
        "instructions": "Answer in one word.", "max_output_tokens": 64, "temperature": 0,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "Capital of France?"}]}]}),
    ("responses input_tokens", "POST", "/v1/responses/input_tokens", {"input": "Say hi in one word."}),
    ("messages", "POST", "/v1/messages", {"model": "x", "max_tokens": 64, "messages": MSG, "temperature": 0}),
    ("messages stream", "POST", "/v1/messages", {"model": "x", "max_tokens": 64, "messages": MSG,
                                                 "temperature": 0, "stream": True}),
    ("messages tools", "POST", "/v1/messages", {
        "model": "x", "max_tokens": 200, "temperature": 0,
        "messages": [{"role": "user", "content": WEATHER_Q}],
        "tools": [{"name": "get_weather", "description": "Current weather of a city",
                   "input_schema": WEATHER_PARAMS}]}),
    ("messages count_tokens", "POST", "/v1/messages/count_tokens", {"model": "x", "messages": MSG}),
    ("tokenize", "POST", "/tokenize", {"content": "Hello world"}),
    ("tokenize with_pieces", "POST", "/tokenize", {"content": "Hello world", "with_pieces": True}),
    ("detokenize", "POST", "/detokenize", {"tokens": [9707, 1879]}),
    ("apply-template", "POST", "/apply-template", {"messages": MSG}),
    ("infill", "POST", "/infill", {"input_prefix": "def add(a, b):\n    ", "input_suffix": "\n", "n_predict": 12,
                                   "temperature": 0}),
    ("embeddings", "POST", "/embeddings", {"content": "hi"}),
    ("v1 embeddings", "POST", "/v1/embeddings", {"input": "hi"}),
    ("rerank", "POST", "/rerank", {"query": "a", "documents": ["a", "b"]}),
    ("lora get", "GET", "/lora-adapters", None),
    ("lora post", "POST", "/lora-adapters", []),
    ("props post", "POST", "/props", {}),
    ("slots", "GET", "/slots", None),
    ("slot erase", "POST", "/slots/0?action=erase", {}),
    ("slot save", "POST", "/slots/0?action=save", {"filename": "x.bin"}),
    ("metrics", "GET", "/metrics", None),
    ("transcriptions", "POST", "/v1/audio/transcriptions", {}),
    ("control", "POST", "/v1/chat/completions/control", {"id": "nope", "action": "reasoning_end"}),
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
    if "text/event-stream" in ctype or text.startswith(("data:", "event:")):
        events, name = [], None
        for line in text.splitlines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                payload = line[5:].strip()
                if payload and payload != "[DONE]":
                    try:
                        ev = json.loads(payload)
                    except json.JSONDecodeError:
                        ev = {"<unparsed>": payload[:40]}
                    if name is not None and isinstance(ev, dict):
                        ev = {"<event>": name, **ev}
                    events.append(ev)
                name = None
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


def event_names(body):
    if isinstance(body, dict) and "<stream>" in body:
        return sorted({e.get("<event>", "") for e in body["<stream>"] if isinstance(e, dict)} - {""})
    return []


def record(url, only=None):
    out = []
    for (label, m, p, b) in CASES:
        if only and label not in only:
            continue
        s, r = call(url, m, p, b)
        out.append({"label": label, "status": s, "body": r})
    return out


def compare(ref, test, show=False):
    test_by = {t["label"]: t for t in test}
    failures = 0
    for r in ref:
        t = test_by.get(r["label"], {"status": None, "body": None})
        rk, tk = keys(r["body"]), keys(t["body"])
        missing, extra = sorted(rk - tk), sorted(tk - rk)
        missing_ev = sorted(set(event_names(r["body"])) - set(event_names(t["body"])))
        ok = r["status"] == t["status"] and not missing and not missing_ev
        failures += not ok
        print(f"{'OK  ' if ok else 'DIFF'} {r['label']}: status {r['status']} vs {t['status']}")
        if missing:
            print(f"       missing in airun: {', '.join(missing[:40])}{' ...' if len(missing) > 40 else ''}")
        if missing_ev:
            print(f"       events missing in airun: {', '.join(missing_ev)}")
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
    ap.add_argument("--only", nargs="*", help="record only these case labels")
    ap.add_argument("--compare", nargs=2, metavar=("REF", "TEST"), help="compare two recordings")
    ap.add_argument("--show", action="store_true", help="print both bodies")
    a = ap.parse_args()
    if a.record:
        json.dump(record(a.url, a.only), open(a.record, "w", encoding="utf-8"), indent=1)
        return 0
    if a.compare:
        ref, test = (json.load(open(f, encoding="utf-8")) for f in a.compare)
        return compare(ref, test, a.show)
    return compare(record(a.ref), record(a.test), a.show)


if __name__ == "__main__":
    sys.exit(main())
