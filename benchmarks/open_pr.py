#!/usr/bin/env python3
"""Open the PR for the current branch via the GitHub API.

Reads the token from $GITHUB_TOKEN (filled by `git credential fill`) and the
body from PR.md. Run:
  $env:GITHUB_TOKEN=<token>; python benchmarks/open_pr.py
"""
import json, os, sys, urllib.request

REPO = "Dspofu/airun"
HEAD = "fix/nvfp4-global-scale-and-serve-draft"
BASE = "main"
TITLE = "NVFP4 global-scale fix, serve --draft, non-silent MTP, docs"

token = os.environ.get("GITHUB_TOKEN")
if not token:
    sys.exit("GITHUB_TOKEN not set")

body = open("PR.md", encoding="utf-8").read()
payload = json.dumps({
    "title": TITLE,
    "head": HEAD,
    "base": BASE,
    "body": body,
}).encode("utf-8")

req = urllib.request.Request(
    f"https://api.github.com/repos/{REPO}/pulls",
    data=payload,
    method="POST",
    headers={
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
    },
)
try:
    with urllib.request.urlopen(req) as r:
        pr = json.load(r)
    print(f"PR #{pr['number']}: {pr['html_url']}")
    print(f"state: {pr['state']}  draft: {pr.get('draft')}  mergeable: {pr.get('mergeable')}")
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')}")
    sys.exit(1)
