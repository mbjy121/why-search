"""
fetch_github.py

Fetches ~500 public GitHub repositories tagged with the topics:
  - recommender-system
  - information-retrieval
  - nlp

For each repo, saves: name, description, README text, URL.
Writes the result to data/repos.json.

Notes on rate limits (this is important for the write-up):
  * GitHub's search API is limited to 10 requests/minute for unauthenticated
    clients and 30/min with a token. We only need ~5 pages per topic.
  * The REST /repos/{owner}/{repo}/readme endpoint is capped at 60/hour
    unauthenticated. To avoid that ceiling, we fetch the README's raw text
    from raw.githubusercontent.com, which has a much more generous cap.
  * If a GITHUB_TOKEN env var is present, we send it — this raises the
    search limit to 30/min and the core REST limit to 5,000/hour.
"""

import base64
import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# --- SSL context (same reason as fetch_arxiv.py: Windows Python often
# can't find a CA bundle) --------------------------------------------------
try:
    import certifi  # type: ignore
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    SSL_CTX = ssl._create_unverified_context()

# --- Config ---------------------------------------------------------------
TOPICS = [
    ("recommender-system", 170),
    ("information-retrieval", 170),
    ("nlp", 170),
]
TARGET_TOTAL = 500  # soft cap after dedupe
PER_PAGE = 100      # GitHub's max
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "").strip()

SEARCH_URL = "https://api.github.com/search/repositories"
RAW_README_TEMPLATES = [
    "https://raw.githubusercontent.com/{full_name}/{branch}/README.md",
    "https://raw.githubusercontent.com/{full_name}/{branch}/README.rst",
    "https://raw.githubusercontent.com/{full_name}/{branch}/README",
    "https://raw.githubusercontent.com/{full_name}/{branch}/readme.md",
]
COMMON_BRANCHES = ["main", "master"]


def _headers():
    h = {
        "User-Agent": "dual-search-demo/1.0",
        "Accept": "application/vnd.github+json",
    }
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h


def http_get(url: str, timeout: int = 30):
    """GET a URL. Returns (status, body_bytes, headers_dict)."""
    req = urllib.request.Request(url, headers=_headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=SSL_CTX) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers or {})


def search_topic(topic: str, want: int):
    """Search GitHub repositories by topic and return metadata dicts."""
    print(f"\nSearching topic: {topic!r}  (target {want})")
    results = []
    page = 1
    while len(results) < want and page <= 10:  # GitHub caps search at 1000 = 10*100
        need = want - len(results)
        per = min(PER_PAGE, need)
        params = {
            "q": f"topic:{topic}",
            "sort": "stars",
            "order": "desc",
            "per_page": per,
            "page": page,
        }
        url = f"{SEARCH_URL}?{urllib.parse.urlencode(params)}"
        print(f"  page {page} (per_page={per}) ...", end=" ", flush=True)
        status, body, headers = http_get(url)
        if status == 403 or status == 429:
            reset = headers.get("X-RateLimit-Reset")
            print(f"rate-limited (status {status}). Stopping this topic.")
            print(f"    remaining={headers.get('X-RateLimit-Remaining')} reset={reset}")
            break
        if status != 200:
            print(f"error status {status}: {body[:200]!r}")
            break
        data = json.loads(body.decode("utf-8"))
        items = data.get("items", [])
        print(f"got {len(items)} repos")
        if not items:
            break
        for r in items:
            results.append(
                {
                    "full_name": r["full_name"],
                    "name": r["name"],
                    "description": (r.get("description") or "").strip(),
                    "url": r["html_url"],
                    "default_branch": r.get("default_branch") or "main",
                    "stars": r.get("stargazers_count", 0),
                    "topic_matched": topic,
                }
            )
        page += 1
        # Politeness delay. Unauth search is 10/min; with token 30/min.
        time.sleep(2.0 if GITHUB_TOKEN else 6.5)
    return results


def fetch_readme(full_name: str, default_branch: str) -> str:
    """Fetch a repo's README as plain text. Returns '' if not found."""
    branches = [default_branch] + [b for b in COMMON_BRANCHES if b != default_branch]
    for branch in branches:
        for tmpl in RAW_README_TEMPLATES:
            url = tmpl.format(full_name=full_name, branch=branch)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": _headers()["User-Agent"]})
                with urllib.request.urlopen(req, timeout=20, context=SSL_CTX) as resp:
                    if resp.status == 200:
                        raw = resp.read()
                        # Try utf-8; fall back to latin-1
                        try:
                            return raw.decode("utf-8")
                        except UnicodeDecodeError:
                            return raw.decode("latin-1", errors="replace")
            except urllib.error.HTTPError:
                continue
            except Exception:
                continue
    return ""


def clean_readme(text: str, max_chars: int = 8000) -> str:
    """Trim README to a manageable size and collapse whitespace."""
    if not text:
        return ""
    # Drop obvious noise: HTML tags in badges, but keep the words
    # (we intentionally keep it simple; the search index tokenizes anyway).
    text = text.replace("\r", "\n")
    # Collapse very long runs of blank lines
    lines = [ln.rstrip() for ln in text.split("\n")]
    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


def dedupe(entries):
    seen = set()
    out = []
    for e in entries:
        key = e["full_name"]
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def main():
    here = Path(__file__).parent
    out_dir = here / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "repos.json"

    if GITHUB_TOKEN:
        print("Using GITHUB_TOKEN from environment (higher rate limits).")
    else:
        print("No GITHUB_TOKEN in environment; using unauthenticated limits.")

    # 1) Search each topic for repo metadata
    all_meta = []
    for topic, want in TOPICS:
        try:
            all_meta.extend(search_topic(topic, want))
        except Exception as e:
            print(f"  ! topic {topic!r} failed: {e}")

    print(f"\nTotal metadata rows (before dedupe): {len(all_meta)}")
    all_meta = dedupe(all_meta)
    print(f"After dedupe by full_name: {len(all_meta)}")

    # Trim to TARGET_TOTAL, keeping the higher-star repos first
    all_meta.sort(key=lambda r: r.get("stars", 0), reverse=True)
    all_meta = all_meta[:TARGET_TOTAL]
    print(f"Trimmed to top {len(all_meta)} by stars.")

    # 2) Fetch READMEs
    print("\nFetching READMEs from raw.githubusercontent.com ...")
    kept = []
    for i, meta in enumerate(all_meta, 1):
        readme = clean_readme(fetch_readme(meta["full_name"], meta["default_branch"]))
        if i % 25 == 0 or i == len(all_meta):
            print(f"  {i}/{len(all_meta)} done (last: {meta['full_name']}, readme={len(readme)} chars)")
        kept.append(
            {
                "name": meta["name"],
                "full_name": meta["full_name"],
                "description": meta["description"],
                "readme": readme,
                "url": meta["url"],
                "stars": meta["stars"],
                "topic_matched": meta["topic_matched"],
            }
        )
        # Small politeness delay for raw.githubusercontent.com
        time.sleep(0.15)

    # Drop repos where we got neither description nor README (no text to search)
    kept = [r for r in kept if r["description"] or r["readme"]]
    print(f"\nRepos with searchable text: {len(kept)}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(kept)} repos to {out_path}")
    if kept:
        s = kept[0]
        # Encode-safe printing for Windows terminals (cp1252 can't render
        # emoji that appear in some GitHub descriptions).
        def _safe(t: str) -> str:
            return t.encode("ascii", errors="replace").decode("ascii")
        print("\nSample entry:")
        print(f"  full_name:   {_safe(s['full_name'])}  (stars: {s['stars']})")
        print(f"  description: {_safe(s['description'][:120])}")
        print(f"  url:         {_safe(s['url'])}")
        print(f"  readme[:150]:{_safe(s['readme'][:150].replace(chr(10),' '))}")


if __name__ == "__main__":
    main()
