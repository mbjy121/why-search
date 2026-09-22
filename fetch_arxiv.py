"""
fetch_arxiv.py

Fetches ~400 paper abstracts from the public arXiv API across three topics:
  - information retrieval
  - recommender systems
  - natural language processing

Saves the combined result to data/papers.json with fields:
  title, abstract, url, published

The arXiv API returns Atom XML, so we parse it with xml.etree.
No API key is needed. We are polite: small sleeps between requests.
"""

import json
import os
import ssl
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

# Windows Python builds often can't find a CA bundle. Try certifi first;
# if it isn't installed, fall back to an unverified context. arXiv is a
# public, read-only API, so this is acceptable for this educational demo.
try:
    import certifi  # type: ignore
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    SSL_CTX = ssl._create_unverified_context()

ARXIV_ENDPOINT = "http://export.arxiv.org/api/query"

# Atom XML namespaces used by arXiv
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

# We split 400 across three topics. arXiv caps a single request at 2000
# results, but smaller pages are more reliable. We request ~134 per topic.
QUERIES = [
    ("information retrieval", 140),
    ("recommender systems", 140),
    ("natural language processing", 140),
]


def build_url(search_query: str, start: int, max_results: int) -> str:
    """Build an arXiv API query URL for a given search term."""
    # `all:` searches title, abstract, and other fields.
    q = f'all:"{search_query}"'
    params = {
        "search_query": q,
        "start": start,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    return f"{ARXIV_ENDPOINT}?{urllib.parse.urlencode(params)}"


def fetch_page(url: str) -> str:
    """Fetch one page of results. Retries a couple of times on network hiccups."""
    last_err = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "dual-search-demo/1.0 (educational use)"},
            )
            with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as resp:
                return resp.read().decode("utf-8")
        except Exception as e:
            last_err = e
            print(f"  attempt {attempt + 1} failed: {e}")
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def parse_entries(xml_text: str):
    """Parse an arXiv Atom XML response into a list of dicts."""
    root = ET.fromstring(xml_text)
    entries = []
    for entry in root.findall("atom:entry", NS):
        title_el = entry.find("atom:title", NS)
        summary_el = entry.find("atom:summary", NS)
        published_el = entry.find("atom:published", NS)
        id_el = entry.find("atom:id", NS)

        title = (title_el.text or "").strip().replace("\n", " ") if title_el is not None else ""
        abstract = (summary_el.text or "").strip().replace("\n", " ") if summary_el is not None else ""
        published = (published_el.text or "").strip() if published_el is not None else ""
        # id_el.text is the canonical arXiv abstract URL, e.g.
        # "http://arxiv.org/abs/2401.12345v1"
        url = (id_el.text or "").strip() if id_el is not None else ""

        # Collapse whitespace runs inside abstracts/titles
        title = " ".join(title.split())
        abstract = " ".join(abstract.split())

        if title and abstract:
            entries.append(
                {
                    "title": title,
                    "abstract": abstract,
                    "url": url,
                    "published": published,
                }
            )
    return entries


def fetch_topic(topic: str, target_count: int, page_size: int = 50):
    """Fetch up to target_count entries for a single topic, paging through the API."""
    print(f"\nTopic: {topic!r}  (target {target_count})")
    collected = []
    start = 0
    while len(collected) < target_count:
        need = target_count - len(collected)
        this_page = min(page_size, need)
        url = build_url(topic, start=start, max_results=this_page)
        print(f"  GET start={start} n={this_page} ...", end=" ", flush=True)
        xml_text = fetch_page(url)
        entries = parse_entries(xml_text)
        print(f"got {len(entries)}")
        if not entries:
            break
        collected.extend(entries)
        start += this_page
        # arXiv asks for at least a few seconds between requests
        time.sleep(3.5)
    return collected[:target_count]


def dedupe(entries):
    """Remove duplicates by URL (same paper can match multiple topics)."""
    seen = set()
    out = []
    for e in entries:
        key = e["url"] or (e["title"] + e["published"])
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def main():
    here = Path(__file__).parent
    out_dir = here / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "papers.json"

    all_entries = []
    for topic, count in QUERIES:
        try:
            all_entries.extend(fetch_topic(topic, count))
        except Exception as e:
            print(f"  ! topic {topic!r} failed: {e}")

    print(f"\nTotal fetched (with possible duplicates): {len(all_entries)}")
    all_entries = dedupe(all_entries)
    print(f"After dedupe: {len(all_entries)}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(all_entries)} papers to {out_path}")
    # Show a tiny sample so a human can eyeball it worked
    if all_entries:
        sample = all_entries[0]
        print("\nSample entry:")
        print(f"  title:     {sample['title'][:100]}...")
        print(f"  published: {sample['published']}")
        print(f"  url:       {sample['url']}")
        print(f"  abstract:  {sample['abstract'][:150]}...")


if __name__ == "__main__":
    main()
