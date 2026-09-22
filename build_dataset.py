"""
build_dataset.py

Merges data/papers.json and data/repos.json into a single searchable file
data/combined.json. Each entry looks like:

  {
    "text":        "<title + abstract>"  OR  "<name + description + README>",
    "source_type": "paper" | "repo",
    "title":       "<paper title or repo full_name>",
    "url":         "<link back to arXiv or GitHub>"
  }

The `text` field is what our TF-IDF search will index in the next step, so
we concatenate the most search-worthy fields into one blob.
"""

import json
from pathlib import Path


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_paper_entry(p: dict) -> dict:
    title = (p.get("title") or "").strip()
    abstract = (p.get("abstract") or "").strip()
    text = f"{title}. {abstract}".strip()
    return {
        "text": text,
        "source_type": "paper",
        "title": title,
        "url": p.get("url", ""),
        "published": p.get("published", ""),
    }


def build_repo_entry(r: dict) -> dict:
    name = (r.get("full_name") or r.get("name") or "").strip()
    description = (r.get("description") or "").strip()
    readme = (r.get("readme") or "").strip()
    # Combine name + description + README into one text blob for search.
    parts = [name, description, readme]
    text = "\n".join(part for part in parts if part)
    return {
        "text": text,
        "source_type": "repo",
        "title": name,
        "url": r.get("url", ""),
        "stars": r.get("stars", 0),
    }


def main():
    here = Path(__file__).parent
    data_dir = here / "data"
    papers_path = data_dir / "papers.json"
    repos_path = data_dir / "repos.json"
    out_path = data_dir / "combined.json"

    papers = load_json(papers_path)
    repos = load_json(repos_path)

    combined = []
    for p in papers:
        entry = build_paper_entry(p)
        if entry["text"]:
            combined.append(entry)
    for r in repos:
        entry = build_repo_entry(r)
        if entry["text"]:
            combined.append(entry)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    # Summary
    n_papers = sum(1 for e in combined if e["source_type"] == "paper")
    n_repos = sum(1 for e in combined if e["source_type"] == "repo")
    print(f"Papers loaded:  {len(papers)}")
    print(f"Repos loaded:   {len(repos)}")
    print(f"Combined saved: {len(combined)}  ->  {out_path}")
    print(f"  papers: {n_papers}")
    print(f"  repos:  {n_repos}")

    # Show one of each so a human can eyeball it
    def _safe(t: str) -> str:
        return t.encode("ascii", errors="replace").decode("ascii")

    for want in ("paper", "repo"):
        sample = next((e for e in combined if e["source_type"] == want), None)
        if sample:
            print(f"\nSample {want}:")
            print(f"  title:  {_safe(sample['title'])[:100]}")
            print(f"  url:    {_safe(sample['url'])}")
            print(f"  text:   {_safe(sample['text'][:180])}...")


if __name__ == "__main__":
    main()
