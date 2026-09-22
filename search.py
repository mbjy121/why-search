"""
search.py

Builds a TF-IDF index over data/combined.json and searches it.

What TF-IDF does, in one line: it turns each document (a paper or repo)
into a bag of words, weights each word by how often it appears in that
document vs. how rare it is across the whole dataset, then ranks
documents against a query by cosine similarity of those weighted vectors.

We also report WHY each result matched by returning the specific words
from the query that overlap with that document (weighted by TF-IDF).

Public functions:
    load_dataset() -> list[dict]
    build_index(entries) -> SearchIndex
    SearchIndex.search(query, top_k=5, source_type=None) -> list[Result]

Run this file directly to print results for a demo query.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = Path(__file__).parent / "data" / "combined.json"
EMBEDDINGS_PATH = Path(__file__).parent / "data" / "embeddings.npy"
SEMANTIC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_dataset(path: Path = DATA_PATH) -> list[dict]:
    """Load the combined papers + repos dataset."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class Result:
    rank: int
    score: float
    source_type: str
    title: str
    url: str
    snippet: str
    matched_words: list[str]  # words from the query that actually hit this doc

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "score": round(float(self.score), 4),
            "source_type": self.source_type,
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "matched_words": self.matched_words,
        }


class SearchIndex:
    """A tiny wrapper over a scikit-learn TF-IDF vectorizer + doc matrix.

    Also holds an optional dense-embedding matrix so we can offer
    semantic ("meaning-based") search alongside the keyword search.
    """

    def __init__(self, entries: list[dict]):
        self.entries = entries
        self.texts = [e["text"] for e in entries]
        # Lazy semantic bits — populated on first semantic query
        self._doc_embeddings = None    # np.ndarray, shape (N, D)
        self._semantic_model = None    # sentence_transformers.SentenceTransformer
        # Reasonable defaults for a small corpus:
        #  - lowercase (default True)
        #  - remove English stopwords ("the", "a", ...)
        #  - unigrams + bigrams (bigrams catch phrases like "recommender system")
        #  - min_df=2 drops words that appear in only 1 doc (noise)
        #  - sublinear_tf tames very-frequent terms
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]+\b",
        )
        self.doc_matrix = self.vectorizer.fit_transform(self.texts)
        # Fast lookup: term -> column index
        self.vocab = self.vectorizer.vocabulary_

    # ------------------------------------------------------------------
    def _query_tokens(self, query: str) -> list[str]:
        """Tokenize a query the same way the vectorizer does, so we can
        report overlapping words back to the user in their natural form."""
        # Use the vectorizer's analyzer to guarantee identical tokenization
        # (lowercase, ngrams, stopword removal, etc.).
        analyzer = self.vectorizer.build_analyzer()
        return list(analyzer(query))

    def _snippet(self, text: str, matched_words: list[str], max_chars: int = 220) -> str:
        """Return a short snippet from `text`, centered near the first match."""
        if not text:
            return ""
        # Try to find the first matched (unigram) word in the raw text
        lower = text.lower()
        pos = -1
        for w in matched_words:
            if " " in w:
                # It's a bigram — search for it directly
                p = lower.find(w)
            else:
                # Whole-word search
                m = re.search(r"\b" + re.escape(w) + r"\b", lower)
                p = m.start() if m else -1
            if p != -1:
                pos = p
                break
        if pos == -1:
            snippet = text[:max_chars]
        else:
            start = max(0, pos - 60)
            snippet = text[start : start + max_chars]
            if start > 0:
                snippet = "..." + snippet
        # Collapse whitespace
        snippet = " ".join(snippet.split())
        if len(snippet) > max_chars + 10:
            snippet = snippet[: max_chars + 10] + "..."
        return snippet

    def _overlapping_terms(self, query: str, doc_idx: int, top_n: int = 6) -> list[str]:
        """Return the query terms that overlap with this doc, ranked by
        how much they contributed to the similarity score."""
        q_vec = self.vectorizer.transform([query])
        # Element-wise product of query vector and doc vector = each term's
        # contribution to the cosine dot product (before length normalization).
        doc_vec = self.doc_matrix[doc_idx]
        contrib = q_vec.multiply(doc_vec)
        # Build (term, weight) pairs
        # contrib is sparse; iterate over nonzero entries
        contrib = contrib.tocoo()
        term_by_col = {v: k for k, v in self.vocab.items()}
        pairs = []
        for col, val in zip(contrib.col, contrib.data):
            term = term_by_col.get(col)
            if term:
                pairs.append((term, float(val)))
        # Prefer bigrams first (they're more informative), then higher weight
        pairs.sort(key=lambda p: (" " in p[0], p[1]), reverse=True)
        seen = set()
        out = []
        for term, _w in pairs:
            if term in seen:
                continue
            seen.add(term)
            out.append(term)
            if len(out) >= top_n:
                break
        return out

    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 5,
        source_type: Optional[str] = None,
    ) -> list[Result]:
        """Return top_k results for `query`. Optionally filter by source_type."""
        query = (query or "").strip()
        if not query:
            return []

        q_vec = self.vectorizer.transform([query])
        # Cosine similarity between query and every doc
        sims = cosine_similarity(q_vec, self.doc_matrix).ravel()

        # Optionally filter by source_type by zeroing out other rows
        if source_type in ("paper", "repo"):
            mask = np.array(
                [e["source_type"] == source_type for e in self.entries]
            )
            sims = np.where(mask, sims, 0.0)

        # Take the top_k highest scores (that are > 0)
        order = np.argsort(-sims)
        results: list[Result] = []
        rank = 0
        for idx in order:
            score = float(sims[idx])
            if score <= 0:
                break
            rank += 1
            entry = self.entries[idx]
            matched = self._overlapping_terms(query, idx, top_n=6)
            snippet = self._snippet(entry["text"], matched)
            results.append(
                Result(
                    rank=rank,
                    score=score,
                    source_type=entry["source_type"],
                    title=entry.get("title", ""),
                    url=entry.get("url", ""),
                    snippet=snippet,
                    matched_words=matched,
                )
            )
            if rank >= top_k:
                break
        return results


    # ------------------------------------------------------------------
    # Semantic (dense embedding) search
    # ------------------------------------------------------------------
    def _ensure_semantic_ready(self) -> bool:
        """Load the precomputed embeddings + the model on first use.
        Returns True if the semantic index is available, False if either
        the embeddings file or sentence_transformers is missing."""
        if self._doc_embeddings is not None and self._semantic_model is not None:
            return True
        # Try to load embeddings
        if not EMBEDDINGS_PATH.exists():
            return False
        try:
            self._doc_embeddings = np.load(EMBEDDINGS_PATH)
        except Exception:
            return False
        # Try to load the model (lazy import so app boots even if the
        # library isn't installed)
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception:
            return False
        self._semantic_model = SentenceTransformer(SEMANTIC_MODEL_NAME)
        return True

    def semantic_available(self) -> bool:
        """Cheap check: do we have the embeddings file on disk?"""
        return EMBEDDINGS_PATH.exists()

    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        source_type: Optional[str] = None,
    ) -> list[Result]:
        """Rank documents by meaning-similarity to `query`."""
        query = (query or "").strip()
        if not query:
            return []
        if not self._ensure_semantic_ready():
            return []

        # Embed the query and cosine-sim against all docs
        q_vec = self._semantic_model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )[0].astype(np.float32)
        # doc_embeddings are already L2-normalized -> cosine == dot
        sims = self._doc_embeddings @ q_vec

        if source_type in ("paper", "repo"):
            mask = np.array([e["source_type"] == source_type for e in self.entries])
            sims = np.where(mask, sims, -1.0)

        order = np.argsort(-sims)
        results: list[Result] = []
        rank = 0
        # For "why it matched" on semantic hits we still surface any keyword
        # overlap between the query and the doc (best-effort). It might be
        # empty; the UI copes.
        for idx in order:
            score = float(sims[idx])
            if score <= 0:
                break
            rank += 1
            entry = self.entries[idx]
            matched = self._overlapping_terms(query, idx, top_n=6)
            snippet = self._snippet(entry["text"], matched or [w for w in query.lower().split()])
            results.append(
                Result(
                    rank=rank,
                    score=score,
                    source_type=entry["source_type"],
                    title=entry.get("title", ""),
                    url=entry.get("url", ""),
                    snippet=snippet,
                    matched_words=matched,
                )
            )
            if rank >= top_k:
                break
        return results

    # ------------------------------------------------------------------
    # Hybrid: run both, average the normalized scores
    # ------------------------------------------------------------------
    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        source_type: Optional[str] = None,
    ) -> list[Result]:
        """Combine TF-IDF and semantic rankings. Score = 0.5 * norm(tfidf) + 0.5 * norm(semantic)."""
        query = (query or "").strip()
        if not query:
            return []
        # Get both rankings over a wide window, then re-rank
        wide_k = max(top_k * 5, 25)
        tfidf_hits = {r.rank_index if hasattr(r, "rank_index") else i: r
                      for i, r in enumerate(self.search(query, top_k=wide_k, source_type=source_type))}
        # We need per-doc scores, not just top-K. Recompute cheaply.
        q_vec = self.vectorizer.transform([query])
        tfidf_scores = cosine_similarity(q_vec, self.doc_matrix).ravel()

        if self._ensure_semantic_ready():
            sem_q = self._semantic_model.encode(
                [query], normalize_embeddings=True, convert_to_numpy=True
            )[0].astype(np.float32)
            sem_scores = self._doc_embeddings @ sem_q
        else:
            sem_scores = np.zeros_like(tfidf_scores)

        # Normalize each to [0, 1] independently so they mix fairly
        def _norm(a: np.ndarray) -> np.ndarray:
            hi = float(a.max()) if len(a) else 0.0
            return (a / hi) if hi > 0 else a
        combined = 0.5 * _norm(tfidf_scores) + 0.5 * _norm(sem_scores)

        if source_type in ("paper", "repo"):
            mask = np.array([e["source_type"] == source_type for e in self.entries])
            combined = np.where(mask, combined, -1.0)

        order = np.argsort(-combined)
        results: list[Result] = []
        rank = 0
        for idx in order:
            score = float(combined[idx])
            if score <= 0:
                break
            rank += 1
            entry = self.entries[idx]
            matched = self._overlapping_terms(query, idx, top_n=6)
            snippet = self._snippet(entry["text"], matched or [w for w in query.lower().split()])
            results.append(
                Result(
                    rank=rank,
                    score=score,
                    source_type=entry["source_type"],
                    title=entry.get("title", ""),
                    url=entry.get("url", ""),
                    snippet=snippet,
                    matched_words=matched,
                )
            )
            if rank >= top_k:
                break
        return results


def build_index(entries: list[dict] | None = None) -> SearchIndex:
    if entries is None:
        entries = load_dataset()
    return SearchIndex(entries)


# ---------------------------------------------------------------------------
# Small CLI so we can smoke-test the search from the terminal
# ---------------------------------------------------------------------------
def _demo(query: str = "diffusion models for recommendation"):
    def _safe(t: str) -> str:
        return t.encode("ascii", errors="replace").decode("ascii")

    print("Loading dataset ...")
    entries = load_dataset()
    print(f"  {len(entries)} entries ({sum(1 for e in entries if e['source_type']=='paper')} papers, {sum(1 for e in entries if e['source_type']=='repo')} repos)")
    print("Building TF-IDF index ...")
    index = build_index(entries)
    print(f"  vocab size: {len(index.vocab):,}")
    print(f"\nQuery: {query!r}")
    results = index.search(query, top_k=5)
    if not results:
        print("(no matches)")
        return
    for r in results:
        print(f"\n  #{r.rank}  [{r.source_type}]  score={r.score:.4f}")
        print(f"    title:   {_safe(r.title)[:110]}")
        print(f"    url:     {_safe(r.url)}")
        print(f"    matched: {', '.join(r.matched_words)}")
        print(f"    snippet: {_safe(r.snippet)}")


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "diffusion models for recommendation"
    _demo(q)
