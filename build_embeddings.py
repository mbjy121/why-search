"""
build_embeddings.py

Runs once. Loads data/combined.json, feeds each entry's text through a
small sentence-transformers model, and saves the resulting embeddings
to data/embeddings.npy so search.py can load them instantly at runtime.

Model: all-MiniLM-L6-v2  (~90 MB, 384-dim vectors, downloaded on first run
from Hugging Face into your local cache).

Semantic search complements the TF-IDF keyword search in search.py:
where TF-IDF matches on exact overlapping words, this model matches on
meaning — so "chatbot" can find "conversational agent" and so on.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
IN_PATH = DATA_DIR / "combined.json"
OUT_PATH = DATA_DIR / "embeddings.npy"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_CHARS = 2000  # trim each doc to keep encoding fast


def main():
    if not IN_PATH.exists():
        raise SystemExit(f"Missing {IN_PATH}. Run build_dataset.py first.")

    print(f"Loading {IN_PATH.name} ...")
    entries = json.loads(IN_PATH.read_text(encoding="utf-8"))
    texts = [(e.get("text") or "")[:MAX_CHARS] for e in entries]
    print(f"  {len(entries):,} documents to embed (trimmed to {MAX_CHARS} chars each)")

    print(f"Loading model {MODEL_NAME} ...")
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME)
    print(f"  model ready in {time.time()-t0:.1f}s")

    print("Encoding ...")
    t0 = time.time()
    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so cosine sim == dot product
    )
    print(f"  encoded in {time.time()-t0:.1f}s, shape={vectors.shape}, dtype={vectors.dtype}")

    # Save as float32 to keep the file small
    vectors = vectors.astype(np.float32)
    np.save(OUT_PATH, vectors)
    size_mb = OUT_PATH.stat().st_size / 1024 / 1024
    print(f"Saved -> {OUT_PATH}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
