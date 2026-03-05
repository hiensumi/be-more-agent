"""RAG retriever – loads the Adventure Time wiki vector DB and retrieves
relevant chunks for a query using cosine similarity.

Usage:
    from be_more_agent.rag import retrieve
    chunks = retrieve("What video games does BMO have?", top_k=5)
    # chunks = ["[BMO] BMO has several games...", ...]
"""

import json
import os

import numpy as np
import ollama

_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "finetune", "at_wiki_db")
_CHUNKS_FILE = os.path.join(_DB_DIR, "chunks.json")
_EMBEDDINGS_FILE = os.path.join(_DB_DIR, "embeddings.npy")
_EMBED_MODEL = "all-minilm"

# Lazy-loaded globals
_chunks: list[dict] | None = None
_embeddings: np.ndarray | None = None


def _load_db():
    """Load the vector DB into memory (lazy, once)."""
    global _chunks, _embeddings
    if _chunks is not None:
        return

    if not os.path.exists(_CHUNKS_FILE) or not os.path.exists(_EMBEDDINGS_FILE):
        print("[RAG] Vector DB not found. Run finetune/5_build_vectordb.py first.", flush=True)
        _chunks = []
        _embeddings = np.array([])
        return

    with open(_CHUNKS_FILE) as f:
        _chunks = json.load(f)
    _embeddings = np.load(_EMBEDDINGS_FILE)

    # Normalise embeddings for fast cosine similarity via dot product
    norms = np.linalg.norm(_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    _embeddings = _embeddings / norms

    print(f"[RAG] Loaded {len(_chunks)} chunks, embedding shape {_embeddings.shape}", flush=True)


def retrieve(query: str, top_k: int = 5) -> list[str]:
    """Return the top-k most relevant wiki chunks for the given query."""
    _load_db()
    if not _chunks or _embeddings is None or _embeddings.size == 0:
        return []

    # Embed the query
    resp = ollama.embed(model=_EMBED_MODEL, input=[query])
    q_emb = np.array(resp["embeddings"][0], dtype=np.float32)
    q_emb = q_emb / (np.linalg.norm(q_emb) or 1)

    # Cosine similarity = dot product (both normalised)
    scores = _embeddings @ q_emb
    top_indices = np.argsort(scores)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        score = scores[idx]
        if score < 0.3:  # relevance threshold
            continue
        results.append(_chunks[idx]["text"])
        print(f"[RAG]   {score:.3f}  {_chunks[idx]['title']}", flush=True)

    return results
