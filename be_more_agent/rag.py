"""RAG retriever – loads the Adventure Time wiki vector DB and retrieves
relevant chunks for a query using cosine similarity + BM25 re-ranking.

Pipeline:
  1. Embed query → cosine similarity → top-K candidates (fast, broad)
  2. BM25 keyword re-rank candidates → top-N final results (precise)

Usage:
    from be_more_agent.rag import retrieve
    chunks = retrieve("What video games does BMO have?", top_k=3)
"""

import json
import math
import os
import re
from collections import Counter

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


# ── BM25 re-ranker (no model needed, pure Python) ──────────────────────

_STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could of in to for on with "
    "at by from and or but not so if then that this it its you your "
    "i me my we our they them their he she him her what which who whom "
    "how when where why all each every both few more most some any no".split()
)


def _tokenize(text: str) -> list[str]:
    """Lowercase, split, remove stop words."""
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP_WORDS]


def _bm25_score(query_tokens: list[str], doc_tokens: list[str],
                avg_dl: float, n_docs: int, df: dict[str, int],
                k1: float = 1.5, b: float = 0.75) -> float:
    """BM25 score for a single document against the query."""
    tf = Counter(doc_tokens)
    dl = len(doc_tokens)
    score = 0.0
    for qt in query_tokens:
        if qt not in tf:
            continue
        doc_freq = df.get(qt, 0)
        idf = math.log((n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
        term_freq = tf[qt]
        tf_norm = (term_freq * (k1 + 1)) / (term_freq + k1 * (1 - b + b * dl / avg_dl))
        score += idf * tf_norm
    return score


def _rerank_bm25(query: str, candidates: list[dict], top_n: int = 3) -> list[dict]:
    """Re-rank candidate chunks using BM25. Each candidate = {idx, score, text, title}."""
    query_tokens = _tokenize(query)
    if not query_tokens:
        return candidates[:top_n]

    # Tokenize all candidate docs
    doc_tokens_list = [_tokenize(c["text"]) for c in candidates]
    n_docs = len(candidates)
    avg_dl = sum(len(dt) for dt in doc_tokens_list) / max(n_docs, 1)

    # Document frequency
    df: dict[str, int] = {}
    for dt in doc_tokens_list:
        for w in set(dt):
            df[w] = df.get(w, 0) + 1

    # Score & sort
    for c, dt in zip(candidates, doc_tokens_list):
        c["bm25"] = _bm25_score(query_tokens, dt, avg_dl, n_docs, df)

    candidates.sort(key=lambda c: c["bm25"], reverse=True)
    return candidates[:top_n]


# ── Main retrieval ──────────────────────────────────────────────────────

# Retrieve more candidates for re-ranking, then keep top results
_CANDIDATE_K = 15   # broad vector search
_FINAL_K = 3        # after re-ranking
_SIM_THRESHOLD = 0.30  # minimum cosine similarity for candidates


def retrieve(query: str, top_k: int = _FINAL_K) -> list[str]:
    """Retrieve top-k chunks: vector search → BM25 re-rank → return best."""
    _load_db()
    if not _chunks or _embeddings is None or _embeddings.size == 0:
        return []

    # Step 1: Broad vector search — get top candidates
    resp = ollama.embed(model=_EMBED_MODEL, input=[query])
    q_emb = np.array(resp["embeddings"][0], dtype=np.float32)
    q_emb = q_emb / (np.linalg.norm(q_emb) or 1)

    scores = _embeddings @ q_emb
    top_indices = np.argsort(scores)[-_CANDIDATE_K:][::-1]

    candidates = []
    for idx in top_indices:
        sim = float(scores[idx])
        if sim < _SIM_THRESHOLD:
            continue
        candidates.append({
            "idx": int(idx),
            "sim": sim,
            "text": _chunks[idx]["text"],
            "title": _chunks[idx]["title"],
        })

    if not candidates:
        return []

    print(f"[RAG] {len(candidates)} candidates above {_SIM_THRESHOLD} threshold", flush=True)

    # Step 2: BM25 re-rank
    reranked = _rerank_bm25(query, candidates, top_n=top_k)

    results = []
    for c in reranked:
        if c["bm25"] <= 0:
            continue  # no keyword overlap at all — skip
        # Truncate for LLM context
        chunk_text = c["text"][:400]
        results.append(chunk_text)
        print(f"[RAG]   sim={c['sim']:.3f}  bm25={c['bm25']:.2f}  {c['title']}", flush=True)

    return results
