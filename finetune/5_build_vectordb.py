#!/usr/bin/env python3
"""Step 5: Build a vector database from scraped Adventure Time wiki pages.

Reads raw_wiki_pages.json, chunks the text, embeds via Ollama's all-minilm,
and stores as numpy arrays + JSON index for fast RAG retrieval.

Output:
  finetune/at_wiki_db/embeddings.npy   - float32 matrix (N x dim)
  finetune/at_wiki_db/chunks.json      - list of {title, text} per chunk

No external DB dependency - just numpy + cosine similarity.
"""

import json
import os
import re

import numpy as np
import ollama

WIKI_FILE = os.path.join(os.path.dirname(__file__), "raw_wiki_pages.json")
DB_DIR = os.path.join(os.path.dirname(__file__), "at_wiki_db")
CHUNKS_FILE = os.path.join(DB_DIR, "chunks.json")
EMBEDDINGS_FILE = os.path.join(DB_DIR, "embeddings.npy")
EMBED_MODEL = "all-minilm"

# Chunking parameters
CHUNK_SIZE = 150       # target words per chunk (fits all-minilm 256-token window)
CHUNK_OVERLAP = 20     # overlap words between consecutive chunks
MIN_CHUNK_CHARS = 80   # skip tiny chunks


def chunk_text(title: str, text: str) -> list[dict]:
    """Split a wiki page into overlapping chunks, prefixed with the page title."""
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if len(text) < MIN_CHUNK_CHARS:
        return []

    paragraphs = text.split("\n\n")
    chunks = []
    current_words = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        words = para.split()

        if len(words) > CHUNK_SIZE:
            # Flush current buffer
            if current_words:
                chunk_str = " ".join(current_words)
                if len(chunk_str) >= MIN_CHUNK_CHARS:
                    chunks.append(chunk_str)
                current_words = []
                current_len = 0
            # Split large paragraph
            for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
                piece = " ".join(words[i : i + CHUNK_SIZE])
                if len(piece) >= MIN_CHUNK_CHARS:
                    chunks.append(piece)
        elif current_len + len(words) > CHUNK_SIZE:
            # Flush
            chunk_str = " ".join(current_words)
            if len(chunk_str) >= MIN_CHUNK_CHARS:
                chunks.append(chunk_str)
            overlap = current_words[-CHUNK_OVERLAP:] if len(current_words) > CHUNK_OVERLAP else []
            current_words = overlap + words
            current_len = len(current_words)
        else:
            current_words.extend(words)
            current_len += len(words)

    if current_words:
        chunk_str = " ".join(current_words)
        if len(chunk_str) >= MIN_CHUNK_CHARS:
            chunks.append(chunk_str)

    return [{"title": title, "text": f"[{title}] {c}"} for c in chunks]


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using Ollama."""
    resp = ollama.embed(model=EMBED_MODEL, input=texts)
    return resp["embeddings"]


def main():
    print(f"Loading wiki pages from {WIKI_FILE}...")
    with open(WIKI_FILE) as f:
        pages = json.load(f)
    print(f"  {len(pages)} pages loaded.")

    print("Chunking pages...")
    all_chunks = []
    for page in pages:
        all_chunks.extend(chunk_text(page["title"], page["content"]))
    print(f"  {len(all_chunks)} chunks created.")

    os.makedirs(DB_DIR, exist_ok=True)

    # Embed in batches
    BATCH_SIZE = 64
    total = len(all_chunks)
    all_embeddings = []
    failed = 0

    for i in range(0, total, BATCH_SIZE):
        batch = all_chunks[i : i + BATCH_SIZE]
        # Truncate to fit all-minilm 256-token context (~500 chars)
        texts = [c["text"][:500] for c in batch]
        try:
            embeddings = embed_batch(texts)
            all_embeddings.extend(embeddings)
        except Exception as e:
            # Fall back to one-at-a-time for this batch
            print(f"  Batch {i} failed ({e}), retrying individually...", flush=True)
            for j, t in enumerate(texts):
                try:
                    emb = embed_batch([t[:300]])  # extra truncation
                    all_embeddings.extend(emb)
                except Exception:
                    # Use zero vector as placeholder
                    dim = len(all_embeddings[0]) if all_embeddings else 384
                    all_embeddings.append([0.0] * dim)
                    failed += 1

        done = min(i + BATCH_SIZE, total)
        print(f"  Embedded {done}/{total} chunks ({done * 100 // total}%)", flush=True)

    if failed:
        print(f"  Warning: {failed} chunks failed to embed (zero vectors).", flush=True)

    # Save embeddings as numpy array
    emb_matrix = np.array(all_embeddings, dtype=np.float32)
    np.save(EMBEDDINGS_FILE, emb_matrix)
    print(f"  Embeddings saved: {EMBEDDINGS_FILE} -- shape {emb_matrix.shape}")

    # Save chunks metadata
    with open(CHUNKS_FILE, "w") as f:
        json.dump(all_chunks, f, indent=2)
    print(f"  Chunks saved: {CHUNKS_FILE}")

    print(f"\nDone! Vector DB at: {DB_DIR}")
    print(f"  {len(all_chunks)} chunks, embedding dim = {emb_matrix.shape[1]}")


if __name__ == "__main__":
    main()
