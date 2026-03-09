#!/usr/bin/env python3
"""Step 5: Build a vector database from scraped Adventure Time wiki pages.

Uses SEMANTIC CHUNKING — splits pages by section headers so each chunk
covers one coherent topic (Appearance, Abilities, Games, etc.).

Output:
  finetune/at_wiki_db/embeddings.npy   - float32 matrix (N x dim)
  finetune/at_wiki_db/chunks.json      - list of {title, section, text} per chunk

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

# Limits — tuned for all-minilm 256-token context window
MAX_SECTION_WORDS = 200   # if a section is longer, sub-split it
MIN_CHUNK_CHARS = 80      # skip tiny sections
OVERLAP_WORDS = 15        # overlap when sub-splitting long sections


def _looks_like_header(line: str, next_line: str | None) -> bool:
    """Heuristic: a header is a short line (< 60 chars) starting with uppercase,
    followed by a longer content line. Filters out false positives like single
    sentences by requiring the next line to be substantially longer."""
    stripped = line.strip()
    if not stripped or len(stripped) >= 60:
        return False
    if not re.match(r'^[A-Z0-9]', stripped):
        return False
    if stripped.startswith(('(', '[', '"', "'", '-', '*')):
        return False
        
    # Reject lines ending in sentence punctuation (likely just a short sentence)
    if stripped.endswith(('.', ':', ',', ';', '"', "'", ')', '?', '!')):
        return False

    # Next line should be content (longer than header)
    if next_line:
        return len(next_line.strip()) > len(stripped)
    return False


def _split_into_sections(title: str, text: str) -> list[dict]:
    """Split a wiki page by section headers.  Returns [{section, text}, ...]."""
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    lines = text.split("\n")

    sections: list[dict] = []
    current_section = "Introduction"
    current_lines: list[str] = []

    for i, line in enumerate(lines):
        next_line = lines[i + 1] if i + 1 < len(lines) else None
        if _looks_like_header(line, next_line):
            # Flush previous section
            body = "\n".join(current_lines).strip()
            if len(body) >= MIN_CHUNK_CHARS:
                sections.append({"section": current_section, "text": body})
            current_section = line.strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Flush last section
    body = "\n".join(current_lines).strip()
    if len(body) >= MIN_CHUNK_CHARS:
        sections.append({"section": current_section, "text": body})

    return sections


def _sub_split_long(text: str) -> list[str]:
    """If a section is too long for the embedding model, split into overlapping pieces."""
    words = text.split()
    if len(words) <= MAX_SECTION_WORDS:
        return [text]

    pieces = []
    for i in range(0, len(words), MAX_SECTION_WORDS - OVERLAP_WORDS):
        piece = " ".join(words[i : i + MAX_SECTION_WORDS])
        if len(piece) >= MIN_CHUNK_CHARS:
            pieces.append(piece)
    return pieces


# Sections smaller than this are merged with their neighbour
MERGE_THRESHOLD = 250  # chars


def _merge_small_sections(sections: list[dict]) -> list[dict]:
    """Merge consecutive small sections so each chunk covers enough context.
    Stop merging once the combined text exceeds the merge cap."""
    if not sections:
        return sections

    MERGE_CAP = 800  # don't merge if combined would exceed this

    merged: list[dict] = [sections[0]]
    for sec in sections[1:]:
        prev = merged[-1]
        combined_len = len(prev["text"]) + len(sec["text"])

        # Merge if previous is small and combined doesn't exceed cap
        if len(prev["text"]) < MERGE_THRESHOLD and combined_len <= MERGE_CAP:
            prev["section"] += f" / {sec['section']}"
            prev["text"] += f"\n\n{sec['section']}\n{sec['text']}"
        # Merge if current is small and combined doesn't exceed cap
        elif len(sec["text"]) < MERGE_THRESHOLD and combined_len <= MERGE_CAP:
            prev["section"] += f" / {sec['section']}"
            prev["text"] += f"\n\n{sec['section']}\n{sec['text']}"
        else:
            merged.append(sec)
    return merged


def chunk_text(title: str, text: str) -> list[dict]:
    """Semantically chunk a wiki page: split by headers, merge small ones, sub-split big ones."""
    sections = _split_into_sections(title, text)
    sections = _merge_small_sections(sections)
    chunks = []

    for sec in sections:
        sub_parts = _sub_split_long(sec["text"])
        for part in sub_parts:
            chunk_str = f"[{title} — {sec['section']}] {part}"
            chunks.append({
                "title": f"{title} — {sec['section']}",
                "text": chunk_str,
            })

    # If no sections found (very short page), use the whole text
    if not chunks and len(text.strip()) >= MIN_CHUNK_CHARS:
        chunks.append({
            "title": title,
            "text": f"[{title}] {text.strip()[:MAX_SECTION_WORDS * 6]}",
        })

    return chunks


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
