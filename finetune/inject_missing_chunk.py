import json
import os
import numpy as np
import ollama

def main():
    WIKI_FILE = os.path.join(os.path.dirname(__file__), "raw_wiki_pages.json")
    DB_DIR = os.path.join(os.path.dirname(__file__), "at_wiki_db")
    CHUNKS_FILE = os.path.join(DB_DIR, "chunks.json")
    EMBEDDINGS_FILE = os.path.join(DB_DIR, "embeddings.npy")
    EMBED_MODEL = "all-minilm"

    # Step 1: Load Raw Wiki
    print(f"Loading raw wiki from {WIKI_FILE}...")
    with open(WIKI_FILE, "r", encoding="utf-8") as f:
        pages = json.load(f)

    # Step 2: Extract Playable Games section
    bmo_page = next((p for p in pages if p['title'] == 'BMO'), None)
    if not bmo_page:
        print("Error: Could not find BMO page.")
        return

    content = bmo_page['content']
    start_idx = content.find("Playable Games")
    if start_idx == -1:
        print("Error: Could not find 'Playable Games' section in BMO's raw text.")
        return

    # Extract up until the next section ("Relationships")
    end_idx = content.find("Relationships", start_idx)
    raw_games_text = content[start_idx:end_idx].strip() if end_idx != -1 else content[start_idx:start_idx+1000].strip()

    # Step 3: Format the Chunk
    half = len(raw_games_text) // 2
    part1 = raw_games_text[:half]
    part2 = raw_games_text[half:]

    chunks_to_add = [
        {"title": "BMO — Playable Games Part 1", "text": f"[BMO — Playable Games Part 1]\n{part1}"},
        {"title": "BMO — Playable Games Part 2", "text": f"[BMO — Playable Games Part 2]\n{part2}"}
    ]

    print("Loading existing database...")
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        existing_chunks = json.load(f)
    embeddings = np.load(EMBEDDINGS_FILE)

    client = ollama.Client(host="http://localhost:7676")
    added = 0

    for c in chunks_to_add:
        if any(exc["title"] == c["title"] for exc in existing_chunks):
            print(f"Chunk '{c['title']}' already exists! Skipping.")
            continue
            
        print(f"Embedding {c['title']}...")
        resp = client.embed(model=EMBED_MODEL, input=[c["text"][:500]]) # ensure it doesn't crash on length
        embedding = np.array(resp["embeddings"][0], dtype=np.float32)
        
        existing_chunks.append(c)
        embeddings = np.vstack([embeddings, embedding])
        added += 1

    if added > 0:
        with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
            json.dump(existing_chunks, f, indent=2)
        np.save(EMBEDDINGS_FILE, embeddings)
        print(f"Successfully injected {added} chunks. New DB size: {len(existing_chunks)} chunks, matrix shape: {embeddings.shape}.")

if __name__ == "__main__":
    main()
