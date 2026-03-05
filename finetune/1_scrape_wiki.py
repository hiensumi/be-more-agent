#!/usr/bin/env python3
"""Step 1: Scrape Adventure Time Wiki using the MediaWiki API.

Uses the Fandom/MediaWiki API to fetch all article pages from key categories.
Extracts plain text content (stripping wiki markup) and saves to raw JSON.

Output: finetune/raw_wiki_pages.json
"""

import json
import os
import re
import time

import mwparserfromhell
import requests

BASE_URL = "https://adventuretime.fandom.com/api.php"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "BMOBot/1.0 (Adventure Time wiki scraper for local LLM fine-tuning)"})
MAX_RETRIES = 3
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "raw_wiki_pages.json")

# Categories to scrape — covers the most important content
CATEGORIES = [
    "Characters",
    "Episodes",
    "Locations",
    "Objects",
    "Songs",
    "Species",
    "Transcripts",
    "Lore",
    "Kingdoms",
    "Candy_People",
    "Ice_Kingdom",
    "Fire_Kingdom",
    "Distant_Lands",
    "Fionna_and_Cake",
    "Princesses",
    "Villains",
    "Heroes",
    "Wizards",
    "Elementals",
    "Demons",
]

# Also fetch these specific important pages directly
EXTRA_PAGES = [
    "Adventure_Time",
    "BMO",
    "Finn",
    "Jake",
    "Princess_Bubblegum",
    "Marceline",
    "Ice_King",
    "Lumpy_Space_Princess",
    "Lady_Rainicorn",
    "Flame_Princess",
    "The_Lich_(character)",
    "Candy_Kingdom",
    "Tree_Fort",
    "Land_of_Ooo",
    "Mushroom_War",
    "Ice_Crown",
    "Enchiridion",
    "Gunter",
    "Peppermint_Butler",
    "Tree_Trunks",
    "Cinnamon_Bun",
    "Hunson_Abadeer",
    "Death",
    "Prismo",
    "Cosmic_Owl",
    "NEPTR",
    "Shelby",
    "Choose_Goose",
    "Magic_Man",
    "Fern",
    "Minerva_Campbell",
    "Martin_Mertens",
    "Simon_Petrikov",
    "Betty_Grof",
    "Fionna",
    "Cake",
    "Marshall_Lee",
    "Ice_Queen",
    "Prince_Gumball",
    "Lumpy_Space_Prince",
    "Football_(character)",
    "Banana_Guards",
    "Lemongrab",
    "Flame_King",
    "Orgalorg",
    "GOLB",
    "Normal_Man",
    "Shermy",
    "Beth",
    "List_of_episodes",
    "Adventure_Time:_Distant_Lands",
    "Adventure_Time:_Fionna_%26_Cake",
    "The_Nightosphere",
    "Fire_Kingdom",
    "Lumpy_Space",
    "Crystal_Dimension",
    "Dead_World",
    "Moe_Giovanni",
    "AMO",
    "ALLMO",
]

# Rate limiting
REQUEST_DELAY = 0.3  # seconds between requests


def get_category_members(category: str, limit: int = 500) -> list[str]:
    """Fetch all page titles in a category (with continuation)."""
    titles = []
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmlimit": str(min(limit, 500)),
        "cmtype": "page",
        "format": "json",
    }

    while True:
        time.sleep(REQUEST_DELAY)
        try:
            resp = SESSION.get(BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  [ERROR] Failed to fetch category '{category}': {e}")
            break

        members = data.get("query", {}).get("categorymembers", [])
        for m in members:
            titles.append(m["title"])

        # Handle pagination
        cont = data.get("continue")
        if cont and "cmcontinue" in cont:
            params["cmcontinue"] = cont["cmcontinue"]
        else:
            break

    return titles


def get_page_content(title: str) -> str | None:
    """Fetch wikitext via action=parse and convert to plaintext."""
    params = {
        "action": "parse",
        "page": title,
        "prop": "wikitext",
        "format": "json",
    }

    for attempt in range(MAX_RETRIES):
        time.sleep(REQUEST_DELAY)
        try:
            resp = SESSION.get(BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = (attempt + 1) * 2
                print(f"  [RETRY] '{title}' attempt {attempt + 1} failed: {e}, waiting {wait}s")
                time.sleep(wait)
            else:
                print(f"  [ERROR] Failed to fetch page '{title}' after {MAX_RETRIES} attempts: {e}")
                return None

    if "error" in data:
        return None

    wikitext = data.get("parse", {}).get("wikitext", {}).get("*", "")
    if not wikitext:
        return None

    # Parse wikitext to plaintext
    try:
        parsed = mwparserfromhell.parse(wikitext)
        text = parsed.strip_code()
    except Exception:
        # Fallback: basic regex stripping
        text = re.sub(r"\{\{[^}]*\}\}", "", wikitext)
        text = re.sub(r"\[\[[^|\]]+\|([^\]]+)\]\]", r"\1", text)
        text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
        text = re.sub(r"<[^>]+>", "", text)

    return clean_extract(text)


def clean_extract(text: str) -> str:
    """Deep cleanup of extracted text."""
    if not text:
        return ""

    # Remove excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove common wiki artifacts
    text = re.sub(r"\[edit\]", "", text)
    text = re.sub(r"↑", "", text)
    # Remove reference markers like [1], [2], etc.
    text = re.sub(r"\[\d+\]", "", text)
    # Remove leftover template/table markers
    text = re.sub(r"\{\|[^}]*\|\}", "", text, flags=re.DOTALL)
    text = re.sub(r"^\|.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^!.*$", "", text, flags=re.MULTILINE)
    # Remove image/file references that survived
    text = re.sub(r"thumb\|.*?\n", "\n", text)
    text = re.sub(r"\d+px\|?", "", text)
    # Clean up spacing
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def main():
    all_titles = set()

    # 1. Gather page titles from categories
    print("=== Gathering page titles from categories ===")
    for cat in CATEGORIES:
        titles = get_category_members(cat)
        print(f"  Category:{cat} → {len(titles)} pages")
        all_titles.update(titles)

    # 2. Add extra important pages
    for page in EXTRA_PAGES:
        all_titles.add(page.replace("_", " "))

    # Filter out non-article pages (User:, Talk:, File:, etc.)
    all_titles = {
        t for t in all_titles
        if not any(t.startswith(ns) for ns in [
            "User:", "Talk:", "File:", "Template:", "Category:",
            "MediaWiki:", "Module:", "User talk:", "Adventure Time Wiki:",
        ])
        and "/Transcript" not in t  # Skip transcript pages (too long, not useful for Q&A)
    }

    print(f"\n=== Total unique pages to fetch: {len(all_titles)} ===\n")

    # 3. Load any previously scraped pages (for resume support)
    pages = []
    scraped_titles = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                pages = json.load(f)
            scraped_titles = {p["title"] for p in pages}
            print(f"  [RESUME] Found {len(pages)} previously scraped pages")
        except Exception:
            pass

    remaining = sorted(all_titles - scraped_titles)
    print(f"  Remaining to fetch: {len(remaining)}\n")

    # 4. Fetch content for each page
    failed = 0
    for i, title in enumerate(remaining, 1):
        if i % 50 == 0 or i == 1:
            print(f"  Fetching page {i}/{len(remaining)}... ({len(pages)} scraped so far)")

        content = get_page_content(title)
        if content and len(content) > 100:
            pages.append({
                "title": title,
                "content": content,
                "char_count": len(content),
            })
        else:
            failed += 1

        # Save progress every 100 pages
        if i % 100 == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(pages, f, ensure_ascii=False, indent=2)
            print(f"  [SAVE] Progress saved ({len(pages)} pages)")

    print(f"\n=== Scraping complete ===")
    print(f"  Scraped: {len(pages)} pages")
    print(f"  Skipped/Failed: {failed} pages")
    total_chars = sum(p["char_count"] for p in pages)
    print(f"  Total text: {total_chars:,} characters (~{total_chars // 4:,} tokens)")

    # 4. Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)
    print(f"  Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
