#!/usr/bin/env python3
"""Step 2: Clean scraped wiki data and format into QLoRA training JSONL.

Reads raw_wiki_pages.json and produces training data in two formats:
  - Instruction/response pairs (Q&A style) for chat fine-tuning
  - Knowledge completion pairs for factual grounding

Output: finetune/train_data.jsonl
"""

import json
import os
import random
import re
import textwrap

INPUT_FILE = os.path.join(os.path.dirname(__file__), "raw_wiki_pages.json")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "train_data.jsonl")

# Qwen 2.5 chat template uses ChatML format
SYSTEM_MSG = (
    "You are BMO (Be More), a cute sentient video game console robot from Adventure Time. "
    "You live in the Tree Fort with Finn and Jake. You are helpful, cheerful, and knowledgeable "
    "about the Land of Ooo and everything in Adventure Time. Answer questions accurately based "
    "on your knowledge. Speak in short, enthusiastic sentences."
)

# Question templates for generating instruction pairs
QUESTION_TEMPLATES = {
    "character": [
        "Who is {title}?",
        "Tell me about {title}.",
        "What do you know about {title}?",
        "BMO, who is {title}?",
        "Can you describe {title}?",
        "What is {title} like?",
    ],
    "episode": [
        "What happens in the episode {title}?",
        "Tell me about the episode {title}.",
        "What is the episode {title} about?",
        "BMO, what happens in {title}?",
        "Summarize the episode {title}.",
    ],
    "location": [
        "Where is {title}?",
        "Tell me about {title}.",
        "What is {title}?",
        "BMO, what do you know about {title}?",
        "Describe {title}.",
    ],
    "general": [
        "What is {title}?",
        "Tell me about {title}.",
        "What do you know about {title}?",
        "BMO, tell me about {title}.",
        "Can you explain {title}?",
    ],
}

# Keywords to classify page type
CHARACTER_KEYWORDS = [
    "is a character", "is a human", "is a dog", "is a vampire",
    "is a princess", "is a wizard", "is a demon", "is a robot",
    "is a candy", "is the ruler", "is the king", "is the queen",
    "is a recurring", "is a minor", "is a major",
    "voiced by", "first appears in", "is one of the main",
]

EPISODE_KEYWORDS = [
    "is the ", "episode of", "season ", "written by", "storyboarded by",
    "aired on", "episode of adventure time", "the episode begins",
    "episode features",
]

LOCATION_KEYWORDS = [
    "is a location", "is a kingdom", "is a dimension", "is a realm",
    "is a city", "is a land", "is located in", "is found in",
    "is a place", "is the home of",
]


def classify_page(title: str, content: str) -> str:
    """Classify page type based on content keywords."""
    lower = content[:500].lower()

    for kw in CHARACTER_KEYWORDS:
        if kw in lower:
            return "character"

    for kw in EPISODE_KEYWORDS:
        if kw in lower:
            return "episode"

    for kw in LOCATION_KEYWORDS:
        if kw in lower:
            return "location"

    return "general"


def extract_sections(content: str) -> dict[str, str]:
    """Split wiki content into sections by headings."""
    sections = {}
    current_heading = "Introduction"
    current_text = []

    for line in content.split("\n"):
        # Match section headings (== Heading ==, === Subheading ===)
        heading_match = re.match(r"^(={2,})\s*(.+?)\s*\1$", line)
        if heading_match:
            # Save previous section
            text = "\n".join(current_text).strip()
            if text:
                sections[current_heading] = text
            current_heading = heading_match.group(2)
            current_text = []
        else:
            current_text.append(line)

    # Save last section
    text = "\n".join(current_text).strip()
    if text:
        sections[current_heading] = text

    return sections


def clean_content(text: str) -> str:
    """Deep clean wiki text for training."""
    # Remove image/file references
    text = re.sub(r"\[\[File:.*?\]\]", "", text)
    text = re.sub(r"\[\[Image:.*?\]\]", "", text)
    # Remove category tags
    text = re.sub(r"\[\[Category:.*?\]\]", "", text)
    # Clean wiki links [[Page|Display]] → Display, [[Page]] → Page
    text = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    # Remove templates {{ }}
    text = re.sub(r"\{\{[^}]*\}\}", "", text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove reference tags
    text = re.sub(r"\[ref\].*?\[/ref\]", "", text, flags=re.DOTALL)
    text = re.sub(r"\[\d+\]", "", text)
    # Remove excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    # Remove "See also", "References", "External links" etc. sections
    text = re.sub(
        r"\n(See also|References|External links|Gallery|Trivia list|Navigation)[\s\S]*$",
        "", text, flags=re.IGNORECASE
    )

    return text.strip()


def truncate_for_response(text: str, max_chars: int = 1500) -> str:
    """Truncate text to a reasonable response length."""
    if len(text) <= max_chars:
        return text

    # Try to cut at a sentence boundary
    truncated = text[:max_chars]
    last_period = truncated.rfind(".")
    if last_period > max_chars // 2:
        return truncated[: last_period + 1]
    return truncated + "..."


def make_bmo_response(text: str) -> str:
    """Lightly reformat text to sound more like BMO speaking."""
    # Don't rewrite — just clean up and ensure it reads well
    text = text.strip()
    if not text:
        return ""
    return text


def generate_qa_pairs(title: str, content: str, page_type: str) -> list[dict]:
    """Generate instruction/response training pairs from a wiki page."""
    pairs = []
    sections = extract_sections(content)
    cleaned_content = clean_content(content)

    if not cleaned_content or len(cleaned_content) < 50:
        return []

    # 1. Main Q&A about the page
    templates = QUESTION_TEMPLATES.get(page_type, QUESTION_TEMPLATES["general"])
    question = random.choice(templates).format(title=title)

    # Use intro section or first ~1500 chars
    intro = sections.get("Introduction", "")
    if not intro:
        intro = cleaned_content

    response = truncate_for_response(clean_content(intro))
    if response:
        pairs.append({
            "instruction": question,
            "response": make_bmo_response(response),
        })

    # 2. Section-specific Q&A pairs
    section_questions = {
        "Personality": [
            f"What is {title}'s personality like?",
            f"How would you describe {title}'s personality?",
        ],
        "Appearance": [
            f"What does {title} look like?",
            f"Describe {title}'s appearance.",
        ],
        "Abilities": [
            f"What are {title}'s abilities?",
            f"What powers does {title} have?",
        ],
        "Relationships": [
            f"Who are {title}'s friends?",
            f"Tell me about {title}'s relationships.",
        ],
        "History": [
            f"What is {title}'s backstory?",
            f"Tell me the history of {title}.",
        ],
        "Synopsis": [
            f"What happens in {title}?",
            f"Summarize {title}.",
        ],
        "Plot": [
            f"What is the plot of {title}?",
            f"What happens in the episode {title}?",
        ],
        "Geography": [
            f"Describe the geography of {title}.",
            f"What is {title} like geographically?",
        ],
    }

    for section_name, questions in section_questions.items():
        if section_name in sections:
            section_text = clean_content(sections[section_name])
            if len(section_text) > 50:
                q = random.choice(questions)
                r = truncate_for_response(section_text)
                pairs.append({
                    "instruction": q,
                    "response": make_bmo_response(r),
                })

    # 3. Knowledge completion — feed a chunk for the model to absorb
    if len(cleaned_content) > 200:
        # Split into chunks for knowledge absorption
        chunks = textwrap.wrap(cleaned_content, width=2000, break_long_words=False, break_on_hyphens=False)
        for i, chunk in enumerate(chunks[:3]):  # Max 3 chunks per page
            if len(chunk) > 100:
                pairs.append({
                    "instruction": f"Tell me everything you know about {title}." if i == 0
                    else f"Continue telling me about {title}.",
                    "response": make_bmo_response(chunk),
                })

    return pairs


def format_chatml(instruction: str, response: str) -> dict:
    """Format a pair into ChatML conversation format for Qwen."""
    return {
        "conversations": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
    }


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] Input file not found: {INPUT_FILE}")
        print("  Run 1_scrape_wiki.py first!")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        pages = json.load(f)

    print(f"=== Processing {len(pages)} wiki pages ===\n")

    all_pairs = []
    for page in pages:
        title = page["title"]
        content = page["content"]
        page_type = classify_page(title, content)
        pairs = generate_qa_pairs(title, content, page_type)
        all_pairs.extend(pairs)

    # Shuffle for training
    random.seed(42)
    random.shuffle(all_pairs)

    # Format into ChatML and write JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            entry = format_chatml(pair["instruction"], pair["response"])
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"=== Data formatting complete ===")
    print(f"  Total training examples: {len(all_pairs)}")
    total_chars = sum(len(p["instruction"]) + len(p["response"]) for p in all_pairs)
    print(f"  Total text: {total_chars:,} characters (~{total_chars // 4:,} tokens)")
    print(f"  Saved to: {OUTPUT_FILE}")

    # Print sample
    print(f"\n=== Sample entries ===")
    for pair in all_pairs[:3]:
        print(f"\n  Q: {pair['instruction'][:80]}")
        print(f"  A: {pair['response'][:120]}...")


if __name__ == "__main__":
    main()
