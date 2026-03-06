"""Chat mixin – conversation logic, tool dispatch, memory, image capture."""

import json
import os
import re
import subprocess
import threading

import ollama
from PIL import Image

from .actions import execute_action_and_get_result
from .classifier import classify_input
from .rag import retrieve as rag_retrieve
from .config import (
    BMO_IMAGE_FILE,
    CURRENT_CONFIG,
    MEMORY_FILE,
    OLLAMA_OPTIONS,
    STATIC_MEMORY_FILE,
    TEXT_MODEL,
    VISION_MODEL,
)
from .prompts import SYSTEM_PROMPT
from .states import BotStates


class ChatMixin:
    """Mixed into BotGUI – provides chat / LLM / memory methods."""

    # -- Helpers ---------------------------------------------------------------

    def extract_json_from_text(self, text):
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return None
        except Exception:
            return None

    # -- Camera ----------------------------------------------------------------

    def capture_image(self):
        self.set_state(BotStates.CAPTURING, "Watching...")
        try:
            subprocess.run(
                [
                    "rpicam-still",
                    "-t",
                    "500",
                    "-n",
                    "--width",
                    "640",
                    "--height",
                    "480",
                    "-o",
                    BMO_IMAGE_FILE,
                ],
                check=True,
            )
            rotation = CURRENT_CONFIG.get("camera_rotation", 0)
            if rotation != 0:
                img = Image.open(BMO_IMAGE_FILE)
                img = img.rotate(rotation, expand=True)
                img.save(BMO_IMAGE_FILE)
            return BMO_IMAGE_FILE
        except Exception as e:
            print(f"Camera Error: {e}")
            return None

    # -- Main conversation loop ------------------------------------------------

    def chat_and_respond(self, text, img_path=None):
        # Memory-wipe shortcut
        if "forget everything" in text.lower() or "reset memory" in text.lower():
            self.session_memory = []
            self.permanent_memory = [{"role": "system", "content": SYSTEM_PROMPT}]
            self.save_chat_history()
            with self.tts_queue_lock:
                self.tts_queue.append("Okay. Memory wiped.")
            self.set_state(BotStates.IDLE, "Memory Wiped")
            return

        model_to_use = VISION_MODEL if img_path else TEXT_MODEL
        self.set_state(BotStates.THINKING, "Thinking...", cam_path=img_path)

        # ── Auto-search for factual questions ────────────────────────
        if not img_path:
            input_class = classify_input(text)
            print(f"[CLASSIFIER] '{text[:60]}' → {input_class}", flush=True)

            if input_class == "search":
                search_ctx = self._auto_search(text)
                if search_ctx:
                    # Truncate context to keep input short for fast inference
                    search_ctx = search_ctx[:800]
                    text_with_ctx = (
                        f"{text}\n\n"
                        f"SEARCH RESULTS (answer briefly using this info):\n{search_ctx}"
                    )
                    user_msg = {"role": "user", "content": text_with_ctx}
                else:
                    user_msg = {"role": "user", "content": text}
            else:
                user_msg = {"role": "user", "content": text}

        if img_path:
            messages = [{"role": "user", "content": text, "images": [img_path]}]
        else:
            messages = self.permanent_memory + self.session_memory + [user_msg]

        self.thinking_sound_active.set()
        threading.Thread(target=self._run_thinking_sound_loop, daemon=True).start()

        full_response_buffer = ""
        sentence_buffer = ""

        try:
            stream = ollama.chat(
                model=model_to_use,
                messages=messages,
                stream=True,
                options=OLLAMA_OPTIONS,
            )

            is_action_mode = False

            for chunk in stream:
                if self.interrupted.is_set():
                    break
                content = chunk["message"]["content"]
                full_response_buffer += content

                if '{"' in content or "action:" in content.lower():
                    is_action_mode = True
                    self.thinking_sound_active.clear()
                    continue

                if is_action_mode:
                    continue

                self.thinking_sound_active.clear()
                if self.current_state != BotStates.SPEAKING:
                    self.set_state(BotStates.SPEAKING, "Speaking...", cam_path=img_path)
                    self.append_to_text("BOT: ", newline=False)

                self._stream_to_text(content)

                sentence_buffer += content
                if any(punct in content for punct in ".!?\n"):
                    clean_sentence = sentence_buffer.strip()
                    if clean_sentence and re.search(r"[a-zA-Z0-9]", clean_sentence):
                        with self.tts_queue_lock:
                            self.tts_queue.append(clean_sentence)
                    sentence_buffer = ""

            if is_action_mode:
                self._handle_action(full_response_buffer, text, model_to_use, img_path)
            else:
                self.append_to_text("")
                self.session_memory.append({"role": "assistant", "content": full_response_buffer})

            self.wait_for_tts()
            self.set_state(BotStates.IDLE, "Ready")

        except Exception as e:
            print(f"LLM Error: {e}")
            self.set_state(BotStates.ERROR, "Brain Freeze!")

    # -- Action / tool-call handling -------------------------------------------

    def _handle_action(self, full_response_buffer, text, model_to_use, img_path):
        action_data = self.extract_json_from_text(full_response_buffer)
        if not action_data:
            return

        tool_result = execute_action_and_get_result(action_data)

        if tool_result and tool_result.startswith("CHAT_FALLBACK::"):
            chat_text = tool_result.split("::", 1)[1]
            self.thinking_sound_active.clear()
            self.set_state(BotStates.SPEAKING, "Speaking...", cam_path=img_path)
            self.append_to_text("BOT: ", newline=False)
            self.append_to_text(chat_text, newline=True)
            with self.tts_queue_lock:
                self.tts_queue.append(chat_text)
            self.session_memory.append({"role": "assistant", "content": chat_text})
            self.wait_for_tts()
            self.set_state(BotStates.IDLE, "Ready")
            return

        if tool_result == "IMAGE_CAPTURE_TRIGGERED":
            new_img_path = self.capture_image()
            if new_img_path:
                self.chat_and_respond(text, img_path=new_img_path)
            return

        quick_replies = {
            "INVALID_ACTION": "I am not sure how to do that.",
            "SEARCH_EMPTY": "I searched, but I couldn't find any news about that.",
            "SEARCH_ERROR": "I cannot reach the internet right now.",
        }

        if tool_result in quick_replies:
            fallback_text = quick_replies[tool_result]
            self.thinking_sound_active.clear()
            self.set_state(BotStates.SPEAKING, "Speaking...", cam_path=img_path)
            self.append_to_text("BOT: ", newline=False)
            self.append_to_text(fallback_text, newline=True)
            with self.tts_queue_lock:
                self.tts_queue.append(fallback_text)
        elif tool_result:
            self._summarize_and_speak(tool_result, text, model_to_use, img_path)

    # -- Auto-search helper ----------------------------------------------------

    def _extract_search_queries(self, user_message: str) -> list[str] | None:
        """Use the LLM to extract search queries, or return None if the model can answer itself.

        The LLM decides in a single call whether to search or rely on its
        fine-tuned Adventure Time knowledge. Returns:
          - list[str]: search queries for web search (only non-AT parts)
          - None: ALL questions are about Adventure Time (no search needed)
        """
        prompt = [
            {"role": "system", "content": (
                "You extract web search queries from a user's message.\n"
                "The user's message may contain multiple questions. Handle EACH question independently.\n\n"
                "For each question, output one line:\n"
                "- If the question is SPECIFICALLY about the TV show Adventure Time "
                "(its characters like Finn, Jake, BMO, Marceline, Princess Bubblegum, Ice King, etc., "
                "its episodes, locations like Land of Ooo, Candy Kingdom, etc., or its plot/lore), "
                "output: AT\n"
                "- For ANYTHING ELSE (real-world topics, products, brands, people, places, science, "
                "technology, current events, history, or anything you're unsure about), "
                "output a concise web search query.\n\n"
                "When in doubt, output a search query. Only output AT if you are CERTAIN it's about the Adventure Time show.\n"
                "Remove greetings and filler. Output ONLY 'AT' or search queries, one per line, nothing else.\n\n"
                "Examples:\n"
                "User: Tell me about Sapuwa water bottle and who is Finn?\n"
                "Sapuwa water bottle\nAT\n\n"
                "User: What is Zalo?\n"
                "Zalo\n\n"
                "User: Who is Marceline and what's the weather today?\n"
                "AT\nweather today\n\n"
                "User: Who created BMO?\n"
                "AT"
            )},
            {"role": "user", "content": user_message},
        ]
        try:
            resp = ollama.chat(
                model=TEXT_MODEL,
                messages=prompt,
                stream=False,
                options={"temperature": 0, "num_predict": 100},
            )
            raw = resp["message"]["content"].strip()
            lines = [l.strip().lstrip("0123456789.-) ") for l in raw.splitlines()]
            lines = [l for l in lines if l]

            # Separate AT lines from search queries
            queries = [l for l in lines if len(l) > 2 and l.upper() != "AT"]
            at_count = sum(1 for l in lines if l.upper() == "AT")

            if at_count:
                print(f"[AUTO-SEARCH] {at_count} Adventure Time topic(s) → fine-tuned model", flush=True)
            if queries:
                print(f"[AUTO-SEARCH] {len(queries)} topic(s) need search", flush=True)
                return queries[:3]
            if at_count and not queries:
                print(f"[AUTO-SEARCH] All Adventure Time → skip search", flush=True)
                return None
        except Exception as e:
            print(f"[AUTO-SEARCH] Query extraction failed: {e}", flush=True)

        # Fallback: use the raw message
        return [user_message.strip()]

    def _auto_search(self, query: str) -> str | None:
        """Try RAG first (fast, no LLM call), then fall back to web search."""
        from ddgs import DDGS
        print(f"[AUTO-SEARCH] Raw: {query}", flush=True)

        # Step 1: Try RAG retrieval first (just an embedding lookup, ~1-2s)
        chunks = rag_retrieve(query, top_k=5)
        if chunks:
            print(f"[AUTO-SEARCH] RAG found {len(chunks)} relevant chunks", flush=True)
            return "\n---\n".join(chunks)

        # Step 2: No RAG results → web search
        print("[AUTO-SEARCH] RAG found nothing → web search", flush=True)
        sub_queries = [query.strip()]

        for i, sq in enumerate(sub_queries):
            print(f"[AUTO-SEARCH] Query {i+1}: {sq}", flush=True)

        all_parts = []
        try:
            with DDGS() as ddgs:
                for sq in sub_queries:
                    results = list(ddgs.text(sq, region="us-en", max_results=2))
                    if not results:
                        results = list(ddgs.news(sq, region="us-en", max_results=2))
                    for r in results:
                        title = r.get("title", "")
                        body = r.get("body", r.get("snippet", ""))
                        all_parts.append(f"- {title}: {body[:200]}")
        except Exception as e:
            print(f"[AUTO-SEARCH] Error: {e}", flush=True)

        return "\n".join(all_parts) if all_parts else None

    def _summarize_and_speak(self, tool_result, user_text, model_to_use, img_path):
        summary_prompt = [
            {"role": "system", "content": "Summarize this result in one short sentence."},
            {"role": "user", "content": f"RESULT: {tool_result}\nUser Question: {user_text}"},
        ]

        self.set_state(BotStates.THINKING, "Reading...")
        self.thinking_sound_active.set()

        final_resp = ollama.chat(
            model=model_to_use,
            messages=summary_prompt,
            stream=False,
            options=OLLAMA_OPTIONS,
        )
        final_text = final_resp["message"]["content"]

        self.thinking_sound_active.clear()
        self.set_state(BotStates.SPEAKING, "Speaking...", cam_path=img_path)

        self.append_to_text("BOT: ", newline=False)
        self.append_to_text(final_text, newline=True)
        with self.tts_queue_lock:
            self.tts_queue.append(final_text)
        self.session_memory.append({"role": "assistant", "content": final_text})

    # -- Persistent chat memory ------------------------------------------------

    def load_static_memory(self):
        """Load static knowledge that is prepended to every conversation."""
        if os.path.exists(STATIC_MEMORY_FILE):
            try:
                with open(STATIC_MEMORY_FILE, "r") as f:
                    entries = json.load(f)
                if isinstance(entries, list):
                    print(f"[MEMORY] Loaded {len(entries)} static memory entries.", flush=True)
                    return entries
            except Exception as e:
                print(f"[MEMORY] Static memory load error: {e}", flush=True)
        return []

    def load_chat_history(self):
        system_msg = [{"role": "system", "content": SYSTEM_PROMPT}]
        static = self.load_static_memory()
        base = system_msg + static

        if CURRENT_CONFIG.get("chat_memory", True) and os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r") as f:
                    saved = json.load(f)
                # Strip the old system message from saved data
                conv = [m for m in saved if m.get("role") != "system"]
                print(f"[MEMORY] Restored {len(conv)} saved messages.", flush=True)
                return base + conv
            except Exception:
                pass
        return base

    def save_chat_history(self):
        if not CURRENT_CONFIG.get("chat_memory", True):
            print("[MEMORY] Auto-save disabled.", flush=True)
            return
        full = self.permanent_memory + self.session_memory
        # Keep only non-system, non-static conversation turns
        conv = [m for m in full if m.get("role") != "system"]
        # Strip static memory entries (they reload from file)
        static_contents = {m.get("content") for m in self.load_static_memory()}
        conv = [m for m in conv if m.get("content") not in static_contents]
        if len(conv) > 10:
            conv = conv[-10:]
        with open(MEMORY_FILE, "w") as f:
            json.dump(conv, f, indent=4)
        print(f"[MEMORY] Saved {len(conv)} messages.", flush=True)
