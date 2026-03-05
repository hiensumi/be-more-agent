import datetime

from ddgs import DDGS


def execute_action_and_get_result(action_data):
    raw_action = action_data.get("action", "").lower().strip()
    value = action_data.get("value") or action_data.get("query")

    valid_tools = {
        "get_time",
        "search_web",
        "capture_image",
    }

    aliases = {
        "google": "search_web",
        "browser": "search_web",
        "news": "search_web",
        "search_news": "search_web",
        "look": "capture_image",
        "see": "capture_image",
        "check_time": "get_time",
    }

    action = aliases.get(raw_action, raw_action)
    print(f"ACTION: {raw_action} -> {action}", flush=True)

    if action not in valid_tools:
        if value and isinstance(value, str) and len(value.split()) > 1:
            return f"CHAT_FALLBACK::{value}"
        return "INVALID_ACTION"

    if action == "get_time":
        now = datetime.datetime.now().strftime("%I:%M %p")
        return f"The current time is {now}."

    if action == "search_web":
        print(f"Searching web for: {value}...", flush=True)
        try:
            with DDGS() as ddgs:
                results = []
                try:
                    results = list(ddgs.text(value, region="us-en", max_results=3))
                    if results:
                        print(f"[DEBUG] Found Text: {results[0].get('title')}", flush=True)
                except Exception as e:
                    print(f"[DEBUG] Text Search Error: {e}", flush=True)

                if not results:
                    print("[DEBUG] No text results, trying news search...", flush=True)
                    try:
                        results = list(ddgs.news(value, region="us-en", max_results=3))
                        if results:
                            print(f"[DEBUG] Found News: {results[0].get('title')}", flush=True)
                    except Exception as e:
                        print(f"[DEBUG] News Search Error: {e}", flush=True)

                if results:
                    parts = []
                    for r in results:
                        title = r.get("title", "No Title")
                        body = r.get("body", r.get("snippet", "No Body"))
                        parts.append(f"Title: {title}\nSnippet: {body[:300]}")
                    return f"SEARCH RESULTS for '{value}':\n" + "\n---\n".join(parts)

                print("[DEBUG] Search returned 0 results.", flush=True)
                return "SEARCH_EMPTY"
        except Exception as e:
            print(f"[DEBUG] Connection/Library Error: {e}", flush=True)
            return "SEARCH_ERROR"

    if action == "capture_image":
        return "IMAGE_CAPTURE_TRIGGERED"

    return None
