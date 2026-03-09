import datetime

from tavily import TavilyClient

from .config import CURRENT_CONFIG


def _get_tavily_client():
    """Get a Tavily client, or None if no API key is configured."""
    api_key = CURRENT_CONFIG.get("tavily_api_key", "").strip()
    if not api_key:
        print("[SEARCH] No Tavily API key configured in config.json", flush=True)
        return None
    return TavilyClient(api_key=api_key)


def execute_action_and_get_result(action_data):
    raw_action = action_data.get("action", "").lower().strip()
    value = action_data.get("value") or action_data.get("query")

    valid_tools = {
        "get_time",
        "get_date",
        "search_web",
        "capture_image",
        "open_game",
    }

    aliases = {
        "google": "search_web",
        "browser": "search_web",
        "news": "search_web",
        "search_news": "search_web",
        "look": "capture_image",
        "see": "capture_image",
        "check_time": "get_time",
        "check_date": "get_date",
        "date": "get_date",
        "today": "get_date",
        "play_game": "open_game",
        "launch_game": "open_game",
        "open_games": "open_game",
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

    if action == "get_date":
        today = datetime.datetime.now().strftime("%A, %B %d, %Y")
        return f"Today's date is {today}."

    if action == "search_web":
        print(f"Searching web for: {value}...", flush=True)
        client = _get_tavily_client()
        if not client:
            return "SEARCH_ERROR"
        try:
            response = client.search(
                query=value,
                search_depth="basic",
                max_results=3,
            )
            results = response.get("results", [])
            if results:
                parts = []
                for r in results:
                    title = r.get("title", "No Title")
                    content = r.get("content", "No Content")
                    parts.append(f"Title: {title}\nSnippet: {content[:300]}")
                return f"SEARCH RESULTS for '{value}':\n" + "\n---\n".join(parts)

            print("[DEBUG] Search returned 0 results.", flush=True)
            return "SEARCH_EMPTY"
        except Exception as e:
            print(f"[DEBUG] Tavily Search Error: {e}", flush=True)
            return "SEARCH_ERROR"

    if action == "capture_image":
        return "IMAGE_CAPTURE_TRIGGERED"

    if action == "open_game":
        game_path = CURRENT_CONFIG.get("game_path", "")
        if not game_path:
            return "BMO cannot find the game file. Please add 'game_path' to config.json!"
        
        import os
        import subprocess
        if not os.path.exists(game_path):
            return f"BMO couldn't find the game at: {game_path}. Please check the path!"
            
        try:
            # Popen spawns the process non-blocking
            subprocess.Popen([game_path])
            return "Opening the game now! Let's have fun playing!"
        except Exception as e:
            return f"Whoops, tried to open it but got an error: {e}"

    return None
