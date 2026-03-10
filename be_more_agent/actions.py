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


def launch_web_game():
    import os
    import subprocess
    import tempfile
    
    game_path = "https://html-classic.itch.zone/html/13484643/D:/BmoV1.1/index.html"
    
    webview_script = f"""
import webview
import time
import threading

def hide_taskbar_icon():
    for _ in range(10):
        time.sleep(0.5)
        try:
            import ctypes
            hwnd = ctypes.windll.user32.FindWindowW(None, 'BMO Game Mode')
            if hwnd:
                GWL_EXSTYLE = -20
                WS_EX_TOOLWINDOW = 0x00000080
                ex_style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
                ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex_style | WS_EX_TOOLWINDOW)
                break
        except Exception:
            pass

def inject_styles(window):
    while True:
        try:
            js = '''
            var checkCanvas = setInterval(function() {{
                var canvas = document.querySelector('canvas') || document.querySelector('#unity-canvas');
                var container = document.querySelector('#unity-container');
                
                if(canvas) {{
                    document.body.style.setProperty('margin', '0', 'important');
                    document.body.style.setProperty('overflow', 'hidden', 'important');
                    document.body.style.setProperty('background-color', '#92E6AE', 'important');
                    
                    if (container) {{
                        container.style.setProperty('background-color', '#92E6AE', 'important');
                        container.style.setProperty('width', '100vw', 'important');
                        container.style.setProperty('height', '100vh', 'important');
                    }}
                    
                    var footer = document.querySelector("#unity-footer");
                    if(footer) footer.style.display = "none";
                    var loading = document.querySelector("#unity-loading-bar");
                    if(loading) loading.style.display = "none";
                    
                    canvas.style.setProperty('position', 'absolute', 'important');
                    canvas.style.setProperty('width', '100vw', 'important');
                    canvas.style.setProperty('height', '100vh', 'important');
                    canvas.style.setProperty('left', '0', 'important');
                    canvas.style.setProperty('top', '0', 'important');
                    
                    canvas.style.setProperty('transform-origin', 'initial', 'important');
                    canvas.style.setProperty('transform', 'none', 'important');
                    
                    canvas.oncontextmenu = function(e){{ e.preventDefault(); }};
                    
                    if(!window._bmo_keys_bound) {{
                        window._bmo_keys_bound = true;
                        window._bmo_active_keys = {{}};
                        window._bmo_active_touches = {{}};
                        
                        const keyMap = {{
                            'ArrowUp': {{x: 394, y: 399, id: 1}}, 'w': {{x: 394, y: 399, id: 1}}, 'W': {{x: 394, y: 399, id: 1}},
                            'ArrowDown': {{x: 398, y: 493, id: 2}}, 's': {{x: 398, y: 493, id: 2}}, 'S': {{x: 398, y: 493, id: 2}},
                            'ArrowLeft': {{x: 355, y: 461, id: 3}}, 'a': {{x: 355, y: 461, id: 3}}, 'A': {{x: 355, y: 461, id: 3}},
                            'ArrowRight': {{x: 427, y: 461, id: 4}}, 'd': {{x: 427, y: 461, id: 4}}, 'D': {{x: 427, y: 461, id: 4}},
                            ' ': {{x: 537, y: 539, id: 5}}, 'Enter': {{x: 537, y: 539, id: 5}},  
                            'Escape': {{x: 533, y: 437, id: 6}}, 'Backspace': {{x: 533, y: 437, id: 6}}
                        }};
                        
                        window.addEventListener('keydown', function(e) {{
                            let target = keyMap[e.key];
                            if (target) {{
                                e.preventDefault();
                                
                                if (!window._bmo_active_keys[e.key]) {{
                                    window._bmo_active_keys[e.key] = true;
                                    try {{ e.stopPropagation(); }} catch(err) {{}} 
                                    
                                    let rect = canvas.getBoundingClientRect();
                                    let s = Math.min(rect.width / 960, rect.height / 600);
                                    let gx = (rect.width - (960 * s)) / 2;
                                    let gy = (rect.height - (600 * s)) / 2;
                                    
                                    let clickX = rect.left + gx + (target.x * s);
                                    let clickY = rect.top + gy + (target.y * s);
                                    
                                    let touch = new Touch({{
                                        identifier: target.id,
                                        target: canvas,
                                        clientX: clickX,
                                        clientY: clickY,
                                        screenX: clickX,
                                        screenY: clickY,
                                        pageX: clickX,
                                        pageY: clickY,
                                        radiusX: 5,
                                        radiusY: 5,
                                        force: 1
                                    }});
                                    
                                    window._bmo_active_touches[target.id] = touch;
                                    
                                    let touchEvent = new TouchEvent('touchstart', {{
                                        cancelable: true, bubbles: true,
                                        touches: Object.values(window._bmo_active_touches),
                                        targetTouches: Object.values(window._bmo_active_touches),
                                        changedTouches: [touch]
                                    }});
                                    
                                    canvas.dispatchEvent(touchEvent);
                                }}
                            }}
                        }}, true);
                        
                        window.addEventListener('keyup', function(e) {{
                            let target = keyMap[e.key];
                            if (target) {{
                                e.preventDefault();
                                
                                if (window._bmo_active_keys[e.key]) {{
                                    delete window._bmo_active_keys[e.key];
                                    try {{ e.stopPropagation(); }} catch(err) {{}} 
                                    
                                    let oldTouch = window._bmo_active_touches[target.id];
                                    delete window._bmo_active_touches[target.id];
                                    
                                    if (oldTouch) {{
                                        let touchEvent = new TouchEvent('touchend', {{
                                            cancelable: true, bubbles: true,
                                            touches: Object.values(window._bmo_active_touches),
                                            targetTouches: Object.values(window._bmo_active_touches),
                                            changedTouches: [oldTouch]
                                        }});
                                        
                                        canvas.dispatchEvent(touchEvent);
                                    }}
                                }}
                            }}
                        }}, true);
                    }}
                }}
            }}, 50);
            '''
            window.evaluate_js(js)
            time.sleep(1)
            break
        except:
            time.sleep(1)

threading.Thread(target=hide_taskbar_icon, daemon=True).start()
window = webview.create_window('BMO Game Mode', '{game_path}', x=0, y=0, width=800, height=480, frameless=True, background_color='#92E6AE')
webview.start(inject_styles, window)
"""
    runner_path = os.path.join(tempfile.gettempdir(), 'bmo_web_game.py')
    with open(runner_path, 'w') as f:
        f.write(webview_script)
        
    try:
        subprocess.Popen(["python", runner_path])
        return True
    except Exception as e:
        print(f"Error launching web game: {e}")
        return False

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
        return "LAUNCH_GAMES_MENU"

    return None
