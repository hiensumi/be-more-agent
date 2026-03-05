"""Lightweight HTTP server for sending text to BMO from a browser.

Runs in a daemon thread so it doesn't block the main tkinter loop.
No external dependencies — uses only the Python stdlib.

Endpoints
---------
GET  /          – simple HTML form
POST /send      – accepts ``text`` form field, queues it for the bot
GET  /health    – JSON liveness check
"""

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs

from .config import WEB_PORT

# Populated by start_web_server()
_bot = None

# ---- HTML served on GET / ---------------------------------------------------

_INDEX_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BMO – Text Input</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: system-ui, sans-serif;
    background: #1e1e2e;
    color: #cdd6f4;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
  }
  .card {
    background: #313244;
    border-radius: 16px;
    padding: 2rem;
    width: min(90vw, 480px);
    box-shadow: 0 8px 32px rgba(0,0,0,.4);
  }
  h1 { text-align: center; margin-bottom: 1rem; font-size: 1.6rem; }
  textarea {
    width: 100%;
    min-height: 100px;
    border: 2px solid #585b70;
    border-radius: 8px;
    background: #1e1e2e;
    color: #cdd6f4;
    padding: .75rem;
    font-size: 1rem;
    resize: vertical;
  }
  textarea:focus { outline: none; border-color: #89b4fa; }
  button {
    margin-top: .75rem;
    width: 100%;
    padding: .75rem;
    border: none;
    border-radius: 8px;
    background: #89b4fa;
    color: #1e1e2e;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: background .2s;
  }
  button:hover { background: #74c7ec; }
  #status {
    text-align: center;
    margin-top: .75rem;
    min-height: 1.4em;
    color: #a6e3a1;
  }
</style>
</head>
<body>
<div class="card">
  <h1>BMO</h1>
  <form id="f">
    <textarea id="msg" name="text" placeholder="Type a message for BMO..."
              autofocus></textarea>
    <button type="submit">Send</button>
  </form>
  <p id="status"></p>
</div>
<script>
  const form = document.getElementById('f');
  const msg  = document.getElementById('msg');
  const stat = document.getElementById('status');
  form.addEventListener('submit', async e => {
    e.preventDefault();
    const text = msg.value.trim();
    if (!text) return;
    stat.textContent = 'Sending...';
    try {
      const res = await fetch('/send', {
        method: 'POST',
        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        body: 'text=' + encodeURIComponent(text),
      });
      const data = await res.json();
      stat.textContent = data.ok ? 'Sent!' : ('Error: ' + data.error);
      if (data.ok) msg.value = '';
    } catch (err) {
      stat.textContent = 'Network error';
    }
  });
</script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    """Minimal request handler."""

    def log_message(self, fmt, *args):
        # Keep console clean — only print to stdout
        print(f"[WEB] {args[0]}", flush=True)

    # -- GET -------------------------------------------------------------------

    def do_GET(self):
        if self.path == "/health":
            self._json_response({"status": "ok"})
        else:
            self._html_response(_INDEX_HTML)

    # -- POST ------------------------------------------------------------------

    def do_POST(self):
        if self.path != "/send":
            self._json_response({"ok": False, "error": "not found"}, code=404)
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode()

        # Support both form-encoded and JSON bodies
        text = ""
        ct = self.headers.get("Content-Type", "")
        if "json" in ct:
            try:
                text = json.loads(body).get("text", "")
            except Exception:
                pass
        else:
            parsed = parse_qs(body)
            text = parsed.get("text", [""])[0]

        text = text.strip()
        if not text:
            self._json_response({"ok": False, "error": "empty text"}, code=400)
            return

        if _bot is None:
            self._json_response({"ok": False, "error": "bot not ready"}, code=503)
            return

        with _bot.web_text_lock:
            _bot.web_text_queue.append(text)
        _bot.web_text_event.set()

        print(f"[WEB] Queued text: {text!r}", flush=True)
        self._json_response({"ok": True})

    # -- helpers ---------------------------------------------------------------

    def _json_response(self, obj, code=200):
        payload = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _html_response(self, html, code=200):
        payload = html.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def start_web_server(bot):
    """Launch the HTTP server on a daemon thread.

    Parameters
    ----------
    bot : BotGUI
        The running bot instance (must have ``web_text_queue`` and
        ``web_text_lock`` attributes).
    """
    global _bot
    _bot = bot

    server = HTTPServer(("0.0.0.0", WEB_PORT), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"[WEB] Listening on http://0.0.0.0:{WEB_PORT}", flush=True)
