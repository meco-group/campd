"""
Browser-based keypress relay for Meshcat visualisation.

Provides :class:`BrowserKeyServer`, a tiny HTTP server that serves a wrapper
page embedding the Meshcat viewer in an ``<iframe>`` with a dark control-bar.
Keyboard events in the browser are POSTed back to Python and placed into a
:class:`queue.Queue`.  An optional background thread also reads raw
keypresses from the terminal (stdin), so both input sources work
simultaneously.
"""

from __future__ import annotations

import json
import select
import sys
import termios
import tty
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from queue import Queue, Empty
from threading import Event, Thread


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """HTTPServer that handles each request in a new daemon thread."""
    daemon_threads = True


class BrowserKeyServer:
    """Tiny HTTP server that relays browser keypresses to Python.

    Serves a wrapper page that embeds the Meshcat viewer in an ``<iframe>``
    with a dark control-bar showing keyboard shortcuts.  ``keydown`` events
    on the bar are POSTed to ``/key`` and placed into a :class:`queue.Queue`
    that the main thread reads via :meth:`get_key`.

    A ``setInterval`` auto-refocus keeps keyboard focus on the control bar
    even after the user clicks in the 3-D viewport to rotate the camera.
    Three.js OrbitControls is mouse-only, so this does not interfere.

    If stdin is a real TTY a background thread also reads raw keypresses
    from the terminal, so both browser **and** terminal input work.
    """

    _BROWSER_KEY_MAP: dict[str, str] = {
        "ArrowLeft": "LEFT",
        "ArrowRight": "RIGHT",
        "ArrowUp": "UP",
        "ArrowDown": "DOWN",
        "Enter": "\n",
    }

    _TERMINAL_ESCAPE_MAP: dict[str, str] = {
        "[D": "LEFT",
        "[C": "RIGHT",
        "[A": "UP",
        "[B": "DOWN",
    }

    # fmt: off
    _WRAPPER_HTML = (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        '<title>CAMPD Viewer</title><style>'
        '*{margin:0;padding:0;box-sizing:border-box}'
        'html,body{height:100%;overflow:hidden;'
        "  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',monospace}"
        'body{display:flex;flex-direction:column}'
        '#bar{height:32px;min-height:32px;background:#1a1a2e;color:#a0a0b0;'
        '  display:flex;align-items:center;padding:0 14px;font-size:12px;'
        '  user-select:none;cursor:default;border-top:1px solid #2a2a4a}'
        '#bar:focus{outline:none;color:#d0d0e0}'
        '.k{display:inline-block;background:#2a2a4a;border:1px solid #3a3a5a;'
        '  border-radius:3px;padding:0 5px;margin:0 2px;'
        '  font-size:11px;line-height:18px}'
        '.sep{margin:0 8px;color:#3a3a5a}'
        '#st{margin-left:auto;font-size:11px}'
        '.on #st{color:#5c5}.off #st{color:#fa5}'
        'iframe{flex:1;border:none}'
        '</style></head><body>'
        '<iframe id="mcf" src="__MESHCAT_URL__"></iframe>'
        '<div id="bar" tabindex="0" class="on">'
        '  <span class="k">\u2190</span><span class="k">\u2192</span> step'
        '  <span class="sep">|</span>'
        '  <span class="k">Space</span> play/pause'
        '  <span class="sep">|</span>'
        '  <span class="k">R</span> replay'
        '  <span class="sep">|</span>'
        '  <span class="k">T</span> toggle trails'
        '  <span class="sep">|</span>'
        '  <span class="k">G</span> toggle goal'
        '  <span class="sep">|</span>'
        '  <span class="k">P</span> screenshot'
        '  <span class="sep">|</span>'
        '  <span class="k">V</span> video'
        '  <span class="sep">|</span>'
        '  <span class="k">Enter</span> next'
        '  <span id="st">\u2713 Keyboard active</span>'
        '</div>'
        '<script>'
        'var B=document.getElementById("bar");'
        'var ready=false;'
        'document.getElementById("mcf").onload=function(){'
        '  ready=true;B.focus();};'
        'function u(){'
        '  var s=document.getElementById("st");'
        '  if(document.activeElement===B){'
        '    B.className="on";s.innerHTML="\u2713 Keyboard active";}'
        '  else{B.className="off";'
        '    s.innerHTML="\u26a0 Click here for keyboard";}}'
        'B.onfocus=B.onblur=u;'
        'B.onkeydown=function(e){'
        '  if(["Shift","Control","Alt","Meta","CapsLock","Tab",'
        '      "Escape","NumLock","ScrollLock"].indexOf(e.key)>=0)return;'
        '  fetch("/key",{method:"POST",'
        '    body:JSON.stringify({key:e.key}),'
        '    headers:{"Content-Type":"application/json"}});'
        '  e.preventDefault();};'
        'setInterval(function(){'
        '  if(ready&&document.activeElement!==B)'
        '    B.focus({preventScroll:true});},150);'
        '</script></body></html>'
    )
    # fmt: on

    def __init__(self, meshcat_url: str) -> None:
        self._queue: Queue[str] = Queue()
        html_bytes = self._WRAPPER_HTML.replace(
            "__MESHCAT_URL__", meshcat_url
        ).encode()

        queue = self._queue  # closure reference for the handler

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path != '/':
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(html_bytes)))
                self.end_headers()
                self.wfile.write(html_bytes)

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                try:
                    key = json.loads(body).get("key", "")
                    if key:
                        queue.put(key)
                except (json.JSONDecodeError, AttributeError):
                    pass
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

            def do_OPTIONS(self):
                """Handle CORS preflight requests."""
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header(
                    "Access-Control-Allow-Methods", "POST, OPTIONS")
                self.send_header(
                    "Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args, **kwargs):
                pass  # silence per-request logs

        self._server = _ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._port = self._server.server_address[1]

        self._ready = Event()
        self._thread = Thread(target=self._serve, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)

        self._stop = False
        self._tty_thread: Thread | None = None

    def _serve(self) -> None:
        """Server thread target — signals readiness, then serves forever."""
        self._ready.set()
        self._server.serve_forever()

    def start_terminal_reader(self) -> None:
        """Begin listening for keypresses on stdin."""
        if self._tty_thread is not None:
            return  # already running
        if not sys.stdin.isatty():
            return
        self._tty_thread = Thread(
            target=self._terminal_reader, daemon=True,
        )
        self._tty_thread.start()

    @property
    def url(self) -> str:
        """URL of the wrapper page (open this instead of the raw Meshcat URL)."""
        return f"http://127.0.0.1:{self._port}"

    def get_key(self, timeout: float | None = None) -> str | None:
        """Block until a key arrives, or return *None* after *timeout* secs."""
        try:
            if timeout is not None:
                raw = self._queue.get(timeout=timeout)
            else:
                # Poll with short timeouts so Ctrl-C is not swallowed.
                while True:
                    try:
                        raw = self._queue.get(timeout=0.5)
                        break
                    except Empty:
                        continue
        except Empty:
            return None
        return self._BROWSER_KEY_MAP.get(raw, raw)

    def _terminal_reader(self) -> None:
        """Background thread: read raw keypresses from stdin into the queue."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self._stop:
                rlist, _, _ = select.select([sys.stdin], [], [], 0.2)
                if not rlist:
                    continue
                ch = sys.stdin.read(1)
                if ch == '\x1b':  # escape sequence (arrow keys)
                    ch2 = sys.stdin.read(1)
                    ch3 = sys.stdin.read(1)
                    mapped = self._TERMINAL_ESCAPE_MAP.get(ch2 + ch3)
                    if mapped:
                        self._queue.put(mapped)
                elif ch == '\r' or ch == '\n':
                    self._queue.put('\n')
                elif ch == '\x03':  # Ctrl-C
                    self._stop = True
                    break
                else:
                    self._queue.put(ch)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def shutdown(self) -> None:
        """Stop the HTTP server and terminal reader thread."""
        self._stop = True
        self._server.shutdown()
