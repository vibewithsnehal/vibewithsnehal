"""Use a phone as the camera — no apps, just the browser.

``courtvision live phone`` starts :class:`PhoneIngestServer` (HTTPS, because
mobile browsers only allow camera access on secure pages; a self-signed
certificate is generated on first run — accept the browser warning, the
connection stays on your own network).  Open the printed URL on the phone:

  1. Start the camera (rear lens, landscape).
  2. Tap the four doubles-court corners to calibrate — or skip and let
     automatic court detection try.
  3. Start streaming.  The phone posts JPEG frames to the laptop, which runs
     the live analyzer; the phone screen turns into a scoreboard showing
     every call as it's made.

Endpoints:
  - ``GET  /``            the capture page
  - ``POST /ingest``      one JPEG frame per request (latest-wins queue)
  - ``POST /corners``     tapped corner pixels -> calibration
  - ``GET  /stats.json``  cumulative stats (the phone scoreboard polls this)
"""

from __future__ import annotations

import json
import socket
import ssl
import subprocess
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

PHONE_PAGE = """<!doctype html>
<html><head><title>CourtVision phone camera</title>
<meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no" />
<style>
 body { background:#0a1628; color:#e2e8f0; font-family:system-ui,sans-serif;
        margin:0; padding:12px; display:flex; flex-direction:column; gap:10px; }
 h2 { margin:4px 0; font-size:18px; }
 video, canvas { width:100%; border-radius:8px; background:#000; }
 button { background:#38bdf8; color:#0a1628; border:0; border-radius:8px;
          padding:12px; font-size:16px; font-weight:600; }
 button.secondary { background:#334155; color:#e2e8f0; }
 button:disabled { opacity:.4; }
 #hint { color:#94a3b8; font-size:14px; }
 #board { font-size:15px; background:#111f35; border-radius:8px; padding:10px 12px;
          white-space:pre-line; min-height:3em; }
 .call-in  { color:#4ade80; font-weight:700; }
 .call-out { color:#f87171; font-weight:700; }
</style></head>
<body>
<h2>🎾 CourtVision — phone camera</h2>
<div id="hint">Mount the phone steady (fence/tripod), landscape, elevated behind
the baseline, whole court in frame. Then:</div>
<video id="v" autoplay playsinline muted style="display:none"></video>
<canvas id="c"></canvas>
<button id="btnCam">1 · Start camera</button>
<button id="btnCal" disabled>2 · Tap the 4 court corners</button>
<button id="btnSkip" class="secondary" disabled>2b · Skip — auto-detect court</button>
<button id="btnGo" disabled>3 · Start streaming</button>
<div id="board">waiting…</div>
<script>
const v = document.getElementById('v'), c = document.getElementById('c'),
      ctx = c.getContext('2d'), board = document.getElementById('board');
const btnCam = document.getElementById('btnCam'), btnCal = document.getElementById('btnCal'),
      btnSkip = document.getElementById('btnSkip'), btnGo = document.getElementById('btnGo');
let corners = [], calibrating = false, streaming = false;
const LABELS = ['NEAR-LEFT', 'NEAR-RIGHT', 'FAR-RIGHT', 'FAR-LEFT'];

function drawPreview() {
  if (v.videoWidth) {
    c.width = v.videoWidth; c.height = v.videoHeight;
    if (!calibrating) ctx.drawImage(v, 0, 0);
    for (let i = 0; i < corners.length; i++) {
      const [x, y] = corners[i];
      ctx.strokeStyle = '#38bdf8'; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.arc(x, y, 14, 0, 7); ctx.stroke();
      ctx.fillStyle = '#38bdf8'; ctx.font = 'bold 20px system-ui';
      ctx.fillText(LABELS[i], x + 18, y);
    }
  }
  if (!streaming) requestAnimationFrame(drawPreview);
}

btnCam.onclick = async () => {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: 'environment', width: {ideal: 1280}, height: {ideal: 720} },
    audio: false });
  v.srcObject = stream; await v.play();
  btnCam.disabled = true; btnCal.disabled = false; btnSkip.disabled = false;
  board.textContent = 'Camera on. Frame the whole court, then calibrate.';
  drawPreview();
};

btnCal.onclick = () => {
  calibrating = true; corners = [];
  ctx.drawImage(v, 0, 0);   // freeze the frame for accurate tapping
  board.textContent = 'Tap: ' + LABELS.join(' → ') +
    '\\n(the OUTER corners of the outermost — doubles — lines)';
};

c.onclick = async (e) => {
  if (!calibrating || corners.length >= 4) return;
  const r = c.getBoundingClientRect();
  const x = (e.clientX - r.left) * (c.width / r.width);
  const y = (e.clientY - r.top) * (c.height / r.height);
  corners.push([x, y]);
  if (corners.length === 4) {
    await fetch('/corners', { method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({corners: corners}) });
    calibrating = false; btnGo.disabled = false; btnCal.textContent = '✓ calibrated (redo?)';
    board.textContent = 'Calibrated. Start streaming when ready.';
  } else {
    board.textContent = 'Tap: ' + LABELS[corners.length];
  }
};

btnSkip.onclick = () => {
  corners = []; calibrating = false; btnGo.disabled = false;
  board.textContent = 'Auto-detection it is - calls start once the court is found.';
};

async function pump() {
  if (!streaming) return;
  const blob = await new Promise(res => c.toBlob(res, 'image/jpeg', 0.75));
  try {
    await fetch('/ingest', { method: 'POST',
      headers: {'Content-Type': 'image/jpeg'}, body: blob });
  } catch (err) { /* transient network hiccup: skip the frame */ }
  setTimeout(pump, 20);  // paced by the round-trip; ~15-25 fps on LAN
}

function drawStream() {
  if (!streaming) return;
  ctx.drawImage(v, 0, 0);
  requestAnimationFrame(drawStream);
}

async function scoreboard() {
  if (!streaming) return;
  try {
    const s = await (await fetch('/stats.json')).json();
    const sm = s.summary || {};
    const calls = s.calls || [];
    const last = calls[calls.length - 1];
    let txt = `rallies ${sm.rallies ?? 0} · calls ${sm.total_bounces_called ?? 0} ` +
              `(IN ${sm.in ?? 0} / OUT ${sm.out ?? 0}) · close ${sm.close_calls ?? 0}`;
    board.innerHTML = last
      ? `<span class="call-${last.decision.toLowerCase()}">${last.decision}</span>` +
        ` ${last.margin_cm > 0 ? '+' : ''}${last.margin_cm} cm ` +
        `(${Math.round(last.confidence * 100)}%)<br/>` + txt
      : txt;
  } catch (err) {}
  setTimeout(scoreboard, 1000);
}

btnGo.onclick = () => {
  streaming = true;
  btnGo.disabled = true; btnCal.disabled = true; btnSkip.disabled = true;
  board.textContent = 'Streaming - calls will appear here.';
  drawStream(); pump(); scoreboard();
};
</script>
</body></html>"""


def lan_ip() -> str:
    """Best-effort LAN address for the printed URL."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def ensure_self_signed_cert(cert_dir: Path) -> tuple[Path, Path] | None:
    """Create (or reuse) a self-signed TLS cert.  None if openssl is missing."""
    cert_dir.mkdir(parents=True, exist_ok=True)
    cert, key = cert_dir / "cert.pem", cert_dir / "key.pem"
    if cert.exists() and key.exists():
        return cert, key
    try:
        subprocess.run(
            [
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", str(key), "-out", str(cert),
                "-days", "365", "-nodes",
                "-subj", "/CN=courtvision.local",
                "-addext", f"subjectAltName=IP:{lan_ip()},DNS:localhost",
            ],
            check=True,
            capture_output=True,
        )
        return cert, key
    except (OSError, subprocess.CalledProcessError):
        return None


class PhoneIngestServer:
    """Receives camera frames and corner taps from the phone page.

    Frames land in a small latest-wins queue read by the analysis loop via
    :meth:`next_frame`.  Tapped corners are exposed via :meth:`take_corners`.
    ``stats_supplier`` feeds the phone's scoreboard.
    """

    def __init__(
        self,
        port: int = 9443,
        cert_dir: Path | None = None,
        stats_supplier: Callable[[], str] | None = None,
        queue_size: int = 4,
    ) -> None:
        self._frames: deque[np.ndarray] = deque(maxlen=queue_size)
        self._cond = threading.Condition()
        self._corners: list[list[float]] | None = None
        self._stats_supplier = stats_supplier or (lambda: "{}")
        self.frames_received = 0
        self.tls = False

        server_self = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args) -> None:
                pass

            def _reply(self, code: int, body: bytes, ctype: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if self.path == "/":
                    self._reply(200, PHONE_PAGE.encode(), "text/html; charset=utf-8")
                elif self.path == "/stats.json":
                    self._reply(
                        200, server_self._stats_supplier().encode(), "application/json"
                    )
                else:
                    self._reply(404, b"not found", "text/plain")

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length) if length else b""
                if self.path == "/ingest":
                    frame = cv2.imdecode(
                        np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    if frame is not None:
                        with server_self._cond:
                            server_self._frames.append(frame)
                            server_self.frames_received += 1
                            server_self._cond.notify_all()
                    self._reply(200, b"ok", "text/plain")
                elif self.path == "/corners":
                    try:
                        data = json.loads(body)
                        pts = data["corners"]
                        assert len(pts) == 4
                        with server_self._cond:
                            server_self._corners = [[float(x), float(y)] for x, y in pts]
                        self._reply(200, b"ok", "text/plain")
                    except (KeyError, ValueError, AssertionError, TypeError):
                        self._reply(400, b"bad corners", "text/plain")
                else:
                    self._reply(404, b"not found", "text/plain")

        self._httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        cert_pair = ensure_self_signed_cert(cert_dir or Path("courtvision-live"))
        if cert_pair is not None:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(certfile=str(cert_pair[0]), keyfile=str(cert_pair[1]))
            self._httpd.socket = ctx.wrap_socket(self._httpd.socket, server_side=True)
            self.tls = True
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        scheme = "https" if self.tls else "http"
        return f"{scheme}://{lan_ip()}:{self.port}/"

    def start(self) -> None:
        self._thread.start()

    def next_frame(self, timeout: float = 1.0) -> np.ndarray | None:
        """Oldest queued frame, or None on timeout.  Overflow drops old frames."""
        with self._cond:
            if not self._frames:
                self._cond.wait(timeout)
            if not self._frames:
                return None
            return self._frames.popleft()

    def take_corners(self) -> list[list[float]] | None:
        """Return freshly tapped corners once, then clear them."""
        with self._cond:
            corners, self._corners = self._corners, None
            return corners

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
