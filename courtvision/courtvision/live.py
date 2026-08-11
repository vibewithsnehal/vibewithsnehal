"""Live analysis: frame-in, events-out, while the match is being played.

:class:`LiveAnalyzer` runs the same detection/tracking/bounce/call machinery
as the batch pipeline, but incrementally — feed it one frame at a time and it
emits events as they happen:

  - ``calibrated``  — the court has been found (calls can start)
  - ``rally_start`` — ball in play
  - ``call``        — a bounce ruled IN/OUT, ~6 frames (≈0.2 s at 30 fps)
    after contact: the verdict needs a few frames of post-bounce trajectory
    to confirm the kink and rule out a racquet hit
  - ``rally_end``   — rally over, per-rally stats attached

Cumulative :class:`~.stats.MatchStats` are kept up to date after every frame,
so a scoreboard/stream overlay can read them at any moment.

:class:`LiveStreamServer` is an optional, dependency-free MJPEG web server:
open it in a browser to watch the annotated feed with live stats
(``/``, ``/stream``, ``/stats.json``).
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from .bounce import BounceEvent, HitEvent, detect_bounces, detect_hits
from .calibration import CourtCalibration, detect_court
from .calls import LineCall, LineCaller
from .detection import BallDetector
from .overlay import (
    CALL_PERSIST_S,
    draw_call_marker,
    draw_court_model,
    draw_text,
    draw_trail,
)
from .pipeline import DIRECTION_HORIZON_FRAMES, AnalyzerConfig, is_true_bounce
from .stats import MatchStats, Rally, build_rally
from .tracking import BallTracker, TrackPoint, is_ball_like

# How often to retry automatic court detection until it succeeds (frames).
CALIBRATION_RETRY_INTERVAL = 30
# Keep bounce scanning cheap: only the tail of the open segment is rescanned.
SCAN_TAIL_POINTS = 60
# Events closer than this to an already-processed event are duplicates.
EVENT_DEDUP_FRAMES = 5


@dataclass(frozen=True)
class LiveEvent:
    type: str  # "calibrated" | "rally_start" | "call" | "rally_end"
    frame: int
    call: LineCall | None = None
    rally: Rally | None = None


@dataclass
class _SegState:
    """The currently open (still growing) track segment."""

    points: list[TrackPoint] = field(default_factory=list)
    path_px: float = 0.0

    def append(self, p: TrackPoint) -> None:
        if self.points:
            prev = self.points[-1]
            self.path_px += float(np.hypot(p.x - prev.x, p.y - prev.y))
        self.points.append(p)

    @property
    def net_px(self) -> float:
        if len(self.points) < 2:
            return 0.0
        a, b = self.points[0], self.points[-1]
        return float(np.hypot(b.x - a.x, b.y - a.y))

    def ball_like(self) -> bool:
        return len(self.points) >= 5 and self.path_px >= 40.0 and self.net_px >= 50.0


class LiveAnalyzer:
    """Incremental analysis engine.  One `process(frame)` call per frame."""

    def __init__(
        self,
        fps: float,
        config: AnalyzerConfig | None = None,
        calibration: CourtCalibration | None = None,
    ) -> None:
        self.fps = fps
        self.config = config or AnalyzerConfig()
        self.detector = BallDetector(use_color_prior=self.config.use_color_prior)
        self.tracker = BallTracker()
        self.calibration = calibration
        if self.calibration is None and self.config.corners_file:
            self.calibration = CourtCalibration.from_corners_file(self.config.corners_file)
        self.caller: LineCaller | None = None
        self.stats = MatchStats(fps=fps)

        self.frame_idx = -1
        self._consumed = 0  # tracker points already ingested
        self._calibrated_announced = self.calibration is not None

        self._seg = _SegState()
        self._rally_open = False
        self._rally_points: list[TrackPoint] = []
        self._rally_bounces: list[BounceEvent] = []
        self._rally_hits: list[HitEvent] = []
        self._rally_calls: list[LineCall] = []
        self._processed_frames: list[int] = []  # bounce/hit events already ruled
        self._last_ball_frame: int | None = None

    # -- public API ---------------------------------------------------------

    def process(self, frame: np.ndarray) -> list[LiveEvent]:
        self.frame_idx += 1
        events: list[LiveEvent] = []

        if self.calibration is None and self.frame_idx >= self.config.calibration_frame:
            if (self.frame_idx - self.config.calibration_frame) % CALIBRATION_RETRY_INTERVAL == 0:
                self.calibration = detect_court(frame)
        if self.calibration is not None and not self._calibrated_announced:
            self._calibrated_announced = True
            events.append(LiveEvent(type="calibrated", frame=self.frame_idx))
        if self.calibration is not None and self.caller is None:
            self.caller = LineCaller(calibration=self.calibration, mode=self.config.mode)

        self.tracker.step(self.frame_idx, self.detector.detect(frame))
        new_points = self.tracker.points[self._consumed :]
        self._consumed = len(self.tracker.points)
        for p in new_points:
            events.extend(self._ingest(p))

        if (
            self._rally_open
            and self._last_ball_frame is not None
            and self.frame_idx - self._last_ball_frame > self.config.rally_gap_frames
        ):
            events.extend(self._close_rally())
        return events

    def finish(self) -> list[LiveEvent]:
        """Flush at end of stream: close the open segment and rally."""
        events: list[LiveEvent] = []
        if self._rally_open:
            events.extend(self._close_rally())
        return events

    def set_calibration(self, calibration: CourtCalibration) -> None:
        """Apply (or replace) the calibration mid-stream, e.g. from tapped corners."""
        self.calibration = calibration
        self.caller = None  # rebuilt with the new homography on the next frame
        self._calibrated_announced = False

    # -- internals ----------------------------------------------------------

    def _ingest(self, p: TrackPoint) -> list[LiveEvent]:
        events: list[LiveEvent] = []
        if self._seg.points:
            prev = self._seg.points[-1]
            dframe = p.frame - prev.frame
            jump = float(np.hypot(p.x - prev.x, p.y - prev.y))
            if dframe > 6 or jump > 45.0 * max(dframe, 1):
                events.extend(self._close_segment())
        self._seg.append(p)
        if self._seg.ball_like():
            self._last_ball_frame = p.frame
            if not self._rally_open:
                self._rally_open = True
                events.append(LiveEvent(type="rally_start", frame=self._seg.points[0].frame))
            events.extend(self._scan_segment(final=False))
        return events

    def _scan_segment(self, final: bool) -> list[LiveEvent]:
        """Look for rulable bounce events in the open segment's tail.

        While the segment is still growing, an event is only ruled once the
        trajectory extends ``DIRECTION_HORIZON_FRAMES`` past it, so the
        bounce-vs-hit direction test sees the same data it would in batch
        mode.  At segment close (``final=True``) whatever is left is ruled
        with the data available — exactly like the batch pipeline.
        """
        if self.calibration is None:
            return []
        seg = self._seg.points
        tail = seg[-SCAN_TAIL_POINTS:]
        events: list[LiveEvent] = []
        for b in detect_bounces(tail):
            if not final and b.frame + DIRECTION_HORIZON_FRAMES > self.frame_idx:
                continue
            if any(abs(b.frame - f) < EVENT_DEDUP_FRAMES for f in self._processed_frames):
                continue
            self._processed_frames.append(b.frame)
            if is_true_bounce(seg, b, self.calibration):
                self._rally_bounces.append(b)
                call = self.caller.call(b.frame, b.image_xy, context="rally")
                self._rally_calls.append(call)
                self.stats.calls.append(call)
                self.stats.zone_histogram[call.zone] = (
                    self.stats.zone_histogram.get(call.zone, 0) + 1
                )
                events.append(LiveEvent(type="call", frame=self.frame_idx, call=call))
            else:
                self._rally_hits.append(HitEvent(frame=b.frame, image_xy=b.image_xy))
        return events

    def _close_segment(self) -> list[LiveEvent]:
        events: list[LiveEvent] = []
        if self._seg.ball_like():
            events.extend(self._scan_segment(final=True))
            seg = self._seg.points
            seg_bounces = [
                b for b in self._rally_bounces if seg[0].frame <= b.frame <= seg[-1].frame
            ]
            self._rally_hits.extend(detect_hits(seg, seg_bounces))
            self._rally_points.extend(seg)
        self._seg = _SegState()
        return events

    def _close_rally(self) -> list[LiveEvent]:
        events = self._close_segment()
        self._rally_open = False
        if self._rally_points and self._rally_bounces:
            rally, speeds = build_rally(
                index=len(self.stats.rallies) + 1,
                points=self._rally_points,
                bounces=self._rally_bounces,
                hits=sorted(self._rally_hits, key=lambda h: h.frame),
                calls=self._rally_calls,
                fps=self.fps,
            )
            self.stats.rallies.append(rally)
            self.stats.shot_speeds_kmh.extend(speeds)
            events.append(LiveEvent(type="rally_end", frame=self.frame_idx, rally=rally))
        self._rally_points = []
        self._rally_bounces = []
        self._rally_hits = []
        self._rally_calls = []
        self._processed_frames = []
        return events


class LiveAnnotator:
    """Draws the live overlay: court model, trail, recent calls, tallies."""

    def __init__(self, analyzer: LiveAnalyzer) -> None:
        self.analyzer = analyzer
        self._positions: deque[tuple[int, tuple[float, float]]] = deque(maxlen=32)
        self._consumed = 0
        self._recent_calls: deque[LineCall] = deque(maxlen=8)

    def annotate(self, frame: np.ndarray, events: list[LiveEvent]) -> np.ndarray:
        a = self.analyzer
        for e in events:
            if e.type == "call" and e.call is not None:
                self._recent_calls.append(e.call)
        for p in a.tracker.points[self._consumed :]:
            self._positions.append((p.frame, (p.x, p.y)))
        self._consumed = len(a.tracker.points)

        canvas = frame.copy()
        if a.calibration is not None:
            draw_court_model(canvas, a.calibration)
        draw_trail(canvas, dict(self._positions), a.frame_idx)
        persist = int(CALL_PERSIST_S * a.fps)
        for call in self._recent_calls:
            if call.frame <= a.frame_idx <= call.frame + persist:
                draw_call_marker(canvas, call)

        n_in = sum(1 for c in a.stats.calls if c.decision == "IN")
        n_out = len(a.stats.calls) - n_in
        rally_no = len(a.stats.rallies) + (1 if a._rally_open else 0)
        status = "LIVE" if a.calibration is not None else "LIVE - calibrating..."
        draw_text(canvas, f"COURTVISION {status}", (12, 24), (255, 255, 255))
        draw_text(
            canvas,
            f"rally {max(rally_no, 1)}  in {n_in}  out {n_out}",
            (12, 46),
            (200, 220, 255),
        )
        return canvas


# ---------------------------------------------------------------------------
# Browser streaming (MJPEG over plain http.server — no dependencies)
# ---------------------------------------------------------------------------

_PAGE = """<!doctype html>
<html><head><title>CourtVision live</title>
<style>
 body { background:#0a1628; color:#e2e8f0; font-family:system-ui,sans-serif;
        display:flex; flex-direction:column; align-items:center; gap:12px; padding:16px; }
 img  { max-width:96vw; border-radius:8px; }
 pre  { background:#111f35; padding:12px 16px; border-radius:8px; max-width:96vw;
        overflow-x:auto; font-size:12px; }
</style></head>
<body>
<h2>🎾 CourtVision — live</h2>
<img src="/stream" alt="live annotated feed" />
<pre id="stats">loading stats…</pre>
<script>
 async function tick() {
   try {
     const r = await fetch('/stats.json');
     document.getElementById('stats').textContent =
       JSON.stringify((await r.json()).summary, null, 2);
   } catch (e) {}
   setTimeout(tick, 1000);
 }
 tick();
</script>
</body></html>"""


class LiveStreamServer:
    """Serves the annotated feed and live stats over HTTP.

    - ``/``           a minimal viewer page
    - ``/stream``     MJPEG stream of the annotated frames
    - ``/stats.json`` current cumulative MatchStats
    """

    def __init__(self, port: int = 8765, jpeg_quality: int = 80) -> None:
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._stats_json = "{}"
        self._quality = jpeg_quality
        self._running = True

        server_self = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:  # silence request logging
                pass

            def do_GET(self) -> None:
                if self.path == "/":
                    body = _PAGE.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/stats.json":
                    with server_self._lock:
                        body = server_self._stats_json.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header(
                        "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                    )
                    self.end_headers()
                    try:
                        while server_self._running:
                            with server_self._lock:
                                jpeg = server_self._jpeg
                            if jpeg is not None:
                                self.wfile.write(b"--frame\r\n")
                                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                                self.wfile.write(
                                    f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                                )
                                self.wfile.write(jpeg)
                                self.wfile.write(b"\r\n")
                            time.sleep(1.0 / 30.0)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                else:
                    self.send_response(404)
                    self.end_headers()

        self._httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def update(self, frame: np.ndarray, stats: MatchStats) -> None:
        ok, buf = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self._quality]
        )
        if not ok:
            return
        with self._lock:
            self._jpeg = buf.tobytes()
            self._stats_json = json.dumps(stats.to_dict())

    def stop(self) -> None:
        self._running = False
        self._httpd.shutdown()
        self._httpd.server_close()
