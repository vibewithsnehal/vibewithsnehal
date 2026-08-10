"""Annotated rendering: court model, ball trail, calls, tallies.

The drawing helpers are shared by the batch replay renderer
(:func:`render_overlay_video`) and the live annotator in :mod:`.live`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from . import court
from .calibration import CourtCalibration
from .calls import LineCall
from .pipeline import AnalysisResult

TRAIL_LEN = 12
CALL_PERSIST_S = 2.5

IN_COLOR = (80, 200, 80)
OUT_COLOR = (60, 60, 230)
COURT_COLOR = (200, 160, 60)


def draw_court_model(canvas: np.ndarray, calibration: CourtCalibration) -> None:
    for (p0, p1) in court.COURT_LINES.values():
        a = calibration.court_to_image(np.array([p0]))[0]
        b = calibration.court_to_image(np.array([p1]))[0]
        cv2.line(
            canvas,
            (int(a[0]), int(a[1])),
            (int(b[0]), int(b[1])),
            COURT_COLOR,
            1,
            cv2.LINE_AA,
        )


def draw_trail(
    canvas: np.ndarray, positions: dict[int, tuple[float, float]], idx: int
) -> None:
    trail = [positions[f] for f in range(max(0, idx - TRAIL_LEN), idx + 1) if f in positions]
    for i, (x, y) in enumerate(trail):
        alpha = (i + 1) / len(trail)
        radius = 2 + int(3 * alpha)
        color = (60, int(160 + 60 * alpha), 235)
        cv2.circle(canvas, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)


def draw_call_marker(canvas: np.ndarray, call: LineCall) -> None:
    x, y = int(call.image_xy[0]), int(call.image_xy[1])
    color = IN_COLOR if call.decision == "IN" else OUT_COLOR
    cv2.drawMarker(canvas, (x, y), color, cv2.MARKER_TILTED_CROSS, 16, 2)
    cv2.circle(canvas, (x, y), 10, color, 2, cv2.LINE_AA)
    label = f"{call.decision} {call.margin_m * 100:+.0f}cm ({call.confidence:.0%})"
    draw_text(canvas, label, (x + 14, y - 10), color)


def draw_text(canvas: np.ndarray, s: str, org: tuple[int, int], color) -> None:
    cv2.putText(canvas, s, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, s, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def render_overlay_video(
    frames: Iterable[np.ndarray],
    result: AnalysisResult,
    output_path: str | Path,
) -> Path:
    """Second pass over the source frames, writing the annotated video."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    positions: dict[int, tuple[float, float]] = {}
    for seg in result.segments:
        for p in seg:
            positions[p.frame] = (p.x, p.y)
    calls = sorted(result.calls, key=lambda c: c.frame)
    persist = int(CALL_PERSIST_S * result.fps)

    writer = None
    for idx, frame in enumerate(frames):
        if writer is None:
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                result.fps,
                (frame.shape[1], frame.shape[0]),
            )
        canvas = frame.copy()
        draw_court_model(canvas, result.calibration)
        draw_trail(canvas, positions, idx)
        for call in calls:
            if call.frame <= idx <= call.frame + persist:
                draw_call_marker(canvas, call)
        _draw_hud(canvas, result, idx)
        writer.write(canvas)
    if writer is not None:
        writer.release()
    return output_path


def _draw_hud(canvas: np.ndarray, result: AnalysisResult, idx: int) -> None:
    done = [c for c in result.calls if c.frame <= idx]
    n_in = sum(1 for c in done if c.decision == "IN")
    n_out = len(done) - n_in
    rally = sum(1 for r in result.stats.rallies if r.start_frame <= idx)
    draw_text(canvas, "COURTVISION", (12, 24), (255, 255, 255))
    draw_text(canvas, f"rally {max(rally, 1)}  in {n_in}  out {n_out}", (12, 46), (200, 220, 255))
