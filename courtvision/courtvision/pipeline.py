"""End-to-end analysis: video in, line calls + match stats out."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np

from . import court
from .bounce import BounceEvent, HitEvent, detect_bounces, detect_hits
from .calibration import CourtCalibration, detect_court
from .calls import LineCall, LineCaller
from .detection import BallDetector
from .stats import MatchStats, compile_stats
from .tracking import BallTracker, TrackPoint, is_ball_like, split_tracks

# A "bounce" whose projected point is this far outside the doubles court is
# almost certainly a racquet contact near a player, not a ground contact —
# reclassify it as a hit instead of issuing a bogus OUT call.
MAX_PLAUSIBLE_BOUNCE_OVERSHOOT_M = 2.0

# Horizon (frames) and minimum per-side displacement (pixels) for the test
# that separates bounces from racquet hits: ground contact cannot reverse the
# ball's horizontal direction of travel, a racquet usually does.
DIRECTION_HORIZON_FRAMES = 6
DIRECTION_MIN_TRAVEL_PX = 8.0


def _reverses_horizontal_direction(seg: list[TrackPoint], event: BounceEvent) -> bool:
    """True if the image-x travel flips sign across the event (a racquet hit).

    Image x is used rather than the court projection because projecting an
    *airborne* ball through the ground homography inflates its depth, which
    fakes a direction reversal at every ordinary bounce.
    """
    by_frame = {p.frame: p.x for p in seg}
    before = next(
        (
            by_frame[f]
            for f in range(event.frame - DIRECTION_HORIZON_FRAMES, event.frame)
            if f in by_frame
        ),
        None,
    )
    after = next(
        (
            by_frame[f]
            for f in range(event.frame + DIRECTION_HORIZON_FRAMES, event.frame, -1)
            if f in by_frame
        ),
        None,
    )
    if before is None or after is None:
        return False
    dx_in = event.image_xy[0] - before
    dx_out = after - event.image_xy[0]
    return (
        abs(dx_in) > DIRECTION_MIN_TRAVEL_PX
        and abs(dx_out) > DIRECTION_MIN_TRAVEL_PX
        and np.sign(dx_in) != np.sign(dx_out)
    )


def is_true_bounce(
    seg: list[TrackPoint], event: BounceEvent, calib: CourtCalibration
) -> bool:
    """Shared bounce-vs-racquet-hit classification for batch and live modes."""
    cx, cy = calib.image_to_court(np.array([event.image_xy]))[0]
    overshoot = court.DOUBLES_COURT.signed_distance(float(cx), float(cy))
    if overshoot > MAX_PLAUSIBLE_BOUNCE_OVERSHOOT_M:
        return False
    return not _reverses_horizontal_direction(seg, event)


@dataclass
class AnalysisResult:
    calibration: CourtCalibration
    segments: list[list[TrackPoint]]
    segment_events: list[tuple[list[BounceEvent], list[HitEvent], list[LineCall]]]
    stats: MatchStats
    fps: float

    @property
    def calls(self) -> list[LineCall]:
        return self.stats.calls


@dataclass
class AnalyzerConfig:
    mode: str = "singles"  # or "doubles"
    corners_file: str | None = None  # manual calibration JSON
    calibration_frame: int = 5  # frame index used for auto-detection
    rally_gap_frames: int = 30  # ball absent this long -> the rally is over
    fps_override: float | None = None
    use_color_prior: bool = True  # optic-yellow ball filter; off for other balls


def _iter_video(path: str | Path) -> tuple[Iterator[np.ndarray], float, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def gen() -> Iterator[np.ndarray]:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
        cap.release()

    return gen(), fps, w, h


def analyze_frames(
    frames: Iterable[np.ndarray],
    fps: float,
    config: AnalyzerConfig | None = None,
    calibration: CourtCalibration | None = None,
) -> AnalysisResult:
    """Analyze an in-memory or streamed frame sequence."""
    config = config or AnalyzerConfig()
    detector = BallDetector(use_color_prior=config.use_color_prior)
    tracker = BallTracker()

    calib = calibration
    if calib is None and config.corners_file:
        calib = CourtCalibration.from_corners_file(config.corners_file)

    for idx, frame in enumerate(frames):
        if calib is None and idx >= config.calibration_frame:
            calib = detect_court(frame)
        tracker.step(idx, detector.detect(frame))

    if calib is None:
        raise RuntimeError(
            "Court calibration failed: no confident court fit. "
            "Provide the four doubles-court corner pixels via a corners file."
        )

    points = tracker.finish()
    # Kinematically continuous ball segments (a tracker re-lock starts a new
    # one), then grouped into rallies by dead time between them.
    subsegments = [s for s in split_tracks(points) if is_ball_like(s)]
    rally_groups: list[list[list[TrackPoint]]] = []
    for s in subsegments:
        if (
            rally_groups
            and s[0].frame - rally_groups[-1][-1][-1].frame <= config.rally_gap_frames
        ):
            rally_groups[-1].append(s)
        else:
            rally_groups.append([s])

    caller = LineCaller(calibration=calib, mode=config.mode)
    segments: list[list[TrackPoint]] = []
    segment_events: list[tuple[list[BounceEvent], list[HitEvent], list[LineCall]]] = []
    for group in rally_groups:
        bounces: list[BounceEvent] = []
        hits: list[HitEvent] = []
        calls: list[LineCall] = []
        for seg in group:
            seg_bounces: list[BounceEvent] = []
            for b in detect_bounces(seg):
                if not is_true_bounce(seg, b, calib):
                    hits.append(HitEvent(frame=b.frame, image_xy=b.image_xy))
                    continue
                seg_bounces.append(b)
                calls.append(caller.call(b.frame, b.image_xy, context="rally"))
            bounces.extend(seg_bounces)
            hits.extend(detect_hits(seg, seg_bounces))
        hits.sort(key=lambda h: h.frame)
        segments.append([p for seg in group for p in seg])
        segment_events.append((bounces, hits, calls))

    stats = compile_stats(segments, segment_events, fps)
    return AnalysisResult(
        calibration=calib,
        segments=segments,
        segment_events=segment_events,
        stats=stats,
        fps=fps,
    )


def analyze_video(
    video_path: str | Path,
    config: AnalyzerConfig | None = None,
    calibration: CourtCalibration | None = None,
) -> AnalysisResult:
    config = config or AnalyzerConfig()
    frames, fps, _, _ = _iter_video(video_path)
    if config.fps_override:
        fps = config.fps_override
    return analyze_frames(frames, fps, config, calibration)
