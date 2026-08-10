import numpy as np

from courtvision.bounce import detect_bounces
from courtvision.tracking import BallTracker, TrackPoint, split_tracks
from courtvision.detection import BallCandidate


def _ballistic_track(bounce_frames, fps=30.0, y0=100.0):
    """Simple 1D image trajectory: falls to y=400 at each bounce frame, rises after."""
    points = []
    frame = 0
    y_floor = 400.0
    for bf in bounce_frames:
        n_down = bf - frame
        for i in range(n_down):
            t = i / max(n_down - 1, 1)
            y = y0 + (y_floor - y0) * t * t  # accelerating fall
            points.append(TrackPoint(frame + i, 300.0 + 0.5 * (frame + i), y, True))
        frame = bf
        # rise for 12 frames
        for i in range(12):
            t = i / 11.0
            y = y_floor - (y_floor - (y0 + 80)) * (2 * t - t * t)
            points.append(TrackPoint(frame + i, 300.0 + 0.5 * (frame + i), y, True))
        frame += 12
        y0 += 0  # next fall starts from the risen height
    return points


def test_detects_single_bounce():
    pts = _ballistic_track([25])
    events = detect_bounces(pts)
    assert len(events) == 1
    assert abs(events[0].frame - 25) <= 2
    assert abs(events[0].image_xy[1] - 400.0) < 12.0


def test_no_bounce_on_smooth_flight():
    pts = [TrackPoint(i, 10.0 * i, 200.0 + 1.5 * i, True) for i in range(60)]
    assert detect_bounces(pts) == []


def test_tracker_follows_moving_ball_through_dropouts():
    tracker = BallTracker()
    for f in range(60):
        x, y = 100.0 + 5.0 * f, 200.0 + 2.0 * f
        if 20 <= f < 24:  # 4-frame dropout
            tracker.step(f, [])
        else:
            tracker.step(f, [BallCandidate(x, y, 20.0, 0.9)])
    pts = tracker.finish()
    frames = [p.frame for p in pts]
    assert frames == list(range(60))  # dropout was coasted and confirmed
    # Coasted positions stay close to the true line.
    for p in pts:
        assert abs(p.x - (100.0 + 5.0 * p.frame)) < 8.0


def test_tracker_gives_up_after_long_dropout_and_splits():
    tracker = BallTracker(max_missed=5)
    for f in range(30):
        tracker.step(f, [BallCandidate(100.0 + 5 * f, 200.0, 20.0, 0.9)])
    for f in range(30, 60):
        tracker.step(f, [])
    for f in range(60, 80):
        tracker.step(f, [BallCandidate(500.0, 300.0, 20.0, 0.9)])
    segments = split_tracks(tracker.finish(), max_gap=12)
    assert len(segments) == 2
    assert segments[0][-1].frame == 29
    assert segments[1][0].frame == 60
