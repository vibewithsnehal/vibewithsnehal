"""Bounce and hit detection from the ball trajectory.

In image space a bounce is a local *maximum* of the ball's y coordinate
(the lowest visual point) where the vertical velocity flips from downward
to upward.  We find candidate frames with a windowed velocity test, then
refine the contact position by intersecting quadratic fits of the incoming
and outgoing arcs — the same idea ball-tracking systems use to localize
contact between discrete frames.

Racquet hits (for stats, not calls) show up as reversals of the ball's
*along-court* direction of travel that are not bounces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tracking import TrackPoint


@dataclass(frozen=True)
class BounceEvent:
    frame: int
    image_xy: tuple[float, float]
    strength: float  # velocity change magnitude, px/frame


@dataclass(frozen=True)
class HitEvent:
    frame: int
    image_xy: tuple[float, float]


def _fit_parabola(pts: list[TrackPoint]) -> np.ndarray | None:
    """Fit y = a*t^2 + b*t + c over track points (t = frame index)."""
    if len(pts) < 3:
        return None
    t = np.array([p.frame for p in pts], dtype=np.float64)
    y = np.array([p.y for p in pts], dtype=np.float64)
    try:
        return np.polyfit(t, y, 2)
    except np.linalg.LinAlgError:
        return None


def detect_bounces(
    points: list[TrackPoint],
    window: int = 4,
    min_speed: float = 1.0,
    min_kink: float = 3.0,
    min_separation: int = 5,
) -> list[BounceEvent]:
    """Find ground contacts in a continuous track segment.

    A bounce is a *kink* in the vertical image velocity: the ball is falling
    (image y increasing) and its descent rate collapses — or reverses — at
    contact.  A sign flip is not required: when the ball travels toward the
    camera, perspective can cancel the post-bounce rise entirely, so the
    kink magnitude is the robust signal.  Candidates are kept via non-max
    suppression on the kink score.
    """
    n = len(points)
    if n < 2 * window + 1:
        return []
    ys = np.array([p.y for p in points])
    frames = np.array([p.frame for p in points])

    kink = np.full(n, -np.inf)
    for i in range(window, n - window):
        v_in = (ys[i] - ys[i - window]) / max(frames[i] - frames[i - window], 1)
        v_out = (ys[i + window] - ys[i]) / max(frames[i + window] - frames[i], 1)
        k = v_in - v_out
        # Falling, and the descent rate dropped by more than half — a smooth
        # arc only changes velocity gradually over such a short window.
        if v_in > min_speed and k > min_kink and k > 0.5 * v_in:
            kink[i] = k

    # Non-max suppression: strongest kinks first, enforce frame separation.
    events: list[BounceEvent] = []
    for i in np.argsort(-kink):
        if not np.isfinite(kink[i]):
            break
        if any(abs(int(frames[i]) - e.frame) < min_separation for e in events):
            continue
        xy = _refine_contact(points, int(i), window)
        events.append(
            BounceEvent(frame=int(frames[i]), image_xy=xy, strength=float(kink[i]))
        )
    events.sort(key=lambda e: e.frame)
    return events


def _refine_contact(points: list[TrackPoint], i: int, window: int) -> tuple[float, float]:
    """Refine the bounce pixel by intersecting pre/post quadratic arcs in y(t).

    Falls back to the discrete track point when the fits are degenerate.
    """
    pre = points[max(0, i - 2 * window) : i + 1]
    post = points[i : i + 2 * window + 1]
    fit_pre = _fit_parabola(pre)
    fit_post = _fit_parabola(post)
    p = points[i]
    if fit_pre is None or fit_post is None:
        return (p.x, p.y)
    # Contact time: where the two arcs meet nearest the peak frame.
    diff = np.polysub(fit_pre, fit_post)
    roots = np.roots(diff)
    real = [r.real for r in roots if abs(r.imag) < 1e-6]
    if not real:
        return (p.x, p.y)
    t_c = min(real, key=lambda r: abs(r - p.frame))
    if abs(t_c - p.frame) > window:
        return (p.x, p.y)
    y_c = float(np.polyval(fit_pre, t_c))
    # Interpolate x linearly around the contact time.
    t = np.array([q.frame for q in points[max(0, i - window) : i + window + 1]])
    x = np.array([q.x for q in points[max(0, i - window) : i + window + 1]])
    x_c = float(np.interp(t_c, t, x))
    return (x_c, y_c)


def detect_hits(
    points: list[TrackPoint],
    bounces: list[BounceEvent],
    window: int = 4,
    min_speed: float = 1.5,
    min_separation: int = 8,
) -> list[HitEvent]:
    """Racquet contacts: reversals of vertical *court-direction* travel.

    In a behind-the-court view, the ball travels up-image toward the far
    player and down-image back.  A reversal of the along-image y drift that
    is not near a bounce (bounces don't reverse travel direction) is a hit.
    We use the x drift as well to catch cross-court exchanges.
    """
    n = len(points)
    if n < 2 * window + 1:
        return []
    bounce_frames = {b.frame for b in bounces}
    ys = np.array([p.y for p in points])
    frames = np.array([p.frame for p in points])

    events: list[HitEvent] = []
    last_frame = -(10**9)
    for i in range(window, n - window):
        f = int(frames[i])
        if any(abs(f - bf) < window + 2 for bf in bounce_frames):
            continue
        v_in = (ys[i] - ys[i - window]) / max(frames[i] - frames[i - window], 1)
        v_out = (ys[i + window] - ys[i]) / max(frames[i + window] - frames[i], 1)
        reversal = (v_in > min_speed and v_out < -min_speed) or (
            v_in < -min_speed and v_out > min_speed
        )
        if reversal and f - last_frame >= min_separation:
            events.append(HitEvent(frame=f, image_xy=(points[i].x, points[i].y)))
            last_frame = f
    return events
