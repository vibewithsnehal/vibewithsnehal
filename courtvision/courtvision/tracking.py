"""Ball tracking: turn per-frame candidates into a continuous trajectory.

A constant-velocity Kalman filter predicts where the ball should be; the
nearest candidate inside a (miss-widened) gate is accepted as the ball.
Short dropouts are coasted through; long ones end the track and a new one
starts on the next stable candidate.  The output is a list of
:class:`TrackPoint` covering every frame where the ball's position is known
(observed or briefly interpolated).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detection import BallCandidate


@dataclass(frozen=True)
class TrackPoint:
    frame: int
    x: float
    y: float
    observed: bool  # False for coasted (predicted-only) points


class BallTracker:
    def __init__(
        self,
        gate_px: float = 40.0,
        max_missed: int = 6,
        process_noise: float = 8.0,
        measurement_noise: float = 2.0,
    ) -> None:
        self.gate_px = gate_px
        self.max_missed = max_missed
        self.q = process_noise
        self.r = measurement_noise
        self._state: np.ndarray | None = None  # [x, y, vx, vy]
        self._P: np.ndarray | None = None
        self._missed = 0
        self._pending: list[TrackPoint] = []  # coasted points awaiting confirmation
        self.points: list[TrackPoint] = []

    def _predict(self) -> None:
        F = np.eye(4)
        F[0, 2] = F[1, 3] = 1.0
        Q = np.diag([0.25, 0.25, 1.0, 1.0]) * self.q
        self._state = F @ self._state
        self._P = F @ self._P @ F.T + Q

    def _update(self, z: np.ndarray) -> None:
        Hm = np.zeros((2, 4))
        Hm[0, 0] = Hm[1, 1] = 1.0
        R = np.eye(2) * self.r
        y = z - Hm @ self._state
        S = Hm @ self._P @ Hm.T + R
        K = self._P @ Hm.T @ np.linalg.inv(S)
        self._state = self._state + K @ y
        self._P = (np.eye(4) - K @ Hm) @ self._P

    def step(self, frame_idx: int, candidates: list[BallCandidate]) -> None:
        if self._state is None:
            if candidates:
                best = candidates[0]
                self._state = np.array([best.x, best.y, 0.0, 0.0])
                self._P = np.diag([4.0, 4.0, 100.0, 100.0])
                self._missed = 0
                self.points.append(TrackPoint(frame_idx, best.x, best.y, True))
            return

        self._predict()
        px, py = self._state[0], self._state[1]
        gate = self.gate_px * (1.0 + 0.5 * self._missed)
        best, best_d = None, gate
        for c in candidates:
            d = float(np.hypot(c.x - px, c.y - py))
            if d < best_d:
                best, best_d = c, d

        if best is not None:
            self._update(np.array([best.x, best.y]))
            self._missed = 0
            # A real re-detection confirms the coasted gap was the ball in flight.
            self.points.extend(self._pending)
            self._pending = []
            self.points.append(
                TrackPoint(frame_idx, float(self._state[0]), float(self._state[1]), True)
            )
        else:
            self._missed += 1
            if self._missed > self.max_missed:
                self._state = None
                self._P = None
                self._pending = []
            else:
                self._pending.append(TrackPoint(frame_idx, float(px), float(py), False))

    def finish(self) -> list[TrackPoint]:
        self._pending = []
        return self.points


def split_tracks(
    points: list[TrackPoint],
    max_gap: int = 6,
    max_jump_px_per_frame: float = 45.0,
) -> list[list[TrackPoint]]:
    """Split into kinematically continuous segments.

    Coasted dropouts are back-filled by the tracker, so any frame gap in the
    final point list means the track died and re-locked — possibly onto a
    different object.  A large spatial jump between consecutive points means
    the same thing.  Either one starts a new segment.
    """
    segments: list[list[TrackPoint]] = []
    current: list[TrackPoint] = []
    for p in points:
        if current:
            prev = current[-1]
            dframe = p.frame - prev.frame
            jump = float(np.hypot(p.x - prev.x, p.y - prev.y))
            if dframe > max_gap or jump > max_jump_px_per_frame * max(dframe, 1):
                segments.append(current)
                current = []
        current.append(p)
    if current:
        segments.append(current)
    return segments


def is_ball_like(
    segment: list[TrackPoint],
    min_points: int = 5,
    min_path_px: float = 40.0,
    min_net_px: float = 50.0,
) -> bool:
    """Reject junk segments: too short, static, or jittering in place.

    An oscillating player fragment accumulates *path* but no *net*
    displacement; a real ball segment travels.
    """
    if len(segment) < min_points:
        return False
    path = sum(
        float(np.hypot(b.x - a.x, b.y - a.y)) for a, b in zip(segment, segment[1:])
    )
    net = float(
        np.hypot(segment[-1].x - segment[0].x, segment[-1].y - segment[0].y)
    )
    return path >= min_path_px and net >= min_net_px
