"""Camera <-> court calibration.

Two ways to get a calibration:

1. ``detect_court(frame)`` — automatic.  Extracts the white line mask, finds
   straight lines with a probabilistic Hough transform, merges them into
   candidate court lines, then searches pairs of (roughly) horizontal and
   vertical lines whose four intersections, mapped to the doubles-court
   corners, produce a homography under which the projected court model lands
   on white pixels.  The best-scoring homography wins.

2. ``CourtCalibration.from_corners(...)`` — manual.  The user supplies the
   four doubles-court corner pixels (near-left, near-right, far-right,
   far-left) and we compute the homography directly.  This is the highest
   accuracy path and the recommended one for fixed-camera recordings.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import court


@dataclass
class CourtCalibration:
    """Ground-plane homography between image pixels and court meters."""

    H_image_to_court: np.ndarray  # 3x3
    score: float = 1.0  # line-mask agreement in [0, 1] (1.0 for manual)
    reprojection_error_px: float = 0.0

    H_court_to_image: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.H_court_to_image = np.linalg.inv(self.H_image_to_court)

    @classmethod
    def from_corners(cls, image_corners: np.ndarray) -> "CourtCalibration":
        """Build from the 4 doubles corners: near-left, near-right, far-right, far-left."""
        src = np.asarray(image_corners, dtype=np.float64).reshape(4, 2)
        H = cv2.getPerspectiveTransform(
            src.astype(np.float32), court.DOUBLES_CORNERS.astype(np.float32)
        )
        return cls(H_image_to_court=np.asarray(H, dtype=np.float64))

    @classmethod
    def from_corners_file(cls, path: str | Path) -> "CourtCalibration":
        """Load corners from JSON: {"corners": [[x,y],[x,y],[x,y],[x,y]]}."""
        data = json.loads(Path(path).read_text())
        return cls.from_corners(np.array(data["corners"], dtype=np.float64))

    def image_to_court(self, points: np.ndarray) -> np.ndarray:
        return _apply_h(self.H_image_to_court, points)

    def court_to_image(self, points: np.ndarray) -> np.ndarray:
        return _apply_h(self.H_court_to_image, points)

    def meters_per_pixel(self, image_xy: np.ndarray) -> float:
        """Local ground-plane scale at an image point (for uncertainty estimates)."""
        p = np.asarray(image_xy, dtype=np.float64).reshape(1, 2)
        eps = 1.0
        c0 = self.image_to_court(p)[0]
        cx = self.image_to_court(p + [eps, 0.0])[0]
        cy = self.image_to_court(p + [0.0, eps])[0]
        return float(max(np.linalg.norm(cx - c0), np.linalg.norm(cy - c0)))


def _apply_h(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homog = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return homog[:, :2] / homog[:, 2:3]


# ---------------------------------------------------------------------------
# Automatic court detection
# ---------------------------------------------------------------------------


def white_line_mask(frame: np.ndarray) -> np.ndarray:
    """Binary mask of bright, low-saturation, locally contrasting pixels (court lines)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Lines are brighter than their local neighborhood: white top-hat.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    _, contrast = cv2.threshold(tophat, 30, 255, cv2.THRESH_BINARY)
    bright = cv2.inRange(hsv, (0, 0, 140), (180, 90, 255))
    return cv2.bitwise_and(contrast, bright)


@dataclass
class _Line:
    """Infinite image line in normal form: x*cos(t) + y*sin(t) = rho."""

    rho: float
    theta: float
    support: float  # total Hough segment length backing this line

    def y_at(self, x: float) -> float:
        s = np.sin(self.theta)
        if abs(s) < 1e-9:
            return np.inf
        return (self.rho - x * np.cos(self.theta)) / s


def _segments_to_lines(segments: np.ndarray, rho_tol: float, theta_tol: float) -> list[_Line]:
    """Merge Hough segments into distinct infinite lines by (rho, theta) clustering."""
    lines: list[_Line] = []
    entries = []
    for x1, y1, x2, y2 in segments.reshape(-1, 4).astype(np.float64):
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        if length < 1e-6:
            continue
        theta = np.arctan2(dy, dx) + np.pi / 2.0  # normal direction
        theta = theta % np.pi
        rho = x1 * np.cos(theta) + y1 * np.sin(theta)
        if rho < 0:
            rho, theta = -rho, (theta + np.pi) % (2 * np.pi)
        entries.append((rho, theta, length))
    entries.sort(key=lambda e: -e[2])
    for rho, theta, length in entries:
        merged = False
        for ln in lines:
            dt = abs(ln.theta - theta)
            dt = min(dt, np.pi - dt)
            if dt < theta_tol and abs(ln.rho - rho) < rho_tol:
                # Weighted average keeps the dominant segment's geometry.
                w = ln.support / (ln.support + length)
                ln.rho = w * ln.rho + (1 - w) * rho
                ln.support += length
                merged = True
                break
        if not merged:
            lines.append(_Line(rho, theta, length))
    return lines


def _intersect(a: _Line, b: _Line) -> np.ndarray | None:
    A = np.array(
        [[np.cos(a.theta), np.sin(a.theta)], [np.cos(b.theta), np.sin(b.theta)]]
    )
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    return np.linalg.solve(A, np.array([a.rho, b.rho]))


def _score_homography(H_court_to_image: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of projected court-model points that land on white-line pixels."""
    h, w = mask.shape
    samples = []
    for (p0, p1) in court.COURT_LINES.values():
        p0, p1 = np.array(p0), np.array(p1)
        n = max(int(np.linalg.norm(p1 - p0) * 4), 2)
        t = np.linspace(0.0, 1.0, n)[:, None]
        samples.append(p0 + t * (p1 - p0))
    pts = _apply_h(H_court_to_image, np.vstack(samples))
    xs = np.round(pts[:, 0]).astype(int)
    ys = np.round(pts[:, 1]).astype(int)
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    if valid.sum() < len(pts) * 0.5:
        return 0.0
    dilated = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    hits = dilated[ys[valid], xs[valid]] > 0
    return float(hits.sum()) / float(len(pts))


def detect_court(
    frame: np.ndarray,
    min_score: float = 0.55,
    max_candidates_per_axis: int = 12,
) -> CourtCalibration | None:
    """Automatically fit the court model to a frame.  None if no confident fit."""
    mask = white_line_mask(frame)
    h, w = mask.shape
    segments = cv2.HoughLinesP(
        mask,
        rho=1,
        theta=np.pi / 360,
        threshold=50,
        minLineLength=w // 12,
        maxLineGap=w // 40,
    )
    if segments is None:
        return None

    lines = _segments_to_lines(segments, rho_tol=max(8.0, h / 80.0), theta_tol=np.radians(2.0))
    # Split by orientation: court's across-lines are near-horizontal in a
    # typical broadcast/behind-the-baseline view; along-lines converge but are
    # steeper.  Use a generous 45-degree split on the line direction.
    horiz: list[_Line] = []
    vert: list[_Line] = []
    for ln in lines:
        direction = (ln.theta + np.pi / 2.0) % np.pi  # line direction angle
        if min(direction, np.pi - direction) < np.pi / 4:
            horiz.append(ln)
        else:
            vert.append(ln)
    horiz = sorted(horiz, key=lambda l: -l.support)[:max_candidates_per_axis]
    vert = sorted(vert, key=lambda l: -l.support)[:max_candidates_per_axis]
    if len(horiz) < 2 or len(vert) < 2:
        return None

    # Order horizontal candidates by image y so (top, bottom) pairs make sense.
    horiz.sort(key=lambda l: l.y_at(w / 2.0))

    court_corners = court.DOUBLES_CORNERS.astype(np.float32)
    best: tuple[float, np.ndarray] | None = None
    for (top, bottom) in itertools.combinations(horiz, 2):
        for (l1, l2) in itertools.combinations(vert, 2):
            corners = [
                _intersect(bottom, l1),
                _intersect(bottom, l2),
                _intersect(top, l2),
                _intersect(top, l1),
            ]
            if any(c is None for c in corners):
                continue
            quad = np.array(corners, dtype=np.float64)
            # near-left, near-right must be left-to-right in the image
            if quad[0, 0] > quad[1, 0]:
                quad = quad[[1, 0, 3, 2]]
            if not _plausible_quad(quad, w, h):
                continue
            H = cv2.getPerspectiveTransform(
                quad.astype(np.float32), court_corners
            )
            H = np.asarray(H, dtype=np.float64)
            score = _score_homography(np.linalg.inv(H), mask)
            if best is None or score > best[0]:
                best = (score, H)

    if best is None or best[0] < min_score:
        return None
    score, H = best
    return CourtCalibration(H_image_to_court=H, score=score)


def _plausible_quad(quad: np.ndarray, w: int, h: int) -> bool:
    """Cheap geometric sanity checks before scoring a candidate quad."""
    # All corners within a generous margin of the frame.
    if np.any(quad[:, 0] < -w) or np.any(quad[:, 0] > 2 * w):
        return False
    if np.any(quad[:, 1] < -h) or np.any(quad[:, 1] > 2 * h):
        return False
    # Convex, non-degenerate, covering a meaningful part of the frame.
    area = cv2.contourArea(quad.astype(np.float32))
    if area < 0.05 * w * h:
        return False
    # Near edge (bottom in image) should be below far edge for a standard view.
    near_y = (quad[0, 1] + quad[1, 1]) / 2.0
    far_y = (quad[2, 1] + quad[3, 1]) / 2.0
    return near_y > far_y
