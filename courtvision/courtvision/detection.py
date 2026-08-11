"""Ball candidate detection.

The ball is a small, fast, roughly circular blob that differs from the
background.  We use background subtraction (MOG2) plus three filters:

- shape: area window and circularity;
- color: an optic-yellow prior — regulation balls are yellow, while the
  usual false positives (player fragments, compression noise on lines,
  shadows) are not.  Disable with ``use_color_prior=False`` for footage
  with a non-yellow ball;
- static-region suppression: a grid cell that has produced candidates in
  most of the trailing window is a quasi-stationary object — a swaying
  player, a flapping net band, a spectator — not a ball.  The ball moves;
  it never lingers in one cell that long.

Each frame yields zero or more :class:`BallCandidate`; the tracker decides
which of them is actually the ball.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class BallCandidate:
    x: float
    y: float
    area: float
    circularity: float

    @property
    def score(self) -> float:
        return self.circularity


class BallDetector:
    def __init__(
        self,
        min_area: float = 4.0,
        max_area: float = 400.0,
        min_circularity: float = 0.45,
        history: int = 120,
        use_color_prior: bool = True,
        hue_range: tuple[int, int] = (18, 55),
        min_saturation: int = 60,
        min_value: int = 90,
        warmup_frames: int = 20,
        static_cell_px: int = 32,
        static_window: int = 60,
        static_threshold: int = 25,
        player_min_area: float = 3000.0,
        player_margin_px: int = 15,
    ) -> None:
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity
        self.use_color_prior = use_color_prior
        self.hue_range = hue_range
        self.min_saturation = min_saturation
        self.min_value = min_value
        self.warmup_frames = warmup_frames
        self.static_cell_px = static_cell_px
        self.static_window = static_window
        self.static_threshold = static_threshold
        self.player_min_area = player_min_area
        self.player_margin_px = player_margin_px
        self._frames_seen = 0
        self._cell_history: deque[set[tuple[int, int]]] = deque()
        self._cell_counts: Counter = Counter()
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=24, detectShadows=False
        )

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        return (int(x // self.static_cell_px), int(y // self.static_cell_px))

    def _update_static_map(self, cells: set[tuple[int, int]]) -> None:
        self._cell_history.append(cells)
        self._cell_counts.update(cells)
        if len(self._cell_history) > self.static_window:
            for cell in self._cell_history.popleft():
                self._cell_counts[cell] -= 1
                if self._cell_counts[cell] <= 0:
                    del self._cell_counts[cell]

    def _is_ball_colored(self, frame: np.ndarray, x: float, y: float) -> bool:
        h, w = frame.shape[:2]
        x0 = int(np.clip(x - 1, 0, w - 3))
        y0 = int(np.clip(y - 1, 0, h - 3))
        mean_bgr = frame[y0 : y0 + 3, x0 : x0 + 3].reshape(-1, 3).mean(axis=0)
        hsv = cv2.cvtColor(mean_bgr.astype(np.uint8).reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0]
        return (
            self.hue_range[0] <= hsv[0] <= self.hue_range[1]
            and hsv[1] >= self.min_saturation
            and hsv[2] >= self.min_value
        )

    def _player_exclusion_mask(self, fg: np.ndarray) -> np.ndarray | None:
        """Mask covering large moving objects (players) plus a safety margin.

        A player's foreground is often fragmented (MOG2 partially absorbs a
        slowly moving body, compression shreds the edges); dilating first
        merges the fragments so the whole body registers as one large blob.
        Ball candidates inside the mask are rejected — tiny yellow-ish
        shards of a moving player are the main source of phantom bounces.
        """
        merged = cv2.dilate(fg, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big = [c for c in contours if cv2.contourArea(c) > self.player_min_area]
        if not big:
            return None
        mask = np.zeros(fg.shape, dtype=np.uint8)
        cv2.drawContours(mask, big, -1, 255, -1)
        k = 2 * self.player_margin_px + 1
        return cv2.dilate(mask, np.ones((k, k), np.uint8))

    def detect(self, frame: np.ndarray) -> list[BallCandidate]:
        fg = self._bg.apply(frame)
        # The background model is unreliable until it has seen some frames.
        self._frames_seen += 1
        if self._frames_seen <= self.warmup_frames:
            return []
        exclusion = self._player_exclusion_mask(fg)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[BallCandidate] = []
        for c in contours:
            area = cv2.contourArea(c)
            if not (self.min_area <= area <= self.max_area):
                continue
            perimeter = cv2.arcLength(c, True)
            if perimeter <= 0:
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < self.min_circularity:
                continue
            m = cv2.moments(c)
            if m["m00"] <= 0:
                continue
            cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
            if exclusion is not None and exclusion[int(cy), int(cx)] > 0:
                continue
            if self.use_color_prior and not self._is_ball_colored(frame, cx, cy):
                continue
            candidates.append(
                BallCandidate(
                    x=cx,
                    y=cy,
                    area=float(area),
                    circularity=float(min(circularity, 1.0)),
                )
            )
        # Static-region suppression: record where candidates appear, reject
        # the ones sitting in a cell that's been occupied for most of the
        # trailing window.  (Cells are recorded before rejection so a hot
        # zone stays hot.)
        cells = {self._cell(c.x, c.y) for c in candidates}
        kept = [
            c
            for c in candidates
            if self._cell_counts[self._cell(c.x, c.y)] < self.static_threshold
        ]
        self._update_static_map(cells)
        kept.sort(key=lambda c: -c.score)
        return kept
