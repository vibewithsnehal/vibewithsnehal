"""Ball candidate detection.

The ball is a small, fast, roughly circular blob that differs from the
background.  We use background subtraction (MOG2) plus shape filters
(area window, circularity) and an optic-yellow color prior: regulation
balls are yellow, while the usual false positives — player fragments,
compression noise on lines, shadows — are not.  Disable the color prior
(``use_color_prior=False``) for footage with a non-yellow ball.

Each frame yields zero or more :class:`BallCandidate`; the tracker decides
which of them is actually the ball.
"""

from __future__ import annotations

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
    ) -> None:
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity
        self.use_color_prior = use_color_prior
        self.hue_range = hue_range
        self.min_saturation = min_saturation
        self.min_value = min_value
        self.warmup_frames = warmup_frames
        self._frames_seen = 0
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=24, detectShadows=False
        )

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

    def detect(self, frame: np.ndarray) -> list[BallCandidate]:
        fg = self._bg.apply(frame)
        # The background model is unreliable until it has seen some frames.
        self._frames_seen += 1
        if self._frames_seen <= self.warmup_frames:
            return []
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
        candidates.sort(key=lambda c: -c.score)
        return candidates
