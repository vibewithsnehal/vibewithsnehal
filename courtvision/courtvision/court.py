"""Official ITF tennis court geometry.

Court coordinate system (meters):
    - Origin at the near-left corner of the *doubles* court.
    - x runs across the court, 0 .. 10.97 (doubles width).
    - y runs along the court, 0 .. 23.77 (near baseline -> far baseline).
    - The net sits at y = 11.885.

All measurements are to the *outside* edge of the lines, per the ITF rules,
which means the painted lines lie inside these rectangles.  A ball touching
any part of a bounding line is IN.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ITF dimensions (meters)
DOUBLES_WIDTH = 10.97
SINGLES_WIDTH = 8.23
COURT_LENGTH = 23.77
SERVICE_FROM_NET = 6.40
NET_Y = COURT_LENGTH / 2.0  # 11.885
ALLEY = (DOUBLES_WIDTH - SINGLES_WIDTH) / 2.0  # 1.37
CENTER_X = DOUBLES_WIDTH / 2.0  # 5.485
SERVICE_NEAR_Y = NET_Y - SERVICE_FROM_NET  # 5.485
SERVICE_FAR_Y = NET_Y + SERVICE_FROM_NET  # 18.285
LINE_WIDTH = 0.05
BALL_RADIUS = 0.033  # regulation ball ~6.54-6.86 cm diameter


@dataclass(frozen=True)
class Rect:
    """Axis-aligned rectangle in court coordinates (outer line edges included)."""

    x0: float
    y0: float
    x1: float
    y1: float

    def signed_distance(self, x: float, y: float) -> float:
        """Signed distance to the rectangle: negative inside, positive outside.

        Outside, this is the Euclidean distance to the nearest boundary point;
        inside, the negated distance to the nearest edge.
        """
        dx = max(self.x0 - x, 0.0, x - self.x1)
        dy = max(self.y0 - y, 0.0, y - self.y1)
        if dx > 0.0 or dy > 0.0:
            return float(np.hypot(dx, dy))
        return -min(x - self.x0, self.x1 - x, y - self.y0, self.y1 - y)

    def contains(self, x: float, y: float) -> bool:
        return self.signed_distance(x, y) <= 0.0


SINGLES_COURT = Rect(ALLEY, 0.0, ALLEY + SINGLES_WIDTH, COURT_LENGTH)
DOUBLES_COURT = Rect(0.0, 0.0, DOUBLES_WIDTH, COURT_LENGTH)

# Service boxes, named by the half of the court the ball must land in.
# "deuce"/"ad" are from the receiver's perspective facing the net.
SERVICE_BOXES = {
    # Far half (receiver at far baseline): serve struck from the near side.
    ("far", "deuce"): Rect(ALLEY, NET_Y, CENTER_X, SERVICE_FAR_Y),
    ("far", "ad"): Rect(CENTER_X, NET_Y, ALLEY + SINGLES_WIDTH, SERVICE_FAR_Y),
    # Near half (receiver at near baseline): serve struck from the far side.
    ("near", "deuce"): Rect(CENTER_X, SERVICE_NEAR_Y, ALLEY + SINGLES_WIDTH, NET_Y),
    ("near", "ad"): Rect(ALLEY, SERVICE_NEAR_Y, CENTER_X, NET_Y),
}

# The four outer corners of the doubles court, in a canonical order:
# near-left, near-right, far-right, far-left.
DOUBLES_CORNERS = np.array(
    [
        [0.0, 0.0],
        [DOUBLES_WIDTH, 0.0],
        [DOUBLES_WIDTH, COURT_LENGTH],
        [0.0, COURT_LENGTH],
    ],
    dtype=np.float64,
)

# Every painted line as a pair of endpoints in court coordinates.
# Used both for rendering overlays and for scoring candidate homographies
# against the white-line mask during automatic calibration.
COURT_LINES: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "baseline_near": ((0.0, 0.0), (DOUBLES_WIDTH, 0.0)),
    "baseline_far": ((0.0, COURT_LENGTH), (DOUBLES_WIDTH, COURT_LENGTH)),
    "sideline_left_doubles": ((0.0, 0.0), (0.0, COURT_LENGTH)),
    "sideline_right_doubles": ((DOUBLES_WIDTH, 0.0), (DOUBLES_WIDTH, COURT_LENGTH)),
    "sideline_left_singles": ((ALLEY, 0.0), (ALLEY, COURT_LENGTH)),
    "sideline_right_singles": ((ALLEY + SINGLES_WIDTH, 0.0), (ALLEY + SINGLES_WIDTH, COURT_LENGTH)),
    "service_line_near": ((ALLEY, SERVICE_NEAR_Y), (ALLEY + SINGLES_WIDTH, SERVICE_NEAR_Y)),
    "service_line_far": ((ALLEY, SERVICE_FAR_Y), (ALLEY + SINGLES_WIDTH, SERVICE_FAR_Y)),
    "center_service_line": ((CENTER_X, SERVICE_NEAR_Y), (CENTER_X, SERVICE_FAR_Y)),
    "center_mark_near": ((CENTER_X, 0.0), (CENTER_X, 0.15)),
    "center_mark_far": ((CENTER_X, COURT_LENGTH - 0.15), (CENTER_X, COURT_LENGTH)),
}


def court_zone(x: float, y: float) -> str:
    """Human-readable zone label for a bounce location (for stat aggregation)."""
    half = "near" if y < NET_Y else "far"
    depth_from_net = abs(y - NET_Y)
    if depth_from_net <= SERVICE_FROM_NET:
        depth = "short"
    elif depth_from_net <= SERVICE_FROM_NET + 3.0:
        depth = "mid"
    else:
        depth = "deep"
    third = SINGLES_WIDTH / 3.0
    if x < ALLEY + third:
        lane = "left"
    elif x < ALLEY + 2 * third:
        lane = "center"
    else:
        lane = "right"
    return f"{half}-{depth}-{lane}"
