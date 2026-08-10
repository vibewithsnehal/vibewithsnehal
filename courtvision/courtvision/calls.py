"""Line calling: project a bounce into court coordinates and rule on it.

Rules implemented (ITF):
  - The lines belong to the court they bound; a ball touching any part of a
    line is IN.  Court rectangles are measured to the outer line edges, so a
    contact patch overlapping the rectangle is IN.
  - The contact patch has a radius: the ball compresses on landing, so the
    center of the visible blob can be up to roughly a ball radius outside
    the paint and still be touching it.

Every call carries a margin (meters, positive = in by that much) and a
confidence derived from the margin versus the measurement uncertainty
(calibration + tracking error at that image location).  Close calls get low
confidence — like a human line judge saying "too close to call".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import court
from .calibration import CourtCalibration


@dataclass(frozen=True)
class LineCall:
    frame: int
    court_xy: tuple[float, float]
    image_xy: tuple[float, float]
    decision: str  # "IN" or "OUT"
    margin_m: float  # distance inside (+) or outside (-) the valid region
    confidence: float  # 0..1
    zone: str
    context: str  # "rally" or "serve"
    nearest_line: str = ""

    def to_dict(self) -> dict:
        return {
            "frame": self.frame,
            "court_x_m": round(self.court_xy[0], 3),
            "court_y_m": round(self.court_xy[1], 3),
            "decision": self.decision,
            "margin_cm": round(self.margin_m * 100.0, 1),
            "confidence": round(self.confidence, 3),
            "zone": self.zone,
            "context": self.context,
            "nearest_line": self.nearest_line,
        }


@dataclass
class LineCaller:
    calibration: CourtCalibration
    mode: str = "singles"  # or "doubles"
    contact_radius_m: float = court.BALL_RADIUS
    # Base measurement noise (meters): tracking jitter + bounce interpolation.
    base_sigma_m: float = 0.03
    calls: list[LineCall] = field(default_factory=list)

    def _region(self, context: str, serve_box: tuple[str, str] | None) -> court.Rect:
        if context == "serve" and serve_box is not None:
            return court.SERVICE_BOXES[serve_box]
        return court.SINGLES_COURT if self.mode == "singles" else court.DOUBLES_COURT

    def call(
        self,
        frame: int,
        image_xy: tuple[float, float],
        context: str = "rally",
        serve_box: tuple[str, str] | None = None,
    ) -> LineCall:
        cx, cy = self.calibration.image_to_court(np.array([image_xy]))[0]
        region = self._region(context, serve_box)
        # signed_distance < 0 means inside.  The ball is IN when the contact
        # patch overlaps the region: signed_distance <= contact_radius.
        sd = region.signed_distance(float(cx), float(cy))
        margin = self.contact_radius_m - sd  # >0 -> IN
        decision = "IN" if margin >= 0.0 else "OUT"

        # Uncertainty: local ground-plane scale converts ~1px of tracking
        # noise; add calibration quality and the interpolation floor.
        mpp = self.calibration.meters_per_pixel(np.array(image_xy))
        sigma = float(
            np.sqrt(self.base_sigma_m**2 + (2.0 * mpp) ** 2)
            + (1.0 - self.calibration.score) * 0.05
        )
        confidence = float(1.0 / (1.0 + np.exp(-abs(margin) / max(sigma, 1e-6))))
        # Map from [0.5, 1.0] to [0, 1]: zero margin = coin flip.
        confidence = (confidence - 0.5) * 2.0

        call = LineCall(
            frame=frame,
            court_xy=(float(cx), float(cy)),
            image_xy=(float(image_xy[0]), float(image_xy[1])),
            decision=decision,
            margin_m=float(margin),
            confidence=confidence,
            zone=court.court_zone(float(cx), float(cy)),
            context=context,
            nearest_line=self._nearest_line(float(cx), float(cy), region),
        )
        self.calls.append(call)
        return call

    @staticmethod
    def _nearest_line(x: float, y: float, region: court.Rect) -> str:
        edges = {
            "near-edge": abs(y - region.y0),
            "far-edge": abs(y - region.y1),
            "left-edge": abs(x - region.x0),
            "right-edge": abs(x - region.x1),
        }
        return min(edges, key=edges.get)
