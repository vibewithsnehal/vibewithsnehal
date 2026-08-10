"""Match statistics compiled purely from what the camera saw.

Inputs are the trajectory segments, hit events, and line calls produced by
the pipeline.  Outputs:

  - rally list: duration, shot count, terminal call
  - in/out tallies and close-call count
  - bounce zone histogram (a text heatmap of where the ball landed)
  - estimated shot speeds (court-plane distance between consecutive
    contact events over elapsed time — an estimate, labeled as such)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .bounce import BounceEvent, HitEvent
from .calls import LineCall


@dataclass
class Rally:
    index: int
    start_frame: int
    end_frame: int
    shots: int
    bounces: int
    duration_s: float
    terminal_call: str | None  # decision of the last bounce in the rally
    avg_shot_speed_kmh: float | None


@dataclass
class MatchStats:
    fps: float
    rallies: list[Rally] = field(default_factory=list)
    calls: list[LineCall] = field(default_factory=list)
    zone_histogram: dict[str, int] = field(default_factory=dict)
    shot_speeds_kmh: list[float] = field(default_factory=list)

    @property
    def total_in(self) -> int:
        return sum(1 for c in self.calls if c.decision == "IN")

    @property
    def total_out(self) -> int:
        return sum(1 for c in self.calls if c.decision == "OUT")

    @property
    def close_calls(self) -> int:
        return sum(1 for c in self.calls if c.confidence < 0.6)

    def to_dict(self) -> dict:
        return {
            "summary": {
                "rallies": len(self.rallies),
                "total_bounces_called": len(self.calls),
                "in": self.total_in,
                "out": self.total_out,
                "close_calls": self.close_calls,
                "longest_rally_shots": max((r.shots for r in self.rallies), default=0),
                "avg_rally_duration_s": round(
                    float(np.mean([r.duration_s for r in self.rallies])), 2
                )
                if self.rallies
                else 0.0,
                "avg_shot_speed_kmh": round(float(np.mean(self.shot_speeds_kmh)), 1)
                if self.shot_speeds_kmh
                else None,
                "max_shot_speed_kmh": round(float(np.max(self.shot_speeds_kmh)), 1)
                if self.shot_speeds_kmh
                else None,
            },
            "rallies": [
                {
                    "index": r.index,
                    "start_frame": r.start_frame,
                    "end_frame": r.end_frame,
                    "shots": r.shots,
                    "bounces": r.bounces,
                    "duration_s": round(r.duration_s, 2),
                    "terminal_call": r.terminal_call,
                    "avg_shot_speed_kmh": round(r.avg_shot_speed_kmh, 1)
                    if r.avg_shot_speed_kmh is not None
                    else None,
                }
                for r in self.rallies
            ],
            "calls": [c.to_dict() for c in self.calls],
            "bounce_zones": dict(sorted(self.zone_histogram.items())),
            "notes": [
                "Shot speeds are court-plane estimates between contact events.",
                "Confidence < 0.6 marks a close call (margin within measurement noise).",
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def _shot_speeds(
    contacts: list[tuple[int, tuple[float, float]]], fps: float
) -> list[float]:
    """km/h between consecutive contact events with court positions."""
    speeds = []
    for (f0, p0), (f1, p1) in zip(contacts, contacts[1:]):
        dt = (f1 - f0) / fps
        if dt <= 0:
            continue
        dist = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        speed = dist / dt * 3.6
        if 10.0 < speed < 260.0:  # discard nonsense
            speeds.append(speed)
    return speeds


def build_rally(
    index: int,
    points: list,  # TrackPoints covering the rally
    bounces: list[BounceEvent],
    hits: list[HitEvent],
    calls: list[LineCall],
    fps: float,
) -> tuple[Rally, list[float]]:
    """Build one rally's stats.  Returns (rally, shot speeds in km/h)."""
    # Contact sequence for speed estimation: bounces carry court positions
    # via their calls; order all contacts by frame.
    contacts = sorted([(c.frame, c.court_xy) for c in calls], key=lambda t: t[0])
    speeds = _shot_speeds(contacts, fps)
    start_f, end_f = points[0].frame, points[-1].frame
    # Shots: every shot ends in a bounce (or a detected racquet contact
    # interrupts it), so take the stronger of the two signals.
    rally = Rally(
        index=index,
        start_frame=start_f,
        end_frame=end_f,
        shots=max(len(hits) + 1, len(bounces)),
        bounces=len(bounces),
        duration_s=(end_f - start_f) / fps,
        terminal_call=calls[-1].decision if calls else None,
        avg_shot_speed_kmh=float(np.mean(speeds)) if speeds else None,
    )
    return rally, speeds


def compile_stats(
    segments: list[list],  # list of track segments (TrackPoint lists)
    segment_events: list[tuple[list[BounceEvent], list[HitEvent], list[LineCall]]],
    fps: float,
) -> MatchStats:
    """One rally per track segment that contains at least one bounce."""
    stats = MatchStats(fps=fps)
    rally_idx = 0
    for seg, (bounces, hits, calls) in zip(segments, segment_events):
        if not seg or not bounces:
            continue
        rally_idx += 1
        rally, speeds = build_rally(rally_idx, seg, bounces, hits, calls, fps)
        stats.rallies.append(rally)
        stats.calls.extend(calls)
        stats.shot_speeds_kmh.extend(speeds)
        for c in calls:
            stats.zone_histogram[c.zone] = stats.zone_histogram.get(c.zone, 0) + 1
    return stats


def render_zone_heatmap(zone_histogram: dict[str, int]) -> str:
    """ASCII heatmap of bounce zones, far half on top (as seen from the camera)."""
    lanes = ["left", "center", "right"]
    rows = [
        ("far", "deep"),
        ("far", "mid"),
        ("far", "short"),
        ("near", "short"),
        ("near", "mid"),
        ("near", "deep"),
    ]
    lines = ["        " + "".join(f"{l:^9}" for l in lanes)]
    for half, depth in rows:
        cells = [zone_histogram.get(f"{half}-{depth}-{lane}", 0) for lane in lanes]
        label = f"{half[:1]}-{depth:<5}"
        lines.append(f"{label:>8}" + "".join(f"{c:^9}" for c in cells))
        if (half, depth) == ("far", "short"):
            lines.append("        " + "―" * 27 + "  net")
    return "\n".join(lines)
