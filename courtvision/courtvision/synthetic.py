"""Synthetic tennis footage with exact ground truth.

Renders a physically simulated ball (gravity, restitution) over an ITF court
through a pinhole camera.  Because we know the camera and the physics, every
bounce's true court position is known exactly — which makes the entire
pipeline testable end to end: detection, tracking, bounce localization,
homography accuracy, and the final IN/OUT decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import court

G = np.array([0.0, 0.0, -9.81])


@dataclass
class Camera:
    """Pinhole camera looking at the court from behind the near baseline."""

    width: int = 960
    height: int = 540
    focal_px: float = 900.0
    position: np.ndarray = field(
        default_factory=lambda: np.array([court.CENTER_X, -14.0, 5.5])
    )
    look_at: np.ndarray = field(
        default_factory=lambda: np.array([court.CENTER_X, 13.0, 0.0])
    )

    def __post_init__(self) -> None:
        self.K = np.array(
            [
                [self.focal_px, 0.0, self.width / 2.0],
                [0.0, self.focal_px, self.height / 2.0],
                [0.0, 0.0, 1.0],
            ]
        )
        forward = self.look_at - self.position
        forward = forward / np.linalg.norm(forward)
        up_world = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up_world)
        right = right / np.linalg.norm(right)
        down = np.cross(forward, right)
        self.R = np.vstack([right, down, forward])
        self.t = -self.R @ self.position

    def project(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        """World point -> (pixel xy, depth)."""
        Xc = self.R @ np.asarray(X, dtype=np.float64) + self.t
        depth = float(Xc[2])
        uvw = self.K @ Xc
        return uvw[:2] / uvw[2], depth

    def ground_homography(self) -> np.ndarray:
        """Court-plane (z=0) homography: court meters -> image pixels."""
        H = self.K @ np.column_stack([self.R[:, 0], self.R[:, 1], self.t])
        return H / H[2, 2]

    def court_corner_pixels(self) -> np.ndarray:
        """Doubles corners in the canonical order (for manual calibration)."""
        return np.array(
            [self.project(np.array([x, y, 0.0]))[0] for x, y in court.DOUBLES_CORNERS]
        )


@dataclass(frozen=True)
class TruthBounce:
    frame: int
    court_xy: tuple[float, float]
    expected_call: str  # vs the singles court


@dataclass
class GroundTruth:
    fps: float
    bounces: list[TruthBounce] = field(default_factory=list)
    hit_frames: list[int] = field(default_factory=list)
    corner_pixels: np.ndarray | None = None


@dataclass(frozen=True)
class Shot:
    """One delivery: re-aim the ball at contact toward a bounce target."""

    target_xy: tuple[float, float]  # intended bounce point (court meters)
    flight_time: float  # seconds from contact to bounce
    after_bounce_frames: int  # frames of follow-through before the next contact


@dataclass
class Rally:
    start_xy_z: tuple[float, float, float]  # first contact point (e.g. a serve toss)
    shots: list[Shot]
    gap_frames: int = 45  # dead time after the rally (ball off court)


def default_match_script() -> list[Rally]:
    """A short match sample: clean winners, clear outs, and one close call."""
    s = court.SINGLES_COURT
    return [
        Rally(  # baseline exchange ending with a deep ball called OUT
            start_xy_z=(3.2, 1.0, 1.1),
            shots=[
                Shot(target_xy=(7.5, 18.0), flight_time=1.05, after_bounce_frames=16),
                Shot(target_xy=(3.4, 4.5), flight_time=1.05, after_bounce_frames=16),
                Shot(target_xy=(6.8, 24.4), flight_time=1.0, after_bounce_frames=14),
            ],
        ),
        Rally(  # rally with a ball landing just inside the far baseline
            start_xy_z=(7.2, 1.2, 1.0),
            shots=[
                Shot(target_xy=(4.0, 17.2), flight_time=1.0, after_bounce_frames=16),
                Shot(target_xy=(6.4, 5.6), flight_time=1.0, after_bounce_frames=16),
                Shot(target_xy=(5.2, 23.55), flight_time=1.0, after_bounce_frames=14),
            ],
        ),
        Rally(  # wide miss past the singles sideline
            start_xy_z=(5.0, 1.5, 1.1),
            shots=[
                Shot(target_xy=(2.2, 16.5), flight_time=0.95, after_bounce_frames=16),
                Shot(target_xy=(10.3, 19.0), flight_time=0.95, after_bounce_frames=14),
            ],
        ),
        Rally(  # short angle winner, comfortably in
            start_xy_z=(6.0, 1.0, 1.2),
            shots=[
                Shot(target_xy=(2.4, 14.6), flight_time=0.95, after_bounce_frames=14),
            ],
        ),
    ]


def _expected_call(x: float, y: float) -> str:
    sd = court.SINGLES_COURT.signed_distance(x, y)
    return "IN" if sd <= court.BALL_RADIUS else "OUT"


class MatchRenderer:
    def __init__(
        self,
        camera: Camera | None = None,
        fps: float = 30.0,
        seed: int = 7,
        draw_players: bool = True,
        noise_sigma: float = 2.0,
    ) -> None:
        self.camera = camera or Camera()
        self.fps = fps
        self.rng = np.random.default_rng(seed)
        self.draw_players = draw_players
        self.noise_sigma = noise_sigma

    # -- static scenery -----------------------------------------------------

    def _paint_court(self, frame: np.ndarray) -> None:
        cam = self.camera
        # Court apron (a bigger rectangle around the doubles court).
        apron = np.array(
            [[-4.0, -6.0], [court.DOUBLES_WIDTH + 4.0, -6.0],
             [court.DOUBLES_WIDTH + 4.0, court.COURT_LENGTH + 6.0],
             [-4.0, court.COURT_LENGTH + 6.0]]
        )
        self._fill_ground_poly(frame, apron, (70, 110, 60))
        inner = np.array(
            [[0.0, 0.0], [court.DOUBLES_WIDTH, 0.0],
             [court.DOUBLES_WIDTH, court.COURT_LENGTH], [0.0, court.COURT_LENGTH]]
        )
        self._fill_ground_poly(frame, inner, (90, 140, 70))
        # Painted lines as true-width ground rectangles.
        wl = court.LINE_WIDTH / 2.0
        for (p0, p1) in court.COURT_LINES.values():
            p0, p1 = np.array(p0), np.array(p1)
            d = p1 - p0
            n = np.array([-d[1], d[0]])
            n = n / (np.linalg.norm(n) + 1e-12) * wl
            quad = np.array([p0 + n, p1 + n, p1 - n, p0 - n])
            self._fill_ground_poly(frame, quad, (245, 245, 245))
        # Net: a translucent band across the net line.
        net0, _ = cam.project(np.array([-0.5, court.NET_Y, 0.0]))
        net1, _ = cam.project(np.array([court.DOUBLES_WIDTH + 0.5, court.NET_Y, 0.0]))
        top0, _ = cam.project(np.array([-0.5, court.NET_Y, 1.07]))
        top1, _ = cam.project(np.array([court.DOUBLES_WIDTH + 0.5, court.NET_Y, 1.07]))
        pts = np.array([net0, net1, top1, top0], dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (40, 40, 40))
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0.0, dst=frame)
        cv2.polylines(frame, [np.array([top0, top1], dtype=np.int32)], False, (250, 250, 250), 2)

    def _fill_ground_poly(self, frame: np.ndarray, poly_xy: np.ndarray, color) -> None:
        pts = []
        for x, y in poly_xy:
            p, depth = self.camera.project(np.array([x, y, 0.0]))
            if depth <= 0.1:
                return
            pts.append(p)
        cv2.fillPoly(frame, [np.array(pts, dtype=np.int32)], color)

    def _draw_players(self, frame: np.ndarray, t: float) -> None:
        sway_near = 0.35 * np.sin(t * 1.3)
        sway_far = 0.3 * np.sin(t * 1.1 + 2.0)
        for base_xy, height, color, sway in [
            ((3.6, -1.0), 1.85, (30, 40, 120), sway_near),
            ((7.2, court.COURT_LENGTH + 1.0), 1.8, (120, 40, 30), sway_far),
        ]:
            foot = np.array([base_xy[0] + sway, base_xy[1], 0.0])
            head = foot + np.array([0.0, 0.0, height])
            pf, df = self.camera.project(foot)
            ph, _ = self.camera.project(head)
            if df <= 0.1:
                continue
            half_w = max(int(self.camera.focal_px * 0.25 / df), 2)
            center = ((pf + ph) / 2.0).astype(int)
            axes = (half_w, max(int(abs(pf[1] - ph[1]) / 2.0), 4))
            cv2.ellipse(frame, tuple(center), axes, 0.0, 0.0, 360.0, color, -1)

    # -- simulation ---------------------------------------------------------

    def render_match(
        self,
        script: list[Rally] | None = None,
        video_path: str | Path | None = None,
    ) -> tuple[list[np.ndarray] | None, GroundTruth]:
        """Simulate + render.  Returns (frames or None if writing to disk, truth)."""
        script = script or default_match_script()
        cam = self.camera
        dt = 1.0 / self.fps
        truth = GroundTruth(fps=self.fps, corner_pixels=cam.court_corner_pixels())

        writer = None
        frames: list[np.ndarray] | None = None
        if video_path is not None:
            Path(video_path).parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (cam.width, cam.height),
            )
        else:
            frames = []

        frame_idx = 0

        def emit(ball_xyz: np.ndarray | None) -> None:
            nonlocal frame_idx
            frame = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
            frame[:] = (60, 90, 55)  # surroundings
            self._paint_court(frame)
            if self.draw_players:
                self._draw_players(frame, frame_idx * dt)
            if ball_xyz is not None:
                p, depth = cam.project(ball_xyz)
                if depth > 0.1:
                    r = max(int(cam.focal_px * court.BALL_RADIUS / depth), 2)
                    cv2.circle(frame, (int(p[0]), int(p[1])), r, (60, 220, 235), -1)
            if self.noise_sigma > 0:
                noise = self.rng.normal(0.0, self.noise_sigma, frame.shape)
                frame = np.clip(frame.astype(np.float64) + noise, 0, 255).astype(np.uint8)
            if writer is not None:
                writer.write(frame)
            else:
                frames.append(frame)
            frame_idx += 1

        # Background warmup so the subtractor learns the empty court.
        for _ in range(30):
            emit(None)

        for rally in script:
            pos = np.array(rally.start_xy_z, dtype=np.float64)
            for shot_i, shot in enumerate(rally.shots):
                truth.hit_frames.append(frame_idx)
                target = np.array([shot.target_xy[0], shot.target_xy[1], 0.0])
                T = shot.flight_time
                vel = (target - pos - 0.5 * G * T * T) / T
                # Integrate to the bounce.
                t_flight = 0.0
                while True:
                    emit(pos)
                    new_pos = pos + vel * dt + 0.5 * G * dt * dt
                    new_vel = vel + G * dt
                    t_flight += dt
                    if new_pos[2] <= 0.0 and new_vel[2] < 0.0:
                        # Exact ground crossing inside this step (quadratic in tau).
                        a, b, c = 0.5 * G[2], vel[2], pos[2]
                        disc = max(b * b - 4 * a * c, 0.0)
                        tau = (-b - np.sqrt(disc)) / (2 * a)
                        tau = float(np.clip(tau, 0.0, dt))
                        contact = pos + vel * tau + 0.5 * G * tau * tau
                        truth.bounces.append(
                            TruthBounce(
                                frame=frame_idx,
                                court_xy=(float(contact[0]), float(contact[1])),
                                expected_call=_expected_call(contact[0], contact[1]),
                            )
                        )
                        pos = np.array([contact[0], contact[1], 0.0])
                        vel = np.array(
                            [new_vel[0] * 0.8, new_vel[1] * 0.8, -new_vel[2] * 0.72]
                        )
                        break
                    pos, vel = new_pos, new_vel
                # Follow-through after the bounce until the next contact.
                for _ in range(shot.after_bounce_frames):
                    emit(pos)
                    pos = pos + vel * dt + 0.5 * G * dt * dt
                    vel = vel + G * dt
                    if pos[2] <= 0.0:
                        pos[2] = 0.0
                        vel[2] = abs(vel[2]) * 0.7
            # Rally over: ball leaves the scene.
            for _ in range(rally.gap_frames):
                emit(None)

        if writer is not None:
            writer.release()
        return frames, truth
