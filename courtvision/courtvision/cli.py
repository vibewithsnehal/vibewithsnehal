"""Command line interface.

    courtvision analyze match.mp4 --out out/ [--corners corners.json] [--mode singles]
    courtvision live 0 --serve 8765            # webcam, watch at http://localhost:8765
    courtvision live rtsp://cam/stream         # ip camera
    courtvision live sim --serve 8765          # simulated live match (no camera needed)
    courtvision demo --out demo/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

from .calibration import CourtCalibration
from .live import LiveAnalyzer, LiveAnnotator, LiveStreamServer
from .phone import PhoneIngestServer
from .pipeline import AnalyzerConfig, analyze_video, _iter_video
from .overlay import render_overlay_video
from .stats import render_zone_heatmap
from .synthetic import MatchRenderer


def _print_summary(result) -> None:
    stats = result.stats
    d = stats.to_dict()["summary"]
    print("\n=== CourtVision analysis ===")
    print(f"calibration score : {result.calibration.score:.2f}")
    print(f"rallies           : {d['rallies']}")
    print(f"bounces called    : {d['total_bounces_called']}  (IN {d['in']} / OUT {d['out']})")
    print(f"close calls       : {d['close_calls']}")
    print(f"longest rally     : {d['longest_rally_shots']} shots")
    if d["avg_shot_speed_kmh"]:
        print(f"avg shot speed    : {d['avg_shot_speed_kmh']} km/h (est)")
        print(f"max shot speed    : {d['max_shot_speed_kmh']} km/h (est)")
    print("\ncalls:")
    for c in stats.calls:
        print(
            f"  frame {c.frame:5d}  {c.decision:3s}  margin {c.margin_m * 100:+6.1f} cm"
            f"  conf {c.confidence:.0%}  at ({c.court_xy[0]:.2f}, {c.court_xy[1]:.2f}) m"
            f"  [{c.zone}]"
        )
    if stats.zone_histogram:
        print("\nbounce heatmap (camera view, far side on top):")
        print(render_zone_heatmap(stats.zone_histogram))
    print()


def cmd_analyze(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = AnalyzerConfig(
        mode=args.mode,
        corners_file=args.corners,
        use_color_prior=not args.no_color_prior,
    )
    result = analyze_video(args.video, config)

    stats_path = out / "stats.json"
    stats_path.write_text(result.stats.to_json())
    print(f"stats written to {stats_path}")

    if not args.no_overlay:
        frames, _, _, _ = _iter_video(args.video)
        overlay_path = render_overlay_video(frames, result, out / "overlay.mp4")
        print(f"annotated replay written to {overlay_path}")

    _print_summary(result)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    video_path = out / "synthetic_match.mp4"
    print("rendering synthetic match (physics-simulated, ground truth known)...")
    renderer = MatchRenderer()
    _, truth = renderer.render_match(video_path=video_path)
    print(f"synthetic match written to {video_path}")

    corners_path = out / "corners.json"
    corners_path.write_text(
        json.dumps({"corners": truth.corner_pixels.tolist()}, indent=2)
    )

    config = AnalyzerConfig(corners_file=None if args.auto_calibrate else str(corners_path))
    result = analyze_video(video_path, config)

    (out / "stats.json").write_text(result.stats.to_json())
    frames, _, _, _ = _iter_video(video_path)
    render_overlay_video(frames, result, out / "overlay.mp4")
    print(f"outputs in {out}/: stats.json, overlay.mp4")

    # Score against ground truth.
    print("\nground truth check:")
    matched = 0
    for tb in truth.bounces:
        best = min(
            result.calls,
            key=lambda c: abs(c.frame - tb.frame),
            default=None,
        )
        if best is None or abs(best.frame - tb.frame) > 6:
            print(f"  MISSED bounce at frame {tb.frame} ({tb.expected_call})")
            continue
        err_cm = (
            ((best.court_xy[0] - tb.court_xy[0]) ** 2 + (best.court_xy[1] - tb.court_xy[1]) ** 2)
            ** 0.5
            * 100.0
        )
        ok = best.decision == tb.expected_call
        matched += ok
        print(
            f"  frame {tb.frame:5d}  truth {tb.expected_call:3s}  called {best.decision:3s}"
            f"  pos err {err_cm:5.1f} cm  {'OK' if ok else 'WRONG'}"
        )
    print(f"  {matched}/{len(truth.bounces)} calls correct")

    _print_summary(result)
    return 0


def _fmt_t(frame: int, fps: float) -> str:
    t = frame / fps
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def _live_phone(args: argparse.Namespace) -> int:
    """Phone-as-camera: browser page streams frames in, phone shows the calls."""
    import numpy as np

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    holder: dict = {}  # {'a': LiveAnalyzer} once enough frames arrived

    ingest = PhoneIngestServer(
        port=args.phone_port,
        cert_dir=out,
        stats_supplier=lambda: holder["a"].stats.to_json() if "a" in holder else "{}",
    )
    ingest.start()
    print("phone camera mode")
    print(f"  1. connect the phone to the same wifi as this machine")
    print(f"  2. open  {ingest.url}  on the phone")
    if ingest.tls:
        print("     (self-signed certificate: tap through the browser warning once)")
    else:
        print("     WARNING: openssl not found, serving plain http - most phone")
        print("     browsers will refuse camera access without https")
    print(f"  3. follow the 3 steps on screen (camera -> tap corners -> stream)")

    server = None
    if args.serve is not None:
        server = LiveStreamServer(port=args.serve)
        server.start()
        print(f"  laptop view: http://localhost:{server.port}/")
    print("waiting for the phone... (Ctrl+C to stop)")

    config = AnalyzerConfig(
        mode=args.mode,
        corners_file=args.corners,
        use_color_prior=not args.no_color_prior,
    )
    analyzer: LiveAnalyzer | None = None
    annotator: LiveAnnotator | None = None
    writer = None
    pending_calib: CourtCalibration | None = None
    buffer: list = []
    arrivals: list[float] = []

    try:
        while True:
            frame = ingest.next_frame(timeout=1.0)

            corners = ingest.take_corners()
            if corners is not None:
                calib = CourtCalibration.from_corners(np.array(corners))
                (out / "corners.json").write_text(
                    json.dumps({"corners": corners}, indent=2)
                )
                if analyzer is not None:
                    analyzer.set_calibration(calib)
                else:
                    pending_calib = calib
                print("corners received - calibration saved to corners.json")

            if frame is None:
                continue
            arrivals.append(time.monotonic())

            if analyzer is None:
                buffer.append(frame)
                if len(buffer) < 15:
                    continue
                fps = (len(arrivals) - 1) / max(arrivals[-1] - arrivals[0], 1e-6)
                fps = float(min(max(fps, 5.0), 60.0))
                print(f"phone connected - measured ~{fps:.0f} fps")
                analyzer = LiveAnalyzer(fps=fps, config=config, calibration=pending_calib)
                annotator = LiveAnnotator(analyzer)
                holder["a"] = analyzer
                frames_now, buffer = buffer, []
            else:
                frames_now = [frame]

            for f in frames_now:
                events = analyzer.process(f)
                for e in events:
                    _print_live_event(e, analyzer.fps)
                annotated = annotator.annotate(f, events)
                if args.record:
                    if writer is None:
                        writer = cv2.VideoWriter(
                            str(out / "live_annotated.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            analyzer.fps,
                            (annotated.shape[1], annotated.shape[0]),
                        )
                    writer.write(annotated)
                if server is not None:
                    server.update(annotated, analyzer.stats)
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        if analyzer is not None:
            for e in analyzer.finish():
                _print_live_event(e, analyzer.fps)
            (out / "live_stats.json").write_text(analyzer.stats.to_json())
            print(f"stats written to {out / 'live_stats.json'}")
        if writer is not None:
            writer.release()
        ingest.stop()
        if server is not None:
            server.stop()

    if analyzer is not None:
        d = analyzer.stats.to_dict()["summary"]
        print(
            f"\nsession: {d['rallies']} rallies, {d['total_bounces_called']} calls "
            f"(IN {d['in']} / OUT {d['out']}), {d['close_calls']} close"
        )
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    source = args.source
    if source == "phone":
        return _live_phone(args)
    if source == "sim":
        sim_path = Path(args.out) / "live_sim.mp4"
        if not sim_path.exists():
            print("rendering simulated match for the live demo...")
            MatchRenderer().render_match(video_path=sim_path)
        source = str(sim_path)
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        print(f"cannot open source: {args.source}", file=sys.stderr)
        return 1
    fps = args.fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
    is_file = not source.isdigit() and Path(source).exists()

    config = AnalyzerConfig(
        mode=args.mode,
        corners_file=args.corners,
        use_color_prior=not args.no_color_prior,
    )
    analyzer = LiveAnalyzer(fps=fps, config=config)
    annotator = LiveAnnotator(analyzer)

    server = None
    if args.serve is not None:
        server = LiveStreamServer(port=args.serve)
        server.start()
        print(f"live view: http://localhost:{server.port}/  (stats at /stats.json)")

    writer = None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    frame_period = 1.0 / fps
    next_deadline = time.monotonic()
    print("watching... (Ctrl+C to stop)")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            events = analyzer.process(frame)
            for e in events:
                _print_live_event(e, fps)
            annotated = annotator.annotate(frame, events)
            if args.record:
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(out / "live_annotated.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (annotated.shape[1], annotated.shape[0]),
                    )
                writer.write(annotated)
            if server is not None:
                server.update(annotated, analyzer.stats)
            # Pace file sources to wall-clock speed so "live" means live;
            # real cameras pace themselves.
            if is_file and not args.fast:
                next_deadline += frame_period
                delay = next_deadline - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        for e in analyzer.finish():
            _print_live_event(e, fps)
        cap.release()
        if writer is not None:
            writer.release()
        (out / "live_stats.json").write_text(analyzer.stats.to_json())
        print(f"stats written to {out / 'live_stats.json'}")
        if server is not None and args.linger and analyzer.frame_idx >= 0:
            print(f"stream ended - viewer stays up at http://localhost:{server.port}/ (Ctrl+C to exit)")
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                pass
        if server is not None:
            server.stop()

    d = analyzer.stats.to_dict()["summary"]
    print(
        f"\nsession: {d['rallies']} rallies, {d['total_bounces_called']} calls "
        f"(IN {d['in']} / OUT {d['out']}), {d['close_calls']} close"
    )
    return 0


def _print_live_event(e, fps: float) -> None:
    t = _fmt_t(e.frame, fps)
    if e.type == "calibrated":
        print(f"[{t}] court calibrated - line calling active")
    elif e.type == "rally_start":
        print(f"[{t}] rally started")
    elif e.type == "call" and e.call is not None:
        c = e.call
        flag = "" if c.confidence >= 0.6 else "  (close call)"
        print(
            f"[{t}] {c.decision:3s}  margin {c.margin_m * 100:+6.1f} cm"
            f"  conf {c.confidence:.0%}  at ({c.court_xy[0]:.2f}, {c.court_xy[1]:.2f}) m{flag}"
        )
    elif e.type == "rally_end" and e.rally is not None:
        r = e.rally
        speed = f", avg {r.avg_shot_speed_kmh:.0f} km/h" if r.avg_shot_speed_kmh else ""
        print(
            f"[{t}] rally over: {r.shots} shots, {r.bounces} bounces, "
            f"{r.duration_s:.1f}s{speed} - last ball {r.terminal_call}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="courtvision",
        description="AI tennis line calling and match stats from video",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_an = sub.add_parser("analyze", help="analyze a match video")
    p_an.add_argument("video", help="path to the match video")
    p_an.add_argument("--out", default="courtvision-out", help="output directory")
    p_an.add_argument("--mode", choices=["singles", "doubles"], default="singles")
    p_an.add_argument(
        "--corners",
        default=None,
        help='JSON file with the 4 doubles-court corner pixels: {"corners": [[x,y]x4]} '
        "(near-left, near-right, far-right, far-left); omit for auto-detection",
    )
    p_an.add_argument("--no-overlay", action="store_true", help="skip the annotated replay")
    p_an.add_argument(
        "--no-color-prior",
        action="store_true",
        help="disable the optic-yellow ball filter (non-yellow balls, odd lighting)",
    )
    p_an.set_defaults(func=cmd_analyze)

    p_live = sub.add_parser("live", help="analyze a live source, calling lines in real time")
    p_live.add_argument(
        "source",
        help="camera index (e.g. 0), stream URL (rtsp/http), video file, 'phone' "
        "to stream from a phone's browser, or 'sim' for a simulated live match",
    )
    p_live.add_argument("--out", default="courtvision-live", help="output directory")
    p_live.add_argument("--mode", choices=["singles", "doubles"], default="singles")
    p_live.add_argument("--corners", default=None, help="corner-pixel JSON (see analyze)")
    p_live.add_argument(
        "--serve",
        type=int,
        nargs="?",
        const=8765,
        default=None,
        metavar="PORT",
        help="serve the annotated feed + live stats over HTTP (default port 8765)",
    )
    p_live.add_argument("--record", action="store_true", help="record the annotated feed")
    p_live.add_argument("--fps", type=float, default=None, help="override source fps")
    p_live.add_argument(
        "--fast", action="store_true", help="file sources: run flat out instead of real time"
    )
    p_live.add_argument(
        "--linger",
        action="store_true",
        help="keep the web viewer up after the stream ends",
    )
    p_live.add_argument(
        "--phone-port",
        type=int,
        default=9443,
        help="port for the phone camera page (source 'phone')",
    )
    p_live.add_argument(
        "--no-color-prior",
        action="store_true",
        help="disable the optic-yellow ball filter (non-yellow balls, odd lighting)",
    )
    p_live.set_defaults(func=cmd_live)

    p_demo = sub.add_parser("demo", help="generate a synthetic match and analyze it")
    p_demo.add_argument("--out", default="courtvision-demo", help="output directory")
    p_demo.add_argument(
        "--auto-calibrate",
        action="store_true",
        help="use automatic court detection instead of the known corner pixels",
    )
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
