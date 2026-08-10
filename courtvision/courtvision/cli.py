"""Command line interface.

    courtvision analyze match.mp4 --out out/ [--corners corners.json] [--mode singles]
    courtvision demo --out demo/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    config = AnalyzerConfig(mode=args.mode, corners_file=args.corners)
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
    p_an.set_defaults(func=cmd_analyze)

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
