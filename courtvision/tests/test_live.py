"""Live mode: same verdicts as batch, bounded latency, working web server."""

import json
import urllib.request

import numpy as np
import pytest

from courtvision.calibration import CourtCalibration
from courtvision.live import LiveAnalyzer, LiveAnnotator, LiveStreamServer
from courtvision.pipeline import analyze_frames
from courtvision.synthetic import MatchRenderer, default_match_script


@pytest.fixture(scope="module")
def match():
    renderer = MatchRenderer(seed=7)
    frames, truth = renderer.render_match(script=default_match_script())
    calib = CourtCalibration.from_corners(truth.corner_pixels)
    return renderer, frames, truth, calib


@pytest.fixture(scope="module")
def live_run(match):
    renderer, frames, truth, calib = match
    analyzer = LiveAnalyzer(fps=renderer.fps, calibration=calib)
    events = []
    for frame in frames:
        for e in analyzer.process(frame):
            events.append((analyzer.frame_idx, e))
    for e in analyzer.finish():
        events.append((analyzer.frame_idx, e))
    return analyzer, events


def test_live_matches_batch_verdicts(match, live_run):
    renderer, frames, truth, calib = match
    analyzer, events = live_run
    batch = analyze_frames(frames, fps=renderer.fps, calibration=calib)

    live_calls = sorted(analyzer.stats.calls, key=lambda c: c.frame)
    batch_calls = sorted(batch.calls, key=lambda c: c.frame)
    assert len(live_calls) == len(batch_calls)
    for lc, bc in zip(live_calls, batch_calls):
        assert abs(lc.frame - bc.frame) <= 2
        assert lc.decision == bc.decision
        assert np.hypot(
            lc.court_xy[0] - bc.court_xy[0], lc.court_xy[1] - bc.court_xy[1]
        ) < 0.05


def test_live_calls_match_ground_truth(match, live_run):
    _, _, truth, _ = match
    analyzer, _ = live_run
    for tb in truth.bounces:
        best = min(analyzer.stats.calls, key=lambda c: abs(c.frame - tb.frame))
        assert abs(best.frame - tb.frame) <= 6
        assert best.decision == tb.expected_call


def test_call_latency_is_bounded(live_run):
    """A verdict arrives shortly after contact.

    Typical latency is ~6 frames (0.2 s at 30 fps): the ruling needs a few
    post-bounce frames to confirm the kink and run the hit test.  Brief
    occlusions right after a bounce extend it — the verdict cannot exist
    before the post-bounce trajectory has been observed — so the hard bound
    here is 18 frames (0.6 s).
    """
    _, events = live_run
    call_events = [(at, e) for at, e in events if e.type == "call"]
    assert call_events
    latencies = [emitted_at - e.call.frame for emitted_at, e in call_events]
    assert all(0 <= l <= 18 for l in latencies), f"latencies: {latencies}"
    assert np.median(latencies) <= 8


def test_rally_lifecycle_events(live_run):
    analyzer, events = live_run
    starts = [e for _, e in events if e.type == "rally_start"]
    ends = [e for _, e in events if e.type == "rally_end"]
    assert len(ends) == len(default_match_script())
    assert len(starts) >= len(ends)
    assert len(analyzer.stats.rallies) == len(ends)
    for r in analyzer.stats.rallies:
        assert r.duration_s > 0
        assert r.bounces >= 1


def test_live_stats_are_cumulative(live_run):
    analyzer, _ = live_run
    d = analyzer.stats.to_dict()
    assert d["summary"]["total_bounces_called"] == len(analyzer.stats.calls)
    assert d["summary"]["in"] + d["summary"]["out"] == len(analyzer.stats.calls)
    assert d["bounce_zones"]


def test_annotator_returns_frames(match):
    renderer, frames, truth, calib = match
    analyzer = LiveAnalyzer(fps=renderer.fps, calibration=calib)
    annotator = LiveAnnotator(analyzer)
    for frame in frames[:80]:
        events = analyzer.process(frame)
        out = annotator.annotate(frame, events)
        assert out.shape == frame.shape
        assert out is not frame


def test_stream_server_serves_page_and_stats(match):
    renderer, frames, truth, calib = match
    analyzer = LiveAnalyzer(fps=renderer.fps, calibration=calib)
    server = LiveStreamServer(port=0)  # ephemeral port
    server.start()
    try:
        analyzer.process(frames[0])
        server.update(frames[0], analyzer.stats)
        base = f"http://127.0.0.1:{server.port}"
        page = urllib.request.urlopen(f"{base}/", timeout=5).read().decode()
        assert "CourtVision" in page
        stats = json.loads(urllib.request.urlopen(f"{base}/stats.json", timeout=5).read())
        assert "summary" in stats
    finally:
        server.stop()
