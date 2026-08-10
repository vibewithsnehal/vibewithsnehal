"""End-to-end: simulate a match with known physics, analyze it, check the calls."""

import numpy as np
import pytest

from courtvision.calibration import CourtCalibration
from courtvision.pipeline import AnalyzerConfig, analyze_frames
from courtvision.synthetic import MatchRenderer, default_match_script


@pytest.fixture(scope="module")
def match():
    renderer = MatchRenderer(seed=7)
    frames, truth = renderer.render_match(script=default_match_script())
    calib = CourtCalibration.from_corners(truth.corner_pixels)
    result = analyze_frames(frames, fps=renderer.fps, calibration=calib)
    return truth, result


def test_all_ground_truth_bounces_are_called(match):
    truth, result = match
    for tb in truth.bounces:
        near = [c for c in result.calls if abs(c.frame - tb.frame) <= 6]
        assert near, f"no call near ground-truth bounce at frame {tb.frame}"


def test_call_decisions_match_ground_truth(match):
    truth, result = match
    wrong = []
    for tb in truth.bounces:
        best = min(result.calls, key=lambda c: abs(c.frame - tb.frame))
        if best.decision != tb.expected_call:
            wrong.append((tb, best))
    assert not wrong, f"wrong calls: {wrong}"


def test_bounce_localization_accuracy(match):
    truth, result = match
    errors = []
    for tb in truth.bounces:
        best = min(result.calls, key=lambda c: abs(c.frame - tb.frame))
        err = np.hypot(
            best.court_xy[0] - tb.court_xy[0], best.court_xy[1] - tb.court_xy[1]
        )
        errors.append(err)
    errors = np.array(errors)
    assert errors.mean() < 0.15, f"mean bounce error {errors.mean():.3f} m"
    assert errors.max() < 0.35, f"worst bounce error {errors.max():.3f} m"


def test_no_spurious_extra_calls(match):
    truth, result = match
    # Every call should correspond to some true bounce (within a few frames).
    for c in result.calls:
        near = [tb for tb in truth.bounces if abs(c.frame - tb.frame) <= 6]
        assert near, f"spurious call at frame {c.frame} -> {c.court_xy}"


def test_rally_segmentation(match):
    truth, result = match
    assert len(result.stats.rallies) == len(default_match_script())


def test_stats_content(match):
    _, result = match
    d = result.stats.to_dict()
    assert d["summary"]["total_bounces_called"] == len(result.calls)
    assert d["summary"]["in"] + d["summary"]["out"] == len(result.calls)
    assert d["bounce_zones"]
    assert d["summary"]["longest_rally_shots"] >= 1
    js = result.stats.to_json()
    assert '"decision"' in js


def test_auto_calibration_end_to_end():
    """Full pipeline with automatic court detection (no corner file)."""
    renderer = MatchRenderer(seed=11)
    script = default_match_script()[:2]
    frames, truth = renderer.render_match(script=script)
    result = analyze_frames(frames, fps=renderer.fps, config=AnalyzerConfig())
    assert result.calibration.score > 0.6
    for tb in truth.bounces:
        best = min(result.calls, key=lambda c: abs(c.frame - tb.frame))
        assert abs(best.frame - tb.frame) <= 6
        assert best.decision == tb.expected_call
