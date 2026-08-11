"""Phone-as-camera ingest: TLS setup, frame round-trip, corners, scoreboard."""

import json
import ssl
import urllib.request

import cv2
import numpy as np
import pytest

from courtvision.calibration import CourtCalibration
from courtvision.live import LiveAnalyzer
from courtvision.phone import PhoneIngestServer, ensure_self_signed_cert
from courtvision.pipeline import AnalyzerConfig
from courtvision.synthetic import MatchRenderer, default_match_script


@pytest.fixture()
def server(tmp_path):
    srv = PhoneIngestServer(port=0, cert_dir=tmp_path, stats_supplier=lambda: '{"ok": 1}')
    srv.start()
    yield srv
    srv.stop()


def _ctx():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _url(server, path):
    scheme = "https" if server.tls else "http"
    return f"{scheme}://127.0.0.1:{server.port}{path}"


def _get(server, path):
    return urllib.request.urlopen(_url(server, path), timeout=5, context=_ctx()).read()


def _post(server, path, body, ctype):
    req = urllib.request.Request(
        _url(server, path), data=body, headers={"Content-Type": ctype}, method="POST"
    )
    return urllib.request.urlopen(req, timeout=5, context=_ctx())


def test_self_signed_cert_created_and_reused(tmp_path):
    pair = ensure_self_signed_cert(tmp_path)
    assert pair is not None
    cert, key = pair
    assert cert.exists() and key.exists()
    first = cert.read_bytes()
    pair2 = ensure_self_signed_cert(tmp_path)
    assert pair2[0].read_bytes() == first  # reused, not regenerated


def test_serves_capture_page_over_tls(server):
    assert server.tls  # openssl is available in CI/dev environments
    page = _get(server, "/").decode()
    assert "getUserMedia" in page
    assert "NEAR-LEFT" in page  # corner-tapping flow present


def test_frame_roundtrip(server):
    img = np.full((90, 160, 3), 40, dtype=np.uint8)
    cv2.circle(img, (80, 45), 6, (60, 220, 235), -1)
    ok, jpeg = cv2.imencode(".jpg", img)
    assert ok
    _post(server, "/ingest", jpeg.tobytes(), "image/jpeg")
    frame = server.next_frame(timeout=5.0)
    assert frame is not None
    assert frame.shape == (90, 160, 3)
    assert server.frames_received == 1


def test_corner_taps_roundtrip(server):
    corners = [[10.0, 200.0], [600.0, 200.0], [500.0, 50.0], [110.0, 50.0]]
    _post(server, "/corners", json.dumps({"corners": corners}).encode(), "application/json")
    got = server.take_corners()
    assert got == corners
    assert server.take_corners() is None  # consumed once


def test_bad_corners_rejected(server):
    with pytest.raises(urllib.error.HTTPError):
        _post(server, "/corners", b'{"corners": [[1,2]]}', "application/json")


def test_stats_endpoint_uses_supplier(server):
    assert json.loads(_get(server, "/stats.json")) == {"ok": 1}


def test_phone_frames_drive_the_analyzer(tmp_path):
    """Frames pushed through the ingest queue produce the same live calls."""
    renderer = MatchRenderer(seed=7)
    script = default_match_script()[:1]
    frames, truth = renderer.render_match(script=script)
    calib = CourtCalibration.from_corners(truth.corner_pixels)

    server = PhoneIngestServer(port=0, cert_dir=tmp_path)
    server.start()
    try:
        analyzer = LiveAnalyzer(fps=renderer.fps, calibration=calib)
        for f in frames:
            ok, jpeg = cv2.imencode(".jpg", f, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            _post(server, "/ingest", jpeg.tobytes(), "image/jpeg")
            frame = server.next_frame(timeout=5.0)
            assert frame is not None
            analyzer.process(frame)
        analyzer.finish()
    finally:
        server.stop()

    assert len(analyzer.stats.calls) == len(truth.bounces)
    for tb in truth.bounces:
        best = min(analyzer.stats.calls, key=lambda c: abs(c.frame - tb.frame))
        assert best.decision == tb.expected_call


def test_color_prior_config_plumbing():
    a = LiveAnalyzer(fps=30.0, config=AnalyzerConfig(use_color_prior=False))
    assert a.detector.use_color_prior is False
    b = LiveAnalyzer(fps=30.0)
    assert b.detector.use_color_prior is True
