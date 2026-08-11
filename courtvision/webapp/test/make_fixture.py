"""Export a parity fixture for the JavaScript engine.

Renders the standard synthetic match, compresses every frame as JPEG (what a
phone camera path produces), runs the *Python* ball detector, and dumps the
per-frame candidates plus ground truth.  The Node test feeds these candidates
to the JS engine, which must reproduce the Python pipeline's calls.

Run from the courtvision/ directory:  python webapp/test/make_fixture.py
"""

import json
from pathlib import Path

import cv2

from courtvision.detection import BallDetector
from courtvision.synthetic import MatchRenderer, default_match_script

out_path = Path(__file__).parent / "fixture.json"

renderer = MatchRenderer(seed=7)
frames, truth = renderer.render_match(script=default_match_script())
detector = BallDetector()

candidates = []
for frame in frames:
    ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    import numpy as np

    decoded = cv2.imdecode(np.frombuffer(jpeg.tobytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    cands = detector.detect(decoded)
    candidates.append([[round(c.x, 2), round(c.y, 2), round(c.area, 1)] for c in cands])

fixture = {
    "fps": renderer.fps,
    "height": renderer.camera.height,
    "corner_pixels": [[round(v, 3) for v in p] for p in truth.corner_pixels],
    "truth_bounces": [
        {
            "frame": b.frame,
            "court_xy": [round(b.court_xy[0], 3), round(b.court_xy[1], 3)],
            "expected_call": b.expected_call,
        }
        for b in truth.bounces
    ],
    "candidates": candidates,
}
out_path.write_text(json.dumps(fixture))
print(f"wrote {out_path} ({out_path.stat().st_size // 1024} KB, {len(candidates)} frames)")
