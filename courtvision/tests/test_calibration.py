import numpy as np

from courtvision import court
from courtvision.calibration import CourtCalibration, detect_court
from courtvision.synthetic import Camera, MatchRenderer


def test_manual_calibration_roundtrip():
    cam = Camera()
    calib = CourtCalibration.from_corners(cam.court_corner_pixels())
    pts = np.array([[2.0, 5.0], [court.CENTER_X, court.NET_Y], [9.0, 20.0]])
    img = calib.court_to_image(pts)
    back = calib.image_to_court(img)
    assert np.allclose(back, pts, atol=1e-6)


def test_manual_calibration_matches_true_homography():
    cam = Camera()
    calib = CourtCalibration.from_corners(cam.court_corner_pixels())
    # Project interior model points through the true camera and compare.
    for x, y in [(1.37, 5.485), (9.6, 18.285), (5.485, 11.885)]:
        true_px, _ = cam.project(np.array([x, y, 0.0]))
        est_px = calib.court_to_image(np.array([[x, y]]))[0]
        assert np.linalg.norm(true_px - est_px) < 0.5


def test_auto_court_detection_on_synthetic_frame():
    renderer = MatchRenderer(draw_players=False, noise_sigma=1.0)
    frames, _ = renderer.render_match(script=[], video_path=None)
    frame = frames[10]
    calib = detect_court(frame)
    assert calib is not None, "auto-detection failed on a clean synthetic frame"
    assert calib.score > 0.6

    # Compare against the true corner pixels.
    truth = renderer.camera.court_corner_pixels()
    est = calib.court_to_image(court.DOUBLES_CORNERS)
    err = np.linalg.norm(truth - est, axis=1)
    assert err.max() < 5.0, f"corner reprojection error too large: {err}"


def test_meters_per_pixel_larger_at_far_end():
    cam = Camera()
    calib = CourtCalibration.from_corners(cam.court_corner_pixels())
    near = calib.court_to_image(np.array([[court.CENTER_X, 2.0]]))[0]
    far = calib.court_to_image(np.array([[court.CENTER_X, 22.0]]))[0]
    assert calib.meters_per_pixel(far) > calib.meters_per_pixel(near)
