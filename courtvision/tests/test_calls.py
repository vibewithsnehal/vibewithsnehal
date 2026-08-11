import numpy as np
import pytest

from courtvision import court
from courtvision.calibration import CourtCalibration
from courtvision.calls import LineCaller
from courtvision.synthetic import Camera


@pytest.fixture()
def caller() -> LineCaller:
    cam = Camera()
    calib = CourtCalibration.from_corners(cam.court_corner_pixels())
    return LineCaller(calibration=calib)


def _image_point(court_xy, caller: LineCaller):
    return caller.calibration.court_to_image(np.array([court_xy]))[0]


def call_at(caller: LineCaller, court_xy, **kw):
    return caller.call(0, tuple(_image_point(court_xy, caller)), **kw)


def test_center_court_is_confidently_in(caller):
    c = call_at(caller, (court.CENTER_X, 6.0))
    assert c.decision == "IN"
    assert c.confidence > 0.95
    assert c.margin_m > 1.0


def test_clearly_long_is_out(caller):
    c = call_at(caller, (court.CENTER_X, court.COURT_LENGTH + 0.5))
    assert c.decision == "OUT"
    assert c.nearest_line == "far-edge"


def test_ball_touching_line_edge_is_in(caller):
    # Center of contact 2 cm beyond the outer edge of the baseline: the
    # contact patch (ball radius 3.3 cm) still overlaps the line -> IN.
    c = call_at(caller, (court.CENTER_X, court.COURT_LENGTH + 0.02))
    assert c.decision == "IN"
    assert c.confidence < 0.6  # and it's flagged as a close call


def test_just_beyond_contact_radius_is_out(caller):
    c = call_at(caller, (court.CENTER_X, court.COURT_LENGTH + 0.06))
    assert c.decision == "OUT"


def test_doubles_alley_out_in_singles_in_in_doubles(caller):
    alley_xy = (0.5, 10.0)  # inside the doubles alley
    c = call_at(caller, alley_xy)
    assert c.decision == "OUT"
    caller.mode = "doubles"
    c2 = call_at(caller, alley_xy)
    assert c2.decision == "IN"


def test_serve_box_call(caller):
    inside_far_deuce = (3.0, 15.0)
    c = call_at(caller, inside_far_deuce, context="serve", serve_box=("far", "deuce"))
    assert c.decision == "IN"
    c2 = call_at(caller, inside_far_deuce, context="serve", serve_box=("far", "ad"))
    assert c2.decision == "OUT"


def test_confidence_grows_with_margin(caller):
    margins = [0.0, 0.05, 0.3, 2.0]
    confs = [
        call_at(caller, (court.CENTER_X, court.COURT_LENGTH - m)).confidence
        for m in margins
    ]
    assert confs == sorted(confs)
