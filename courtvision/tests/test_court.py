import numpy as np

from courtvision import court


def test_dimensions():
    assert court.DOUBLES_WIDTH == 10.97
    assert court.SINGLES_WIDTH == 8.23
    assert court.COURT_LENGTH == 23.77
    assert abs(court.NET_Y - 11.885) < 1e-9
    assert abs(court.ALLEY - 1.37) < 1e-9


def test_signed_distance_inside_outside():
    r = court.SINGLES_COURT
    # dead center
    assert r.signed_distance(court.CENTER_X, court.NET_Y) < 0
    # just outside the right singles sideline
    assert r.signed_distance(court.ALLEY + court.SINGLES_WIDTH + 0.10, 10.0) > 0
    # exactly on the outer line edge
    assert abs(r.signed_distance(court.ALLEY, 10.0)) < 1e-9


def test_signed_distance_corner_is_euclidean():
    r = court.SINGLES_COURT
    d = r.signed_distance(court.ALLEY - 0.3, -0.4)
    assert abs(d - 0.5) < 1e-9


def test_service_boxes_tile_the_singles_midcourt():
    boxes = court.SERVICE_BOXES
    widths = {round(b.x1 - b.x0, 3) for b in boxes.values()}
    assert widths == {round(court.SINGLES_WIDTH / 2, 3)}
    depths = {round(b.y1 - b.y0, 3) for b in boxes.values()}
    assert depths == {6.40}


def test_zone_labels():
    assert court.court_zone(court.CENTER_X, 1.0) == "near-deep-center"
    assert court.court_zone(2.0, 13.0) == "far-short-left"
    assert court.court_zone(9.0, 22.0) == "far-deep-right"
