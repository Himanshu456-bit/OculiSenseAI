"""
Unit tests for geometric math, distance, and Eye Aspect Ratio (EAR).
"""

import math
from oculisense.vision.geometry import (
    calculate_ear,
    euclidean_distance,
    smooth_exponential,
)


def test_euclidean_distance_2d():
    p1 = (0.0, 0.0)
    p2 = (3.0, 4.0)
    assert math.isclose(euclidean_distance(p1, p2), 5.0)


def test_euclidean_distance_3d():
    p1 = (1.0, 2.0, 3.0)
    p2 = (4.0, 6.0, 3.0)
    assert math.isclose(euclidean_distance(p1, p2), 5.0)


def test_euclidean_distance_zero_and_empty():
    assert euclidean_distance((0, 0), (0, 0)) == 0.0
    assert euclidean_distance([], (1, 1)) == 0.0
    assert euclidean_distance(None, (1, 1)) == 0.0


def test_calculate_ear_standard_open_eye():
    # Canonical open eye: width = 20, vertical spans = 6
    # p1, p2, p3, p4, p5, p6
    landmarks = [
        (0.0, 0.0),    # p1: corner left
        (5.0, 3.0),    # p2: upper 1
        (15.0, 3.0),   # p3: upper 2
        (20.0, 0.0),   # p4: corner right
        (15.0, -3.0),  # p5: lower 2
        (5.0, -3.0),   # p6: lower 1
    ]
    # v1 = ||p2 - p6|| = 6.0, v2 = ||p3 - p5|| = 6.0, h = ||p1 - p4|| = 20.0
    # EAR = (6.0 + 6.0) / (2.0 * 20.0) = 12.0 / 40.0 = 0.30
    ear = calculate_ear(landmarks)
    assert math.isclose(ear, 0.30, rel_tol=1e-3)


def test_calculate_ear_closed_eye():
    # Flat / closed eye: vertical distances near zero
    landmarks = [
        (0.0, 0.0),
        (5.0, 0.5),
        (15.0, 0.5),
        (20.0, 0.0),
        (15.0, -0.5),
        (5.0, -0.5),
    ]
    # v1 = 1.0, v2 = 1.0, h = 20.0 -> EAR = 2.0 / 40.0 = 0.05
    ear = calculate_ear(landmarks)
    assert math.isclose(ear, 0.05, rel_tol=1e-3)


def test_calculate_ear_degenerate_inputs():
    # Coincident points: width = 0
    coincident = [(0, 0)] * 6
    assert calculate_ear(coincident) == 0.0

    # Fewer than 6 landmarks
    assert calculate_ear([(0, 0), (1, 1)]) == 0.0
    assert calculate_ear([]) == 0.0


def test_smooth_exponential():
    prev = 0.30
    curr = 0.20
    # alpha=0.3 -> 0.3*0.20 + 0.7*0.30 = 0.06 + 0.21 = 0.27
    smoothed = smooth_exponential(curr, prev, alpha=0.3)
    assert math.isclose(smoothed, 0.27)
    # If prev is None, returns current
    assert smooth_exponential(0.25, None) == 0.25


def test_gaze_eccentricity_centered_and_lateral():
    from oculisense.vision.geometry import calculate_gaze_eccentricity

    # Centered eyes: left eye around x=280, right eye around x=360 on 640px screen
    # Midpoint = (280 + 360) / 2 = 320 = center
    left_centered = [(270.0, 200.0), (280.0, 195.0), (290.0, 195.0), (300.0, 200.0), (290.0, 205.0), (280.0, 205.0)]
    right_centered = [(340.0, 200.0), (350.0, 195.0), (360.0, 195.0), (370.0, 200.0), (360.0, 205.0), (350.0, 205.0)]

    ecc_center = calculate_gaze_eccentricity(left_centered, right_centered, frame_width=640.0)
    assert ecc_center < 0.05

    # Lateral gaze (looking at secondary monitor on the far right):
    # Eyes shifted toward x=560
    left_lateral = [(500.0, 200.0), (510.0, 195.0), (520.0, 195.0), (530.0, 200.0), (520.0, 205.0), (510.0, 205.0)]
    right_lateral = [(570.0, 200.0), (580.0, 195.0), (590.0, 195.0), (600.0, 200.0), (590.0, 205.0), (580.0, 205.0)]

    ecc_lateral = calculate_gaze_eccentricity(left_lateral, right_lateral, frame_width=640.0)
    assert ecc_lateral > 0.60

    # Degenerate inputs
    assert calculate_gaze_eccentricity([], [], frame_width=640.0) == 0.0
    assert calculate_gaze_eccentricity(left_centered, right_centered, frame_width=0.0) == 0.0
