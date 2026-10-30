"""
Geometric Calculations for Eye Aspect Ratio (EAR) and Landmark Tracking.
Supports 2D and 3D landmark points, robust against division by zero and noisy inputs.
"""

from __future__ import annotations
import math
from typing import List, Sequence, Tuple, Union

Point = Union[Tuple[float, float], Tuple[float, float, float], List[float]]

# Canonical MediaPipe Face Mesh landmarks for 6-point EAR calculation
# Left eye (subject's left eye, viewer's right):
# p1 (corner), p2 (upper1), p3 (upper2), p4 (corner), p5 (lower2), p6 (lower1)
LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

# Right eye (subject's right eye, viewer's left):
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]

# Eyebrow landmarks for furrowing/squint detection
LEFT_EYEBROW_LANDMARKS = [70, 63, 105, 66, 107]
RIGHT_EYEBROW_LANDMARKS = [336, 296, 334, 293, 300]


def euclidean_distance(p1: Point, p2: Point) -> float:
    """
    Computes Euclidean distance between two 2D or 3D points.
    Robust against mismatched dimensions (uses min length).
    """
    if not p1 or not p2:
        return 0.0
    dim = min(len(p1), len(p2))
    sum_sq = sum((float(p1[i]) - float(p2[i])) ** 2 for i in range(dim))
    return math.sqrt(sum_sq)


def calculate_ear(eye_landmarks: Sequence[Point]) -> float:
    """
    Computes Eye Aspect Ratio (EAR) based on Soukupová and Čech (2016).
    Formula:
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2.0 * ||p1 - p4||)

    Parameters:
        eye_landmarks: A sequence of 6 points ordered [p1, p2, p3, p4, p5, p6].
                       p1 and p4 are horizontal eye corners.
                       p2 and p6 are top/bottom vertical pair.
                       p3 and p5 are second top/bottom vertical pair.

    Returns:
        float: EAR value. 0.0 if invalid landmarks or zero horizontal width.
    """
    if len(eye_landmarks) < 6:
        return 0.0

    p1, p2, p3, p4, p5, p6 = (
        eye_landmarks[0],
        eye_landmarks[1],
        eye_landmarks[2],
        eye_landmarks[3],
        eye_landmarks[4],
        eye_landmarks[5],
    )

    # Compute vertical distances
    v1 = euclidean_distance(p2, p6)
    v2 = euclidean_distance(p3, p5)

    # Compute horizontal distance
    h = euclidean_distance(p1, p4)

    # Prevent zero division or degenerate geometry
    if h <= 1e-6:
        return 0.0

    ear = (v1 + v2) / (2.0 * h)
    return max(0.0, float(ear))


def smooth_exponential(current: float, previous: float, alpha: float = 0.3) -> float:
    """
    Exponential moving average smoother for jitter reduction.
    alpha: smoothing factor between 0.0 (keep previous) and 1.0 (no smoothing).
    """
    if previous is None or math.isnan(previous):
        return current
    return alpha * current + (1.0 - alpha) * previous


def calculate_gaze_eccentricity(
    left_eye: Sequence[Point],
    right_eye: Sequence[Point],
    frame_width: float = 640.0,
) -> float:
    """
    Computes horizontal gaze eccentricity for multi-monitor workstation ergonomics.
    Returns normalized offset from center [0.0 = centered, 1.0 = extreme lateral gaze].
    """
    if not left_eye or not right_eye or frame_width <= 0:
        return 0.0
    lx = sum(float(p[0]) for p in left_eye) / len(left_eye)
    rx = sum(float(p[0]) for p in right_eye) / len(right_eye)
    mid_x = (lx + rx) / 2.0
    center_x = frame_width / 2.0
    offset = abs(mid_x - center_x)
    return max(0.0, min(1.0, offset / center_x))
