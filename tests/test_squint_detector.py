"""
Unit tests for the SquintDetector and micro-expression strain engine.
"""

from oculisense.vision.squint_detector import SquintDetector


def test_normal_open_eyes_no_squint():
    detector = SquintDetector(
        closed_threshold=0.20,
        squint_upper_threshold=0.26,
        min_squint_duration_sec=1.5,
    )
    t = 1000.0

    # Normal open eyes EAR 0.32 for 3 seconds
    for _ in range(30):
        res = detector.process_frame(0.32, timestamp=t)
        t += 0.1
        assert not res.is_squinting
        assert res.total_squint_episodes == 0


def test_sustained_squint_detection():
    detector = SquintDetector(
        closed_threshold=0.20,
        squint_upper_threshold=0.26,
        min_squint_duration_sec=1.0,
    )
    t = 1000.0

    # Narrowed eyes EAR 0.23 (in [0.20, 0.26]) for 0.8 seconds (under min duration)
    for _ in range(8):
        res = detector.process_frame(0.23, timestamp=t)
        t += 0.1
        assert not res.is_squinting

    # Keep squinting past 1.0 second threshold
    for _ in range(4):
        res = detector.process_frame(0.23, timestamp=t)
        t += 0.1

    assert res.is_squinting
    assert res.total_squint_episodes == 1
    assert res.current_squint_duration_sec >= 1.0
    assert res.squint_intensity > 0.0


def test_squint_reset_on_reopening():
    detector = SquintDetector(
        closed_threshold=0.20,
        squint_upper_threshold=0.26,
        min_squint_duration_sec=1.0,
    )
    t = 1000.0

    # Squint for 1.2s
    for _ in range(12):
        detector.process_frame(0.23, timestamp=t)
        t += 0.1

    # Eyes open fully back to 0.32
    res = detector.process_frame(0.32, timestamp=t)
    assert not res.is_squinting
    assert res.current_squint_duration_sec == 0.0
    # Still records episode
    assert res.total_squint_episodes == 1
