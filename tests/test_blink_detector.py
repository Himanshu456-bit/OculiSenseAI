"""
Unit tests for the BlinkDetector state machine, PERCLOS, and frequency metrics.
"""

from oculisense.vision.blink_detector import BlinkDetector


def test_single_blink_cycle():
    detector = BlinkDetector(
        closed_threshold=0.20,
        min_blink_frames=2,
        prolonged_closure_sec=0.5,
    )
    t = 1000.0

    # Eyes open for 3 frames
    for _ in range(3):
        res = detector.process_frame(0.32, 0.32, timestamp=t)
        t += 0.033
        assert not res.is_eyes_closed
        assert res.total_blinks == 0
        assert not res.blink_detected_this_frame

    # Eyes close for 3 frames (100ms)
    for _ in range(3):
        res = detector.process_frame(0.12, 0.12, timestamp=t)
        t += 0.033
        assert res.is_eyes_closed
        assert not res.blink_detected_this_frame

    # Eyes open again -> blink completes
    res = detector.process_frame(0.32, 0.32, timestamp=t)
    assert not res.is_eyes_closed
    assert res.total_blinks == 1
    assert res.blink_detected_this_frame
    assert res.last_closure_duration_ms > 50.0
    assert not res.is_prolonged_closure


def test_jitter_rejection():
    detector = BlinkDetector(closed_threshold=0.20, min_blink_frames=3)
    t = 1000.0

    # 1 single frame of eye closure (noise/glitch)
    detector.process_frame(0.30, 0.30, timestamp=t)
    t += 0.033
    detector.process_frame(0.10, 0.10, timestamp=t)  # only 1 frame closed
    t += 0.033
    res = detector.process_frame(0.30, 0.30, timestamp=t)  # reopened

    # Should NOT count as a blink
    assert res.total_blinks == 0
    assert not res.blink_detected_this_frame


def test_prolonged_closure_microsleep():
    detector = BlinkDetector(
        closed_threshold=0.20,
        min_blink_frames=2,
        prolonged_closure_sec=0.4,
    )
    t = 1000.0

    # Closed for 0.6 seconds (> 0.4s threshold)
    for _ in range(18):
        res = detector.process_frame(0.08, 0.08, timestamp=t)
        t += 0.033

    assert res.is_prolonged_closure
    assert res.last_closure_duration_ms >= 400.0


def test_perclos_calculation():
    detector = BlinkDetector(closed_threshold=0.20, perclos_window_sec=10.0)
    t = 1000.0

    # 5 frames closed, 5 frames open -> 50% closure
    for _ in range(5):
        detector.process_frame(0.10, 0.10, timestamp=t)
        t += 0.1
    for _ in range(5):
        detector.process_frame(0.30, 0.30, timestamp=t)
        t += 0.1

    perclos = detector.get_perclos(t)
    assert abs(perclos - 0.50) < 1e-2


def test_eyes_not_present():
    detector = BlinkDetector(closed_threshold=0.20)
    t = 1000.0

    res = detector.process_frame(0.0, 0.0, timestamp=t, eyes_present=False)
    assert not res.eyes_present
    assert not res.is_eyes_closed
    assert res.total_blinks == 0


def test_realtime_blink_increment_and_rolling_bpm():
    """Verify blinks immediately increment total_blinks and rolling BPM calculates responsively."""
    detector = BlinkDetector(closed_threshold=0.20, min_blink_frames=2)
    t = 1000.0

    # 1. Initially 0 blinks, 0.0 BPM
    for _ in range(5):
        res = detector.process_frame(0.30, 0.30, timestamp=t)
        t += 0.033
    assert res.total_blinks == 0
    assert res.blink_rate_bpm == 0.0

    # 2. First blink: 3 frames closed, then reopen
    for _ in range(3):
        detector.process_frame(0.10, 0.10, timestamp=t)
        t += 0.033
    res1 = detector.process_frame(0.30, 0.30, timestamp=t)
    t += 0.033

    # Must increment immediately in real time
    assert res1.total_blinks == 1
    assert res1.blink_detected_this_frame
    assert detector.total_blinks == 1

    # Open for ~3.0 seconds
    for _ in range(90):
        res = detector.process_frame(0.30, 0.30, timestamp=t)
        t += 0.033
    assert res.total_blinks == 1

    # 3. Second blink at ~3.0s interval
    for _ in range(3):
        detector.process_frame(0.10, 0.10, timestamp=t)
        t += 0.033
    res2 = detector.process_frame(0.30, 0.30, timestamp=t)
    t += 0.033

    assert res2.total_blinks == 2
    assert res2.blink_detected_this_frame
    # BPM should be in normal human range ~14-24 BPM (not 1.0 or 2.0!)
    assert 14.0 <= res2.blink_rate_bpm <= 24.0

    # 4. Third blink at ~3.0s interval
    for _ in range(90):
        detector.process_frame(0.30, 0.30, timestamp=t)
        t += 0.033
    for _ in range(3):
        detector.process_frame(0.10, 0.10, timestamp=t)
        t += 0.033
    res3 = detector.process_frame(0.30, 0.30, timestamp=t)
    t += 0.033

    assert res3.total_blinks == 3
    assert 14.0 <= res3.blink_rate_bpm <= 24.0

    # 5. Staring for 15 seconds without blinking: BPM should decay responsively
    for _ in range(450):
        res_stare = detector.process_frame(0.30, 0.30, timestamp=t)
        t += 0.033
    assert res_stare.total_blinks == 3
    assert res_stare.blink_rate_bpm < 10.0  # Decayed due to sustained staring

