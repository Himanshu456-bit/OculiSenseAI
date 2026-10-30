"""
Unit tests for the FatigueModel and drowsiness scoring.
"""

from oculisense.analytics.fatigue_model import FatigueModel


def test_alert_baseline_fatigue():
    model = FatigueModel()
    t = 1000000.0  # arbitrary epoch

    # Low PERCLOS (0.02), alert blinks (180ms), 18 BPM
    res = model.predict(
        perclos=0.02,
        blink_rate_bpm=18.0,
        recent_blink_durations_ms=[180.0, 190.0, 175.0],
        is_prolonged_closure=False,
        timestamp=t,
    )
    assert res.fatigue_score < 40.0
    assert res.fatigue_level in (FatigueModel.LEVEL_LOW, FatigueModel.LEVEL_MODERATE)
    assert not res.microsleep_risk


def test_drowsy_perclos_and_long_blinks():
    model = FatigueModel()
    t = 1000000.0

    # High PERCLOS (0.22), long sluggish blinks (550ms)
    res = model.predict(
        perclos=0.22,
        blink_rate_bpm=8.0,
        recent_blink_durations_ms=[500.0, 600.0, 550.0],
        is_prolonged_closure=False,
        timestamp=t,
    )
    assert res.fatigue_score >= 60.0
    assert res.is_warning


def test_microsleep_override():
    model = FatigueModel()
    t = 1000000.0

    # Prolonged closure detected
    res = model.predict(
        perclos=0.10,
        blink_rate_bpm=15.0,
        recent_blink_durations_ms=[200.0],
        is_prolonged_closure=True,
        timestamp=t,
    )
    assert res.microsleep_risk
    assert res.fatigue_score >= 85.0
    assert res.fatigue_level == FatigueModel.LEVEL_CRITICAL


def test_dlib_consecutive_closed_frames_drowsiness():
    """Validates the akshaybahadur21/Drowsiness_Detection consecutive frame standard."""
    model = FatigueModel()
    t = 1000000.0

    # 20 consecutive closed frames (~0.66s) triggers immediate severe drowsiness
    res = model.predict(
        perclos=0.08,
        blink_rate_bpm=14.0,
        recent_blink_durations_ms=[220.0],
        is_prolonged_closure=False,
        consecutive_closed_frames=20,
        timestamp=t,
    )
    assert res.microsleep_risk
    assert res.fatigue_score >= 85.0
    assert res.fatigue_level == FatigueModel.LEVEL_CRITICAL
    assert res.metrics_summary["consecutive_closed_frames"] == 20
