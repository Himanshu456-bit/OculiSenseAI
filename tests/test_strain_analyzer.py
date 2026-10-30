"""
Unit tests for the StrainAnalyzer and Digital Eye Strain Index (DESI).
"""

from oculisense.analytics.strain_analyzer import StrainAnalyzer


def test_initial_optimal_strain():
    analyzer = StrainAnalyzer()
    t = 1000.0

    # User just started session with healthy 18 BPM and no squint
    res = analyzer.evaluate(blink_rate_bpm=18.0, squint_ratio=0.0, timestamp=t)
    assert res.strain_score == 0.0
    assert res.strain_level == StrainAnalyzer.LEVEL_OPTIMAL
    assert not res.is_warning


def test_screen_time_and_blink_suppression_accumulation():
    analyzer = StrainAnalyzer()
    t = 1000.0

    # Simulate 30 minutes of continuous presence (360 ticks of 5s = 1800 seconds)
    for _ in range(360):
        analyzer.update_presence(True, timestamp=t)
        t += 5.0

    # Intensely staring at screen: blink rate plunged to 5 BPM, 20% squinting
    res = analyzer.evaluate(blink_rate_bpm=5.0, squint_ratio=0.20, timestamp=t)

    assert res.strain_score > 30.0
    assert res.screen_time_factor > 15.0
    assert res.blink_suppression_factor > 15.0
    assert res.squint_factor > 10.0


def test_strain_reset_after_break():
    analyzer = StrainAnalyzer()
    t = 1000.0

    # Accumulate focus
    for _ in range(240):
        analyzer.update_presence(True, timestamp=t)
        t += 5.0

    # User takes a break
    analyzer.reset_focus(timestamp=t)
    res = analyzer.evaluate(blink_rate_bpm=18.0, squint_ratio=0.0, timestamp=t)

    assert res.continuous_focus_sec == 0.0
    assert res.screen_time_factor == 0.0
    assert res.strain_score == 0.0


def test_strain_auto_reset_on_extended_away():
    analyzer = StrainAnalyzer()
    t = 1000.0

    # Accumulate 20 minutes of focus
    for _ in range(240):
        analyzer.update_presence(True, timestamp=t)
        t += 5.0
    res1 = analyzer.evaluate(blink_rate_bpm=15.0, squint_ratio=0.0, timestamp=t)
    assert res1.continuous_focus_sec > 1000.0

    # User steps away for 10 seconds (short away - should NOT reset)
    for _ in range(10):
        analyzer.update_presence(False, timestamp=t)
        t += 1.0
    assert analyzer._accumulated_focus_sec > 1000.0

    # User steps away for another 40 seconds (total away = 50s > 45s threshold)
    for _ in range(40):
        analyzer.update_presence(False, timestamp=t)
        t += 1.0
    assert analyzer._accumulated_focus_sec == 0.0
    res2 = analyzer.evaluate(blink_rate_bpm=15.0, squint_ratio=0.0, timestamp=t)
    assert res2.continuous_focus_sec == 0.0


def test_multi_monitor_lateral_gaze_strain():
    analyzer = StrainAnalyzer()
    t = 1000.0

    # Looking straight ahead (eccentricity = 0.05) -> no lateral gaze penalty
    res_center = analyzer.evaluate(
        blink_rate_bpm=18.0,
        squint_ratio=0.0,
        timestamp=t,
        gaze_eccentricity=0.05,
    )
    assert res_center.gaze_eccentricity_factor == 0.0
    assert res_center.strain_score == 0.0

    # Looking sideways at secondary screen (eccentricity = 0.70) -> lateral gaze penalty applied
    res_lateral = analyzer.evaluate(
        blink_rate_bpm=18.0,
        squint_ratio=0.0,
        timestamp=t,
        gaze_eccentricity=0.70,
    )
    assert res_lateral.gaze_eccentricity_factor > 5.0
    assert res_lateral.strain_score > 5.0
    assert "eccentricity_pts" in res_lateral.contributing_factors
