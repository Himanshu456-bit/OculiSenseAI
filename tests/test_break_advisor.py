"""
Unit tests for the BreakAdvisor and 20-20-20 rule state engine.
"""

from oculisense.analytics.break_advisor import BreakAdvisor
from oculisense.config import BreakConfig


def test_20_20_20_scheduled_trigger():
    cfg = BreakConfig(rule_20_20_20_interval_sec=10, rule_20_20_20_duration_sec=5)
    advisor = BreakAdvisor(config=cfg)
    t = 1000.0

    # Step through 8 seconds: working
    res = advisor.update(eyes_present=True, is_eyes_closed=False, timestamp=t)
    assert not res.is_break_recommended
    assert not res.is_in_break
    assert res.remaining_until_break_sec > 0

    # Advance past 10 seconds -> break triggered
    t += 11.0
    res = advisor.update(eyes_present=True, is_eyes_closed=False, timestamp=t)
    assert res.is_break_recommended
    assert res.reason == "SCHEDULED_20_20_20"
    assert res.total_prompted == 1


def test_break_rest_and_completion():
    cfg = BreakConfig(rule_20_20_20_interval_sec=10, rule_20_20_20_duration_sec=3)
    advisor = BreakAdvisor(config=cfg)
    t = 1000.0
    advisor.update(eyes_present=True, is_eyes_closed=False, timestamp=t)

    # Trigger break
    advisor.trigger_dynamic_break(reason="MANUAL")
    advisor.start_break(timestamp=t)

    # Rest for 2 seconds (eyes closed)
    t += 2.0
    res = advisor.update(eyes_present=True, is_eyes_closed=True, timestamp=t)
    assert res.is_in_break
    assert res.remaining_break_duration_sec <= 1.0

    # Complete rest period
    t += 1.5
    res = advisor.update(eyes_present=True, is_eyes_closed=True, timestamp=t)
    assert not res.is_in_break
    assert not res.is_break_recommended
    assert advisor._total_completed == 1
    assert advisor.compliance_rate == 100.0


def test_snooze_and_skip():
    cfg = BreakConfig(snooze_duration_sec=30)
    advisor = BreakAdvisor(config=cfg)
    advisor.trigger_dynamic_break("TEST")

    # Snooze break
    ok = advisor.snooze_break()
    assert ok
    assert advisor._snooze_count == 1

    # Skip break
    advisor.skip_break()
    assert advisor._total_skipped == 1
    assert advisor.compliance_rate == 0.0


def test_pomodoro_integration():
    cfg = BreakConfig(
        enable_pomodoro=True,
        pomodoro_work_sec=10,
        pomodoro_short_rest_sec=3,
        pomodoro_long_rest_sec=6,
        pomodoro_cycles_before_long_rest=2,
    )
    advisor = BreakAdvisor(config=cfg)
    t = 1000.0

    # Step before interval
    res = advisor.update(eyes_present=True, is_eyes_closed=False, timestamp=t)
    assert not res.is_break_recommended
    assert res.remaining_until_break_sec == 10.0

    # Trigger cycle 1 (short rest)
    t += 11.0
    res = advisor.update(eyes_present=True, is_eyes_closed=False, timestamp=t)
    assert res.is_break_recommended
    assert res.reason == "POMODORO_SHORT_REST"

    # Start and complete break 1
    advisor.start_break(timestamp=t)
    t += 3.5
    res = advisor.update(eyes_present=True, is_eyes_closed=True, timestamp=t)
    assert not res.is_in_break
    assert advisor._pomodoro_completed_cycles == 1

    # Work cycle 2 (triggers long rest because cycle 2 reaches cycles_before_long_rest=2)
    t += 11.0
    res = advisor.update(eyes_present=True, is_eyes_closed=False, timestamp=t)
    assert res.is_break_recommended
    assert res.reason == "POMODORO_LONG_REST"


def test_break_event_callback_logging():
    logged_events = []

    def on_event(reason: str, outcome: str, duration_sec: float):
        logged_events.append((reason, outcome, duration_sec))

    advisor = BreakAdvisor(on_break_event=on_event)

    # 1. Complete break event
    advisor.trigger_dynamic_break("FATIGUE_THRESHOLD")
    advisor.start_break(timestamp=100.0)
    advisor.complete_break(timestamp=120.0)

    assert len(logged_events) == 1
    assert logged_events[0] == ("FATIGUE_THRESHOLD", "COMPLETED", 20.0)

    # 2. Snooze break event
    advisor.trigger_dynamic_break("SCHEDULED_20_20_20")
    advisor.snooze_break(timestamp=150.0)
    assert len(logged_events) == 2
    assert logged_events[1][0] == "SCHEDULED_20_20_20"
    assert logged_events[1][1] == "SNOOZED"

    # 3. Skip break event
    advisor.trigger_dynamic_break("STRAIN_THRESHOLD")
    advisor.skip_break(timestamp=200.0)
    assert len(logged_events) == 3
    assert logged_events[2] == ("STRAIN_THRESHOLD", "SKIPPED", 0.0)


def test_absence_does_not_accumulate_into_work_interval():
    cfg = BreakConfig(rule_20_20_20_interval_sec=20, rule_20_20_20_duration_sec=5)
    advisor = BreakAdvisor(config=cfg)
    t = 1000.0

    # User works for 10 seconds (present)
    for _ in range(10):
        t += 1.0
        res = advisor.update(eyes_present=True, is_eyes_closed=False, timestamp=t)
        assert not res.is_break_recommended

    # User is absent for 15 seconds (eyes not present)
    for _ in range(15):
        t += 1.0
        res = advisor.update(eyes_present=False, is_eyes_closed=False, timestamp=t)
        assert not res.is_break_recommended

    # Total elapsed wall-clock time is 25s (> 20s interval), but active work was only 10s!
    # Break should NOT be recommended because absence was excluded from work interval.
    res = advisor.update(eyes_present=True, is_eyes_closed=False, timestamp=t)
    assert not res.is_break_recommended
    assert res.remaining_until_break_sec > 5.0


def test_extended_absence_resets_work_timer():
    cfg = BreakConfig(rule_20_20_20_interval_sec=20, rule_20_20_20_duration_sec=5)
    advisor = BreakAdvisor(config=cfg)
    t = 1000.0

    # User works for 18 seconds (close to 20s threshold)
    for _ in range(18):
        t += 1.0
        advisor.update(eyes_present=True, is_eyes_closed=False, timestamp=t)

    # User steps away from desk for 50 seconds (>= 45s natural rest)
    t += 50.0
    res = advisor.update(eyes_present=False, is_eyes_closed=False, timestamp=t)
    assert not res.is_break_recommended

    # User returns to desk -> timer should have reset, full 20s remaining
    t += 1.0
    res = advisor.update(eyes_present=True, is_eyes_closed=False, timestamp=t)
    assert not res.is_break_recommended
    assert res.remaining_until_break_sec >= 18.0


def test_skip_cooldown_prevents_immediate_reprompt():
    cfg = BreakConfig(snooze_duration_sec=30)
    advisor = BreakAdvisor(config=cfg, eval_window_sec=0.0)
    t = 1000.0

    # High strain triggers dynamic break
    res = advisor.update(eyes_present=True, is_eyes_closed=False, strain_score=85.0, timestamp=t, ear_avg=0.15)
    assert res.is_break_recommended
    assert res.reason == "CRITICAL_LOW_EAR_SCORE"

    # User skips the break
    advisor.skip_break(timestamp=t)
    assert advisor._state == advisor.STATE_WORKING

    # Next frame (33ms later) strain is still elevated, but skip cooldown prevents instant re-prompting
    t += 0.033
    res = advisor.update(eyes_present=True, is_eyes_closed=False, strain_score=85.0, timestamp=t, ear_avg=0.15)
    assert not res.is_break_recommended, "Break must be debounced by skip cooldown!"

    # 10 seconds later, still within cooldown
    t += 10.0
    res = advisor.update(eyes_present=True, is_eyes_closed=False, strain_score=85.0, timestamp=t, ear_avg=0.15)
    assert not res.is_break_recommended

    # After cooldown expires (60s+), new prompt can fire
    t += 65.0
    res = advisor.update(eyes_present=True, is_eyes_closed=False, strain_score=85.0, timestamp=t, ear_avg=0.15)
    assert res.is_break_recommended


def test_critical_low_overall_score_triggers_forced_rest():
    cfg = BreakConfig(snooze_duration_sec=30)
    advisor = BreakAdvisor(config=cfg, eval_window_sec=0.0)
    t = 1000.0

    # Moderate individual strain and fatigue (neither alone triggers >=80/85),
    # but combined they drop the overall wellness score below 40.0
    res = advisor.update(
        eyes_present=True,
        is_eyes_closed=False,
        strain_score=72.0,
        fatigue_score=72.0,
        timestamp=t,
        ear_avg=0.25,
    )
    assert res.is_break_recommended
    assert res.reason == "CRITICAL_LOW_SCORE"


def test_20s_ear_observation_window_suppresses_overlay_while_observing():
    """Verify that during the initial 20 seconds, the rest overlay is NEVER prompted."""
    advisor = BreakAdvisor(eval_window_sec=20.0)
    t = 1000.0

    # User starts working; EAR is 0.31
    res = advisor.update(eyes_present=True, is_eyes_closed=False, timestamp=t, ear_avg=0.31)
    assert not res.is_break_recommended
    assert res.is_evaluating
    assert res.eval_elapsed_sec == 0.0
    assert res.least_recorded_ear == 0.31

    # Step through 15 seconds: even with high fatigue, rest overlay MUST NOT appear before 20s
    for i in range(1, 16):
        t += 1.0
        res = advisor.update(
            eyes_present=True,
            is_eyes_closed=False,
            strain_score=85.0,
            fatigue_score=90.0,
            timestamp=t,
            ear_avg=0.28,
        )
        assert not res.is_break_recommended, f"Overlay should not appear at t={res.eval_elapsed_sec}s (<20s)!"
        assert res.is_evaluating
        assert res.eval_decision == "OBSERVING"


def test_20s_ear_observation_healthy_eyes_suppress_overlay():
    """Healthy open eyes (least EAR >= 0.18, normal score) over 20s must NOT show overlay."""
    advisor = BreakAdvisor(eval_window_sec=20.0)
    t = 1000.0

    # 20 full seconds with open healthy eyes (EAR ~0.30 - 0.34)
    for i in range(21):
        t += 1.0
        ear = 0.32 if i % 2 == 0 else 0.29
        res = advisor.update(
            eyes_present=True,
            is_eyes_closed=False,
            strain_score=20.0,
            fatigue_score=15.0,
            timestamp=t,
            ear_avg=ear,
        )

    # After 20 seconds: healthy passed!
    assert not res.is_break_recommended
    assert res.least_recorded_ear >= 0.29
    assert res.eval_decision == "HEALTHY_PASSED"


def test_20s_ear_observation_normal_brief_blinks_suppress_overlay():
    """Normal brief blinks (brief dip in EAR, closed < 1.5s in 20s) must NOT show overlay."""
    advisor = BreakAdvisor(eval_window_sec=20.0)
    t = 1000.0

    # Step 20 seconds at 30 FPS (0.033s increments = ~600 frames)
    # Simulate 4 normal 100ms blinks where EAR touches 0.14
    blink_times = {3.0, 7.0, 12.0, 17.0}
    for frame_idx in range(610):
        t += 0.033
        elapsed = frame_idx * 0.033
        # Check if in a blink (duration ~100ms)
        is_blink = any(abs(elapsed - bt) < 0.10 for bt in blink_times)
        ear = 0.14 if is_blink else 0.31
        res = advisor.update(
            eyes_present=True,
            is_eyes_closed=is_blink,
            strain_score=25.0,
            fatigue_score=20.0,
            timestamp=t,
            ear_avg=ear,
        )

    # Least recorded EAR did touch 0.14 during the blink,
    # BUT total closure was under 1.5s and eyes were open healthy 98% of the time.
    # Therefore, overlay MUST NOT be displayed!
    assert not res.is_break_recommended
    assert res.eval_decision == "HEALTHY_PASSED"


def test_20s_ear_observation_critical_low_ear_triggers_overlay():
    """Sustained low EAR / prolonged eye closure across 20s must trigger CRITICAL_LOW_EAR_SCORE overlay."""
    advisor = BreakAdvisor(eval_window_sec=20.0)
    t = 1000.0

    # 20 seconds with eyes closed / drooping (EAR = 0.12, eyes closed)
    for _ in range(21):
        t += 1.0
        res = advisor.update(
            eyes_present=True,
            is_eyes_closed=True,
            strain_score=60.0,
            fatigue_score=80.0,
            timestamp=t,
            ear_avg=0.12,
        )

    # 20 seconds complete with least EAR = 0.12 and prolonged closures -> BREAK TRIGGERED!
    assert res.is_break_recommended
    assert res.reason == "CRITICAL_LOW_EAR_SCORE"
    assert res.least_recorded_ear <= 0.12
    assert res.eval_decision == "REST_RECOMMENDED"


def test_skip_break_resets_20s_eval_window_and_cooldown():
    """Skipping a break engages cooldown and restarts the 20s evaluation window."""
    advisor = BreakAdvisor(eval_window_sec=20.0)
    t = 1000.0

    # Trigger break after 20s
    for _ in range(21):
        t += 1.0
        res = advisor.update(eyes_present=True, is_eyes_closed=True, timestamp=t, ear_avg=0.12)
    assert res.is_break_recommended

    # User skips
    advisor.skip_break(timestamp=t)
    assert advisor._state == advisor.STATE_WORKING

    # Next frame: evaluation window is reset to 0, and cooldown active
    t += 0.033
    res = advisor.update(eyes_present=True, is_eyes_closed=True, timestamp=t, ear_avg=0.12)
    assert not res.is_break_recommended, "Overlay must not appear immediately after skip!"
    assert res.eval_elapsed_sec < 1.0


def test_ergonomic_condition_1_microsleep_droop():
    """Condition 1: Continuous closure > 1.8s immediately triggers acute microsleep rest."""
    advisor = BreakAdvisor(eval_window_sec=20.0)
    t = 1000.0

    # 1.0 second of closure (normal slow blink or brief rest) -> does NOT trigger yet
    for _ in range(10):
        t += 0.1
        res = advisor.update(eyes_present=True, is_eyes_closed=True, timestamp=t, ear_avg=0.10)
    assert not res.is_break_recommended

    # Continued closure exceeding 1.8s -> triggers immediate acute microsleep break!
    for _ in range(10):
        t += 0.1
        res = advisor.update(eyes_present=True, is_eyes_closed=True, timestamp=t, ear_avg=0.10)
    assert res.is_break_recommended
    assert res.reason == "CRITICAL_LOW_EAR_SCORE"
    assert res.eval_condition_detail == "CONDITION_1_MICROSLEEP_DROOP"


def test_ergonomic_condition_2_sustained_micro_squinting():
    """Condition 2: Average EAR compressed to 0.21 - 0.25 across 2 consecutive cycles triggers strain."""
    advisor = BreakAdvisor(eval_window_sec=20.0)
    t = 1000.0

    # Cycle 1: 20 seconds with compressed EAR = 0.23 (squinting)
    for _ in range(21):
        t += 1.0
        res = advisor.update(eyes_present=True, is_eyes_closed=False, timestamp=t, ear_avg=0.23, blink_rate_bpm=14.0)
    # Cycle 1 ends: squint cycle count is 1, not yet 2
    assert not res.is_break_recommended
    assert advisor._squint_cycle_count == 1

    # Cycle 2: another 20 seconds with compressed EAR = 0.23
    for _ in range(21):
        t += 1.0
        res = advisor.update(eyes_present=True, is_eyes_closed=False, timestamp=t, ear_avg=0.23, blink_rate_bpm=14.0)
    # Cycle 2 ends: persists across 2 consecutive cycles -> TRIGGERED!
    assert res.is_break_recommended
    assert res.reason == "STRAIN_THRESHOLD"
    assert res.eval_condition_detail == "CONDITION_2_SUSTAINED_SQUINT"


def test_ergonomic_condition_3_severe_blink_suppression_stare_freeze():
    """Condition 3: Rolling BPM < 6 with high EAR > 0.30 after sustained work triggers 20-20-20 break."""
    advisor = BreakAdvisor(eval_window_sec=20.0)
    t = 1000.0
    advisor._last_break_end_time = t - 350.0  # worked for 350 seconds (> 300s threshold)

    # 20 seconds with stare freeze: BPM = 4.0, EAR = 0.33
    for _ in range(21):
        t += 1.0
        res = advisor.update(eyes_present=True, is_eyes_closed=False, timestamp=t, ear_avg=0.33, blink_rate_bpm=4.0)

    assert res.is_break_recommended
    assert res.reason == "SCHEDULED_20_20_20"
    assert res.eval_condition_detail == "CONDITION_3_STARE_FREEZE"


def test_ergonomic_condition_4_dual_threshold_fatigue_index():
    """Condition 4: Composite Fatigue Index >= 70 triggers lockout overlay, < 70 suppresses."""
    advisor = BreakAdvisor(eval_window_sec=20.0)
    t = 1000.0

    # Sub-case A: Normal brief blinks (4 blinks of 100ms in 20s, BPM=16, min EAR=0.14)
    # Fatigue Score = 15 + (0.02 * 60) + 0 + (0.24 - 0.14)*100 = 15 + 1.2 + 10 = 26.2 << 70
    blink_times = {3.0, 8.0, 13.0, 18.0}
    for frame_idx in range(610):
        t += 0.033
        elapsed = frame_idx * 0.033
        is_blk = any(abs(elapsed - bt) < 0.10 for bt in blink_times)
        res = advisor.update(
            eyes_present=True,
            is_eyes_closed=is_blk,
            timestamp=t,
            ear_avg=0.14 if is_blk else 0.32,
            blink_rate_bpm=16.0,
        )
    assert not res.is_break_recommended
    assert res.eval_decision == "HEALTHY_PASSED"
    assert res.dual_threshold_fatigue_score < 40.0

    # Sub-case B: High fatigue state:
    # 10 seconds closed intermittently across 20s (PERCLOS = 0.50), BPM = 4.0, min EAR = 0.08
    # Fatigue Score = 15 + (0.50 * 60) + (12 - 4)*3 + (0.24 - 0.08)*100 = 15 + 30 + 24 + 16 = 85.0 >= 70!
    advisor.reset(timestamp=t)
    for frame_idx in range(21):
        t += 1.0
        is_closed = (frame_idx % 2 == 0)
        ear_val = 0.08 if is_closed else 0.28
        res = advisor.update(
            eyes_present=True,
            is_eyes_closed=is_closed,
            timestamp=t,
            ear_avg=ear_val,
            blink_rate_bpm=4.0,
        )
    assert res.is_break_recommended
    assert res.dual_threshold_fatigue_score >= 70.0
    assert res.eval_condition_detail == "CONDITION_4_DUAL_THRESHOLD_FATIGUE"




