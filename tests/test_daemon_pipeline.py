"""
Integration tests for the full OculiSenseDaemon processing pipeline and demo mode.
"""

import os
import tempfile
import time
from oculisense.config import AppConfig
from oculisense.daemon import OculiSenseDaemon
from oculisense.storage.exporter import ReportExporter


def test_daemon_step_synthetic_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_daemon.db")
        cfg = AppConfig()
        cfg.privacy.database_filename = db_path
        cfg.demo_mode = True

        daemon = OculiSenseDaemon(config=cfg, force_synthetic=True)
        t = 1000.0

        # Step through 10 frames
        snapshots = []
        for _ in range(10):
            snap = daemon.step(custom_time=t)
            snapshots.append(snap)
            t += 0.033

        assert len(snapshots) == 10
        latest = snapshots[-1]
        assert latest.vision.is_synthetic
        assert latest.strain.strain_score >= 0.0
        assert latest.fatigue.fatigue_score >= 0.0
        assert latest.breaks.remaining_until_break_sec > 0

        # Stop daemon and verify database persistence
        daemon.stop()
        sessions = daemon.db.get_recent_sessions(limit=1)
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == daemon.session_id


def test_report_export_lifecycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "export_test.db")
        html_out = os.path.join(tmpdir, "report.html")
        csv_out = os.path.join(tmpdir, "report.csv")
        json_out = os.path.join(tmpdir, "report.json")

        cfg = AppConfig()
        cfg.privacy.database_filename = db_path
        cfg.demo_mode = True
        daemon = OculiSenseDaemon(config=cfg, force_synthetic=True)

        # Run 5 steps
        for _ in range(5):
            daemon.step()

        exporter = ReportExporter(daemon.db)
        summary = daemon.aggregator.get_session_summary()

        exporter.export_html_report(html_out, session_summary=summary)
        exporter.export_csv(csv_out)
        exporter.export_json(json_out, session_summary=summary)

        assert os.path.exists(html_out)
        assert os.path.exists(csv_out)
        assert os.path.exists(json_out)

        with open(html_out, "r", encoding="utf-8") as f:
            content = f.read()
            assert "OculiSenseAI" in content
            assert "Biometric Ergonomics Report" in content

        daemon.stop()


def test_synthetic_scenarios_and_modes():
    from oculisense.vision.synthetic_feed import SyntheticVideoSource
    source = SyntheticVideoSource()

    # Test cyclic mode
    f_cyclic = source.read_frame(100.0)
    assert f_cyclic.face_detected

    # Test forced away
    source.set_mode(SyntheticVideoSource.STATE_AWAY)
    f_away = source.read_frame(101.0)
    assert not f_away.face_detected
    assert len(f_away.left_eye) == 0

    # Test forced strained
    source.set_mode(SyntheticVideoSource.STATE_STRAINED)
    f_strained = source.read_frame(102.0)
    assert f_strained.face_detected
    assert f_strained.current_state_name == "STRAINED"

    # Test forced fatigued
    source.set_mode(SyntheticVideoSource.STATE_FATIGUED)
    f_fatigued = source.read_frame(103.0)
    assert f_fatigued.face_detected
    assert f_fatigued.current_state_name == "FATIGUED"


def test_calibration_lifecycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "calib_test.db")
        cfg = AppConfig()
        cfg.privacy.database_filename = db_path
        cfg.demo_mode = True
        daemon = OculiSenseDaemon(config=cfg, force_synthetic=True)

        calib_snapshots = []
        daemon.register_listener(lambda s: calib_snapshots.append(s))

        daemon.start_calibration(duration_sec=0.2)
        assert daemon._is_calibrating

        t = 1000.0
        for _ in range(10):
            snap = daemon.step(custom_time=t)
            t += 0.033

        assert not daemon._is_calibrating
        assert cfg.profile.baseline_open_ear > 0.0
        assert cfg.profile.baseline_closed_ear > 0.0
        assert cfg.profile.calibrated_at is not None

        # Verify listeners received calibration progress
        assert len(calib_snapshots) == 10
        assert any(s.is_calibrating for s in calib_snapshots)

        # Verify calibrated profile persisted to SQLite database
        db_prof = daemon.db.get_user_profile(cfg.profile.profile_id)
        assert db_prof is not None
        assert abs(db_prof.baseline_open_ear - cfg.profile.baseline_open_ear) < 1e-4

        daemon.stop()


def test_synthetic_cyclic_progression():
    """Verify that cyclic mode transitions smoothly through all 4 ergonomic states."""
    from oculisense.vision.synthetic_feed import SyntheticVideoSource
    source = SyntheticVideoSource()

    # Phase 0: 0s -> NORMAL (face present, healthy EAR ~0.32)
    f0 = source.read_frame(1000.0)
    assert f0.face_detected
    assert f0.current_state_name == "NORMAL"

    # Phase 1: +15s -> STRAINED (squinting, EAR ~0.23)
    f1 = source.read_frame(1015.0)
    assert f1.face_detected
    assert f1.current_state_name == "STRAINED"

    # Phase 2: +30s -> FATIGUED (sluggish blinks, heavy eyelids)
    f2 = source.read_frame(1030.0)
    assert f2.face_detected
    assert f2.current_state_name == "FATIGUED"

    # Phase 3: +45s -> AWAY (face absent, eye landmarks empty)
    f3 = source.read_frame(1045.0)
    assert not f3.face_detected
    assert f3.current_state_name == "AWAY_FROM_DESK"

    # Phase 0 cycle repeat: +60s -> NORMAL again
    f4 = source.read_frame(1060.0)
    assert f4.face_detected
    assert f4.current_state_name == "NORMAL"


def test_multi_user_profile_switching():
    """Verify switching between user profiles updates daemon and detector thresholds."""
    from oculisense.config import UserProfile
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "multi_user.db")
        cfg = AppConfig()
        cfg.privacy.database_filename = db_path
        cfg.demo_mode = True
        daemon = OculiSenseDaemon(config=cfg, force_synthetic=True)

        # Create Profile A (wide eyes: open 0.36, closed 0.18)
        prof_a = UserProfile(
            profile_id="prof_alice",
            name="Alice",
            baseline_open_ear=0.36,
            baseline_closed_ear=0.18,
        )
        # Create Profile B (narrow eyes: open 0.28, closed 0.14)
        prof_b = UserProfile(
            profile_id="prof_bob",
            name="Bob",
            baseline_open_ear=0.28,
            baseline_closed_ear=0.14,
        )

        daemon.create_or_update_user_profile(prof_a)
        assert daemon.config.profile.name == "Alice"
        th_a = daemon.blink_detector.closed_threshold
        assert abs(th_a - prof_a.dynamic_ear_threshold) < 1e-4

        # Switch to Bob
        daemon.create_or_update_user_profile(prof_b)
        assert daemon.config.profile.name == "Bob"
        th_b = daemon.blink_detector.closed_threshold
        assert abs(th_b - prof_b.dynamic_ear_threshold) < 1e-4
        assert th_b < th_a  # Bob's threshold should be narrower

        # Switch back to Alice
        ok = daemon.switch_user_profile("prof_alice")
        assert ok
        assert daemon.config.profile.name == "Alice"
        assert abs(daemon.blink_detector.closed_threshold - th_a) < 1e-4

        # Invalid profile id
        assert not daemon.switch_user_profile("nonexistent_id")

        daemon.stop()


def test_voice_and_audio_alerts():
    """Verify sound and speech alert dispatch execute cleanly without exceptions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "voice_test.db")
        cfg = AppConfig()
        cfg.privacy.database_filename = db_path
        cfg.breaks.enable_audio_cue = True
        cfg.breaks.enable_voice_alert = True
        daemon = OculiSenseDaemon(config=cfg, force_synthetic=True)

        # Trigger audio & voice dispatch
        daemon.play_alert_sound("STRAIN_THRESHOLD")
        daemon.play_alert_sound("FATIGUE_THRESHOLD")
        daemon.play_alert_sound("SCHEDULED_20_20_20")
        daemon.speak_alert("OculiSense test voice notification")

        # Step daemon with dynamic break to verify end-to-end integration
        daemon.break_advisor.trigger_dynamic_break(reason="STRAIN_THRESHOLD")
        snap = daemon.step()
        assert snap.breaks.is_break_recommended

        daemon.stop()

