"""
Unit tests for the SQLite DatabaseManager.
"""

import os
import tempfile
import time
from oculisense.analytics.aggregator import MinuteSummary, SessionSummary
from oculisense.storage.database import DatabaseManager


def test_database_lifecycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_oculisense.db")
        db = DatabaseManager(db_path)

        # 1. Start session
        session_id = "sess_001"
        db.start_session(session_id, user_id="user_test")

        # 2. Log minute telemetry
        m = MinuteSummary(
            timestamp=time.time(),
            avg_ear=0.31,
            avg_fatigue=25.0,
            avg_strain=18.0,
            blinks_in_minute=16,
            squint_duration_sec=1.2,
            perclos_avg=0.04,
            presence_ratio=1.0,
        )
        db.log_minute_telemetry(session_id, m)

        # 3. Log break event
        db.log_break_event(session_id, reason="20_20_20", outcome="COMPLETED")

        # 4. End session
        s = SessionSummary(
            start_time=time.time() - 3600,
            duration_minutes=60.0,
            total_blinks=960,
            avg_blink_rate_bpm=16.0,
            avg_fatigue_score=25.0,
            peak_fatigue_score=35.0,
            avg_strain_score=20.0,
            peak_strain_score=30.0,
            total_squint_sec=45.0,
            breaks_prompted=3,
            breaks_completed=3,
            compliance_rate_pct=100.0,
            screen_time_minutes=55.0,
        )
        db.end_session(session_id, s)

        # Verify queries
        sessions = db.get_recent_sessions(limit=5)
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == session_id
        assert sessions[0]["total_blinks"] == 960

        daily = db.get_daily_metrics(days=1)
        assert len(daily) >= 1
        assert daily[0]["total_blinks"] == 16


def test_privacy_purge():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "purge_test.db")
        db = DatabaseManager(db_path)

        # Insert old record
        session_id = "old_sess"
        db.start_session(session_id)
        m_old = MinuteSummary(
            timestamp=time.time() - (40 * 86400),  # 40 days ago
            avg_ear=0.30,
            avg_fatigue=20.0,
            avg_strain=15.0,
            blinks_in_minute=15,
            squint_duration_sec=0.0,
            perclos_avg=0.03,
            presence_ratio=1.0,
        )
        db.log_minute_telemetry(session_id, m_old)

        # Purge records older than 30 days
        deleted = db.purge_older_than(days=30)
        assert deleted == 1


def test_database_user_profile_crud():
    from oculisense.config import UserProfile
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "profile_test.db")
        db = DatabaseManager(db_path)

        p1 = UserProfile(
            profile_id="u1",
            name="Alice",
            baseline_open_ear=0.34,
            baseline_closed_ear=0.18,
            calibrated_at="2026-09-22 10:00:00",
        )
        p2 = UserProfile(
            profile_id="u2",
            name="Bob",
            baseline_open_ear=0.29,
            baseline_closed_ear=0.15,
            calibrated_at="2026-09-22 11:00:00",
        )

        db.save_user_profile(p1)
        db.save_user_profile(p2)

        fetched = db.get_user_profile("u1")
        assert fetched is not None
        assert fetched.name == "Alice"
        assert fetched.baseline_open_ear == 0.34

        all_profiles = db.list_user_profiles()
        assert len(all_profiles) == 2
        assert all_profiles[0].name == "Alice"
        assert all_profiles[1].name == "Bob"

        # Update profile
        p1.baseline_open_ear = 0.36
        db.save_user_profile(p1)
        updated = db.get_user_profile("u1")
        assert updated.baseline_open_ear == 0.36


def test_database_context_exception_safety():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "exc_test.db")
        db = DatabaseManager(db_path)

        # Trigger an invalid SQL syntax inside connection context
        try:
            with db._get_connection() as conn:
                conn.execute("INVALID SQL SYNTAX HERE")
        except Exception:
            pass

        # The connection should have been closed cleanly and subsequent queries should succeed
        sessions = db.get_recent_sessions()
        assert sessions == []


def test_database_break_events_query():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "break_test.db")
        db = DatabaseManager(db_path)

        db.log_break_event("sess_1", "20_20_20", "COMPLETED", 20.0)
        db.log_break_event("sess_1", "STRAIN", "SKIPPED", 0.0)
        db.log_break_event("sess_2", "FATIGUE", "SNOOZED", 300.0)

        all_events = db.get_break_events()
        assert len(all_events) == 3

        sess1_events = db.get_break_events(session_id="sess_1")
        assert len(sess1_events) == 2
        assert sess1_events[0]["session_id"] == "sess_1"
