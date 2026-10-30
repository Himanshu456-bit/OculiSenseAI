"""
Privacy-First SQLite Database Manager for OculiSenseAI.
Stores only anonymized numerical telemetry, session rollups, and compliance logs locally.
No raw images or video frames are ever recorded.
"""

from __future__ import annotations
import contextlib
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

from oculisense.analytics.aggregator import MinuteSummary, SessionSummary
from oculisense.config import UserProfile


class DatabaseManager:
    """
    Manages local SQLite database for personal analytics and compliance tracking.
    """

    def __init__(self, db_path: str = "oculisense.db") -> None:
        self.db_path = db_path
        # Ensure directory exists
        dirname = os.path.dirname(os.path.abspath(db_path))
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        self.init_db()

    @contextlib.contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def close(self) -> None:
        """Explicit close hook for database resources."""
        pass

    def init_db(self) -> None:
        """Create tables and indices if not already present."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Sessions table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL,
                    duration_minutes REAL DEFAULT 0,
                    total_blinks INTEGER DEFAULT 0,
                    avg_blink_rate_bpm REAL DEFAULT 0,
                    avg_fatigue REAL DEFAULT 0,
                    peak_fatigue REAL DEFAULT 0,
                    avg_strain REAL DEFAULT 0,
                    peak_strain REAL DEFAULT 0,
                    total_squint_sec REAL DEFAULT 0,
                    breaks_prompted INTEGER DEFAULT 0,
                    breaks_completed INTEGER DEFAULT 0,
                    compliance_rate REAL DEFAULT 100.0,
                    screen_time_minutes REAL DEFAULT 0
                )
                """
            )

            # Minute-level rolling telemetry
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS telemetry_minute (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    avg_ear REAL,
                    avg_fatigue REAL,
                    avg_strain REAL,
                    blinks_in_minute INTEGER,
                    squint_duration_sec REAL,
                    perclos_avg REAL,
                    presence_ratio REAL,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp ON telemetry_minute(timestamp)"
            )

            # Break compliance events
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS break_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    reason TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    duration_sec REAL DEFAULT 0,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
                """
            )

            # User profiles
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS user_profiles (
                    profile_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    baseline_open_ear REAL NOT NULL,
                    baseline_closed_ear REAL NOT NULL,
                    calibrated_at TEXT
                )
                """
            )

            conn.commit()

    def start_session(self, session_id: str, user_id: str = "default") -> None:
        """Create new session record."""
        now = time.time()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, user_id, start_time)
                VALUES (?, ?, ?)
                """,
                (session_id, user_id, now),
            )
            conn.commit()

    def end_session(self, session_id: str, summary: SessionSummary) -> None:
        """Update session record with final metrics."""
        now = time.time()
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE sessions SET
                    end_time = ?,
                    duration_minutes = ?,
                    total_blinks = ?,
                    avg_blink_rate_bpm = ?,
                    avg_fatigue = ?,
                    peak_fatigue = ?,
                    avg_strain = ?,
                    peak_strain = ?,
                    total_squint_sec = ?,
                    breaks_prompted = ?,
                    breaks_completed = ?,
                    compliance_rate = ?,
                    screen_time_minutes = ?
                WHERE session_id = ?
                """,
                (
                    now,
                    summary.duration_minutes,
                    summary.total_blinks,
                    summary.avg_blink_rate_bpm,
                    summary.avg_fatigue_score,
                    summary.peak_fatigue_score,
                    summary.avg_strain_score,
                    summary.peak_strain_score,
                    summary.total_squint_sec,
                    summary.breaks_prompted,
                    summary.breaks_completed,
                    summary.compliance_rate_pct,
                    summary.screen_time_minutes,
                    session_id,
                ),
            )
            conn.commit()

    def log_minute_telemetry(self, session_id: str, m: MinuteSummary) -> None:
        """Insert aggregated 1-minute telemetry record."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO telemetry_minute (
                    session_id, timestamp, avg_ear, avg_fatigue, avg_strain,
                    blinks_in_minute, squint_duration_sec, perclos_avg, presence_ratio
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    m.timestamp,
                    m.avg_ear,
                    m.avg_fatigue,
                    m.avg_strain,
                    m.blinks_in_minute,
                    m.squint_duration_sec,
                    m.perclos_avg,
                    m.presence_ratio,
                ),
            )
            conn.commit()

    def log_break_event(
        self,
        session_id: str,
        reason: str,
        outcome: str,
        duration_sec: float = 20.0,
    ) -> None:
        """Log rest event (e.g. COMPLETED, SNOOZED, SKIPPED)."""
        now = time.time()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO break_events (session_id, timestamp, reason, outcome, duration_sec)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, now, reason, outcome, duration_sec),
            )
            conn.commit()

    def get_break_events(
        self, session_id: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Retrieve recorded break events for compliance tracking."""
        with self._get_connection() as conn:
            if session_id:
                cursor = conn.execute(
                    """
                    SELECT id, session_id, timestamp, reason, outcome, duration_sec
                    FROM break_events
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (session_id, limit),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT id, session_id, timestamp, reason, outcome, duration_sec
                    FROM break_events
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
            return [dict(row) for row in cursor.fetchall()]

    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve recent session summaries."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM sessions ORDER BY start_time DESC LIMIT ?
                """,
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_daily_metrics(self, days: int = 7) -> List[Dict[str, Any]]:
        """Aggregate metrics per day for trend charts."""
        cutoff = time.time() - (days * 86400.0)
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT
                    date(datetime(timestamp, 'unixepoch', 'localtime')) as log_date,
                    COUNT(*) as active_minutes,
                    ROUND(AVG(avg_fatigue), 1) as mean_fatigue,
                    ROUND(MAX(avg_fatigue), 1) as max_fatigue,
                    ROUND(AVG(avg_strain), 1) as mean_strain,
                    ROUND(MAX(avg_strain), 1) as max_strain,
                    SUM(blinks_in_minute) as total_blinks,
                    ROUND(SUM(squint_duration_sec) / 60.0, 1) as total_squint_mins
                FROM telemetry_minute
                WHERE timestamp >= ?
                GROUP BY log_date
                ORDER BY log_date ASC
                """,
                (cutoff,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def purge_older_than(self, days: int = 30) -> int:
        """Privacy cleanup: delete telemetry records older than retention threshold."""
        cutoff = time.time() - (days * 86400.0)
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM telemetry_minute WHERE timestamp < ?", (cutoff,)
            )
            deleted = cursor.rowcount
            conn.commit()
            return deleted

    def save_user_profile(self, profile: UserProfile) -> None:
        """Upsert user profile into local database."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO user_profiles (profile_id, name, baseline_open_ear, baseline_closed_ear, calibrated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(profile_id) DO UPDATE SET
                    name = excluded.name,
                    baseline_open_ear = excluded.baseline_open_ear,
                    baseline_closed_ear = excluded.baseline_closed_ear,
                    calibrated_at = excluded.calibrated_at
                """,
                (
                    profile.profile_id,
                    profile.name,
                    profile.baseline_open_ear,
                    profile.baseline_closed_ear,
                    profile.calibrated_at,
                ),
            )
            conn.commit()

    def get_user_profile(self, profile_id: str) -> Optional[UserProfile]:
        """Fetch user profile by profile_id."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM user_profiles WHERE profile_id = ?",
                (profile_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return UserProfile(
                profile_id=row["profile_id"],
                name=row["name"],
                baseline_open_ear=float(row["baseline_open_ear"]),
                baseline_closed_ear=float(row["baseline_closed_ear"]),
                calibrated_at=row["calibrated_at"],
            )

    def list_user_profiles(self) -> List[UserProfile]:
        """List all stored user profiles."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM user_profiles ORDER BY name ASC")
            return [
                UserProfile(
                    profile_id=row["profile_id"],
                    name=row["name"],
                    baseline_open_ear=float(row["baseline_open_ear"]),
                    baseline_closed_ear=float(row["baseline_closed_ear"]),
                    calibrated_at=row["calibrated_at"],
                )
                for row in cursor.fetchall()
            ]
