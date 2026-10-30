"""
Telemetry Aggregator and Trend Analyzer.
Maintains rolling window analytics, session rollups, and fatigue trajectory detection.
"""

from __future__ import annotations
import collections
import time
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple


@dataclass
class MinuteSummary:
    """Summary of metrics over a 60-second window."""
    timestamp: float
    avg_ear: float
    avg_fatigue: float
    avg_strain: float
    blinks_in_minute: int
    squint_duration_sec: float
    perclos_avg: float
    presence_ratio: float


@dataclass
class SessionSummary:
    """Session-level aggregate metrics for reporting."""
    start_time: float
    duration_minutes: float
    total_blinks: int
    avg_blink_rate_bpm: float
    avg_fatigue_score: float
    peak_fatigue_score: float
    avg_strain_score: float
    peak_strain_score: float
    total_squint_sec: float
    breaks_prompted: int
    breaks_completed: int
    compliance_rate_pct: float
    screen_time_minutes: float


class TelemetryAggregator:
    """
    Buffers sub-second frame telemetry and provides statistical rollups
    for database logging, dashboard charting, and daily/weekly trend reports.
    """

    def __init__(self, buffer_minutes: int = 120) -> None:
        self.buffer_minutes = buffer_minutes
        self.session_start = time.time()

        # Rolling minute summaries
        self._minute_history: Deque[MinuteSummary] = collections.deque(maxlen=buffer_minutes)

        # Current minute accumulator
        self._curr_minute_start: float = time.time()
        self._curr_ears: List[float] = []
        self._curr_fatigues: List[float] = []
        self._curr_strains: List[float] = []
        self._curr_perclos: List[float] = []
        self._curr_presences: List[bool] = []
        self._curr_blinks: int = 0
        self._curr_squint_sec: float = 0.0

        # Session running stats
        self._session_blinks: int = 0
        self._peak_fatigue: float = 0.0
        self._peak_strain: float = 0.0
        self._total_squint_sec: float = 0.0
        self._screen_active_sec: float = 0.0

    def add_sample(
        self,
        ear: float,
        fatigue: float,
        strain: float,
        perclos: float,
        is_present: bool,
        is_squinting: bool,
        blink_occurred: bool,
        dt_sec: float = 0.033,
        timestamp: Optional[float] = None,
    ) -> Optional[MinuteSummary]:
        """
        Record a telemetry sample. Returns MinuteSummary when a 60-second window completes.
        """
        now = timestamp or time.time()

        # Update running session stats
        if is_present:
            self._screen_active_sec += dt_sec
        if blink_occurred:
            self._session_blinks += 1
            self._curr_blinks += 1
        if is_squinting:
            self._total_squint_sec += dt_sec
            self._curr_squint_sec += dt_sec

        self._peak_fatigue = max(self._peak_fatigue, fatigue)
        self._peak_strain = max(self._peak_strain, strain)

        if is_present and ear > 0.0:
            self._curr_ears.append(ear)
        self._curr_fatigues.append(fatigue)
        self._curr_strains.append(strain)
        self._curr_perclos.append(perclos)
        self._curr_presences.append(is_present)

        # Check if 60 seconds have elapsed
        if (now - self._curr_minute_start) >= 60.0:
            summary = self._finalize_minute(now)
            return summary
        return None

    def _finalize_minute(self, now: float) -> MinuteSummary:
        avg_ear = sum(self._curr_ears) / len(self._curr_ears) if self._curr_ears else 0.0
        avg_fatigue = sum(self._curr_fatigues) / len(self._curr_fatigues) if self._curr_fatigues else 0.0
        avg_strain = sum(self._curr_strains) / len(self._curr_strains) if self._curr_strains else 0.0
        avg_perclos = sum(self._curr_perclos) / len(self._curr_perclos) if self._curr_perclos else 0.0
        presence_ratio = (
            sum(1 for p in self._curr_presences if p) / len(self._curr_presences)
            if self._curr_presences
            else 0.0
        )

        summary = MinuteSummary(
            timestamp=self._curr_minute_start,
            avg_ear=round(avg_ear, 3),
            avg_fatigue=round(avg_fatigue, 1),
            avg_strain=round(avg_strain, 1),
            blinks_in_minute=self._curr_blinks,
            squint_duration_sec=round(self._curr_squint_sec, 2),
            perclos_avg=round(avg_perclos, 3),
            presence_ratio=round(presence_ratio, 2),
        )
        self._minute_history.append(summary)

        # Reset minute accumulators
        self._curr_minute_start = now
        self._curr_ears = []
        self._curr_fatigues = []
        self._curr_strains = []
        self._curr_perclos = []
        self._curr_presences = []
        self._curr_blinks = 0
        self._curr_squint_sec = 0.0

        return summary

    def get_recent_timeline(self, limit_minutes: int = 60) -> List[MinuteSummary]:
        """Return chronological list of recent minute summaries."""
        items = list(self._minute_history)
        return items[-limit_minutes:]

    def get_session_summary(
        self,
        breaks_prompted: int = 0,
        breaks_completed: int = 0,
        compliance_rate: float = 100.0,
    ) -> SessionSummary:
        """Compute full session summary."""
        now = time.time()
        duration_mins = max(0.1, (now - self.session_start) / 60.0)

        fatigues = [m.avg_fatigue for m in self._minute_history]
        strains = [m.avg_strain for m in self._minute_history]

        avg_fatigue = sum(fatigues) / len(fatigues) if fatigues else 0.0
        avg_strain = sum(strains) / len(strains) if strains else 0.0
        avg_bpm = float(self._session_blinks) / duration_mins

        return SessionSummary(
            start_time=self.session_start,
            duration_minutes=round(duration_mins, 1),
            total_blinks=self._session_blinks,
            avg_blink_rate_bpm=round(avg_bpm, 1),
            avg_fatigue_score=round(avg_fatigue, 1),
            peak_fatigue_score=round(self._peak_fatigue, 1),
            avg_strain_score=round(avg_strain, 1),
            peak_strain_score=round(self._peak_strain, 1),
            total_squint_sec=round(self._total_squint_sec, 1),
            breaks_prompted=breaks_prompted,
            breaks_completed=breaks_completed,
            compliance_rate_pct=round(compliance_rate, 1),
            screen_time_minutes=round(self._screen_active_sec / 60.0, 1),
        )
