"""
Digital Eye Strain Analyzer.
Computes the Digital Eye Strain Index (DESI) combining continuous screen focus time,
blink suppression (dry-eye risk), and squinting frequency.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from oculisense.config import AnalyticsConfig


@dataclass
class StrainResult:
    """Telemetry report on digital eye strain."""
    timestamp: float
    strain_score: float  # [0.0 - 100.0]
    strain_level: str    # "OPTIMAL", "MILD", "MODERATE", "HIGH", "CRITICAL"
    continuous_focus_sec: float
    squint_factor: float
    blink_suppression_factor: float
    screen_time_factor: float
    is_warning: bool
    gaze_eccentricity_factor: float = 0.0
    contributing_factors: Dict[str, float] = field(default_factory=dict)


class StrainAnalyzer:
    """
    Evaluates ocular stress in real-time based on ergonomic and biometric factors.
    """

    LEVEL_OPTIMAL = "OPTIMAL"
    LEVEL_MILD = "MILD"
    LEVEL_MODERATE = "MODERATE"
    LEVEL_HIGH = "HIGH"
    LEVEL_CRITICAL = "CRITICAL"

    def __init__(self, config: Optional[AnalyticsConfig] = None) -> None:
        self.config = config or AnalyticsConfig()

        self._session_start_time: Optional[float] = None
        self._last_active_time: Optional[float] = None
        self._last_break_time: Optional[float] = None
        self._away_start_time: Optional[float] = None
        self._accumulated_focus_sec: float = 0.0

    def reset_focus(self, timestamp: Optional[float] = None) -> None:
        """Call when user takes a break or looks away for an extended period."""
        now = timestamp or time.time()
        self._last_break_time = now
        self._accumulated_focus_sec = 0.0

    def update_presence(self, is_present: bool, timestamp: Optional[float] = None) -> None:
        """Track continuous screen presence."""
        now = timestamp or time.time()
        if self._last_active_time is None or abs(now - self._last_active_time) > 86400:
            self._session_start_time = now
            self._last_active_time = now
            self._last_break_time = now
            self._away_start_time = None

        if is_present:
            dt = max(0.0, min(5.0, now - self._last_active_time))
            self._accumulated_focus_sec += dt
            self._last_active_time = now
            self._away_start_time = None
        else:
            if self._away_start_time is None:
                self._away_start_time = now
            # If away for more than 45 seconds, automatically grant a focus reset
            if (now - self._away_start_time) > 45.0:
                self.reset_focus(now)
            self._last_active_time = now

    def evaluate(
        self,
        blink_rate_bpm: float,
        squint_ratio: float,
        timestamp: Optional[float] = None,
        gaze_eccentricity: float = 0.0,
        is_present: bool = True,
    ) -> StrainResult:
        """
        Calculates the 0-100 Digital Eye Strain Index (DESI).
        """
        now = timestamp or time.time()

        # 1. Screen Time Factor (0.0 to 35.0 pts)
        # Reaches max after 45 minutes of continuous un-rested screen time
        focus_minutes = self._accumulated_focus_sec / 60.0
        screen_time_factor = min(35.0, (focus_minutes / 45.0) * 35.0)

        # 2. Blink Suppression Factor (0.0 to 35.0 pts)
        # Normal blink rate is 15-20 BPM. When staring at screens, it drops to 4-8 BPM.
        # Tear film breaks down, causing dry eyes and digital asthenopia.
        # When user is absent, do not penalize them with dry-eye blink suppression points.
        if not is_present or blink_rate_bpm >= self.config.normal_blink_rate_cpm:
            blink_suppression = 0.0
        else:
            deficiency = self.config.normal_blink_rate_cpm - max(0.0, blink_rate_bpm)
            blink_suppression = min(35.0, (deficiency / self.config.normal_blink_rate_cpm) * 35.0)

        # 3. Squint Factor (0.0 to 30.0 pts)
        # Persistent eyelid narrowing indicates uncorrected glare or small font strain
        squint_factor = min(30.0, squint_ratio * 100.0 * 0.75) if is_present else 0.0

        # 4. Multi-Monitor Lateral Gaze Factor (0.0 to 15.0 pts)
        # Lateral fixation towards secondary monitors increases extraocular mechanical strain
        if is_present and gaze_eccentricity > 0.35:
            eccentricity_factor = min(15.0, ((gaze_eccentricity - 0.35) / 0.65) * 15.0)
        else:
            eccentricity_factor = 0.0

        total_score = min(
            100.0,
            max(0.0, screen_time_factor + blink_suppression + squint_factor + eccentricity_factor),
        )

        if total_score < 25.0:
            level = self.LEVEL_OPTIMAL
        elif total_score < 50.0:
            level = self.LEVEL_MILD
        elif total_score < 70.0:
            level = self.LEVEL_MODERATE
        elif total_score < 85.0:
            level = self.LEVEL_HIGH
        else:
            level = self.LEVEL_CRITICAL

        is_warning = total_score >= self.config.strain_warning_score

        factors = {
            "screen_time_pts": round(screen_time_factor, 1),
            "blink_suppression_pts": round(blink_suppression, 1),
            "squint_pts": round(squint_factor, 1),
            "eccentricity_pts": round(eccentricity_factor, 1),
            "focus_minutes": round(focus_minutes, 1),
        }

        return StrainResult(
            timestamp=now,
            strain_score=round(total_score, 1),
            strain_level=level,
            continuous_focus_sec=self._accumulated_focus_sec,
            squint_factor=round(squint_factor, 2),
            blink_suppression_factor=round(blink_suppression, 2),
            screen_time_factor=round(screen_time_factor, 2),
            is_warning=is_warning,
            gaze_eccentricity_factor=round(eccentricity_factor, 2),
            contributing_factors=factors,
        )
