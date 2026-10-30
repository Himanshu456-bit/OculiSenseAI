"""
Workload Fatigue Analytics and Prediction Model.
Predicts ocular fatigue and drowsiness based on PERCLOS, blink duration dynamics,
circadian rhythms, and continuous session load.
"""

from __future__ import annotations
import datetime
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from oculisense.config import AnalyticsConfig

try:
    from sklearn.linear_model import Ridge  # type: ignore
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class FatigueResult:
    """Comprehensive workload fatigue assessment."""
    timestamp: float
    fatigue_score: float  # [0.0 - 100.0]
    fatigue_level: str    # "LOW", "MODERATE", "ELEVATED", "CRITICAL"
    perclos_score: float  # [0.0 - 100.0 contribution]
    blink_duration_score: float
    circadian_factor: float
    session_factor: float
    is_warning: bool
    microsleep_risk: bool
    metrics_summary: Dict[str, float] = field(default_factory=dict)


class FatigueModel:
    """
    Edge-AI model for cognitive and physical workload fatigue estimation.
    Uses calibrated multi-modal biometric indicators with fallback or sklearn predictor.
    """

    LEVEL_LOW = "LOW"
    LEVEL_MODERATE = "MODERATE"
    LEVEL_ELEVATED = "ELEVATED"
    LEVEL_CRITICAL = "CRITICAL"

    def __init__(self, config: Optional[AnalyticsConfig] = None) -> None:
        self.config = config or AnalyticsConfig()
        self.session_start_time: Optional[float] = None

        # Optional scikit-learn model wrapper
        self._sklearn_model = None
        if HAS_SKLEARN:
            # Initialize with pre-calibrated weights for [perclos, avg_duration_norm, bpm_norm, circadian, session_hrs]
            self._sklearn_model = Ridge()
            # Fit synthetic calibration anchor points
            X_anchor = [
                [0.02, 0.2, 0.8, 0.0, 0.2],  # fresh morning
                [0.08, 0.35, 0.5, 0.3, 1.5], # moderate afternoon
                [0.18, 0.6, 0.3, 0.7, 4.0],  # high fatigue evening
                [0.35, 0.9, 0.1, 0.9, 8.0],  # severe exhaustion
            ]
            y_anchor = [12.0, 42.0, 76.0, 96.0]
            self._sklearn_model.fit(X_anchor, y_anchor)

    def _compute_circadian_factor(self, dt: datetime.datetime) -> float:
        """
        Calculates circadian drowsiness factor (0.0 to 1.0) based on local time.
        Known circadian fatigue peaks:
        - Post-lunch dip: 13:30 - 15:30 (approx 0.4 - 0.6)
        - Nighttime dip: 23:00 - 05:00 (approx 0.7 - 1.0)
        - Morning peak alertness: 09:00 - 11:30 (approx 0.05 - 0.15)
        """
        hour = dt.hour + dt.minute / 60.0

        if 23.0 <= hour or hour < 5.0:
            return 0.85 + 0.15 * math.sin((hour % 24) * 0.2)
        elif 13.0 <= hour < 15.5:
            # Postprandial dip
            return 0.55
        elif 9.0 <= hour < 12.0:
            return 0.10
        elif 18.0 <= hour < 22.0:
            # Evening fatigue buildup
            return 0.35 + (hour - 18.0) * 0.1
        else:
            return 0.25

    def predict(
        self,
        perclos: float,
        blink_rate_bpm: float,
        recent_blink_durations_ms: List[float],
        is_prolonged_closure: bool = False,
        timestamp: Optional[float] = None,
        eyes_present: bool = True,
        consecutive_closed_frames: int = 0,
    ) -> FatigueResult:
        """
        Calculates the real-time Fatigue Score [0.0 - 100.0].
        Incorporates PERCLOS, blink elongation, blink frequency, circadian cycle, session load,
        and Dlib consecutive eye closure thresholding (akshaybahadur21/Drowsiness_Detection standard).
        """
        now = timestamp or time.time()
        dt = datetime.datetime.fromtimestamp(now)

        # 1. PERCLOS Score (0 - 100)
        # Healthy PERCLOS is < 0.08. Drowsiness threshold is > 0.15 (15% closure).
        # Extreme fatigue is > 0.25.
        if not eyes_present:
            perclos_clamped = 0.0
            perclos_score = 0.0
        else:
            perclos_clamped = max(0.0, min(1.0, perclos))
            perclos_score = min(100.0, (perclos_clamped / 0.25) * 100.0)

        # 2. Blink Duration Score (0 - 100)
        # Normal alert blink duration: 100ms - 250ms
        # Fatigued blink duration: 350ms - 800ms
        if recent_blink_durations_ms and eyes_present:
            avg_duration = sum(recent_blink_durations_ms) / len(recent_blink_durations_ms)
        else:
            avg_duration = 200.0  # nominal default

        if not eyes_present or avg_duration <= 220.0:
            duration_score = 0.0
        else:
            duration_score = min(100.0, ((avg_duration - 220.0) / 380.0) * 100.0)

        # 3. Blink Frequency Factor (0 - 100)
        # Hypo-blinking (< 8 BPM) indicates sluggishness or strain; very high erratic blinking (> 32) can indicate irritation
        if not eyes_present or 12.0 <= blink_rate_bpm <= 24.0:
            freq_score = 0.0
        elif blink_rate_bpm < 12.0:
            freq_score = min(100.0, ((12.0 - blink_rate_bpm) / 12.0) * 70.0)
        else:
            freq_score = min(100.0, ((blink_rate_bpm - 24.0) / 20.0) * 60.0)

        # 4. Circadian Factor (0 - 100)
        circadian_val = self._compute_circadian_factor(dt)
        circadian_score = circadian_val * 100.0

        # 5. Session Duration Factor (0 - 100)
        # Work session duration in hours
        if self.session_start_time is None or self.session_start_time > now:
            self.session_start_time = now
        session_hours = max(0.0, (now - self.session_start_time) / 3600.0)
        session_score = min(100.0, (session_hours / 6.0) * 100.0)

        # Weighted synthesis
        raw_fatigue = (
            self.config.weight_perclos * perclos_score
            + self.config.weight_blink_duration * duration_score
            + self.config.weight_blink_frequency * freq_score
            + self.config.weight_circadian * circadian_score
            + self.config.weight_continuous_work * session_score
        )

        # 6. Dlib Drowsiness & Consecutive Eye Closure Detection (akshaybahadur21 standard)
        # EYE_AR_CONSEC_FRAMES = 20 (~0.66s continuous closure at 30 FPS)
        consec_drowsiness = consecutive_closed_frames >= 20

        # Immediate critical penalty if prolonged closure, microsleep, or >=20 consecutive closed frames
        microsleep_risk = eyes_present and (is_prolonged_closure or consec_drowsiness or (perclos > 0.30))
        if microsleep_risk:
            raw_fatigue = max(raw_fatigue, 85.0)
        elif eyes_present and consecutive_closed_frames >= 8:
            # Progressive drowsiness penalty for sluggish eye-drooping
            droop_penalty = min(35.0, ((consecutive_closed_frames - 8) / 12.0) * 35.0)
            raw_fatigue = min(84.0, raw_fatigue + droop_penalty)

        final_score = max(0.0, min(100.0, raw_fatigue))

        # Categorize
        if final_score < 30.0:
            level = self.LEVEL_LOW
        elif final_score < 55.0:
            level = self.LEVEL_MODERATE
        elif final_score < 75.0:
            level = self.LEVEL_ELEVATED
        else:
            level = self.LEVEL_CRITICAL

        is_warning = final_score >= self.config.fatigue_warning_score

        metrics = {
            "perclos_raw": round(perclos_clamped, 3),
            "perclos_pts": round(perclos_score, 1),
            "avg_blink_duration_ms": round(avg_duration, 1),
            "duration_pts": round(duration_score, 1),
            "circadian_factor": round(circadian_val, 2),
            "session_hours": round(session_hours, 2),
            "consecutive_closed_frames": consecutive_closed_frames,
        }

        return FatigueResult(
            timestamp=now,
            fatigue_score=round(final_score, 1),
            fatigue_level=level,
            perclos_score=round(perclos_score, 1),
            blink_duration_score=round(duration_score, 1),
            circadian_factor=round(circadian_score, 1),
            session_factor=round(session_score, 1),
            is_warning=is_warning,
            microsleep_risk=microsleep_risk,
            metrics_summary=metrics,
        )
