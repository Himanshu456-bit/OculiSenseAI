"""
Configuration and Profile Management for OculiSenseAI.
Handles thresholds, camera preferences, privacy flags, and user personalization.
"""

from __future__ import annotations
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class VisionConfig:
    """Vision and facial landmark tracking thresholds."""
    camera_id: int = 0
    target_fps: int = 30
    # Baseline EAR thresholds
    ear_closed_threshold: float = 0.21
    ear_squint_ratio: float = 0.75  # EAR < open_baseline * squint_ratio is squint
    blink_min_frames: int = 1
    blink_max_frames: int = 20
    perclos_window_sec: float = 60.0
    squint_duration_threshold_sec: float = 2.0
    # MediaPipe vs Haar Cascade preference
    use_mediapipe: bool = True
    detection_confidence: float = 0.5
    tracking_confidence: float = 0.5
    autostart_camera: bool = True


@dataclass
class AnalyticsConfig:
    """Fatigue and strain calculation hyperparameters."""
    normal_blink_rate_cpm: float = 17.0  # blinks per minute baseline
    low_blink_rate_threshold: float = 9.0  # indicator of dry eye / intense screen fixation
    prolonged_blink_duration_ms: float = 400.0  # fatigued blinks last > 400ms
    # Fatigue weights
    weight_perclos: float = 0.35
    weight_blink_duration: float = 0.25
    weight_blink_frequency: float = 0.20
    weight_continuous_work: float = 0.10
    weight_circadian: float = 0.10
    # Thresholds for triggering status alerts
    strain_warning_score: float = 60.0
    strain_critical_score: float = 80.0
    fatigue_warning_score: float = 60.0
    fatigue_critical_score: float = 85.0


@dataclass
class BreakConfig:
    """20-20-20 rule, Pomodoro timer, and smart rest advisor settings."""
    rule_20_20_20_interval_sec: int = 20 * 60  # 20 minutes = 1200 seconds
    rule_20_20_20_duration_sec: int = 20       # 20 seconds
    enable_break_overlays: bool = True
    gentle_lockout: bool = False
    enable_audio_cue: bool = True
    enable_voice_alert: bool = True
    auto_snooze_limit: int = 3
    snooze_duration_sec: int = 5 * 60          # 5 minutes
    # Pomodoro mode integration
    enable_pomodoro: bool = False
    pomodoro_work_sec: int = 25 * 60           # 25 minutes
    pomodoro_short_rest_sec: int = 5 * 60      # 5 minutes
    pomodoro_long_rest_sec: int = 15 * 60      # 15 minutes
    pomodoro_cycles_before_long_rest: int = 4


@dataclass
class SecurityConfig:
    """Eye-presence workstation security settings."""
    enable_presence_lock: bool = False
    away_timeout_sec: float = 15.0             # trigger lock after 15s without face
    dry_run_lock: bool = True                  # simulate lock instead of actual OS lock by default


@dataclass
class PrivacyConfig:
    """Privacy-first edge processing guarantees."""
    store_video_frames: bool = False           # strictly FALSE - never stores raw frames
    upload_telemetry: bool = False             # strictly FALSE - 100% on-device
    retention_days: int = 30
    database_filename: str = "oculisense.db"


@dataclass
class UserProfile:
    """Personalized calibration profile."""
    profile_id: str = "default"
    name: str = "Primary User"
    baseline_open_ear: float = 0.32
    baseline_closed_ear: float = 0.17
    calibrated_at: Optional[str] = None

    @property
    def dynamic_ear_threshold(self) -> float:
        """Computes personalized midpoint threshold between open and closed baseline."""
        return self.baseline_closed_ear + 0.40 * (self.baseline_open_ear - self.baseline_closed_ear)

    @property
    def dynamic_squint_threshold(self) -> float:
        """EAR below this value while above closed threshold signifies squinting."""
        return self.baseline_closed_ear + 0.65 * (self.baseline_open_ear - self.baseline_closed_ear)


@dataclass
class AppConfig:
    """Master Application Configuration."""
    vision: VisionConfig = field(default_factory=VisionConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    breaks: BreakConfig = field(default_factory=BreakConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    profile: UserProfile = field(default_factory=UserProfile)
    demo_mode: bool = False
    config_dir: str = field(default_factory=lambda: str(Path.home() / ".oculisense"))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AppConfig:
        cfg = cls()
        if "vision" in data:
            cfg.vision = VisionConfig(**data["vision"])
        if "analytics" in data:
            cfg.analytics = AnalyticsConfig(**data["analytics"])
        if "breaks" in data:
            cfg.breaks = BreakConfig(**data["breaks"])
        if "security" in data:
            cfg.security = SecurityConfig(**data["security"])
        if "privacy" in data:
            cfg.privacy = PrivacyConfig(**data["privacy"])
        if "profile" in data:
            cfg.profile = UserProfile(**data["profile"])
        if "demo_mode" in data:
            cfg.demo_mode = bool(data["demo_mode"])
        if "config_dir" in data:
            cfg.config_dir = str(data["config_dir"])
        return cfg

    def save(self, filepath: Optional[str] = None) -> None:
        if filepath is None:
            os.makedirs(self.config_dir, exist_ok=True)
            filepath = os.path.join(self.config_dir, "config.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Optional[str] = None) -> AppConfig:
        if filepath is None:
            default_path = os.path.join(str(Path.home() / ".oculisense"), "config.json")
            if not os.path.exists(default_path):
                return cls()
            filepath = default_path
        if not os.path.exists(filepath):
            return cls()
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
