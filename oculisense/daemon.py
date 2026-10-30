"""
OculiSenseAI Core Daemon and Pipeline Coordinator.
Orchestrates vision processing, ergonomic telemetry, fatigue analytics,
rest advisor, presence security, and database logging in real time.
"""

from __future__ import annotations
import logging
import platform
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from oculisense.analytics.aggregator import MinuteSummary, SessionSummary, TelemetryAggregator
from oculisense.analytics.break_advisor import BreakAdvisor, BreakStatus
from oculisense.analytics.fatigue_model import FatigueModel, FatigueResult
from oculisense.analytics.strain_analyzer import StrainAnalyzer, StrainResult
from oculisense.config import AppConfig, UserProfile
from oculisense.security.presence_lock import PresenceLockManager, PresenceState
from oculisense.storage.database import DatabaseManager
from oculisense.vision.blink_detector import BlinkDetector, BlinkFrameResult
from oculisense.vision.squint_detector import SquintDetector, SquintFrameResult
from oculisense.vision.tracker import EyeTracker, VisionTelemetry

logger = logging.getLogger(__name__)


@dataclass
class DaemonSnapshot:
    """Consolidated state snapshot broadcast to UI or daemon clients."""
    timestamp: float
    session_id: str
    vision: VisionTelemetry
    blink: BlinkFrameResult
    squint: SquintFrameResult
    strain: StrainResult
    fatigue: FatigueResult
    breaks: BreakStatus
    presence: PresenceState
    is_calibrating: bool = False
    calibration_progress: float = 0.0
    is_camera_active: bool = False

    @property
    def overall_score(self) -> float:
        """
        Overall Eye Wellness & Health Score [0.0 - 100.0].
        100 = Peak fresh eyes, 60-79 = Moderate, <40 = Critical eye exhaustion/low score.
        Combines digital eye strain and neurological drowsiness penalties.
        """
        if not self.is_camera_active or not self.vision.face_detected:
            return 100.0
        s = getattr(self.strain, "strain_score", 0.0)
        f = getattr(self.fatigue, "fatigue_score", 0.0)
        max_penalty = max(s, f)
        avg_penalty = 0.5 * s + 0.5 * f
        effective_penalty = 0.6 * max_penalty + 0.4 * avg_penalty
        return round(max(0.0, min(100.0, 100.0 - effective_penalty)), 1)

    @property
    def overall_health_level(self) -> str:
        score = self.overall_score
        if score >= 80.0:
            return "EXCELLENT"
        elif score >= 60.0:
            return "GOOD"
        elif score >= 40.0:
            return "MODERATE"
        else:
            return "CRITICAL_LOW"

    @property
    def is_force_stop_needed(self) -> bool:
        """Determines if user has low score requiring forced activity stoppage."""
        return self.breaks.is_break_recommended and (
            self.overall_score < 40.0
            or getattr(self.strain, "strain_score", 0.0) >= 80.0
            or getattr(self.fatigue, "fatigue_score", 0.0) >= 85.0
            or self.breaks.reason in ("CRITICAL_LOW_SCORE", "CRITICAL_LOW_EAR_SCORE", "MANUAL_FORCE_REST")
        )


class OculiSenseDaemon:
    """
    Central background coordinator for OculiSenseAI.
    Can run headlessly as a service or feed live state to desktop GUIs.
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        force_synthetic: bool = False,
        autostart_camera: Optional[bool] = None,
    ) -> None:
        self.config = config or AppConfig.load()
        if force_synthetic:
            self.config.demo_mode = True

        self.session_id = str(uuid.uuid4())[:8]
        self.db = DatabaseManager(self.config.privacy.database_filename)

        # Synchronize active user profile with local SQLite store
        existing_profile = self.db.get_user_profile(self.config.profile.profile_id)
        if existing_profile is not None:
            self.config.profile = existing_profile
        else:
            self.db.save_user_profile(self.config.profile)

        self.db.start_session(self.session_id, user_id=self.config.profile.profile_id)

        # Camera state management
        if autostart_camera is not None:
            self._is_camera_active = bool(autostart_camera)
        else:
            self._is_camera_active = bool(self.config.vision.autostart_camera or force_synthetic)

        # Initialize vision and tracking
        self.tracker = EyeTracker(
            config=self.config.vision,
            force_synthetic=self.config.demo_mode,
            autostart=self._is_camera_active,
        )

        # Use calibrated thresholds from user profile
        calibrated_closed = self.config.profile.dynamic_ear_threshold
        calibrated_squint = self.config.profile.dynamic_squint_threshold

        self.blink_detector = BlinkDetector(
            closed_threshold=calibrated_closed,
            min_blink_frames=self.config.vision.blink_min_frames,
            max_blink_frames=self.config.vision.blink_max_frames,
            perclos_window_sec=self.config.vision.perclos_window_sec,
        )

        self.squint_detector = SquintDetector(
            closed_threshold=calibrated_closed,
            squint_upper_threshold=calibrated_squint,
            min_squint_duration_sec=self.config.vision.squint_duration_threshold_sec,
        )

        # Analytics
        self.strain_analyzer = StrainAnalyzer(config=self.config.analytics)
        self.fatigue_model = FatigueModel(config=self.config.analytics)
        self.break_advisor = BreakAdvisor(
            config=self.config.breaks,
            on_break_event=self._handle_break_event,
        )
        self.aggregator = TelemetryAggregator()

        # Security
        self.presence_manager = PresenceLockManager(
            config=self.config.security,
            on_lock_callback=self._handle_auto_lock,
        )

        # Threading and listeners
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._listeners: List[Callable[[DaemonSnapshot], None]] = []
        self._last_snapshot: Optional[DaemonSnapshot] = None
        self._api_server: Optional[Any] = None

        # Calibration state
        self._is_calibrating = False
        self._calibration_start_time: float = 0.0
        self._calibration_duration: float = 5.0
        self._calibration_ears: List[float] = []
        self._camera_stopped_time: Optional[float] = None

    def register_listener(self, callback: Callable[[DaemonSnapshot], None]) -> None:
        """Register a callback for telemetry updates."""
        self._listeners.append(callback)

    def unregister_listener(self, callback: Callable[[DaemonSnapshot], None]) -> None:
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _handle_break_event(self, reason: str, outcome: str, duration_sec: float) -> None:
        """Persist rest event in local SQLite store."""
        try:
            self.db.log_break_event(
                session_id=self.session_id,
                reason=reason,
                outcome=outcome,
                duration_sec=duration_sec,
            )
        except Exception as e:
            logger.debug("Failed to log break event: %s", e)

    def _handle_auto_lock(self) -> None:
        logger.warning("Auto-lock executed: session paused.")

    def speak_alert(self, text: str) -> None:
        """Non-blocking spoken voice reminder via system text-to-speech."""
        def _speak():
            sys_platform = platform.system()
            try:
                if sys_platform == "Windows":
                    try:
                        import win32com.client
                        speaker = win32com.client.Dispatch("SAPI.SpVoice")
                        speaker.Speak(text)
                    except Exception as e:
                        logger.debug("Windows SAPI voice alert unavailable: %s", e)
                elif sys_platform == "Darwin":
                    import subprocess
                    subprocess.run(["say", text], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    import subprocess
                    subprocess.run(["espeak", text], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                logger.debug("Voice alert speech synthesis failed: %s", e)

        threading.Thread(target=_speak, daemon=True).start()

    def play_alert_sound(self, reason: Optional[str] = None) -> None:
        """Play a non-blocking gentle chime and optional spoken voice reminder for rest."""
        def _play():
            try:
                if platform.system() == "Windows":
                    import winsound
                    # Play gentle ascending two-tone chime
                    winsound.Beep(587, 180)  # D5
                    winsound.Beep(880, 240)  # A5
                else:
                    # ANSI bell fallback
                    print("\a", end="", flush=True)
            except Exception:
                pass

            # Spoken voice reminder if enabled
            if getattr(self.config.breaks, "enable_voice_alert", False):
                if reason == "STRAIN_THRESHOLD":
                    msg = "Elevated eye strain detected. Please take a 20-second rest."
                elif reason == "FATIGUE_THRESHOLD":
                    msg = "Workload fatigue is elevated. Please rest your eyes."
                else:
                    msg = "Time for your 20-20-20 eye rest. Look 20 feet away for 20 seconds."
                self.speak_alert(msg)

        threading.Thread(target=_play, daemon=True).start()

    def switch_user_profile(self, profile_id: str) -> bool:
        """Switch active user profile and apply personalized calibration thresholds."""
        profile = self.db.get_user_profile(profile_id)
        if profile is None:
            return False
        self.config.profile = profile
        new_closed_th = profile.dynamic_ear_threshold
        new_squint_th = profile.dynamic_squint_threshold
        self.blink_detector.update_thresholds(new_closed_th)
        self.squint_detector.update_thresholds(new_closed_th, new_squint_th)
        logger.info("Switched to user profile '%s' (%s).", profile.name, profile_id)
        return True

    def create_or_update_user_profile(self, profile: UserProfile) -> None:
        """Save and switch to a personalized user profile."""
        self.db.save_user_profile(profile)
        self.switch_user_profile(profile.profile_id)

    @property
    def is_camera_active(self) -> bool:
        """Returns True if the camera and biometric tracking are actively running."""
        return self._is_camera_active

    @property
    def total_blinks(self) -> int:
        """Total completed blinks recorded in real time."""
        return self.blink_detector.total_blinks

    @property
    def blink_rate_bpm(self) -> float:
        """Real-time responsive blink rate (BPM)."""
        return self.blink_detector.blink_rate_bpm

    def start_camera(self, timestamp: Optional[float] = None) -> bool:
        """Explicitly activate camera hardware and resume biometric tracking."""
        self._is_camera_active = True
        now = timestamp if timestamp is not None else time.time()
        self.presence_manager.reset()
        self.blink_detector.reset()
        self.squint_detector.reset()
        self.strain_analyzer._last_active_time = None
        self.strain_analyzer._away_start_time = None
        if self._camera_stopped_time is not None:
            stopped_duration = max(0.0, now - self._camera_stopped_time)
            if stopped_duration >= 45.0:
                self.strain_analyzer.reset_focus(now)
                self.break_advisor.reset(now)
            else:
                self.break_advisor.shift_work_time(stopped_duration)
        else:
            self.break_advisor.reset(now)
        self.break_advisor._last_frame_time = None
        success = self.tracker.start_camera()
        logger.info("Camera activated (status: %s).", "OK" if success else "FALLBACK")
        return success

    def stop_camera(self, timestamp: Optional[float] = None) -> None:
        """Deactivate camera hardware and pause all biometric tracking and calculations."""
        self._is_camera_active = False
        self._is_calibrating = False
        self._calibration_ears.clear()
        self._camera_stopped_time = timestamp if timestamp is not None else time.time()
        self.presence_manager.reset()
        self.blink_detector.reset()
        self.squint_detector.reset()
        self.strain_analyzer._last_active_time = None
        self.strain_analyzer._away_start_time = None
        self.break_advisor._last_frame_time = None
        self.tracker.stop_camera()
        logger.info("Camera deactivated. Biometric tracking paused.")

    def toggle_camera(self) -> bool:
        """Toggle camera between active and paused states."""
        if self._is_camera_active:
            self.stop_camera()
            return False
        else:
            self.start_camera()
            return self._is_camera_active

    def switch_camera(self, camera_id: int) -> bool:
        """Switch active webcam device index."""
        self.config.vision.camera_id = camera_id
        res = self.tracker.switch_camera(camera_id)
        if res:
            self.blink_detector.reset()
            self.squint_detector.reset()
            self.presence_manager.reset()
        return res

    def step(self, custom_time: Optional[float] = None) -> DaemonSnapshot:
        """
        Execute one processing cycle across vision, analytics, security, and persistence.
        """
        now = custom_time or time.time()

        # If camera is disabled/paused: do zero calculations and return an idle snapshot
        if not self._is_camera_active:
            v_telemetry = self.tracker.read(timestamp=now)
            b_res = BlinkFrameResult(
                timestamp=now,
                ear_left=0.0,
                ear_right=0.0,
                ear_avg=0.0,
                is_eyes_closed=False,
                blink_detected_this_frame=False,
                total_blinks=self.blink_detector._total_blinks,
                blink_rate_bpm=0.0,
                perclos=0.0,
                last_closure_duration_ms=0.0,
                is_prolonged_closure=False,
                eyes_present=False,
            )
            sq_res = SquintFrameResult(
                timestamp=now,
                is_squinting=False,
                current_squint_duration_sec=0.0,
                total_squint_episodes=self.squint_detector._total_squint_episodes,
                squint_time_ratio=0.0,
                squint_intensity=0.0,
            )
            presence_res = PresenceState(
                is_present=False,
                away_duration_sec=0.0,
                lock_triggered=False,
                status_label="CAMERA_OFF",
            )
            last_strain = self._last_snapshot.strain.strain_score if self._last_snapshot else 0.0
            last_fatigue = self._last_snapshot.fatigue.fatigue_score if self._last_snapshot else 0.0

            strain_res = StrainResult(
                timestamp=now,
                strain_score=last_strain,
                strain_level="IDLE",
                continuous_focus_sec=self.strain_analyzer._accumulated_focus_sec,
                squint_factor=0.0,
                blink_suppression_factor=0.0,
                screen_time_factor=0.0,
                is_warning=False,
                gaze_eccentricity_factor=0.0,
            )
            fatigue_res = FatigueResult(
                timestamp=now,
                fatigue_score=last_fatigue,
                fatigue_level="IDLE",
                perclos_score=0.0,
                blink_duration_score=0.0,
                circadian_factor=0.0,
                session_factor=0.0,
                is_warning=False,
                microsleep_risk=False,
            )
            break_res = BreakStatus(
                is_break_recommended=False,
                is_in_break=False,
                remaining_until_break_sec=self.break_advisor.remaining_until_break_sec,
                remaining_break_duration_sec=0.0,
                reason="IDLE",
                total_prompted=self.break_advisor._total_prompted,
                total_completed=self.break_advisor._total_completed,
                total_skipped=self.break_advisor._total_skipped,
                compliance_rate_pct=self.break_advisor.compliance_rate,
                audio_cue_needed=False,
                least_recorded_ear=1.0,
                eval_elapsed_sec=0.0,
                is_evaluating=False,
                eval_decision="IDLE",
            )
            snapshot = DaemonSnapshot(
                timestamp=now,
                session_id=self.session_id,
                vision=v_telemetry,
                blink=b_res,
                squint=sq_res,
                strain=strain_res,
                fatigue=fatigue_res,
                breaks=break_res,
                presence=presence_res,
                is_calibrating=False,
                calibration_progress=0.0,
                is_camera_active=False,
            )
            self._last_snapshot = snapshot
            for listener in self._listeners:
                try:
                    listener(snapshot)
                except Exception as e:
                    logger.error("Error in daemon listener: %s", e)
            return snapshot

        # 1. Vision Capture & Landmark Tracking
        v_telemetry = self.tracker.read(timestamp=now)

        # Handle calibration if active
        if self._is_calibrating:
            calib_snap = self._process_calibration_tick(v_telemetry, now)
            self._last_snapshot = calib_snap
            for listener in self._listeners:
                try:
                    listener(calib_snap)
                except Exception as e:
                    logger.error("Error in daemon listener: %s", e)
            return calib_snap

        # 2. Blink & PERCLOS Processing
        b_res = self.blink_detector.process_frame(
            ear_left=v_telemetry.ear_left,
            ear_right=v_telemetry.ear_right,
            timestamp=now,
            eyes_present=v_telemetry.face_detected,
        )

        # 3. Squint & Micro-Expression Detection
        sq_res = self.squint_detector.process_frame(
            ear_avg=v_telemetry.ear_avg,
            timestamp=now,
            eyes_present=v_telemetry.face_detected,
        )

        # 4. Presence & Security Monitoring
        presence_res = self.presence_manager.update(
            eyes_present=v_telemetry.face_detected,
            timestamp=now,
        )

        # 5. Digital Eye Strain Index (DESI)
        self.strain_analyzer.update_presence(
            is_present=v_telemetry.face_detected,
            timestamp=now,
        )
        strain_res = self.strain_analyzer.evaluate(
            blink_rate_bpm=b_res.blink_rate_bpm,
            squint_ratio=sq_res.squint_time_ratio,
            timestamp=now,
            gaze_eccentricity=getattr(v_telemetry, "gaze_eccentricity", 0.0),
            is_present=v_telemetry.face_detected,
        )

        # 6. Workload Fatigue Scoring
        recent_durations = self.blink_detector.get_recent_blink_durations()
        fatigue_res = self.fatigue_model.predict(
            perclos=b_res.perclos,
            blink_rate_bpm=b_res.blink_rate_bpm,
            recent_blink_durations_ms=recent_durations,
            is_prolonged_closure=b_res.is_prolonged_closure,
            timestamp=now,
            eyes_present=v_telemetry.face_detected,
            consecutive_closed_frames=getattr(b_res, "consecutive_closed_frames", 0),
        )

        # 7. Break Advisor & 20-20-20 Rule (Observes EAR for at least 20s with 4 Ergonomic Conditions)
        break_res = self.break_advisor.update(
            eyes_present=v_telemetry.face_detected,
            is_eyes_closed=b_res.is_eyes_closed,
            strain_score=strain_res.strain_score,
            fatigue_score=fatigue_res.fatigue_score,
            timestamp=now,
            ear_avg=v_telemetry.ear_avg,
            blink_rate_bpm=b_res.blink_rate_bpm,
        )

        if break_res.audio_cue_needed and self.config.breaks.enable_audio_cue:
            self.play_alert_sound(reason=break_res.reason)

        # 8. Aggregation & Persistence
        min_summary = self.aggregator.add_sample(
            ear=v_telemetry.ear_avg,
            fatigue=fatigue_res.fatigue_score,
            strain=strain_res.strain_score,
            perclos=b_res.perclos,
            is_present=v_telemetry.face_detected,
            is_squinting=sq_res.is_squinting,
            blink_occurred=b_res.blink_detected_this_frame,
            dt_sec=0.033,
            timestamp=now,
        )

        if min_summary is not None:
            self.db.log_minute_telemetry(self.session_id, min_summary)

        snapshot = DaemonSnapshot(
            timestamp=now,
            session_id=self.session_id,
            vision=v_telemetry,
            blink=b_res,
            squint=sq_res,
            strain=strain_res,
            fatigue=fatigue_res,
            breaks=break_res,
            presence=presence_res,
            is_calibrating=False,
            calibration_progress=0.0,
            is_camera_active=True,
        )
        self._last_snapshot = snapshot

        # Notify UI listeners
        for listener in self._listeners:
            try:
                listener(snapshot)
            except Exception as e:
                logger.error("Error in daemon listener: %s", e)

        return snapshot

    def start_calibration(self, duration_sec: float = 5.0, start_time: Optional[float] = None) -> bool:
        """Initiate user baseline calibration. Requires active camera."""
        if not self._is_camera_active:
            logger.warning("Cannot start calibration: camera is inactive. Turn on camera first.")
            return False
        self._is_calibrating = True
        self._calibration_start_time = start_time or time.time()
        self._calibration_duration = duration_sec
        self._calibration_ears.clear()
        logger.info("Beginning EAR calibration for %s seconds.", duration_sec)
        return True

    def _process_calibration_tick(
        self, v_telemetry: VisionTelemetry, now: float
    ) -> DaemonSnapshot:
        if not self._is_camera_active:
            self._is_calibrating = False
            return self.step(custom_time=now)

        if self._calibration_start_time == 0.0 or self._calibration_start_time > now:
            self._calibration_start_time = now
        elapsed = now - self._calibration_start_time
        progress = min(1.0, elapsed / self._calibration_duration)

        if v_telemetry.face_detected and v_telemetry.ear_avg > 0.05:
            self._calibration_ears.append(v_telemetry.ear_avg)

        if progress >= 1.0:
            self._finalize_calibration()

        # Return snapshot with calibrating markers without mutating running detectors or models
        return DaemonSnapshot(
            timestamp=now,
            session_id=self.session_id,
            vision=v_telemetry,
            blink=BlinkFrameResult(
                timestamp=now,
                ear_left=v_telemetry.ear_left,
                ear_right=v_telemetry.ear_right,
                ear_avg=v_telemetry.ear_avg,
                is_eyes_closed=False,
                blink_detected_this_frame=False,
                total_blinks=self.blink_detector._total_blinks,
                blink_rate_bpm=0.0,
                perclos=0.0,
                last_closure_duration_ms=0.0,
                is_prolonged_closure=False,
                eyes_present=v_telemetry.face_detected,
            ),
            squint=SquintFrameResult(
                timestamp=now,
                is_squinting=False,
                current_squint_duration_sec=0.0,
                total_squint_episodes=self.squint_detector._total_squint_episodes,
                squint_time_ratio=0.0,
                squint_intensity=0.0,
            ),
            strain=StrainResult(
                timestamp=now,
                strain_score=0.0,
                strain_level="CALIBRATING",
                continuous_focus_sec=0.0,
                squint_factor=0.0,
                blink_suppression_factor=0.0,
                screen_time_factor=0.0,
                is_warning=False,
                gaze_eccentricity_factor=0.0,
            ),
            fatigue=FatigueResult(
                timestamp=now,
                fatigue_score=0.0,
                fatigue_level="CALIBRATING",
                perclos_score=0.0,
                blink_duration_score=0.0,
                circadian_factor=0.0,
                session_factor=0.0,
                is_warning=False,
                microsleep_risk=False,
            ),
            breaks=BreakStatus(
                is_break_recommended=False,
                is_in_break=False,
                remaining_until_break_sec=self.break_advisor.remaining_until_break_sec,
                remaining_break_duration_sec=0.0,
                reason="CALIBRATING",
                total_prompted=self.break_advisor._total_prompted,
                total_completed=self.break_advisor._total_completed,
                total_skipped=self.break_advisor._total_skipped,
                compliance_rate_pct=self.break_advisor.compliance_rate,
                audio_cue_needed=False,
            ),
            presence=PresenceState(True, 0.0, False, "CALIBRATING"),
            is_calibrating=True,
            calibration_progress=progress,
            is_camera_active=True,
        )

    def _finalize_calibration(self) -> None:
        self._is_calibrating = False
        if not self._calibration_ears:
            logger.warning("Calibration completed with no valid eye samples.")
            return

        sorted_ears = sorted(self._calibration_ears)
        # 90th percentile as open EAR baseline
        p90_idx = int(len(sorted_ears) * 0.90)
        open_baseline = sorted_ears[min(p90_idx, len(sorted_ears) - 1)]

        # Estimated closed baseline is ~50% of open baseline
        closed_baseline = open_baseline * 0.52

        self.config.profile.baseline_open_ear = round(open_baseline, 3)
        self.config.profile.baseline_closed_ear = round(closed_baseline, 3)
        self.config.profile.calibrated_at = time.strftime("%Y-%m-%d %H:%M:%S")

        # Update running detector thresholds
        new_closed_th = self.config.profile.dynamic_ear_threshold
        new_squint_th = self.config.profile.dynamic_squint_threshold

        self.blink_detector.update_thresholds(new_closed_th)
        self.squint_detector.update_thresholds(new_closed_th, new_squint_th)
        try:
            self.config.save()
        except Exception as e:
            logger.debug("Could not persist config file: %s", e)
        self.db.save_user_profile(self.config.profile)
        logger.info(
            "Calibration successful! Open EAR: %.3f, Closed: %.3f, Threshold: %.3f",
            open_baseline,
            closed_baseline,
            new_closed_th,
        )

    def start_api(self, host: str = "127.0.0.1", port: int = 8765) -> bool:
        """Start local REST API service for third-party health and wellness tools."""
        from oculisense.api import OculiSenseAPIServer
        if self._api_server is not None:
            return True
        try:
            self._api_server = OculiSenseAPIServer(self, host=host, port=port)
            self._api_server.start()
            return True
        except Exception as e:
            logger.error("Failed to start API server on %s:%d: %s", host, port, e)
            return False

    def stop_api(self) -> None:
        """Stop local REST API server."""
        if self._api_server is not None:
            self._api_server.stop()
            self._api_server = None

    def start(self, target_fps: int = 30) -> None:
        """Start daemon loop in background thread with adaptive rate limiting."""
        if self._running:
            return
        self._running = True
        active_interval = 1.0 / float(target_fps)
        idle_interval = 0.20  # 5 Hz when paused/idle to consume near-zero CPU

        def _loop():
            while self._running:
                t0 = time.time()
                snap = self.step()
                elapsed = time.time() - t0

                if not self._is_camera_active:
                    # Idle cadence: sleep up to 200ms
                    sleep_time = max(0.01, idle_interval - elapsed)
                elif snap.vision.source_status == "NO_FRAME":
                    # Backoff when frames are unavailable to prevent CPU spin loops
                    sleep_time = max(0.05, active_interval * 2.0 - elapsed)
                elif snap.vision.source_status == "SEARCHING_FACE" or not snap.vision.face_detected:
                    # Moderate pacing when camera is active but face is absent (~15 FPS to prevent high CPU)
                    sleep_time = max(0.02, 0.067 - elapsed)
                else:
                    # Active target FPS cadence (30 FPS)
                    sleep_time = max(0.001, active_interval - elapsed)

                time.sleep(sleep_time)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()
        logger.info("OculiSense Daemon started running at %d FPS.", target_fps)

    def stop(self) -> None:
        """Gracefully stop daemon and commit session summary."""
        self.stop_api()
        if not self._running and self._thread is None:
            return
        self._running = False
        self._is_camera_active = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        self.tracker.release()

        # Save session summary to DB
        summary = self.aggregator.get_session_summary(
            breaks_prompted=self.break_advisor._total_prompted,
            breaks_completed=self.break_advisor._total_completed,
            compliance_rate=self.break_advisor.compliance_rate,
        )
        self.db.end_session(self.session_id, summary)
        self.db.close()
        logger.info("OculiSense Daemon stopped. Session %s finalized.", self.session_id)
