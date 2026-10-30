"""
Smart Break Reminders and 20-20-20 Rule Advisor.
Monitors continuous focus intervals, manages the 20-20-20 rest cycle,
verifies user compliance (looking away), and logs rest compliance history.
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Callable, Optional

from oculisense.config import BreakConfig


@dataclass
class BreakStatus:
    """Current state of the 20-20-20 break advisor."""
    is_break_recommended: bool
    is_in_break: bool
    remaining_until_break_sec: float
    remaining_break_duration_sec: float
    reason: str  # "SCHEDULED_20_20_20", "CRITICAL_LOW_EAR_SCORE", "FATIGUE_THRESHOLD", "STRAIN_THRESHOLD"
    total_prompted: int
    total_completed: int
    total_skipped: int
    compliance_rate_pct: float
    audio_cue_needed: bool
    least_recorded_ear: float = 1.0
    eval_elapsed_sec: float = 0.0
    is_evaluating: bool = False
    eval_decision: str = "OBSERVING"
    dual_threshold_fatigue_score: float = 15.0
    eval_condition_detail: str = "NORMAL"


class BreakAdvisor:
    """
    Enforces the clinical 20-20-20 rule:
    'Every 20 minutes, look at something 20 feet away for at least 20 seconds.'
    """

    STATE_WORKING = "WORKING"
    STATE_PROMPTED = "PROMPTED"
    STATE_RESTING = "RESTING"
    STATE_SNOOZED = "SNOOZED"

    def __init__(
        self,
        config: Optional[BreakConfig] = None,
        on_break_event: Optional[Callable[[str, str, float], None]] = None,
        eval_window_sec: float = 20.0,
    ) -> None:
        self.config = config or BreakConfig()
        self.on_break_event = on_break_event
        self.eval_window_sec: float = eval_window_sec

        self._state: str = self.STATE_WORKING
        self._last_break_end_time: Optional[float] = None
        self._break_prompt_time: Optional[float] = None
        self._break_start_time: Optional[float] = None
        self._snooze_until_time: Optional[float] = None

        # Rest compliance verification counters
        self._rest_eyes_away_sec: float = 0.0
        self._last_frame_time: Optional[float] = None
        self._away_start_time: Optional[float] = None

        # Debounce and cooldown controls
        self._last_skip_time: Optional[float] = None
        self._min_skip_cooldown_sec: float = 60.0

        # 20-Second Continuous EAR Observation Window & 4 Ergonomic Conditions
        self._eval_start_time: Optional[float] = None
        self._eval_ears: List[float] = []
        self._eval_least_ear: float = 1.0
        self._eval_closed_sec: float = 0.0
        self._eval_decision: str = "OBSERVING"
        self._last_eval_outcome: str = "OBSERVING"
        self._consecutive_closed_sec: float = 0.0
        self._squint_cycle_count: int = 0
        self._last_fatigue_index: float = 15.0
        self._eval_condition_detail: str = "NORMAL"

        # Stats
        self._total_prompted: int = 0
        self._total_completed: int = 0
        self._total_skipped: int = 0
        self._snooze_count: int = 0

        self._pending_audio_cue: bool = False
        self._active_reason: str = "SCHEDULED_20_20_20"
        self._pomodoro_completed_cycles: int = 0

    def reset(self, timestamp: Optional[float] = None) -> None:
        """Reset work timer, state, and tracking timestamps."""
        now = timestamp if timestamp is not None else time.time()
        self._state = self.STATE_WORKING
        self._last_break_end_time = now
        self._last_frame_time = now
        self._break_prompt_time = None
        self._break_start_time = None
        self._snooze_until_time = None
        self._rest_eyes_away_sec = 0.0
        self._pending_audio_cue = False
        self._away_start_time = None
        self._last_skip_time = None

        # Reset 20-second evaluation window
        self._eval_start_time = now
        self._eval_least_ear = 1.0
        self._eval_ears.clear()
        self._eval_closed_sec = 0.0
        self._eval_decision = "OBSERVING"
        self._last_eval_outcome = "OBSERVING"
        self._consecutive_closed_sec = 0.0
        self._squint_cycle_count = 0
        self._last_fatigue_index = 15.0
        self._eval_condition_detail = "NORMAL"

    def shift_work_time(self, delta_sec: float) -> None:
        """Shift work timer forward by delta_sec to exclude idle/paused time."""
        if delta_sec <= 0:
            return
        if self._last_break_end_time is not None:
            self._last_break_end_time += delta_sec

    def _get_target_work_interval(self) -> float:
        """Returns work interval duration based on 20-20-20 or Pomodoro mode."""
        if getattr(self.config, "enable_pomodoro", False):
            return float(self.config.pomodoro_work_sec)
        return float(self.config.rule_20_20_20_interval_sec)

    def _get_target_rest_duration(self) -> float:
        """Returns rest duration based on 20-20-20 or Pomodoro mode."""
        if getattr(self.config, "enable_pomodoro", False):
            is_long = ((self._pomodoro_completed_cycles + 1) % self.config.pomodoro_cycles_before_long_rest == 0)
            return float(self.config.pomodoro_long_rest_sec if is_long else self.config.pomodoro_short_rest_sec)
        return float(self.config.rule_20_20_20_duration_sec)

    @property
    def compliance_rate(self) -> float:
        if self._total_prompted == 0:
            return 100.0
        return (float(self._total_completed) / float(self._total_prompted)) * 100.0

    @property
    def remaining_until_break_sec(self) -> float:
        now = self._last_frame_time if self._last_frame_time is not None else time.time()
        target_interval = self._get_target_work_interval()
        if self._last_break_end_time is None:
            return target_interval
        elapsed = now - self._last_break_end_time
        return max(0.0, target_interval - elapsed)

    def trigger_dynamic_break(self, reason: str = "STRAIN_THRESHOLD", timestamp: Optional[float] = None) -> None:
        """Manually or threshold-triggered rest recommendation."""
        if self._state not in (self.STATE_PROMPTED, self.STATE_RESTING):
            self._state = self.STATE_PROMPTED
            now = timestamp if timestamp is not None else (self._last_frame_time if self._last_frame_time is not None else time.time())
            self._break_prompt_time = now
            self._active_reason = reason
            self._total_prompted += 1
            self._pending_audio_cue = self.config.enable_audio_cue

    def start_break(self, timestamp: Optional[float] = None) -> None:
        """User begins looking away or confirms rest."""
        now = timestamp or time.time()
        self._state = self.STATE_RESTING
        self._break_start_time = now
        self._rest_eyes_away_sec = 0.0

    def complete_break(self, timestamp: Optional[float] = None) -> None:
        """Successfully finished break."""
        now = timestamp or time.time()
        duration = round(now - self._break_start_time, 1) if self._break_start_time else self._get_target_rest_duration()
        self._total_completed += 1
        self._pomodoro_completed_cycles += 1
        self._state = self.STATE_WORKING
        self._last_break_end_time = now
        self._last_skip_time = None
        self._break_prompt_time = None
        self._break_start_time = None
        self._snooze_count = 0

        # Reset 20-second evaluation window
        self._eval_start_time = now
        self._eval_least_ear = 1.0
        self._eval_ears.clear()
        self._eval_closed_sec = 0.0
        self._eval_decision = "OBSERVING"
        self._last_eval_outcome = "OBSERVING"
        self._consecutive_closed_sec = 0.0
        self._squint_cycle_count = 0
        self._last_fatigue_index = 15.0
        self._eval_condition_detail = "NORMAL"

        if self.on_break_event:
            try:
                self.on_break_event(self._active_reason, "COMPLETED", duration)
            except Exception:
                pass

    def snooze_break(self, timestamp: Optional[float] = None) -> bool:
        """Delay break prompt by snooze_duration_sec."""
        now = timestamp or time.time()
        if self._snooze_count >= self.config.auto_snooze_limit:
            self.skip_break(now)
            return False  # Snooze limit exceeded, dismiss to avoid infinite popup loop
        self._state = self.STATE_SNOOZED
        self._snooze_count += 1
        self._snooze_until_time = now + self.config.snooze_duration_sec
        self._break_prompt_time = None
        self._break_start_time = None
        if self.on_break_event:
            try:
                self.on_break_event(self._active_reason, "SNOOZED", float(self.config.snooze_duration_sec))
            except Exception:
                pass
        return True

    def skip_break(self, timestamp: Optional[float] = None) -> None:
        """User dismisses break without resting."""
        now = timestamp or time.time()
        self._total_skipped += 1
        self._state = self.STATE_WORKING
        self._last_break_end_time = now
        self._last_skip_time = now
        self._break_prompt_time = None
        self._break_start_time = None
        self._snooze_until_time = None

        # Reset 20-second evaluation window and engage cooldown
        self._eval_start_time = now
        self._eval_least_ear = 1.0
        self._eval_ears.clear()
        self._eval_closed_sec = 0.0
        self._eval_decision = "OBSERVING"
        self._last_eval_outcome = "OBSERVING"
        self._consecutive_closed_sec = 0.0
        self._squint_cycle_count = 0
        self._last_fatigue_index = 15.0
        self._eval_condition_detail = "NORMAL"

        if self.on_break_event:
            try:
                self.on_break_event(self._active_reason, "SKIPPED", 0.0)
            except Exception:
                pass

    def update(
        self,
        eyes_present: bool,
        is_eyes_closed: bool,
        strain_score: float = 0.0,
        fatigue_score: float = 0.0,
        timestamp: Optional[float] = None,
        ear_avg: float = 0.30,
        blink_rate_bpm: float = 16.0,
    ) -> BreakStatus:
        """
        Update advisor state with the latest frame biometrics.
        Evaluates rest overlay triggers under four specific ergonomic conditions
        over a rolling 20-second window:
          1. Prolonged Eye Droop / Microsleep Episode (EAR < 0.20 continuously for > 1.8s)
          2. Sustained Micro-Squinting (Average EAR compressed to 0.21 - 0.25 across 2 cycles)
          3. Severe Blink Suppression / Stare Freeze (BPM < 6 while EAR remains high > 0.30)
          4. Dual-Threshold Fatigue Index:
             Fatigue Score = 15 + (PERCLOS_20s * 60) + max(0, (12 - BPM) * 3) + max(0, (0.24 - EAR_min) * 100)
             Trigger when Fatigue Score >= 70.
        """
        now = timestamp or time.time()
        prev_frame_time = self._last_frame_time
        if (
            self._last_break_end_time is None
            or prev_frame_time is None
            or abs(now - prev_frame_time) > 86400
            or abs(now - self._last_break_end_time) > 86400
        ):
            if self._last_break_end_time is None or abs(now - self._last_break_end_time) > 86400:
                self._last_break_end_time = now
            dt = 0.0
            raw_dt = 0.0
        else:
            raw_dt = max(0.0, now - prev_frame_time)
            dt = min(2.0, raw_dt)
        self._last_frame_time = now

        # Track user presence to prevent accumulating absence time into focus interval
        if not eyes_present:
            self._consecutive_closed_sec = 0.0
            if self._away_start_time is None:
                self._away_start_time = prev_frame_time if prev_frame_time is not None else now

            away_duration = max(raw_dt, now - self._away_start_time)
            # If away for >= 45 seconds, user had natural break away from screen -> reset
            if away_duration >= 45.0:
                self._last_break_end_time = now
                if self._state in (self.STATE_PROMPTED, self.STATE_RESTING):
                    self._state = self.STATE_WORKING
                self._break_prompt_time = None
                self._break_start_time = None
                self._snooze_until_time = None
                self._eval_start_time = now
                self._eval_least_ear = 1.0
                self._eval_ears.clear()
                self._eval_closed_sec = 0.0
                self._consecutive_closed_sec = 0.0
                self._squint_cycle_count = 0
            else:
                # Do not accumulate absence time into continuous focus/work interval
                if self._state == self.STATE_WORKING and self._last_break_end_time is not None:
                    self._last_break_end_time += raw_dt
                if self._eval_start_time is not None:
                    self._eval_start_time += raw_dt
            eval_elapsed = max(0.0, now - self._eval_start_time) if self._eval_start_time else 0.0
        else:
            self._away_start_time = None
            # Track continuous eye droop / closure (EAR < 0.20 or confirmed closed)
            if ear_avg < 0.20 or is_eyes_closed:
                self._consecutive_closed_sec += dt
            else:
                self._consecutive_closed_sec = 0.0

            # Accumulate EAR samples and closure time in the 20-second evaluation window
            if self._eval_start_time is None:
                self._eval_start_time = now
                self._eval_least_ear = ear_avg
                self._eval_ears = [ear_avg]
                self._eval_closed_sec = dt if (ear_avg < 0.21 or is_eyes_closed) else 0.0
            else:
                self._eval_ears.append(ear_avg)
                if ear_avg < self._eval_least_ear:
                    self._eval_least_ear = ear_avg
                if ear_avg < 0.21 or is_eyes_closed:
                    self._eval_closed_sec += dt
            eval_elapsed = max(0.0, now - self._eval_start_time)

        audio_cue = False
        if self._pending_audio_cue:
            audio_cue = True
            self._pending_audio_cue = False

        # Check threshold triggers
        if self._state == self.STATE_WORKING:
            skip_cooldown_active = (
                self._last_skip_time is not None
                and (now - self._last_skip_time) < self._min_skip_cooldown_sec
            )
            # Check scheduled interval (20-20-20 or Pomodoro)
            work_interval = self._get_target_work_interval()
            elapsed_work = now - self._last_break_end_time
            if elapsed_work >= work_interval and not skip_cooldown_active:
                if getattr(self.config, "enable_pomodoro", False):
                    is_long = ((self._pomodoro_completed_cycles + 1) % self.config.pomodoro_cycles_before_long_rest == 0)
                    reason = "POMODORO_LONG_REST" if is_long else "POMODORO_SHORT_REST"
                else:
                    reason = "SCHEDULED_20_20_20"
                self.trigger_dynamic_break(reason=reason, timestamp=now)

            # Condition 1: Immediate Prolonged Eye Droop / Microsleep Episode
            # EAR < 0.20 continuously for > 1.8 seconds (~45-60 frames at 30 FPS)
            elif eyes_present and (self._consecutive_closed_sec >= 1.8):
                self._eval_decision = "REST_RECOMMENDED"
                self._last_eval_outcome = "REST_RECOMMENDED"
                self._eval_condition_detail = "CONDITION_1_MICROSLEEP_DROOP"
                if not skip_cooldown_active:
                    self.trigger_dynamic_break(reason="CRITICAL_LOW_EAR_SCORE", timestamp=now)

            # 20-Second Dynamic EAR & Biometric Observation
            elif not eyes_present:
                pass
            elif eval_elapsed < self.eval_window_sec:
                # Count the EAR for at least 20s before deciding: DO NOT DISPLAY OVERLAY
                self._eval_decision = "OBSERVING"
                self._eval_condition_detail = "OBSERVING"
            else:
                # 20 SECONDS COMPLETE: Full Ergonomic Evaluation across the entire 20s window!
                eval_time = max(1.0, eval_elapsed)
                perclos_20s = min(1.0, max(0.0, self._eval_closed_sec / eval_time))
                ear_min = self._eval_least_ear
                ear_mean = sum(self._eval_ears) / max(1, len(self._eval_ears))
                bpm = max(0.0, blink_rate_bpm)

                # Condition 4: Dual-Threshold Fatigue Index (Combined Score)
                # Fatigue Score = 15 + (PERCLOS_20s * 60) + max(0, (12 - BPM) * 3.0) + max(0, (0.24 - EAR_min) * 100)
                fatigue_index = (
                    15.0
                    + (perclos_20s * 60.0)
                    + max(0.0, (12.0 - bpm) * 3.0)
                    + max(0.0, (0.24 - ear_min) * 100.0)
                )
                self._last_fatigue_index = round(fatigue_index, 1)

                # Condition 2: Sustained Micro-Squinting (Asthenopia / Strain)
                # Average EAR compressed between 0.21 and 0.25 (vs open baseline >= 0.30)
                is_micro_squinting = (0.21 <= ear_mean <= 0.25 and ear_min >= 0.18)
                if is_micro_squinting:
                    self._squint_cycle_count += 1
                else:
                    self._squint_cycle_count = 0
                cond2_triggered = (self._squint_cycle_count >= 2)

                # Condition 3: Severe Blink Suppression (Stare Freeze)
                # Rolling BPM < 6.0 while EAR remains high (> 0.30)
                cond3_triggered = (bpm < 6.0 and ear_mean > 0.30 and elapsed_work >= 300.0)

                # Condition 4 Trigger Check (Threshold: >= 70.0)
                cond4_triggered = (fatigue_index >= 70.0)

                # Overall wellness score
                max_penalty = max(strain_score, fatigue_score)
                avg_penalty = 0.5 * strain_score + 0.5 * fatigue_score
                effective_penalty = 0.6 * max_penalty + 0.4 * avg_penalty
                overall_score = max(0.0, min(100.0, 100.0 - effective_penalty))
                critical_low_overall = (overall_score < 40.0)
                severe_fatigue = (fatigue_score >= 85.0 or strain_score >= 80.0)

                has_trigger = (
                    cond4_triggered
                    or cond2_triggered
                    or cond3_triggered
                    or critical_low_overall
                    or severe_fatigue
                    or (ear_min < 0.18 and perclos_20s >= 0.10)
                    or (self._eval_closed_sec >= 1.5 and ear_mean < 0.22)
                )

                if not has_trigger:
                    # Healthy user: DO NOT DISPLAY OVERLAY!
                    self._eval_decision = "HEALTHY_PASSED"
                    self._last_eval_outcome = "HEALTHY_PASSED"
                    self._eval_condition_detail = "HEALTHY_PASSED"
                    # Reset observation window to begin next 20s cycle
                    self._eval_start_time = now
                    self._eval_least_ear = ear_avg
                    self._eval_ears = [ear_avg]
                    self._eval_closed_sec = 0.0
                    self._consecutive_closed_sec = 0.0
                    if strain_score < 70.0 and fatigue_score < 75.0 and overall_score >= 45.0:
                        self._last_skip_time = None
                else:
                    # Ergonomic condition triggered in 20s window!
                    self._eval_decision = "REST_RECOMMENDED"
                    self._last_eval_outcome = "REST_RECOMMENDED"
                    if not skip_cooldown_active:
                        if cond4_triggered or ear_min < 0.18 or self._eval_closed_sec >= 1.5:
                            reason = "CRITICAL_LOW_EAR_SCORE"
                            self._eval_condition_detail = "CONDITION_4_DUAL_THRESHOLD_FATIGUE"
                        elif cond2_triggered:
                            reason = "STRAIN_THRESHOLD"
                            self._eval_condition_detail = "CONDITION_2_SUSTAINED_SQUINT"
                        elif cond3_triggered:
                            reason = "SCHEDULED_20_20_20"
                            self._eval_condition_detail = "CONDITION_3_STARE_FREEZE"
                        elif critical_low_overall:
                            reason = "CRITICAL_LOW_SCORE"
                            self._eval_condition_detail = "CRITICAL_LOW_SCORE"
                        elif fatigue_score >= 85.0:
                            reason = "FATIGUE_THRESHOLD"
                            self._eval_condition_detail = "FATIGUE_THRESHOLD"
                        else:
                            reason = "STRAIN_THRESHOLD"
                            self._eval_condition_detail = "STRAIN_THRESHOLD"
                        self.trigger_dynamic_break(reason=reason, timestamp=now)

        elif self._state == self.STATE_SNOOZED:
            if self._snooze_until_time and now >= self._snooze_until_time:
                self.trigger_dynamic_break(reason="SNOOZE_EXPIRED", timestamp=now)

        elif self._state == self.STATE_PROMPTED:
            # If user already looks away or closes eyes for > 2 seconds, auto-start resting
            if not eyes_present or is_eyes_closed:
                self.start_break(now)

        elif self._state == self.STATE_RESTING:
            # In resting state, count seconds eyes are relaxed/away from screen
            if not eyes_present or is_eyes_closed:
                self._rest_eyes_away_sec += dt
            else:
                # User looking back at screen prematurely
                pass

            target_duration = self._get_target_rest_duration()
            rest_elapsed = now - (self._break_start_time or now)
            if rest_elapsed >= target_duration:
                # Finished break!
                self.complete_break(now)

        # Compute remaining countdowns
        target_duration = self._get_target_rest_duration()
        target_interval = self._get_target_work_interval()
        if self._state == self.STATE_RESTING:
            rem_break = max(
                0.0,
                target_duration - (now - (self._break_start_time or now)),
            )
            rem_until = 0.0
        elif self._state == self.STATE_PROMPTED:
            rem_break = target_duration
            rem_until = 0.0
        elif self._state == self.STATE_SNOOZED:
            rem_break = target_duration
            rem_until = max(0.0, (self._snooze_until_time or now) - now)
        else:
            elapsed = now - self._last_break_end_time
            rem_until = max(0.0, target_interval - elapsed)
            rem_break = target_duration

        return BreakStatus(
            is_break_recommended=(self._state in (self.STATE_PROMPTED, self.STATE_RESTING)),
            is_in_break=(self._state == self.STATE_RESTING),
            remaining_until_break_sec=round(rem_until, 1),
            remaining_break_duration_sec=round(rem_break, 1),
            reason=self._active_reason,
            total_prompted=self._total_prompted,
            total_completed=self._total_completed,
            total_skipped=self._total_skipped,
            compliance_rate_pct=round(self.compliance_rate, 1),
            audio_cue_needed=audio_cue,
            least_recorded_ear=round(self._eval_least_ear if self._eval_least_ear <= 1.0 else 0.30, 3),
            eval_elapsed_sec=round(eval_elapsed, 1),
            is_evaluating=(self._state == self.STATE_WORKING and eval_elapsed < self.eval_window_sec),
            eval_decision=(self._last_eval_outcome if self._last_eval_outcome != "OBSERVING" else self._eval_decision),
            dual_threshold_fatigue_score=round(self._last_fatigue_index, 1),
            eval_condition_detail=self._eval_condition_detail,
        )
