"""
Real-time Blink Detector and PERCLOS Telemetry Engine.
Tracks blink count, duration, frequency (BPM), and percentage of eye closure.
"""

from __future__ import annotations
import collections
import time
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple


@dataclass
class BlinkEvent:
    """Record of a completed blink."""
    timestamp: float
    duration_ms: float
    min_ear: float
    is_prolonged: bool = False


@dataclass
class BlinkFrameResult:
    """Result of blink analysis for a single frame."""
    timestamp: float
    ear_left: float
    ear_right: float
    ear_avg: float
    is_eyes_closed: bool
    blink_detected_this_frame: bool
    total_blinks: int
    blink_rate_bpm: float
    perclos: float  # Percentage of eye closure [0.0 - 1.0]
    last_closure_duration_ms: float
    is_prolonged_closure: bool
    eyes_present: bool
    consecutive_closed_frames: int = 0


class BlinkDetector:
    """
    Robust blink detection state machine with rolling PERCLOS and frequency metrics.
    """

    def __init__(
        self,
        closed_threshold: float = 0.21,
        min_blink_frames: int = 2,
        max_blink_frames: int = 15,
        prolonged_closure_sec: float = 0.5,
        perclos_window_sec: float = 60.0,
    ) -> None:
        self.closed_threshold = closed_threshold
        self.min_blink_frames = min_blink_frames
        self.max_blink_frames = max_blink_frames
        self.prolonged_closure_sec = prolonged_closure_sec
        self.perclos_window_sec = perclos_window_sec

        # State tracking
        self._consecutive_closed_frames: int = 0
        self._closure_start_time: Optional[float] = None
        self._min_ear_in_blink: float = 1.0
        self._total_blinks: int = 0
        self._last_closure_duration_ms: float = 0.0

        # Rolling history buffers (timestamp, was_closed)
        self._frame_history: Deque[Tuple[float, bool]] = collections.deque()
        # Completed blinks history (timestamp, duration_ms)
        self._blink_history: Deque[BlinkEvent] = collections.deque()

        # Real-time responsive BPM tracking
        self._last_blink_time: Optional[float] = None
        self._last_frame_time: Optional[float] = None
        self._realtime_bpm: float = 0.0
        self._recent_ibis: Deque[float] = collections.deque(maxlen=8)

    @property
    def total_blinks(self) -> int:
        return self._total_blinks

    @property
    def blink_rate_bpm(self) -> float:
        return self.get_blink_rate()

    def update_thresholds(self, closed_threshold: float) -> None:
        """Dynamically update threshold (e.g. after user calibration)."""
        self.closed_threshold = max(0.05, min(0.5, closed_threshold))

    def reset(self) -> None:
        """Reset internal counters and histories."""
        self._consecutive_closed_frames = 0
        self._closure_start_time = None
        self._min_ear_in_blink = 1.0
        self._total_blinks = 0
        self._last_closure_duration_ms = 0.0
        self._frame_history.clear()
        self._blink_history.clear()
        self._last_blink_time = None
        self._last_frame_time = None
        self._realtime_bpm = 0.0
        self._recent_ibis.clear()

    def process_frame(
        self,
        ear_left: float,
        ear_right: float,
        timestamp: Optional[float] = None,
        eyes_present: bool = True,
    ) -> BlinkFrameResult:
        """
        Process a single video frame's EAR values.
        """
        if timestamp is None:
            timestamp = time.time()

        if not eyes_present:
            # Eyes not in frame: do not record as closed blink, but clear ongoing closure
            self._consecutive_closed_frames = 0
            self._closure_start_time = None
            self._last_frame_time = timestamp
            return BlinkFrameResult(
                timestamp=timestamp,
                ear_left=0.0,
                ear_right=0.0,
                ear_avg=0.0,
                is_eyes_closed=False,
                blink_detected_this_frame=False,
                total_blinks=self._total_blinks,
                blink_rate_bpm=self.get_blink_rate(timestamp),
                perclos=self.get_perclos(timestamp),
                last_closure_duration_ms=self._last_closure_duration_ms,
                is_prolonged_closure=False,
                eyes_present=False,
            )

        self._last_frame_time = timestamp
        ear_avg = (ear_left + ear_right) / 2.0
        is_closed = ear_avg < self.closed_threshold

        # Record to rolling history for PERCLOS
        self._frame_history.append((timestamp, is_closed))
        self._prune_history(timestamp)

        blink_detected = False
        is_prolonged = False

        if is_closed:
            self._consecutive_closed_frames += 1
            if self._closure_start_time is None:
                self._closure_start_time = timestamp
                self._min_ear_in_blink = ear_avg
            else:
                self._min_ear_in_blink = min(self._min_ear_in_blink, ear_avg)

            # Check for prolonged closure (e.g. eye closure > 500ms)
            current_closure_duration = timestamp - self._closure_start_time
            if current_closure_duration >= self.prolonged_closure_sec:
                is_prolonged = True
                self._last_closure_duration_ms = current_closure_duration * 1000.0
        else:
            # Eyes are currently OPEN
            closure_duration = (
                timestamp - self._closure_start_time
                if self._closure_start_time
                else 0.0
            )
            duration_ms = closure_duration * 1000.0

            # Register blink if minimum frame criteria met or duration matches human blink (>= 45ms)
            if self._consecutive_closed_frames >= self.min_blink_frames or (
                self.min_blink_frames <= 1 and duration_ms >= 45.0
            ):
                self._last_closure_duration_ms = duration_ms

                # Register blink event
                self._total_blinks += 1
                blink_detected = True
                was_prolonged = closure_duration >= self.prolonged_closure_sec

                event = BlinkEvent(
                    timestamp=timestamp,
                    duration_ms=duration_ms,
                    min_ear=self._min_ear_in_blink,
                    is_prolonged=was_prolonged,
                )
                self._blink_history.append(event)

                # Real-time IBI and instantaneous BPM computation
                if self._last_blink_time is not None:
                    ibi = max(0.1, timestamp - self._last_blink_time)
                    if 0.15 <= ibi <= 25.0:
                        self._recent_ibis.append(ibi)
                        inst_bpm = 60.0 / ibi
                        if self._realtime_bpm <= 0.0:
                            self._realtime_bpm = inst_bpm
                        else:
                            self._realtime_bpm = 0.55 * self._realtime_bpm + 0.45 * inst_bpm
                else:
                    if self._realtime_bpm <= 0.0:
                        self._realtime_bpm = 16.0
                self._last_blink_time = timestamp

            # Reset closure trackers
            self._consecutive_closed_frames = 0
            self._closure_start_time = None
            self._min_ear_in_blink = 1.0

        # Decay real-time BPM if user is actively staring
        if self._last_blink_time is not None:
            time_since_blink = max(0.0, timestamp - self._last_blink_time)
            if time_since_blink > 3.0:
                max_possible = 60.0 / time_since_blink
                if self._realtime_bpm > max_possible:
                    self._realtime_bpm = max_possible
            if time_since_blink > 45.0:
                self._realtime_bpm = 0.0

        bpm = self.get_blink_rate(timestamp)
        perclos = self.get_perclos(timestamp)

        return BlinkFrameResult(
            timestamp=timestamp,
            ear_left=ear_left,
            ear_right=ear_right,
            ear_avg=ear_avg,
            is_eyes_closed=is_closed,
            blink_detected_this_frame=blink_detected,
            total_blinks=self._total_blinks,
            blink_rate_bpm=bpm,
            perclos=perclos,
            last_closure_duration_ms=self._last_closure_duration_ms,
            is_prolonged_closure=is_prolonged,
            eyes_present=True,
            consecutive_closed_frames=self._consecutive_closed_frames if is_closed else 0,
        )

    def _prune_history(self, now: float) -> None:
        """Evict records older than perclos_window_sec."""
        cutoff = now - self.perclos_window_sec
        while self._frame_history and self._frame_history[0][0] < cutoff:
            self._frame_history.popleft()
        while self._blink_history and self._blink_history[0].timestamp < cutoff:
            self._blink_history.popleft()

    def get_perclos(self, now: Optional[float] = None) -> float:
        """
        Compute Percentage of Eye Closure (PERCLOS) over rolling window.
        Returns float 0.0 to 1.0.
        """
        if not self._frame_history:
            return 0.0
        if now is not None:
            self._prune_history(now)
        if len(self._frame_history) < 10:
            return 0.0
        closed_count = sum(1 for _, closed in self._frame_history if closed)
        return float(closed_count) / float(len(self._frame_history))

    def get_blink_rate(self, now: Optional[float] = None) -> float:
        """
        Compute real-time responsive blinks per minute (BPM).
        Combines inter-blink interval (IBI), dynamic screen-stare decay,
        and rolling-window frequency for instantaneous accuracy.
        Query-safe and does not mutate internal detector state.
        """
        if now is None:
            now = self._last_frame_time if self._last_frame_time is not None else time.time()
        self._prune_history(now)

        if not self._frame_history:
            return 0.0

        decayed_bpm = self._realtime_bpm
        # Dynamic decay if user is staring without blinking
        if self._last_blink_time is not None:
            time_since_blink = max(0.0, now - self._last_blink_time)
            if time_since_blink > 3.0:
                max_possible = 60.0 / time_since_blink
                if decayed_bpm > max_possible:
                    decayed_bpm = max_possible
            if time_since_blink > 45.0:
                decayed_bpm = 0.0

        # Rolling window calculation
        window_start = self._frame_history[0][0]
        elapsed_sec = max(1.0, now - window_start)
        scale_factor = 60.0 / min(60.0, elapsed_sec)
        rolling_bpm = float(len(self._blink_history)) * scale_factor

        if decayed_bpm > 0.0:
            if len(self._blink_history) >= 2:
                final_bpm = 0.65 * decayed_bpm + 0.35 * rolling_bpm
            else:
                final_bpm = decayed_bpm
        else:
            final_bpm = rolling_bpm

        return round(max(0.0, min(60.0, final_bpm)), 1)

    def get_recent_blink_durations(self) -> List[float]:
        """Returns durations in ms of recent blinks in window."""
        return [b.duration_ms for b in self._blink_history]
