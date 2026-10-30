"""
Digital Eye Strain Squint and Micro-Expression Detector.
Detects sustained eyelid narrowing (squinting) indicating refractive or digital strain.
"""

from __future__ import annotations
import collections
import time
from dataclasses import dataclass
from typing import Deque, Optional, Tuple


@dataclass
class SquintFrameResult:
    """Result of squint analysis for a single frame."""
    timestamp: float
    is_squinting: bool
    current_squint_duration_sec: float
    total_squint_episodes: int
    squint_time_ratio: float  # Fraction of recent window spent squinting [0.0 - 1.0]
    squint_intensity: float   # 0.0 to 1.0 how pronounced the squint is


class SquintDetector:
    """
    Detects persistent eyelid narrowing (squinting) as a primary indicator of digital eye strain.
    """

    def __init__(
        self,
        closed_threshold: float = 0.21,
        squint_upper_threshold: float = 0.26,
        min_squint_duration_sec: float = 1.5,
        window_sec: float = 300.0,  # 5 minute rolling window
    ) -> None:
        self.closed_threshold = closed_threshold
        self.squint_upper_threshold = squint_upper_threshold
        self.min_squint_duration_sec = min_squint_duration_sec
        self.window_sec = window_sec

        # State tracking
        self._squint_start_time: Optional[float] = None
        self._total_squint_episodes: int = 0
        self._is_in_squint_episode: bool = False

        # Rolling history of (timestamp, is_squinting) for ratio computation
        self._history: Deque[Tuple[float, bool]] = collections.deque()

    def update_thresholds(self, closed_threshold: float, squint_upper_threshold: float) -> None:
        """Update thresholds based on user calibration."""
        self.closed_threshold = closed_threshold
        self.squint_upper_threshold = max(closed_threshold + 0.02, squint_upper_threshold)

    def reset(self) -> None:
        """Reset state and counters."""
        self._squint_start_time = None
        self._total_squint_episodes = 0
        self._is_in_squint_episode = False
        self._history.clear()

    def process_frame(
        self,
        ear_avg: float,
        timestamp: Optional[float] = None,
        eyes_present: bool = True,
    ) -> SquintFrameResult:
        """
        Process a single frame EAR for squint detection.
        """
        if timestamp is None:
            timestamp = time.time()

        if not eyes_present or ear_avg <= 0.0:
            self._squint_start_time = None
            self._is_in_squint_episode = False
            return SquintFrameResult(
                timestamp=timestamp,
                is_squinting=False,
                current_squint_duration_sec=0.0,
                total_squint_episodes=self._total_squint_episodes,
                squint_time_ratio=self.get_squint_ratio(timestamp),
                squint_intensity=0.0,
            )

        # A squint candidate is an EAR between closed and the squint upper bound
        in_squint_zone = (self.closed_threshold < ear_avg <= self.squint_upper_threshold)
        
        current_duration = 0.0
        is_squinting = False
        intensity = 0.0

        if in_squint_zone:
            if self._squint_start_time is None:
                self._squint_start_time = timestamp
            current_duration = timestamp - self._squint_start_time

            # Intensity scales as EAR nears closed threshold from squint upper bound
            span = max(0.001, self.squint_upper_threshold - self.closed_threshold)
            intensity = max(0.0, min(1.0, (self.squint_upper_threshold - ear_avg) / span))

            # Only flag as a true squint once it has persisted beyond transient eye movements
            if current_duration >= self.min_squint_duration_sec:
                is_squinting = True
                if not self._is_in_squint_episode:
                    self._is_in_squint_episode = True
                    self._total_squint_episodes += 1
        else:
            # Not in squint zone (either eyes are fully open OR completely closed/blinking)
            self._squint_start_time = None
            self._is_in_squint_episode = False

        self._history.append((timestamp, is_squinting))
        self._prune_history(timestamp)

        ratio = self.get_squint_ratio(timestamp)

        return SquintFrameResult(
            timestamp=timestamp,
            is_squinting=is_squinting,
            current_squint_duration_sec=current_duration if is_squinting else 0.0,
            total_squint_episodes=self._total_squint_episodes,
            squint_time_ratio=ratio,
            squint_intensity=intensity if is_squinting else 0.0,
        )

    def _prune_history(self, now: float) -> None:
        cutoff = now - self.window_sec
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

    def get_squint_ratio(self, now: Optional[float] = None) -> float:
        """Fraction of time spent squinting in the rolling window."""
        if not self._history:
            return 0.0
        if now is not None:
            self._prune_history(now)
        if not self._history:
            return 0.0
        squint_frames = sum(1 for _, sq in self._history if sq)
        return float(squint_frames) / float(len(self._history))
