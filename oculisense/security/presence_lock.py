"""
Workstation Presence Monitor and Automated Screen Lock Security Layer.
Detects when the user steps away from their desk and can trigger an automatic workstation lock.
"""

from __future__ import annotations
import ctypes
import logging
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, Optional

from oculisense.config import SecurityConfig

logger = logging.getLogger(__name__)


@dataclass
class PresenceState:
    """Current state of workstation presence monitoring."""
    is_present: bool
    away_duration_sec: float
    lock_triggered: bool
    status_label: str  # "PRESENT", "AWAY_PENDING_LOCK", "LOCKED", "AWAY"


class PresenceLockManager:
    """
    Tracks presence at desk using biometric eye tracking.
    Triggers OS screen lock when absence exceeds threshold.
    """

    def __init__(
        self,
        config: Optional[SecurityConfig] = None,
        on_lock_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        self.config = config or SecurityConfig()
        self.on_lock_callback = on_lock_callback

        self._last_seen_time: Optional[float] = None
        self._is_present: bool = True
        self._lock_executed: bool = False
        self._total_away_events: int = 0

    def reset(self) -> None:
        """Reset presence timers and lock flags (e.g., when camera toggles or re-activates)."""
        self._last_seen_time = None
        self._is_present = True
        self._lock_executed = False

    def update(self, eyes_present: bool, timestamp: Optional[float] = None) -> PresenceState:
        """
        Update presence state with frame observation.
        """
        now = timestamp or time.time()
        if self._last_seen_time is None or abs(now - self._last_seen_time) > 86400:
            self._last_seen_time = now

        if eyes_present:
            if not self._is_present:
                logger.info("User returned to workstation.")
            self._is_present = True
            self._last_seen_time = now
            self._lock_executed = False
            return PresenceState(
                is_present=True,
                away_duration_sec=0.0,
                lock_triggered=False,
                status_label="PRESENT",
            )

        # Eyes absent
        away_duration = now - self._last_seen_time

        if self._is_present and away_duration > 1.0:
            self._is_present = False
            self._total_away_events += 1

        lock_triggered = False
        if away_duration >= self.config.away_timeout_sec and not self._lock_executed:
            if self.config.enable_presence_lock:
                self._execute_lock()
                lock_triggered = True
            self._lock_executed = True

        status = "LOCKED" if self._lock_executed else ("AWAY_PENDING_LOCK" if away_duration > 5.0 else "PRESENT")

        return PresenceState(
            is_present=False,
            away_duration_sec=round(away_duration, 1),
            lock_triggered=lock_triggered,
            status_label=status,
        )

    def _execute_lock(self) -> None:
        """Execute platform-specific workstation lock or dry-run simulation."""
        logger.warning("Workstation lock triggered due to user absence.")

        if self.on_lock_callback:
            try:
                self.on_lock_callback()
            except Exception as e:
                logger.error("Error in on_lock_callback: %s", e)

        if self.config.dry_run_lock:
            logger.info("[DRY RUN] Workstation screen lock simulated successfully.")
            return

        sys_platform = platform.system()
        try:
            if sys_platform == "Windows":
                # Windows API call to lock desktop
                ctypes.windll.user32.LockWorkStation()
            elif sys_platform == "Darwin":
                # macOS lock command
                subprocess.run(
                    ["pmset", "displaysleepnow"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                # Linux (freedesktop / xscreensaver / gnome-screensaver)
                subprocess.run(
                    ["xdg-screensaver", "lock"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except Exception as e:
            logger.error("Failed to execute OS workstation lock: %s", e)
