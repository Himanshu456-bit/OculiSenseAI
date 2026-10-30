"""
Synthetic Video and Landmark Feed for OculiSenseAI.
Enables full headless testing, automated unit verification, and interactive demo mode
without requiring a physical webcam.
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

Point2D = Tuple[float, float]


@dataclass
class SyntheticFrame:
    """A synthetic vision frame containing landmark coordinates and metadata."""
    timestamp: float
    left_eye: List[Point2D]
    right_eye: List[Point2D]
    face_detected: bool
    ear_left: float
    ear_right: float
    ear_avg: float
    current_state_name: str
    gaze_eccentricity: float = 0.0
    raw_image: Optional[object] = None  # RGB image representation if PIL/cv2 available


class SyntheticVideoSource:
    """
    Simulates human eyelid dynamics and facial presence across multiple behavioral states:
    - NORMAL: Healthy blink rate (16 bpm), normal EAR (0.32), brief blinks.
    - STRAINED: Reduced blink rate (5 bpm), sustained squinting (EAR 0.23).
    - FATIGUED: Prolonged closure, sluggish blinks (500ms), high PERCLOS.
    - AWAY: User left desk (face_detected=False).
    - CYCLIC: Cycles through all states for end-to-end demonstrations.
    """

    STATE_NORMAL = "NORMAL"
    STATE_STRAINED = "STRAINED"
    STATE_FATIGUED = "FATIGUED"
    STATE_AWAY = "AWAY"
    STATE_CYCLIC = "CYCLIC"

    def __init__(self, mode: str = STATE_CYCLIC, fps: int = 30) -> None:
        self.mode = mode
        self.fps = fps
        self.frame_duration = 1.0 / float(fps)
        self.start_time: Optional[float] = None
        self._current_time: Optional[float] = None
        self._cycle_duration = 15.0  # seconds per cycle phase

    def reset(self, start_time: Optional[float] = None) -> None:
        """Reset internal timeline and start timestamp."""
        self.start_time = start_time
        self._current_time = start_time

    def set_mode(self, mode: str) -> None:
        if mode != self.mode and mode == self.STATE_CYCLIC and self._current_time is not None:
            self.start_time = self._current_time
        self.mode = mode

    def _get_active_state(self, t: float) -> str:
        if self.mode != self.STATE_CYCLIC:
            return self.mode

        if self.start_time is None:
            self.start_time = t

        # In cyclic mode, transition every 15 seconds
        elapsed = max(0.0, t - self.start_time)
        phase = int(elapsed // 15.0) % 4
        if phase == 0:
            return self.STATE_NORMAL
        elif phase == 1:
            return self.STATE_STRAINED
        elif phase == 2:
            return self.STATE_FATIGUED
        else:
            return self.STATE_AWAY

    def read_frame(self, t: Optional[float] = None) -> SyntheticFrame:
        """
        Advance simulation and return synthetic landmarks for the current timestamp.
        """
        if t is None:
            if self._current_time is None:
                self._current_time = time.time()
                self.start_time = self._current_time
            else:
                self._current_time += self.frame_duration
            t = self._current_time
        else:
            if self.start_time is None or t < self.start_time:
                self.start_time = t
            self._current_time = t

        state = self._get_active_state(t)

        if state == self.STATE_AWAY:
            return SyntheticFrame(
                timestamp=t,
                left_eye=[],
                right_eye=[],
                face_detected=False,
                ear_left=0.0,
                ear_right=0.0,
                ear_avg=0.0,
                current_state_name="AWAY_FROM_DESK",
                raw_image=self._render_synthetic_image(state, 0.0, [], []),
            )

        # Baseline EAR parameters per state
        if state == self.STATE_NORMAL:
            base_ear = 0.32
            # Blink every ~3.5 seconds for 0.15s (4-5 frames at 30fps)
            blink_period = 3.5
            blink_duration = 0.15
            cycle_pos = t % blink_period
            is_blinking = cycle_pos < blink_duration
            if is_blinking:
                # Dip EAR down to 0.10
                ear = 0.10 + 0.05 * math.sin(math.pi * (cycle_pos / blink_duration))
            else:
                # Slight micro-fluctuation
                ear = base_ear + 0.01 * math.sin(t * 2.0)

        elif state == self.STATE_STRAINED:
            # Low blink rate (blinks every 12 seconds), persistent squint
            blink_period = 12.0
            blink_duration = 0.18
            cycle_pos = t % blink_period
            is_blinking = cycle_pos < blink_duration

            if is_blinking:
                ear = 0.11
            else:
                # Squint: narrowed eyelids in the 0.22 - 0.24 zone
                ear = 0.23 + 0.015 * math.sin(t * 1.5)

        elif state == self.STATE_FATIGUED:
            # Sluggish blinks (0.6s) and high closure periods
            blink_period = 4.0
            blink_duration = 0.65  # prolonged closure!
            cycle_pos = t % blink_period
            is_blinking = cycle_pos < blink_duration
            if is_blinking:
                ear = 0.08
            else:
                # Heavy eyelids
                ear = 0.26 + 0.01 * math.sin(t * 1.0)
        else:
            ear = 0.30

        # Construct 6-point eye landmarks matching this EAR
        left_eye, right_eye = self._generate_landmarks_for_ear(ear)

        # Generate synthetic video frame image for visualizer canvas
        raw_img = self._render_synthetic_image(state, ear, left_eye, right_eye)

        return SyntheticFrame(
            timestamp=t,
            left_eye=left_eye,
            right_eye=right_eye,
            face_detected=True,
            ear_left=ear,
            ear_right=ear,
            ear_avg=ear,
            current_state_name=state,
            gaze_eccentricity=0.0,
            raw_image=raw_img,
        )

    def _generate_landmarks_for_ear(self, target_ear: float) -> Tuple[List[Point2D], List[Point2D]]:
        """
        Creates 6 landmark points whose calculated EAR precisely equals target_ear.
        Width = 20.0 pixels.
        Height = target_ear * Width.
        """
        w = 20.0
        h = target_ear * w

        # Eye centered at (100, 100) for left, (160, 100) for right
        def make_eye(cx: float, cy: float) -> List[Point2D]:
            p1 = (cx - w / 2.0, cy)                # outer corner
            p2 = (cx - w / 4.0, cy - h / 2.0)      # top left
            p3 = (cx + w / 4.0, cy - h / 2.0)      # top right
            p4 = (cx + w / 2.0, cy)                # inner corner
            p5 = (cx + w / 4.0, cy + h / 2.0)      # bottom right
            p6 = (cx - w / 4.0, cy + h / 2.0)      # bottom left
            return [p1, p2, p3, p4, p5, p6]

        return make_eye(100.0, 100.0), make_eye(160.0, 100.0)

    def _render_synthetic_image(
        self,
        state: str,
        ear_val: float,
        left_pts: List[Point2D],
        right_pts: List[Point2D],
    ) -> Optional[object]:
        """Render a 640x480 BGR synthetic frame simulating face landmarks for visualizer canvas."""
        try:
            import cv2
            import numpy as np
        except Exception:
            return None

        w, h = 640, 480
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (24, 18, 15)  # dark slate background

        if state == self.STATE_AWAY:
            cv2.putText(img, "USER AWAY FROM DESK", (140, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
            cv2.putText(img, "No face detected in camera viewport", (150, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
            return img

        # Stylized head oval
        cv2.ellipse(img, (320, 235), (145, 185), 0, 0, 360, (55, 42, 35), -1)
        cv2.ellipse(img, (320, 235), (145, 185), 0, 0, 360, (110, 85, 70), 2)

        # Eyebrows
        cv2.line(img, (220, 175), (280, 170), (40, 30, 25), 3)
        cv2.line(img, (360, 170), (420, 175), (40, 30, 25), 3)

        # Draw left and right eyes
        for cx_eye, cy_eye, tag in [(250, 205, "L"), (390, 205, "R")]:
            ew = 36
            eh = max(2, int(ear_val * 48.0))
            box_col = (0, 255, 128) if ear_val >= 0.21 else (0, 165, 255)

            # Eye sclera
            cv2.ellipse(img, (cx_eye, cy_eye), (ew, eh), 0, 0, 360, (235, 235, 235), -1)
            # Iris
            iris_r = max(2, min(eh, 13))
            cv2.circle(img, (cx_eye, cy_eye), iris_r, (40, 75, 30), -1)
            # Pupil
            cv2.circle(img, (cx_eye, cy_eye), max(1, min(eh // 2, 6)), (10, 10, 10), -1)
            # Eyelid contour
            cv2.ellipse(img, (cx_eye, cy_eye), (ew, eh), 0, 0, 360, box_col, 2)

            # Bounding box & EAR tag
            x1, y1 = cx_eye - ew - 4, cy_eye - eh - 4
            x2, y2 = cx_eye + ew + 4, cy_eye + eh + 4
            cv2.rectangle(img, (x1, y1), (x2, y2), box_col, 1)
            cv2.putText(img, f"{tag}:{ear_val:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, box_col, 1)

            # 6 Landmark dots
            pts = [
                (cx_eye - ew, cy_eye),
                (cx_eye - ew // 2, cy_eye - eh),
                (cx_eye + ew // 2, cy_eye - eh),
                (cx_eye + ew, cy_eye),
                (cx_eye + ew // 2, cy_eye + eh),
                (cx_eye - ew // 2, cy_eye + eh),
            ]
            for px, py in pts:
                cv2.circle(img, (int(px), int(py)), 2, (0, 255, 0), -1)

        cv2.putText(img, f"DEMO SIMULATOR • {state}", (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)
        return img
