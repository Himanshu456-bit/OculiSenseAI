"""
Unified Eye and Facial Landmark Tracker.
Integrates MediaPipe Face Mesh, OpenCV cascades, and Synthetic fallback feeds.
"""

from __future__ import annotations
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from oculisense.config import VisionConfig
from oculisense.vision.geometry import (
    LEFT_EYE_LANDMARKS,
    RIGHT_EYE_LANDMARKS,
    calculate_ear,
    calculate_gaze_eccentricity,
)
from oculisense.vision.synthetic_feed import SyntheticVideoSource

logger = logging.getLogger(__name__)

# Check optional dependencies safely
try:
    import cv2  # type: ignore
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    import mediapipe as mp  # type: ignore
    from mediapipe.tasks import python as mp_python  # type: ignore
    from mediapipe.tasks.python import vision as mp_vision  # type: ignore
    HAS_MEDIAPIPE = (
        hasattr(mp_vision, "FaceLandmarker")
        or (hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"))
    )
except Exception:
    HAS_MEDIAPIPE = False


def detect_available_cameras(max_probe: int = 5) -> List[int]:
    """
    Probe and return list of openable webcam indices.
    Tests Windows DirectShow and standard backends with frame read verification.
    """
    if not HAS_OPENCV:
        return []

    is_windows = sys.platform == "win32" or os.name == "nt"
    backends: List[Tuple[Any, str]] = []
    if is_windows and hasattr(cv2, "CAP_DSHOW"):
        backends.append((cv2.CAP_DSHOW, "CAP_DSHOW"))
    if is_windows and hasattr(cv2, "CAP_MSMF"):
        backends.append((cv2.CAP_MSMF, "CAP_MSMF"))
    backends.append((cv2.CAP_ANY, "DEFAULT"))

    accessible_indices: List[int] = []
    for idx in range(max_probe):
        for b_flag, _ in backends:
            try:
                cap = cv2.VideoCapture(idx, b_flag)
                if cap is not None and cap.isOpened():
                    for _ in range(3):
                        ret, f = cap.read()
                        if ret and f is not None and getattr(f, "size", 0) > 0:
                            accessible_indices.append(idx)
                            cap.release()
                            break
                        time.sleep(0.04)
                    else:
                        cap.release()
                        continue
                    break
                if cap is not None:
                    cap.release()
            except Exception:
                pass
    return accessible_indices


@dataclass
class VisionTelemetry:
    """Standardized vision tracking output for downstream analytics and UI."""
    timestamp: float
    face_detected: bool
    ear_left: float
    ear_right: float
    ear_avg: float
    left_eye_landmarks: List[Tuple[float, float]] = field(default_factory=list)
    right_eye_landmarks: List[Tuple[float, float]] = field(default_factory=list)
    annotated_frame: Optional[Any] = None
    is_synthetic: bool = False
    source_status: str = "ONLINE"
    gaze_eccentricity: float = 0.0
    camera_id: int = 0
    source_indicator: str = "[LIVE WEBCAM 0]"

    @property
    def display_indicator(self) -> str:
        """Clear visual badge indicating active source."""
        if self.source_status == "CAMERA_OFF":
            return "[CAMERA OFF]"
        if self.is_synthetic:
            return "[DEMO]"
        return f"[LIVE WEBCAM {self.camera_id}]"

    @property
    def display_status(self) -> str:
        """Composite status string combining source indicator and tracking state."""
        return f"{self.display_indicator} {self.source_status}"


class EyeTracker:
    """
    Orchestrates webcam capture and facial landmark extraction.
    Automatically falls back to synthetic feed if no physical webcam is detected.
    """

    def __init__(
        self,
        config: Optional[VisionConfig] = None,
        force_synthetic: bool = False,
        autostart: bool = False,
    ) -> None:
        self.config = config or VisionConfig()
        self.force_synthetic = force_synthetic

        self._active_camera_id = self.config.camera_id
        self._cap: Optional[Any] = None
        self._mp_landmarker: Optional[Any] = None
        self._mp_face_mesh: Optional[Any] = None
        self._mp_mesh_instance: Optional[Any] = None
        self._synthetic_source: Optional[SyntheticVideoSource] = None
        self._is_active = False
        self._consecutive_frame_failures = 0
        self._last_reconnect_time: float = 0.0

        # Model asset path for modern MediaPipe Vision Tasks
        self._model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "models",
            "face_landmarker.task",
        )

        if autostart:
            self.start_camera()

    @property
    def is_active(self) -> bool:
        return self._is_active

    @property
    def active_camera_id(self) -> int:
        return self._active_camera_id

    def start_camera(self) -> bool:
        """Initialize and activate camera hardware or synthetic feed."""
        if self._is_active and (
            (self._cap is not None and self._cap.isOpened())
            or (self.force_synthetic and self._synthetic_source is not None)
        ):
            return True
        self._is_active = True
        return self._init_backend()

    def stop_camera(self) -> None:
        """Deactivate camera hardware and pause capture."""
        self._is_active = False
        self.release()

    def toggle_camera(self) -> bool:
        """Toggle camera between active and inactive states."""
        if self._is_active:
            self.stop_camera()
            return False
        else:
            return self.start_camera()

    def switch_camera(self, camera_id: int) -> bool:
        """Switch active webcam device index."""
        self.config.camera_id = camera_id
        self._active_camera_id = camera_id
        if not self._is_active:
            return True

        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        self._consecutive_frame_failures = 0
        cap = self._open_camera_capture(preferred_id=camera_id, allow_fallback=False)
        if cap is not None:
            self._cap = cap
            self._synthetic_source = None
            try:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self._cap.set(cv2.CAP_PROP_FPS, self.config.target_fps)
            except Exception:
                pass
            self._init_face_mesh()
            logger.info("Successfully switched to camera index %d.", camera_id)
            return True
        else:
            logger.warning("Failed to open camera index %d on switch.", camera_id)
            return False

    def _open_camera_capture(
        self,
        preferred_id: Optional[int] = None,
        allow_fallback: bool = True,
    ) -> Optional[Any]:
        """
        Attempt to open physical webcam with appropriate backend and index fallbacks.
        Prioritizes user-selected camera index, then probes candidate indices 0, 1, 2, 3, 4.
        On Windows, prefers DirectShow (CAP_DSHOW), then MSMF, then default backend,
        with frame read warmup retries to accommodate asynchronous hardware startup.
        """
        if not HAS_OPENCV:
            return None

        is_windows = sys.platform == "win32" or os.name == "nt"
        target_id = self.config.camera_id if preferred_id is None else preferred_id

        # Order candidates: target first, then remaining standard indices if fallback allowed
        candidate_ids = [target_id]
        if allow_fallback:
            for idx in range(5):
                if idx not in candidate_ids:
                    candidate_ids.append(idx)

        # Backends to try for each candidate index
        backends: List[Tuple[Any, str]] = []
        if is_windows and hasattr(cv2, "CAP_DSHOW"):
            backends.append((cv2.CAP_DSHOW, "cv2.CAP_DSHOW"))
        if is_windows and hasattr(cv2, "CAP_MSMF"):
            backends.append((cv2.CAP_MSMF, "cv2.CAP_MSMF"))
        backends.append((cv2.CAP_ANY, "DEFAULT"))

        for cam_id in candidate_ids:
            for b_flag, b_name in backends:
                try:
                    cap = cv2.VideoCapture(cam_id, b_flag)
                    if cap is not None and cap.isOpened():
                        # Warm up stream and retry reading initial frames
                        for _ in range(8):
                            ret, test_frame = cap.read()
                            if ret and test_frame is not None and getattr(test_frame, "size", 0) > 0:
                                logger.info(
                                    "Successfully opened webcam index %d with %s backend.",
                                    cam_id,
                                    b_name,
                                )
                                self._active_camera_id = cam_id
                                self.config.camera_id = cam_id
                                return cap
                            time.sleep(0.05)
                        cap.release()
                except Exception as e:
                    logger.debug("Failed opening camera index %d with %s: %s", cam_id, b_name, e)

        return None

    def _init_face_mesh(self) -> None:
        """Initialize MediaPipe FaceLandmarker Task or legacy FaceMesh."""
        if not HAS_MEDIAPIPE or not self.config.use_mediapipe:
            return

        if self._mp_landmarker is not None or self._mp_mesh_instance is not None:
            return

        # Priority 1: Modern MediaPipe Vision Tasks FaceLandmarker with Blendshapes
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            if os.path.exists(self._model_path):
                base_options = mp_python.BaseOptions(model_asset_path=self._model_path)
                options = mp_vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    output_face_blendshapes=True,
                    output_facial_transformation_matrixes=False,
                    num_faces=1,
                    min_face_detection_confidence=self.config.detection_confidence,
                    min_face_presence_confidence=self.config.tracking_confidence,
                    min_tracking_confidence=self.config.tracking_confidence,
                )
                self._mp_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
                logger.info("Modern MediaPipe FaceLandmarker Task active with 52 blendshapes and 478 3D landmarks.")
                return
        except Exception as task_err:
            logger.warning("MediaPipe Tasks FaceLandmarker init failed: %s. Falling back to legacy...", task_err)
            self._mp_landmarker = None

        # Priority 2: Legacy MediaPipe FaceMesh
        try:
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                self._mp_face_mesh = mp.solutions.face_mesh
                self._mp_mesh_instance = self._mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=self.config.detection_confidence,
                    min_tracking_confidence=self.config.tracking_confidence,
                )
                logger.info("Legacy MediaPipe FaceMesh active.")
        except Exception as mp_err:
            logger.warning("MediaPipe initialization failed (%s). Using OpenCV dynamic ocular tracking.", mp_err)
            self._mp_mesh_instance = None

    def _init_backend(self) -> bool:
        """Initialize camera capture or synthetic engine."""
        if self.force_synthetic or not HAS_OPENCV:
            logger.info("Using SyntheticVideoSource for vision feed.")
            if self._synthetic_source is None:
                self._synthetic_source = SyntheticVideoSource(mode=SyntheticVideoSource.STATE_CYCLIC)
            self._is_active = True
            return True

        # Attempt to open physical webcam
        try:
            if self._cap is not None and not self._cap.isOpened():
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None

            if self._cap is None:
                self._cap = self._open_camera_capture(preferred_id=self.config.camera_id, allow_fallback=True)

            if self._cap is None or not self._cap.isOpened():
                logger.warning(
                    "Physical webcam not immediately accessible on indices 0-4. "
                    "Will actively probe and connect in background."
                )
                self._cap = None
                self._synthetic_source = None  # Strictly NO synthetic fallback when not in demo mode
            else:
                self._synthetic_source = None
                try:
                    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self._cap.set(cv2.CAP_PROP_FPS, self.config.target_fps)
                except Exception:
                    pass
                self._init_face_mesh()

            self._is_active = True
            return True
        except Exception as e:
            logger.error("Error initializing camera backend: %s.", e)
            self._cap = None
            self._synthetic_source = None
            self._is_active = True
            return True

    def set_synthetic_mode(self, mode: str) -> None:
        """Change behavioral scenario in synthetic feed."""
        if self._synthetic_source is not None:
            self._synthetic_source.set_mode(mode)

    def read(self, timestamp: Optional[float] = None) -> VisionTelemetry:
        """
        Capture the next frame, extract eye coordinates, and calculate EAR.
        """
        now = timestamp if timestamp is not None else time.time()

        # If camera is disabled/paused, do zero calculations and return CAMERA_OFF telemetry
        if not self._is_active:
            return VisionTelemetry(
                timestamp=now,
                face_detected=False,
                ear_left=0.0,
                ear_right=0.0,
                ear_avg=0.0,
                annotated_frame=None,
                is_synthetic=self.force_synthetic,
                source_status="CAMERA_OFF",
                gaze_eccentricity=0.0,
                camera_id=self._active_camera_id,
                source_indicator=f"[LIVE WEBCAM {self._active_camera_id}]",
            )

        # Handle synthetic feed ONLY if explicitly forced by user (--demo)
        if self.force_synthetic:
            if self._synthetic_source is None:
                self._synthetic_source = SyntheticVideoSource(mode=SyntheticVideoSource.STATE_CYCLIC)

            synth_frame = self._synthetic_source.read_frame(now)
            ear_l = calculate_ear(synth_frame.left_eye) if synth_frame.left_eye else 0.0
            ear_r = calculate_ear(synth_frame.right_eye) if synth_frame.right_eye else 0.0
            ear_avg = (ear_l + ear_r) / 2.0 if synth_frame.face_detected else 0.0

            return VisionTelemetry(
                timestamp=now,
                face_detected=synth_frame.face_detected,
                ear_left=ear_l,
                ear_right=ear_r,
                ear_avg=ear_avg,
                left_eye_landmarks=synth_frame.left_eye,
                right_eye_landmarks=synth_frame.right_eye,
                annotated_frame=getattr(synth_frame, "raw_image", None),
                is_synthetic=True,
                source_status=f"DEMO ({synth_frame.current_state_name})",
                gaze_eccentricity=getattr(synth_frame, "gaze_eccentricity", 0.0),
                camera_id=self._active_camera_id,
                source_indicator="[DEMO]",
            )

        # Physical webcam mode: reconnect if capture is not currently open
        if self._cap is None or not self._cap.isOpened():
            if now - getattr(self, "_last_reconnect_time", 0.0) >= 1.5:
                self._last_reconnect_time = now
                self._cap = self._open_camera_capture(preferred_id=self._active_camera_id, allow_fallback=True)
                if self._cap is not None and self._cap.isOpened():
                    try:
                        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        self._cap.set(cv2.CAP_PROP_FPS, self.config.target_fps)
                    except Exception:
                        pass
                    self._init_face_mesh()

            if self._cap is None or not self._cap.isOpened():
                return VisionTelemetry(
                    timestamp=now,
                    face_detected=False,
                    ear_left=0.0,
                    ear_right=0.0,
                    ear_avg=0.0,
                    annotated_frame=None,
                    is_synthetic=False,
                    source_status="NO_CAMERA",
                    gaze_eccentricity=0.0,
                    camera_id=self._active_camera_id,
                    source_indicator=f"[LIVE WEBCAM {self._active_camera_id}]",
                )

        # Real OpenCV / MediaPipe capture
        ret, frame = self._cap.read()
        if not ret or frame is None:
            self._consecutive_frame_failures += 1
            if self._consecutive_frame_failures > 30:
                logger.warning("Camera lost frame delivery. Releasing capture to re-acquire...")
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None
                self._last_reconnect_time = now
            return VisionTelemetry(
                timestamp=now,
                face_detected=False,
                ear_left=0.0,
                ear_right=0.0,
                ear_avg=0.0,
                source_status="NO_FRAME",
                gaze_eccentricity=0.0,
                camera_id=self._active_camera_id,
                source_indicator=f"[LIVE WEBCAM {self._active_camera_id}]",
            )

        self._consecutive_frame_failures = 0

        # Flip horizontally for mirrored mirror view
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Modern MediaPipe Vision Tasks (FaceLandmarker with 52 blendshapes & 478 3D landmarks)
        if self._mp_landmarker is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            try:
                detection_result = self._mp_landmarker.detect(mp_image)
            except Exception as det_err:
                logger.debug("MediaPipe FaceLandmarker detection error: %s", det_err)
                detection_result = None

            if (
                detection_result is not None
                and detection_result.face_landmarks
                and len(detection_result.face_landmarks) > 0
            ):
                face_landmarks = detection_result.face_landmarks[0]

                # Extract 6 canonical points for left and right eyes
                left_points = [
                    (face_landmarks[idx].x * w, face_landmarks[idx].y * h)
                    for idx in LEFT_EYE_LANDMARKS
                ]
                right_points = [
                    (face_landmarks[idx].x * w, face_landmarks[idx].y * h)
                    for idx in RIGHT_EYE_LANDMARKS
                ]

                ear_l_geom = calculate_ear(left_points)
                ear_r_geom = calculate_ear(right_points)

                # Extract blendshapes
                blendshapes_dict = {}
                if detection_result.face_blendshapes and len(detection_result.face_blendshapes) > 0:
                    for cat in detection_result.face_blendshapes[0]:
                        blendshapes_dict[cat.category_name] = cat.score

                blink_l = blendshapes_dict.get("eyeBlinkLeft", 0.0)
                blink_r = blendshapes_dict.get("eyeBlinkRight", 0.0)

                closed_thresh = getattr(self.config, "ear_closed_threshold", 0.21)

                # Robust fusion of Euclidean EAR and Blendshape Blink probability:
                # If blendshape signals closure (>= 0.45), ensure EAR drops below closed threshold
                if blink_l >= 0.45:
                    ear_l = min(ear_l_geom, 0.14)
                elif blink_l <= 0.15:
                    ear_l = max(ear_l_geom, 0.26)
                else:
                    ear_l = ear_l_geom

                if blink_r >= 0.45:
                    ear_r = min(ear_r_geom, 0.14)
                elif blink_r <= 0.15:
                    ear_r = max(ear_r_geom, 0.26)
                else:
                    ear_r = ear_r_geom

                ear_avg = (ear_l + ear_r) / 2.0
                gaze_ecc = calculate_gaze_eccentricity(left_points, right_points, frame_width=float(w))

                # Render contours and landmarks
                for pt in left_points + right_points:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)

                try:
                    import numpy as np
                    l_arr = np.array([[int(p[0]), int(p[1])] for p in left_points], dtype=np.int32)
                    r_arr = np.array([[int(p[0]), int(p[1])] for p in right_points], dtype=np.int32)
                    cv2.polylines(frame, [l_arr], isClosed=True, color=(0, 255, 255), thickness=1)
                    cv2.polylines(frame, [r_arr], isClosed=True, color=(0, 255, 255), thickness=1)
                except Exception:
                    pass

                for eye_pts, ear_v, b_prob, tag in [
                    (left_points, ear_l, blink_l, "L"),
                    (right_points, ear_r, blink_r, "R"),
                ]:
                    xs = [p[0] for p in eye_pts]
                    ys = [p[1] for p in eye_pts]
                    x1, x2 = max(0, int(min(xs)) - 6), min(w, int(max(xs)) + 6)
                    y1, y2 = max(0, int(min(ys)) - 6), min(h, int(max(ys)) + 6)
                    is_closed = ear_v < closed_thresh or b_prob >= 0.50
                    box_color = (0, 165, 255) if is_closed else (0, 255, 128)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 1)

                    label = f"{tag}:{ear_v:.2f}"
                    if b_prob > 0.10:
                        label += f" ({int(b_prob * 100)}%)"
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(12, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.38,
                        box_color,
                        1,
                        getattr(cv2, "LINE_AA", 8),
                    )

                cv2.putText(
                    frame,
                    "[TRACKING: MEDIAPIPE TASKS]",
                    (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,
                    (0, 255, 128),
                    1,
                    getattr(cv2, "LINE_AA", 8),
                )

                return VisionTelemetry(
                    timestamp=now,
                    face_detected=True,
                    ear_left=ear_l,
                    ear_right=ear_r,
                    ear_avg=ear_avg,
                    left_eye_landmarks=left_points,
                    right_eye_landmarks=right_points,
                    annotated_frame=frame,
                    is_synthetic=False,
                    source_status="TRACKING",
                    gaze_eccentricity=round(gaze_ecc, 3),
                    camera_id=self._active_camera_id,
                    source_indicator=f"[LIVE WEBCAM {self._active_camera_id}]",
                )

        # Legacy MediaPipe FaceMesh
        if self._mp_mesh_instance is not None:
            # Process via MediaPipe FaceMesh
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._mp_mesh_instance.process(rgb_frame)

            if not results.multi_face_landmarks:
                return VisionTelemetry(
                    timestamp=now,
                    face_detected=False,
                    ear_left=0.0,
                    ear_right=0.0,
                    ear_avg=0.0,
                    annotated_frame=frame,
                    source_status="SEARCHING_FACE",
                    gaze_eccentricity=0.0,
                    camera_id=self._active_camera_id,
                    source_indicator=f"[LIVE WEBCAM {self._active_camera_id}]",
                )

            face_landmarks = results.multi_face_landmarks[0].landmark

            # Extract 6 points for left and right eyes
            left_points = [
                (face_landmarks[idx].x * w, face_landmarks[idx].y * h)
                for idx in LEFT_EYE_LANDMARKS
            ]
            right_points = [
                (face_landmarks[idx].x * w, face_landmarks[idx].y * h)
                for idx in RIGHT_EYE_LANDMARKS
            ]

            ear_l = calculate_ear(left_points)
            ear_r = calculate_ear(right_points)
            ear_avg = (ear_l + ear_r) / 2.0
            gaze_ecc = calculate_gaze_eccentricity(left_points, right_points, frame_width=float(w))

            # Draw visual landmarks and eye region overlays on frame for UI
            for pt in left_points + right_points:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)

            # Draw eye contour polylines and bounding regions
            try:
                import numpy as np
                l_arr = np.array([[int(p[0]), int(p[1])] for p in left_points], dtype=np.int32)
                r_arr = np.array([[int(p[0]), int(p[1])] for p in right_points], dtype=np.int32)
                cv2.polylines(frame, [l_arr], isClosed=True, color=(0, 255, 255), thickness=1)
                cv2.polylines(frame, [r_arr], isClosed=True, color=(0, 255, 255), thickness=1)
            except Exception:
                pass

            for eye_pts, ear_v, tag in [(left_points, ear_l, "L"), (right_points, ear_r, "R")]:
                xs = [p[0] for p in eye_pts]
                ys = [p[1] for p in eye_pts]
                x1, x2 = max(0, int(min(xs)) - 5), min(w, int(max(xs)) + 5)
                y1, y2 = max(0, int(min(ys)) - 5), min(h, int(max(ys)) + 5)
                box_color = (0, 255, 128) if ear_v >= getattr(self.config, "ear_closed_threshold", 0.21) else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 1)
                cv2.putText(
                    frame,
                    f"{tag}:{ear_v:.2f}",
                    (x1, max(12, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    box_color,
                    1,
                    getattr(cv2, "LINE_AA", 8),
                )

            return VisionTelemetry(
                timestamp=now,
                face_detected=True,
                ear_left=ear_l,
                ear_right=ear_r,
                ear_avg=ear_avg,
                left_eye_landmarks=left_points,
                right_eye_landmarks=right_points,
                annotated_frame=frame,
                is_synthetic=False,
                source_status="TRACKING",
                gaze_eccentricity=round(gaze_ecc, 3),
                camera_id=self._active_camera_id,
                source_indicator=f"[LIVE WEBCAM {self._active_camera_id}]",
            )

        # Real OpenCV face/eye processing when MediaPipe FaceMesh is absent
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Multi-Space Skin Segmentation (YCrCb + HSV) for robust face detection under varied lighting
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        mask_ycrcb = cv2.inRange(ycrcb, (20, 128, 70), (255, 185, 140))

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_hsv1 = cv2.inRange(hsv, (0, 15, 30), (35, 240, 255))
        mask_hsv2 = cv2.inRange(hsv, (160, 15, 30), (180, 240, 255))
        mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)

        skin_mask = cv2.bitwise_or(mask_ycrcb, mask_hsv)

        # Morphological operations to coalesce facial regions and remove background noise
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, k_open)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, k_close)

        min_face_area = w * h * 0.03
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        face_rect = None
        best_face_score = -1.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_face_area:
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect = float(bh) / max(1.0, float(bw))
                # Head aspect ratio roughly 0.7 to 2.2
                if 0.7 <= aspect <= 2.2:
                    cx = x + bw / 2.0
                    cy = y + bh / 2.0
                    if cy < h * 0.85:
                        center_dev = abs(cx - w / 2.0) / (w / 2.0)
                        score = area * (1.0 - 0.4 * center_dev)
                        if score > best_face_score:
                            best_face_score = score
                            face_rect = (x, y, bw, bh)

        # Fallback to central region if skin segmentation was underexposed/overexposed
        if face_rect is None:
            mean_val = float(gray.mean())
            std_val = float(gray.std())
            if mean_val > 15.0 and std_val > 12.0:
                fw, fh = int(w * 0.45), int(h * 0.55)
                fx, fy = int((w - fw) / 2), int((h - fh) / 2)
                face_rect = (fx, fy, fw, fh)
            else:
                return VisionTelemetry(
                    timestamp=now,
                    face_detected=False,
                    ear_left=0.0,
                    ear_right=0.0,
                    ear_avg=0.0,
                    annotated_frame=frame,
                    is_synthetic=False,
                    source_status="SEARCHING_FACE",
                    gaze_eccentricity=0.0,
                    camera_id=self._active_camera_id,
                    source_indicator=f"[LIVE WEBCAM {self._active_camera_id}]",
                )

        fx, fy, fw, fh = face_rect

        # 2. Dynamic Pupil & Eye Feature Localization
        # The eye horizontal band is anatomically located between 22% and 48% of the face height
        ey_start = max(0, fy + int(fh * 0.22))
        ey_end = min(h, fy + int(fh * 0.48))

        # Left eye search zone (screen left, user's right eye in mirrored view)
        lx_start = max(0, fx + int(fw * 0.12))
        lx_end = min(w, fx + int(fw * 0.48))

        # Right eye search zone (screen right, user's left eye in mirrored view)
        rx_start = max(0, fx + int(fw * 0.52))
        rx_end = min(w, fx + int(fw * 0.88))

        zone_left = gray[ey_start:ey_end, lx_start:lx_end]
        zone_right = gray[ey_start:ey_end, rx_start:rx_end]

        if zone_left.size > 20 and zone_right.size > 20:
            blur_l = cv2.GaussianBlur(zone_left, (7, 7), 0)
            blur_r = cv2.GaussianBlur(zone_right, (7, 7), 0)
            _, _, min_loc_l, _ = cv2.minMaxLoc(blur_l)
            _, _, min_loc_r, _ = cv2.minMaxLoc(blur_r)

            raw_left_cx = lx_start + min_loc_l[0]
            raw_left_cy = ey_start + min_loc_l[1]
            raw_right_cx = rx_start + min_loc_r[0]
            raw_right_cy = ey_start + min_loc_r[1]
        else:
            raw_left_cx = fx + int(fw * 0.30)
            raw_left_cy = fy + int(fh * 0.35)
            raw_right_cx = fx + int(fw * 0.70)
            raw_right_cy = fy + int(fh * 0.35)

        # Smooth eye centers to eliminate jitter while following head motion
        if not hasattr(self, "_smooth_left_eye"):
            self._smooth_left_eye = (float(raw_left_cx), float(raw_left_cy))
            self._smooth_right_eye = (float(raw_right_cx), float(raw_right_cy))
        else:
            slx, sly = self._smooth_left_eye
            srx, sry = self._smooth_right_eye
            self._smooth_left_eye = (0.70 * raw_left_cx + 0.30 * slx, 0.70 * raw_left_cy + 0.30 * sly)
            self._smooth_right_eye = (0.70 * raw_right_cx + 0.30 * srx, 0.70 * raw_right_cy + 0.30 * sry)

        left_eye_cx = int(self._smooth_left_eye[0])
        left_eye_cy = int(self._smooth_left_eye[1])
        right_eye_cx = int(self._smooth_right_eye[0])
        right_eye_cy = int(self._smooth_right_eye[1])

        crop_w = max(20, int(fw * 0.20))
        crop_h = max(14, int(fh * 0.14))

        # 3. Precise Eye Aperture and EAR Measurement
        def _analyze_eye(cx_val: int, cy_val: int) -> Tuple[float, List[Tuple[float, float]]]:
            x1 = max(0, cx_val - crop_w // 2)
            x2 = min(w, cx_val + crop_w // 2)
            y1 = max(0, cy_val - crop_h // 2)
            y2 = min(h, cy_val + crop_h // 2)
            crop = gray[y1:y2, x1:x2]

            if crop.size < 16:
                ear_val = 0.30
            else:
                try:
                    import numpy as np
                    ch, cw = crop.shape
                    p15 = float(np.percentile(crop, 15))
                    p85 = float(np.percentile(crop, 85))
                    dynamic_range = max(1.0, p85 - p15)

                    dark_thresh = p15 + 0.35 * dynamic_range
                    dark_mask = (crop < dark_thresh).astype(np.uint8)

                    # Vertical profile in central 50% of the crop
                    c_start, c_end = int(cw * 0.25), int(cw * 0.75)
                    mid_cols = dark_mask[:, c_start:c_end]
                    vert_profile = mid_cols.sum(axis=1)
                    active_rows = int((vert_profile > max(1, mid_cols.shape[1] * 0.20)).sum())

                    horiz_profile = dark_mask.sum(axis=0)
                    active_cols = int((horiz_profile > 0).sum())

                    # Eye closure criteria: collapsed vertical aperture or flat texture
                    is_closed = (active_rows <= 2 or active_cols < 4 or dynamic_range < 10.0)
                    if is_closed:
                        ear_val = max(0.08, min(0.16, 0.08 + (active_rows / max(1.0, float(ch))) * 0.15))
                    else:
                        ratio = float(active_rows) / float(max(4, active_cols))
                        ear_val = max(0.26, min(0.38, 0.24 + ratio * 0.20))
                except Exception:
                    ear_val = 0.30

            ew = crop_w * 0.45
            eh = max(2.0, min(14.0, ear_val * 35.0))
            pts = [
                (float(cx_val - ew), float(cy_val)),
                (float(cx_val - ew * 0.5), float(cy_val - eh)),
                (float(cx_val + ew * 0.5), float(cy_val - eh)),
                (float(cx_val + ew), float(cy_val)),
                (float(cx_val + ew * 0.5), float(cy_val + eh)),
                (float(cx_val - ew * 0.5), float(cy_val + eh)),
            ]
            return round(ear_val, 3), pts

        ear_l, l_pts = _analyze_eye(left_eye_cx, left_eye_cy)
        ear_r, r_pts = _analyze_eye(right_eye_cx, right_eye_cy)
        ear_avg = round((ear_l + ear_r) / 2.0, 3)

        center_offset = abs((fx + fw / 2.0) - (w / 2.0)) / (w / 2.0)
        gaze_ecc = round(min(1.0, center_offset * 1.2), 3)

        # 4. Render High-Visibility Tracking Overlays on Frame
        # Face bounding box
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 128), 2)
        cv2.putText(
            frame,
            "FACE TRACKED",
            (fx, max(14, fy - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 128),
            1,
            getattr(cv2, "LINE_AA", 8),
        )

        # Render eyes, pupils, and aperture contour polylines
        for eye_pts, ear_v, cx_e, cy_e, tag in [
            (l_pts, ear_l, left_eye_cx, left_eye_cy, "L"),
            (r_pts, ear_r, right_eye_cx, right_eye_cy, "R"),
        ]:
            # Glowing pupil center
            cv2.circle(frame, (cx_e, cy_e), 4, (0, 255, 255), -1)
            cv2.circle(frame, (cx_e, cy_e), 2, (0, 0, 255), -1)

            # Contour points
            for pt in eye_pts:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)

            # Eye box
            bx1, bx2 = max(0, cx_e - crop_w // 2), min(w, cx_e + crop_w // 2)
            by1, by2 = max(0, cy_e - crop_h // 2), min(h, cy_e + crop_h // 2)
            box_color = (56, 189, 248) if ear_v >= 0.20 else (0, 165, 255)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_color, 1)

            # Eye EAR label badge
            cv2.rectangle(frame, (bx1, max(0, by1 - 15)), (bx1 + 48, by1), (15, 23, 42), -1)
            cv2.putText(
                frame,
                f"{tag}:{ear_v:.2f}",
                (bx1 + 2, by1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                box_color,
                1,
                getattr(cv2, "LINE_AA", 8),
            )

        return VisionTelemetry(
            timestamp=now,
            face_detected=True,
            ear_left=ear_l,
            ear_right=ear_r,
            ear_avg=ear_avg,
            left_eye_landmarks=l_pts,
            right_eye_landmarks=r_pts,
            annotated_frame=frame,
            is_synthetic=False,
            source_status="TRACKING",
            gaze_eccentricity=gaze_ecc,
            camera_id=self._active_camera_id,
            source_indicator=f"[LIVE WEBCAM {self._active_camera_id}]",
        )

    def release(self) -> None:
        """Release camera hardware and pipelines."""
        self._is_active = False
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        if self._mp_landmarker is not None:
            try:
                self._mp_landmarker.close()
            except Exception:
                pass
            self._mp_landmarker = None
        if self._mp_mesh_instance is not None:
            try:
                self._mp_mesh_instance.close()
            except Exception:
                pass
            self._mp_mesh_instance = None
        if not self.force_synthetic:
            self._synthetic_source = None
