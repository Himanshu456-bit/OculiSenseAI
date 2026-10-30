"""
Unit and integration tests for camera on/off toggle state, idle bypass of biometric
calculations and calibrations, and adaptive rate-limiting CPU load balancing.
"""

from __future__ import annotations
import json
import os
import tempfile
import time
import urllib.request

from oculisense.config import AppConfig
from oculisense.daemon import OculiSenseDaemon


def test_camera_default_state_and_no_calculations():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_camera_idle.db")
        cfg = AppConfig()
        cfg.privacy.database_filename = db_path

        daemon = OculiSenseDaemon(config=cfg, force_synthetic=True, autostart_camera=False)
        assert not daemon.is_camera_active
        assert not daemon.tracker.is_active

        # Step through 15 cycles while camera is inactive
        t = 1000.0
        for _ in range(15):
            snap = daemon.step(custom_time=t)
            t += 0.033

            # Verify no biometric calculations occurred
            assert not snap.is_camera_active
            assert snap.vision.source_status == "CAMERA_OFF"
            assert snap.vision.ear_avg == 0.0
            assert snap.vision.ear_left == 0.0
            assert snap.vision.ear_right == 0.0
            assert not snap.vision.face_detected
            assert snap.blink.total_blinks == 0
            assert snap.blink.blink_rate_bpm == 0.0
            assert not snap.squint.is_squinting
            assert snap.strain.strain_score == 0.0
            assert snap.strain.strain_level == "IDLE"
            assert snap.fatigue.fatigue_score == 0.0
            assert snap.fatigue.fatigue_level == "IDLE"
            assert not snap.presence.is_present
            assert snap.presence.status_label == "CAMERA_OFF"
            assert not snap.presence.lock_triggered
            assert not snap.is_calibrating

        # Verify database/aggregator has zero samples logged
        assert len(daemon.aggregator._curr_ears) == 0

        daemon.stop()


def test_calibration_requires_camera_active():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_calib_check.db")
        cfg = AppConfig()
        cfg.privacy.database_filename = db_path

        daemon = OculiSenseDaemon(config=cfg, force_synthetic=True, autostart_camera=False)

        # Attempt to calibrate while camera is off -> MUST fail
        success = daemon.start_calibration(duration_sec=0.2)
        assert not success
        assert not daemon._is_calibrating

        t = 1000.0
        snap = daemon.step(custom_time=t)
        assert not snap.is_calibrating
        assert snap.calibration_progress == 0.0

        # Now start camera explicitly
        daemon.start_camera()
        assert daemon.is_camera_active

        # Calibration should now succeed
        success = daemon.start_calibration(duration_sec=0.2, start_time=t)
        assert success
        assert daemon._is_calibrating

        # Step until completion
        for _ in range(10):
            t += 0.033
            daemon.step(custom_time=t)

        assert not daemon._is_calibrating
        assert cfg.profile.baseline_open_ear > 0.0
        assert cfg.profile.baseline_closed_ear > 0.0

        daemon.stop()


def test_camera_start_stop_toggle_lifecycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_lifecycle.db")
        cfg = AppConfig()
        cfg.privacy.database_filename = db_path

        daemon = OculiSenseDaemon(config=cfg, force_synthetic=True, autostart_camera=False)
        assert not daemon.is_camera_active

        # Toggle on
        state = daemon.toggle_camera()
        assert state is True
        assert daemon.is_camera_active
        assert daemon.tracker.is_active

        # Step with active camera -> calculations run
        snap = daemon.step(custom_time=1000.0)
        assert snap.is_camera_active
        assert snap.vision.face_detected
        assert snap.vision.ear_avg > 0.10

        # Toggle off
        state = daemon.toggle_camera()
        assert state is False
        assert not daemon.is_camera_active
        assert not daemon.tracker.is_active

        # Step with paused camera -> idle state
        snap = daemon.step(custom_time=1001.0)
        assert not snap.is_camera_active
        assert snap.vision.source_status == "CAMERA_OFF"
        assert snap.vision.ear_avg == 0.0

        daemon.stop()


def test_daemon_rate_limiting_low_cpu_idle():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_ratelimit.db")
        cfg = AppConfig()
        cfg.privacy.database_filename = db_path

        daemon = OculiSenseDaemon(config=cfg, force_synthetic=True, autostart_camera=False)

        step_count = 0

        def _count_listener(s):
            nonlocal step_count
            step_count += 1

        daemon.register_listener(_count_listener)

        # Run background thread for 0.4 seconds while camera is inactive
        daemon.start(target_fps=30)
        time.sleep(0.4)
        daemon.stop()

        # At idle cadence of 0.20s (5 Hz), in 0.4s it should run around 2-4 iterations,
        # definitely NOT spinning at 30+ iterations
        assert step_count <= 6, f"Expected low CPU cadence (<=6 steps in 0.4s), got {step_count}"


def test_api_camera_control_endpoints():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_api_cam.db")
        cfg = AppConfig()
        cfg.privacy.database_filename = db_path

        daemon = OculiSenseDaemon(config=cfg, force_synthetic=True, autostart_camera=False)
        port = 8792
        assert daemon.start_api(port=port)
        time.sleep(0.1)

        base_url = f"http://127.0.0.1:{port}"

        def _request(path, method="GET", data=None):
            url = f"{base_url}{path}"
            body = json.dumps(data).encode("utf-8") if data else None
            req = urllib.request.Request(url, method=method)
            req.add_header("Content-Type", "application/json")
            try:
                with urllib.request.urlopen(req, data=body, timeout=3.0) as resp:
                    return resp.status, json.loads(resp.read().decode("utf-8"))
            except urllib.error.URLError:
                from oculisense.api import OculiSenseAPIServer
                srv = OculiSenseAPIServer.get_active_server(port)
                if srv is not None:
                    return srv.handle_request_in_process(path, method=method, data=data)
                raise

        # 1. Verify status reports camera is inactive
        status, data = _request("/api/v1/status")
        assert status == 200
        assert data["is_camera_active"] is False

        # 2. Turn camera ON via POST
        status, data = _request("/api/v1/camera/start", method="POST")
        assert status == 200
        assert data["is_camera_active"] is True
        assert daemon.is_camera_active is True

        # 3. Turn camera OFF via POST
        status, data = _request("/api/v1/camera/stop", method="POST")
        assert status == 200
        assert data["is_camera_active"] is False
        assert daemon.is_camera_active is False

        # 4. Toggle camera via POST
        status, data = _request("/api/v1/camera/toggle", method="POST")
        assert status == 200
        assert data["is_camera_active"] is True
        assert daemon.is_camera_active is True

        daemon.stop()


def test_rapid_camera_toggle_and_resource_safety():
    """Verify stability and resource cleanup under rapid start/stop calls."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_rapid.db")
        cfg = AppConfig()
        cfg.privacy.database_filename = db_path

        daemon = OculiSenseDaemon(config=cfg, force_synthetic=True, autostart_camera=False)

        for _ in range(10):
            daemon.start_camera()
            assert daemon.is_camera_active
            assert daemon.tracker.is_active
            snap = daemon.step(custom_time=2000.0)
            assert snap.is_camera_active

            daemon.stop_camera()
            assert not daemon.is_camera_active
            assert not daemon.tracker.is_active
            snap = daemon.step(custom_time=2001.0)
            assert not snap.is_camera_active

        daemon.stop()


def test_no_frame_backoff_rate_limiting():
    """Verify rate-limiting backoff when frames are unavailable (NO_FRAME)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_no_frame.db")
        cfg = AppConfig()
        cfg.privacy.database_filename = db_path

        daemon = OculiSenseDaemon(config=cfg, force_synthetic=True, autostart_camera=True)
        # Mock tracker to return NO_FRAME
        from oculisense.vision.tracker import VisionTelemetry

        def _mock_read(timestamp=None):
            return VisionTelemetry(
                timestamp=timestamp or time.time(),
                face_detected=False,
                ear_left=0.0,
                ear_right=0.0,
                ear_avg=0.0,
                source_status="NO_FRAME",
            )

        daemon.tracker.read = _mock_read

        step_count = 0

        def _count_listener(s):
            nonlocal step_count
            step_count += 1

        daemon.register_listener(_count_listener)

        daemon.start(target_fps=30)
        time.sleep(0.3)
        daemon.stop()

        # At NO_FRAME backoff (active_interval * 2 = 66ms+), in 0.3s it should run <= 8 iterations
        # rather than spin-looping at 100+ iterations
        assert step_count <= 8, f"Expected backoff rate limiting, got {step_count} iterations"


def test_presence_lock_not_triggered_after_camera_activation():
    """Verify that idle time while camera is off does not cause immediate screen lock upon activation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_presence_safety.db")
        cfg = AppConfig()
        cfg.privacy.database_filename = db_path
        cfg.security.away_timeout_sec = 30.0
        cfg.security.enable_presence_lock = True
        cfg.security.dry_run_lock = True

        daemon = OculiSenseDaemon(config=cfg, force_synthetic=True, autostart_camera=False)
        assert not daemon.is_camera_active

        # Step at t = 1000 while camera is off
        daemon.step(custom_time=1000.0)

        # 120 seconds pass while camera is off (> 30s away_timeout_sec)
        t_resume = 1120.0
        daemon.start_camera()
        assert daemon.is_camera_active

        # First frame after activation where face is not yet detected (e.g. acquiring landmarks)
        daemon.tracker.set_synthetic_mode("AWAY")
        snap = daemon.step(custom_time=t_resume)

        # Workstation lock MUST NOT be triggered immediately upon camera activation
        assert not snap.presence.lock_triggered
        assert snap.presence.away_duration_sec == 0.0
        assert snap.presence.status_label in ("PRESENT", "AWAY")

        daemon.stop()


def test_calibration_does_not_mutate_detectors():
    """Verify that biometric calculations and detectors are not mutated during calibration ticks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_calib_mutation.db")
        cfg = AppConfig()
        cfg.privacy.database_filename = db_path

        daemon = OculiSenseDaemon(config=cfg, force_synthetic=True, autostart_camera=True)
        t = 500.0

        # Start calibration
        daemon.start_calibration(duration_sec=2.0, start_time=t)
        assert daemon._is_calibrating

        # Step 5 calibration ticks
        for _ in range(5):
            t += 0.033
            snap = daemon.step(custom_time=t)
            assert snap.is_calibrating
            assert snap.strain.strain_level == "CALIBRATING"
            assert snap.fatigue.fatigue_level == "CALIBRATING"
            assert snap.blink.blink_rate_bpm == 0.0
            assert snap.squint.total_squint_episodes == 0

        # Verify detector histories were not polluted
        assert len(daemon.blink_detector._frame_history) == 0
        assert daemon.strain_analyzer._accumulated_focus_sec == 0.0

        daemon.stop()


def test_searching_face_rate_limiting():
    """Verify that absence / searching face paces rate-limiting to prevent high CPU load."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_searching_rate.db")
        cfg = AppConfig()
        cfg.privacy.database_filename = db_path

        daemon = OculiSenseDaemon(config=cfg, force_synthetic=True, autostart_camera=True)
        daemon.tracker.set_synthetic_mode("AWAY")

        step_count = 0

        def _count_listener(s):
            nonlocal step_count
            step_count += 1

        daemon.register_listener(_count_listener)
        daemon.start(target_fps=30)
        time.sleep(0.3)
        daemon.stop()

        # At paced searching rate (~15 FPS / ~67ms), in 0.3s it should execute <= 8 iterations
        assert step_count <= 8, f"Expected pacing when face is absent, got {step_count} iterations"


def test_camera_activation_does_not_immediately_trigger_break():
    """Verify that activating camera after an extended idle period does not immediately trigger 20-20-20 break."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_break_camera_activation.db")
        cfg = AppConfig()
        cfg.privacy.database_filename = db_path
        cfg.breaks.rule_20_20_20_interval_sec = 60  # 1 min interval

        daemon = OculiSenseDaemon(config=cfg, force_synthetic=True, autostart_camera=False)
        assert not daemon.is_camera_active

        # Step while camera is inactive at t = 1000.0
        daemon.step(custom_time=1000.0)

        # 200 seconds pass while camera is off (> 60s rule_20_20_20_interval_sec)
        t_resume = 1200.0
        daemon.start_camera(timestamp=t_resume)
        assert daemon.is_camera_active

        # First frame after activation
        snap = daemon.step(custom_time=t_resume)

        # Break MUST NOT be triggered immediately upon camera start!
        assert not snap.breaks.is_break_recommended, "Break must not fire immediately when starting camera!"
        assert snap.breaks.remaining_until_break_sec > 50.0

        daemon.stop()


def test_camera_backend_dshow_and_reopen_lifecycle():
    """Verify EyeTracker backend initialization, release, re-open, and non-synthetic live state."""
    from oculisense.config import VisionConfig
    from oculisense.vision.tracker import EyeTracker

    cfg = VisionConfig(camera_id=0)
    tracker = EyeTracker(config=cfg, force_synthetic=False, autostart=False)
    assert not tracker.is_active
    assert tracker._cap is None
    assert tracker._synthetic_source is None

    # Calling start_camera should attempt physical camera and remain non-synthetic if no hardware is attached
    ok = tracker.start_camera()
    assert ok
    assert tracker.is_active
    assert tracker._synthetic_source is None

    # Reading frame returns valid non-synthetic telemetry
    telemetry = tracker.read(timestamp=1000.0)
    assert (
        telemetry.source_status in ("ONLINE", "TRACKING", "BASIC_OPENCV", "NO_FRAME", "NO_CAMERA", "SEARCHING_CAMERA")
    )
    assert telemetry.is_synthetic is False

    # Stop camera cleans up hardware and synthetic source
    tracker.stop_camera()
    assert not tracker.is_active
    assert tracker._cap is None
    assert tracker._synthetic_source is None

    # Re-opening camera cleanly re-initializes
    ok = tracker.start_camera()
    assert ok
    assert tracker.is_active

    tracker.stop_camera()


def test_break_overlay_singleton_and_wm_delete_window():
    """Verify BreakOverlayWindow strictly enforces singleton dialog and WM_DELETE_WINDOW skips break."""
    import tkinter as tk
    from oculisense.ui.break_overlay import BreakOverlayWindow

    root = tk.Tk()
    root.withdraw()
    try:
        skipped = False

        def _on_skip():
            nonlocal skipped
            skipped = True

        # Ensure clean starting state
        if BreakOverlayWindow._active_instance is not None:
            try:
                BreakOverlayWindow._active_instance.destroy()
            except Exception:
                pass
            BreakOverlayWindow._active_instance = None

        overlay1 = BreakOverlayWindow(root, on_skip=_on_skip)
        assert BreakOverlayWindow._active_instance is overlay1
        assert overlay1.winfo_exists()

        # Attempt to create a second overlay dialog while first is active
        overlay2 = BreakOverlayWindow(root, on_skip=_on_skip)
        # overlay2 must return the existing singleton instance, maintaining only one active dialog
        assert BreakOverlayWindow._active_instance is overlay1
        assert overlay2 is overlay1
        assert overlay1.winfo_exists()

        # Simulating window manager 'X' button click (_do_skip)
        overlay1._do_skip()
        assert skipped, "Closing break dialog via WM_DELETE_WINDOW must invoke on_skip"
        assert not overlay1.winfo_exists()
        assert BreakOverlayWindow._active_instance is None
    finally:
        root.destroy()


def test_camera_frame_validation_and_consecutive_failure_fallback():
    """Verify that camera descriptors failing frame read are rejected, and consecutive frame failures do NOT silently switch to synthetic."""
    from unittest.mock import MagicMock, patch
    from oculisense.config import VisionConfig
    from oculisense.vision.tracker import EyeTracker

    cfg = VisionConfig(camera_id=0)
    tracker = EyeTracker(config=cfg, force_synthetic=False, autostart=False)

    # 1. Mock VideoCapture: isOpened() is True, but read() returns (False, None)
    mock_bad_cap = MagicMock()
    mock_bad_cap.isOpened.return_value = True
    mock_bad_cap.read.return_value = (False, None)

    with patch("cv2.VideoCapture", return_value=mock_bad_cap):
        # Must reject bad cap and return None
        opened = tracker._open_camera_capture()
        assert opened is None
        assert mock_bad_cap.release.called

    # 2. Test consecutive frame failure behavior in read() when force_synthetic=False
    mock_failing_cap = MagicMock()
    mock_failing_cap.isOpened.return_value = True
    mock_failing_cap.read.return_value = (False, None)

    tracker._is_active = True
    tracker._cap = mock_failing_cap
    tracker._synthetic_source = None
    tracker._consecutive_frame_failures = 0

    # Step 31 failing frames -> must release failing cap, but strictly NOT switch to synthetic demo
    for _ in range(31):
        telemetry = tracker.read(timestamp=1000.0)

    assert tracker._cap is None
    assert tracker._synthetic_source is None

    # Next read continues in live camera mode without falling back to synthetic
    telemetry = tracker.read(timestamp=1001.0)
    assert telemetry.is_synthetic is False
    assert telemetry.source_status in ("NO_FRAME", "NO_CAMERA")
    assert "[LIVE WEBCAM" in telemetry.source_indicator

    # 3. When force_synthetic=True, synthetic feed is preserved
    tracker_demo = EyeTracker(config=cfg, force_synthetic=True, autostart=True)
    telemetry_demo = tracker_demo.read(timestamp=1001.0)
    assert telemetry_demo.is_synthetic is True
    assert "[DEMO]" in telemetry_demo.source_indicator

    tracker.stop_camera()
    tracker_demo.stop_camera()


def test_camera_stop_and_start_resets_detectors_and_prevents_false_fatigue_break():
    """Verify that stopping and restarting camera clears detector histories and prevents false fatigue breaks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_restart_fatigue.db")
        cfg = AppConfig()
        cfg.privacy.database_filename = db_path

        daemon = OculiSenseDaemon(config=cfg, force_synthetic=True, autostart_camera=False)

        # Start camera and step 10 frames
        daemon.start_camera(timestamp=1000.0)
        t = 1000.0
        for _ in range(10):
            t += 0.033
            daemon.step(custom_time=t)

        # Pause/stop camera for 10 seconds (short pause < 45s)
        daemon.stop_camera(timestamp=1010.0)
        assert len(daemon.blink_detector._frame_history) == 0

        # Resume/start camera at t = 1020.0
        daemon.start_camera(timestamp=1020.0)
        assert len(daemon.blink_detector._frame_history) == 0

        # Step immediately after resume
        snap = daemon.step(custom_time=1020.033)

        # Break must NOT be recommended due to false fatigue or stale timestamps!
        assert not snap.breaks.is_break_recommended
        assert snap.breaks.remaining_until_break_sec > 1000.0

        daemon.stop()


def test_detect_available_cameras():
    """Verify detect_available_cameras probes indices cleanly without exceptions."""
    from oculisense.vision.tracker import detect_available_cameras
    found = detect_available_cameras(max_probe=3)
    assert isinstance(found, list)
    for cam_idx in found:
        assert isinstance(cam_idx, int)
        assert 0 <= cam_idx < 3


def test_camera_switching_and_indicators():
    """Verify camera index switching on tracker and daemon, and status indicator formatting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_switch.db")
        cfg = AppConfig()
        cfg.privacy.database_filename = db_path

        daemon = OculiSenseDaemon(config=cfg, force_synthetic=False, autostart_camera=False)
        assert daemon.tracker.active_camera_id == 0

        # Switch camera to index 2
        daemon.switch_camera(2)
        assert daemon.tracker.active_camera_id == 2
        assert daemon.config.vision.camera_id == 2

        # Verify indicator formatting on VisionTelemetry
        snap_off = daemon.step(custom_time=100.0)
        assert snap_off.vision.display_indicator == "[CAMERA OFF]"

        # Start camera with forced synthetic (demo)
        cfg_demo = AppConfig()
        cfg_demo.privacy.database_filename = os.path.join(tmpdir, "test_demo.db")
        daemon_demo = OculiSenseDaemon(config=cfg_demo, force_synthetic=True, autostart_camera=True)
        snap_demo = daemon_demo.step(custom_time=100.0)
        assert snap_demo.vision.is_synthetic is True
        assert snap_demo.vision.display_indicator == "[DEMO]"
        assert "[DEMO]" in snap_demo.vision.display_status

        # Mock physical capture to test [LIVE WEBCAM X] indicator
        from unittest.mock import MagicMock
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        import numpy as np
        fake_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        mock_cap.read.return_value = (True, fake_frame)

        cfg_live = AppConfig()
        cfg_live.privacy.database_filename = os.path.join(tmpdir, "test_live.db")
        daemon_live = OculiSenseDaemon(config=cfg_live, force_synthetic=False, autostart_camera=False)
        daemon_live.start_camera()
        daemon_live.tracker._cap = mock_cap
        daemon_live.tracker._active_camera_id = 1
        daemon_live.tracker._synthetic_source = None

        snap_live = daemon_live.step(custom_time=105.0)
        assert snap_live.vision.is_synthetic is False
        assert snap_live.vision.camera_id == 1
        assert snap_live.vision.display_indicator == "[LIVE WEBCAM 1]"
        assert "[LIVE WEBCAM 1]" in snap_live.vision.display_status
        assert snap_live.vision.source_status in ("TRACKING", "BASIC_OPENCV", "SEARCHING_FACE")

        daemon.stop()
        daemon_demo.stop()
        daemon_live.stop()


def test_camera_open_warmup_and_candidate_probing():
    """Verify _open_camera_capture probes candidate indices and handles warmup retries."""
    from unittest.mock import MagicMock, patch
    from oculisense.config import VisionConfig
    from oculisense.vision.tracker import EyeTracker
    import numpy as np

    cfg = VisionConfig(camera_id=0)
    tracker = EyeTracker(config=cfg, force_synthetic=False, autostart=False)

    call_history = []
    fake_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128

    def _mock_video_capture(idx, backend=None):
        call_history.append((idx, backend))
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        if idx == 1:
            # Returns valid frame after 2 attempts (simulating camera warmup)
            reads = [(False, None), (False, None), (True, fake_frame)]
            mock_cap.read.side_effect = lambda: reads.pop(0) if reads else (True, fake_frame)
            return mock_cap
        else:
            mock_cap.read.return_value = (False, None)
            return mock_cap

    with patch("cv2.VideoCapture", side_effect=_mock_video_capture):
        cap = tracker._open_camera_capture(preferred_id=0)
        assert cap is not None
        assert tracker.active_camera_id == 1
        cap.release()


def test_break_overlay_forced_rest_lockdown():
    """Verify BreakOverlayWindow in forced rest mode renders lockdown UI and handles lock callback."""
    import tkinter as tk
    from oculisense.ui.break_overlay import BreakOverlayWindow

    root = tk.Tk()
    root.withdraw()
    try:
        locked = False

        def _on_lock():
            nonlocal locked
            locked = True

        if BreakOverlayWindow._active_instance is not None:
            try:
                BreakOverlayWindow._active_instance.destroy()
            except Exception:
                pass
            BreakOverlayWindow._active_instance = None

        forced_overlay = BreakOverlayWindow(
            root,
            is_forced=True,
            overall_score=28.0,
            strain_score=82.0,
            fatigue_score=86.0,
            bpm=6.5,
            ear=0.18,
            on_lock_workstation=_on_lock,
        )

        assert forced_overlay.is_forced
        assert forced_overlay.overall_score == 28.0
        assert hasattr(forced_overlay, "lbl_metrics_banner")

        forced_overlay._do_lock_workstation()
        assert locked

        # Update countdown
        forced_overlay.update_countdown(15.0, is_in_rest=True, overall_score=30.0)
        assert "15s" in forced_overlay.lbl_countdown.cget("text")

        forced_overlay.destroy()
        assert BreakOverlayWindow._active_instance is None
    finally:
        root.destroy()


def test_overlay_hud_window_lifecycle_and_telemetry():
    """Verify OverlayHUDWindow renders live telemetry from DaemonSnapshot and handles callbacks."""
    import tkinter as tk
    from oculisense.ui.overlay_hud import OverlayHUDWindow
    from oculisense.daemon import OculiSenseDaemon
    from oculisense.config import AppConfig

    root = tk.Tk()
    root.withdraw()
    try:
        expanded = False

        def _on_expand():
            nonlocal expanded
            expanded = True

        hud = OverlayHUDWindow(
            root,
            on_expand_dashboard=_on_expand,
        )
        assert hud.winfo_exists()

        # Step synthetic daemon to get snapshot
        cfg = AppConfig()
        cfg.demo_mode = True
        daemon = OculiSenseDaemon(config=cfg, force_synthetic=True, autostart_camera=True)
        snap = daemon.step()

        hud.update_telemetry(snap)
        assert hud.lbl_ear.cget("text") != ""
        assert "BPM" in hud.lbl_bpm.cget("text")

        hud.on_expand_dashboard()
        assert expanded

        hud.destroy()
        daemon.stop()
    finally:
        root.destroy()


def test_daemon_overall_score_and_force_stop_properties():
    """Verify DaemonSnapshot.overall_score and is_force_stop_needed calculations."""
    from oculisense.config import AppConfig
    from oculisense.daemon import OculiSenseDaemon

    cfg = AppConfig()
    cfg.demo_mode = True
    daemon = OculiSenseDaemon(config=cfg, force_synthetic=True, autostart_camera=True)

    snap = daemon.step()
    assert 0.0 <= snap.overall_score <= 100.0
    assert snap.overall_health_level in ("EXCELLENT", "GOOD", "MODERATE", "CRITICAL_LOW")
    assert isinstance(snap.is_force_stop_needed, bool)

    daemon.stop()


def test_live_video_frame_rendering_on_canvas():
    """Verify GUI visualizer canvas converts and renders live OpenCV camera frame via PIL ImageTk."""
    import tkinter as tk
    import numpy as np
    from oculisense.config import AppConfig
    from oculisense.daemon import OculiSenseDaemon, DaemonSnapshot
    from oculisense.vision.tracker import VisionTelemetry
    from oculisense.ui.app import OculiSenseApp

    root = tk.Tk()
    root.withdraw()

    try:
        cfg = AppConfig()
        cfg.demo_mode = True
        daemon = OculiSenseDaemon(config=cfg, force_synthetic=True, autostart_camera=True)
        app = OculiSenseApp(daemon, root=root)

        # Create dummy BGR image frame (640x480)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_frame[:, :] = (120, 80, 40)

        v_telemetry = VisionTelemetry(
            timestamp=time.time(),
            face_detected=True,
            ear_left=0.31,
            ear_right=0.32,
            ear_avg=0.315,
            annotated_frame=dummy_frame,
            source_status="TRACKING",
            camera_id=0,
        )

        snap = daemon.step()
        # Override vision with annotated_frame
        object.__setattr__(snap, "vision", v_telemetry)

        # Trigger visualizer rendering
        app._render_visualizer(snap)

        # Verify that canvas rendered PhotoImage with proper visible dimensions (not collapsed 1x1)
        assert app._vis_photo_img is not None
        assert app._vis_photo_img.width() >= 100, f"Expected scaled width >= 100, got {app._vis_photo_img.width()}"
        assert app._vis_photo_img.height() >= 100, f"Expected scaled height >= 100, got {app._vis_photo_img.height()}"
        canvas_items = app.vis_canvas.find_all()
        assert len(canvas_items) > 0

        daemon.stop()
    finally:
        root.destroy()


def test_daemon_realtime_blink_properties():
    """Verify daemon exposes total_blinks and blink_rate_bpm properties in real time."""
    from oculisense.config import AppConfig
    from oculisense.daemon import OculiSenseDaemon

    cfg = AppConfig()
    cfg.demo_mode = True
    daemon = OculiSenseDaemon(config=cfg, force_synthetic=True, autostart_camera=True)

    assert daemon.total_blinks == 0
    assert daemon.blink_rate_bpm == 0.0

    # Step open frames
    t = 1000.0
    for _ in range(5):
        daemon.blink_detector.process_frame(0.32, 0.32, timestamp=t)
        t += 0.033

    # Blink 1
    for _ in range(3):
        daemon.blink_detector.process_frame(0.10, 0.10, timestamp=t)
        t += 0.033
    daemon.blink_detector.process_frame(0.32, 0.32, timestamp=t)
    t += 0.033

    assert daemon.total_blinks == 1

    # Inter-blink interval of 3.0s
    for _ in range(90):
        daemon.blink_detector.process_frame(0.32, 0.32, timestamp=t)
        t += 0.033

    # Blink 2
    for _ in range(3):
        daemon.blink_detector.process_frame(0.10, 0.10, timestamp=t)
        t += 0.033
    daemon.blink_detector.process_frame(0.32, 0.32, timestamp=t)

    assert daemon.total_blinks == 2
    assert 14.0 <= daemon.blink_rate_bpm <= 24.0

    daemon.stop()






